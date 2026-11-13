"""Tests for WebSocket gateway and HTTP endpoints."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from orbiter_claw.app import create_app
from orbiter_claw.gateway import (
    Channel,
    ConnectionManager,
    Envelope,
    MessageType,
)

# ---------------------------------------------------------------------------
# Unit tests for Envelope model
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_envelope_creation(self):
        env = Envelope(channel=Channel.CHAT, type=MessageType.MESSAGE, payload={"text": "hello"})
        assert env.channel == "chat"
        assert env.type == "message"
        assert env.payload == {"text": "hello"}
        assert isinstance(env.timestamp, float)

    def test_envelope_defaults(self):
        env = Envelope(channel=Channel.SYSTEM, type=MessageType.PING)
        assert env.payload == {}
        assert env.timestamp > 0

    def test_envelope_serialization(self):
        env = Envelope(channel=Channel.CHAT, type=MessageType.MESSAGE, payload={"text": "hi"})
        data = env.model_dump()
        assert data["channel"] == "chat"
        assert data["type"] == "message"
        assert "timestamp" in data

    def test_envelope_validation_rejects_bad_channel(self):
        with pytest.raises(ValidationError):
            Envelope.model_validate({"channel": "invalid", "type": "message", "payload": {}})

    def test_envelope_validation_rejects_bad_type(self):
        with pytest.raises(ValidationError):
            Envelope.model_validate({"channel": "chat", "type": "invalid", "payload": {}})


# ---------------------------------------------------------------------------
# Unit tests for ConnectionManager
# ---------------------------------------------------------------------------


class TestConnectionManager:
    def test_fresh_manager_has_no_connections(self):
        mgr = ConnectionManager()
        assert len(mgr.active_connections) == 0


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.anyio
    async def test_health_check(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestConnectionsEndpoint:
    @pytest.mark.anyio
    async def test_list_connections_empty(self, client):
        resp = await client.get("/api/v1/connections")
        assert resp.status_code == 200
        assert resp.json() == {"connections": []}


# ---------------------------------------------------------------------------
# WebSocket tests (using Starlette TestClient for sync WS)
# ---------------------------------------------------------------------------


class TestWebSocket:
    def setup_method(self):
        self.app = create_app()
        self.test_client = TestClient(self.app)

    def test_websocket_connect_receives_welcome(self):
        with self.test_client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["channel"] == "system"
            assert data["type"] == "ack"
            assert "client_id" in data["payload"]
            assert data["payload"]["message"] == "connected"

    def test_websocket_ping_pong(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json({"channel": "system", "type": "ping", "payload": {}})
            resp = ws.receive_json()
            assert resp["type"] == "pong"
            assert resp["channel"] == "system"

    def test_websocket_send_message_gets_ack(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json(
                {
                    "channel": "chat",
                    "type": "message",
                    "payload": {"text": "hello"},
                }
            )
            resp = ws.receive_json()
            assert resp["type"] == "ack"
            assert resp["channel"] == "chat"
            assert resp["payload"]["received"] is True

    def test_websocket_invalid_message_returns_error(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json({"invalid": "data"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid message format" in resp["payload"]["message"]

    def test_websocket_subscribe(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json(
                {
                    "channel": "system",
                    "type": "subscribe",
                    "payload": {"channel": "chat"},
                }
            )
            resp = ws.receive_json()
            assert resp["type"] == "ack"
            assert resp["payload"]["action"] == "subscribed"
            assert resp["payload"]["channel"] == "chat"

    def test_websocket_unsubscribe(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json(
                {
                    "channel": "system",
                    "type": "unsubscribe",
                    "payload": {"channel": "notifications"},
                }
            )
            resp = ws.receive_json()
            assert resp["type"] == "ack"
            assert resp["payload"]["action"] == "unsubscribed"
            assert resp["payload"]["channel"] == "notifications"

    def test_websocket_subscribe_invalid_channel(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json(
                {
                    "channel": "system",
                    "type": "subscribe",
                    "payload": {"channel": "nonexistent"},
                }
            )
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Unknown channel" in resp["payload"]["message"]

    def test_websocket_unsubscribe_invalid_channel(self):
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            ws.send_json(
                {
                    "channel": "system",
                    "type": "unsubscribe",
                    "payload": {"channel": "nonexistent"},
                }
            )
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Unknown channel" in resp["payload"]["message"]

    def test_websocket_broadcast_between_clients(self):
        with self.test_client.websocket_connect("/ws") as ws1:
            ws1.receive_json()  # welcome

            with self.test_client.websocket_connect("/ws") as ws2:
                ws2.receive_json()  # welcome

                # Client 1 sends a chat message
                ws1.send_json(
                    {
                        "channel": "chat",
                        "type": "message",
                        "payload": {"text": "hello from ws1"},
                    }
                )

                # Client 2 should receive the broadcast
                broadcast = ws2.receive_json()
                assert broadcast["channel"] == "chat"
                assert broadcast["type"] == "message"
                assert broadcast["payload"]["text"] == "hello from ws1"

                # Client 1 receives its ACK
                ack = ws1.receive_json()
                assert ack["type"] == "ack"

    def test_websocket_message_routing_performance(self):
        """Message routing overhead should be under 50ms."""
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            start = time.perf_counter()
            ws.send_json(
                {
                    "channel": "chat",
                    "type": "message",
                    "payload": {"text": "benchmark"},
                }
            )
            ws.receive_json()  # ack
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 50, f"Message routing took {elapsed_ms:.1f}ms (limit: 50ms)"

    def test_websocket_multiple_channels(self):
        """Test that messages are routed to the correct channel."""
        with self.test_client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome

            for channel in ["chat", "system", "notifications"]:
                ws.send_json(
                    {
                        "channel": channel,
                        "type": "message",
                        "payload": {"text": f"msg on {channel}"},
                    }
                )
                resp = ws.receive_json()
                assert resp["channel"] == channel
                assert resp["type"] == "ack"
