"""Tests for encryption module: encrypt/decrypt roundtrip, key derivation, key storage."""

from __future__ import annotations

import os

import pytest

from orbiter_claw.encryption import (
    EncryptionManager,
    derive_key,
    generate_key,
    generate_salt,
    get_encryption_key,
)


class TestKeyDerivation:
    """Tests for PBKDF2 key derivation."""

    def test_derive_key_returns_bytes(self) -> None:
        salt = generate_salt()
        key = derive_key("test-passphrase", salt)
        assert isinstance(key, bytes)

    def test_derive_key_is_44_bytes_base64(self) -> None:
        """Fernet keys are 32 bytes, url-safe base64 encoded = 44 bytes."""
        salt = generate_salt()
        key = derive_key("test-passphrase", salt)
        assert len(key) == 44

    def test_derive_key_deterministic(self) -> None:
        """Same passphrase + salt should produce the same key."""
        salt = generate_salt()
        key1 = derive_key("my-passphrase", salt)
        key2 = derive_key("my-passphrase", salt)
        assert key1 == key2

    def test_derive_key_different_salts(self) -> None:
        """Different salts should produce different keys."""
        salt1 = generate_salt()
        salt2 = generate_salt()
        key1 = derive_key("same-passphrase", salt1)
        key2 = derive_key("same-passphrase", salt2)
        assert key1 != key2

    def test_derive_key_different_passphrases(self) -> None:
        """Different passphrases should produce different keys."""
        salt = generate_salt()
        key1 = derive_key("passphrase-one", salt)
        key2 = derive_key("passphrase-two", salt)
        assert key1 != key2

    def test_generate_salt_is_16_bytes(self) -> None:
        salt = generate_salt()
        assert isinstance(salt, bytes)
        assert len(salt) == 16

    def test_generate_salt_is_random(self) -> None:
        salt1 = generate_salt()
        salt2 = generate_salt()
        assert salt1 != salt2


class TestEncryptionManager:
    """Tests for EncryptionManager encrypt/decrypt."""

    def test_encrypt_decrypt_roundtrip(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        plaintext = "Hello, World!"
        ciphertext = mgr.encrypt(plaintext)
        assert ciphertext != plaintext
        assert mgr.decrypt(ciphertext) == plaintext

    def test_encrypt_decrypt_unicode(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        plaintext = "Bonjour le monde! \u2603 \U0001f600"
        ciphertext = mgr.encrypt(plaintext)
        assert mgr.decrypt(ciphertext) == plaintext

    def test_encrypt_decrypt_empty_string(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        ciphertext = mgr.encrypt("")
        assert mgr.decrypt(ciphertext) == ""

    def test_encrypt_decrypt_long_text(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        plaintext = "x" * 100_000
        ciphertext = mgr.encrypt(plaintext)
        assert mgr.decrypt(ciphertext) == plaintext

    def test_ciphertext_is_different_each_time(self) -> None:
        """Fernet includes a timestamp + IV, so same plaintext produces different ciphertext."""
        key = generate_key()
        mgr = EncryptionManager(key)
        c1 = mgr.encrypt("same text")
        c2 = mgr.encrypt("same text")
        assert c1 != c2

    def test_decrypt_with_wrong_key_raises(self) -> None:
        key1 = generate_key()
        key2 = generate_key()
        mgr1 = EncryptionManager(key1)
        mgr2 = EncryptionManager(key2)
        ciphertext = mgr1.encrypt("secret")
        with pytest.raises(ValueError, match="Decryption failed"):
            mgr2.decrypt(ciphertext)

    def test_decrypt_corrupted_data_raises(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        with pytest.raises(ValueError, match="Decryption failed"):
            mgr.decrypt("not-valid-ciphertext")

    def test_from_passphrase(self) -> None:
        salt = generate_salt()
        mgr = EncryptionManager.from_passphrase("my-passphrase", salt)
        plaintext = "test data"
        ciphertext = mgr.encrypt(plaintext)
        assert mgr.decrypt(ciphertext) == plaintext

    def test_from_passphrase_deterministic(self) -> None:
        """Same passphrase+salt creates managers that can decrypt each other's data."""
        salt = generate_salt()
        mgr1 = EncryptionManager.from_passphrase("shared-pass", salt)
        mgr2 = EncryptionManager.from_passphrase("shared-pass", salt)
        ciphertext = mgr1.encrypt("cross-decrypt")
        assert mgr2.decrypt(ciphertext) == "cross-decrypt"

    def test_from_key(self) -> None:
        key = generate_key()
        mgr = EncryptionManager.from_key(key)
        ciphertext = mgr.encrypt("test")
        assert mgr.decrypt(ciphertext) == "test"


class TestGenerateKey:
    """Tests for Fernet key generation."""

    def test_generate_key_returns_bytes(self) -> None:
        key = generate_key()
        assert isinstance(key, bytes)

    def test_generate_key_is_valid_fernet_key(self) -> None:
        key = generate_key()
        # Should not raise
        EncryptionManager(key)

    def test_generate_key_is_random(self) -> None:
        key1 = generate_key()
        key2 = generate_key()
        assert key1 != key2


class TestGetEncryptionKey:
    """Tests for key retrieval from env var."""

    def test_returns_env_var_key(self) -> None:
        key = generate_key()
        os.environ["ORBITER_CLAW_KEY"] = key.decode("ascii")
        try:
            result = get_encryption_key()
            assert result == key
        finally:
            del os.environ["ORBITER_CLAW_KEY"]

    def test_returns_none_when_no_key(self) -> None:
        os.environ.pop("ORBITER_CLAW_KEY", None)
        result = get_encryption_key()
        assert result is None
