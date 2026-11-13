-- Migration 002: encryption metadata table
CREATE TABLE IF NOT EXISTS _encryption_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
