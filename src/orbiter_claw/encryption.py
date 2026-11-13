"""AES-256 Fernet encryption with PBKDF2 key derivation.

Provides transparent encrypt/decrypt for the database layer.
Key storage priority: OS keyring > ORBITER_CLAW_KEY env var > prompt on startup.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken

_PBKDF2_ITERATIONS = 600_000
_SALT_LENGTH = 16


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible key from a passphrase using PBKDF2.

    Returns a 32-byte key encoded as url-safe base64 (44 bytes) for Fernet.
    """
    raw_key = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(raw_key)


def generate_salt() -> bytes:
    """Generate a random salt for key derivation."""
    return os.urandom(_SALT_LENGTH)


class EncryptionManager:
    """Manages Fernet encryption/decryption with a derived key."""

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes) -> EncryptionManager:
        """Create an EncryptionManager from a passphrase and salt."""
        key = derive_key(passphrase, salt)
        return cls(key)

    @classmethod
    def from_key(cls, key: bytes) -> EncryptionManager:
        """Create an EncryptionManager from a raw Fernet key."""
        return cls(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string, returning base64-encoded ciphertext."""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext string, returning plaintext."""
        try:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except InvalidToken as err:
            raise ValueError("Decryption failed: invalid key or corrupted data") from err


def get_encryption_key() -> bytes | None:
    """Get encryption key using priority: OS keyring > env var > None.

    Returns a Fernet-compatible key (url-safe base64 encoded 32 bytes) or None.
    The caller should prompt the user if None is returned.
    """
    # 1. Try OS keyring
    try:
        import keyring  # pyright: ignore[reportMissingImports]

        stored = keyring.get_password("orbiter-claw", "encryption-key")
        if stored:
            return stored.encode("ascii")
    except Exception:
        pass

    # 2. Try environment variable
    env_key = os.environ.get("ORBITER_CLAW_KEY")
    if env_key:
        return env_key.encode("ascii")

    return None


def store_encryption_key(key: bytes) -> bool:
    """Store encryption key in OS keyring. Returns True on success."""
    try:
        import keyring  # pyright: ignore[reportMissingImports]

        keyring.set_password("orbiter-claw", "encryption-key", key.decode("ascii"))
        return True
    except Exception:
        return False


def generate_key() -> bytes:
    """Generate a new random Fernet key."""
    return Fernet.generate_key()
