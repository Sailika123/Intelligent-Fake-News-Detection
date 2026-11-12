"""Local login and signup using a small SQLite database.

The app does not need a separate database server. A users.db file is created
next to the application the first time an account is used.
"""

import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "users.db"
PBKDF2_ROUNDS = 120_000

def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            display_name TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection

def _hash_password(password: str, salt: bytes) -> str:
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    return derived.hex()

def create_user(username: str, password: str, display_name: str = ""):
    # Usernames are intentionally flexible. Any non-empty username is allowed,
    # including spaces and special characters. Only surrounding whitespace is removed.
    username = (username or "").strip()
    display_name = (display_name or username).strip()

    if not username:
        return False, "Please enter a username."
    if len(password or "") < 6:
        return False, "Password must contain at least 6 characters."

    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)

    connection = _connect()
    try:
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            (username, password_hash, salt.hex(), display_name or username),
        )
        connection.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username is already registered."
    finally:
        connection.close()

def verify(username: str, password: str) -> bool:
    username = (username or "").strip()

    connection = _connect()
    try:
        row = connection.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        connection.close()

    if not row:
        return False

    saved_hash, salt_hex = row
    candidate = _hash_password(password or "", bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, saved_hash)

def display_name(username: str) -> str:
    username = (username or "").strip()

    connection = _connect()
    try:
        row = connection.execute(
            "SELECT display_name FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        connection.close()

    return row[0] if row else (username or "there")

def ensure_default_user():
    """Keep the original demo account available for local testing."""
    connection = _connect()
    try:
        exists = connection.execute(
            "SELECT 1 FROM users WHERE username = ?",
            ("admin",),
        ).fetchone()
    finally:
        connection.close()

    if not exists:
        create_user("admin", "changeme123", "Sailika")
