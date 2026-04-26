"""Auth primitives shared by the API surface, MCP server, and CLI.

Task 3 only persists the *records*. Task 4 wires bearer-token verification
into FastAPI/MCP request handling, and Task 5 layers CLI bootstrap and
rotation on top. The pieces here are deliberately small so both follow-on
tasks can import them without circular dependencies.

Bearer token wire format::

    lsk_<secret_id>_<32-byte-urlsafe-material>

* ``lsk_`` is a static prefix that callers can safely log/scan for.
* ``<secret_id>`` is the UUID primary key of the ``client_secrets`` row
  so we can look up the record without scanning every hash.
* ``<material>`` is the actual secret. Only the Argon2 hash of the full
  bearer token (including prefix and secret_id) is persisted; the
  plaintext only exists in the response of ``create_api_client``.
"""
from __future__ import annotations

from dataclasses import dataclass
from secrets import token_urlsafe

from pwdlib import PasswordHash

# pwdlib's recommended() defaults to Argon2id with sane parameters. Reusing a
# single instance avoids re-initializing the hasher on every call.
password_hash = PasswordHash.recommended()


@dataclass(frozen=True)
class AuthContext:
    """The authenticated identity attached to a request.

    ``is_loopback_bypass`` is True only when the caller hit a loopback
    address while ``settings.allow_unauthenticated_loopback`` was on and
    presented no bearer credential. Task 4 reads this flag to log the
    bypass without conflating it with a real credentialed client.
    """

    client_id: str
    client_name: str
    secret_id: str | None
    scopes: frozenset[str]
    is_loopback_bypass: bool = False


def mint_client_secret(secret_id: str) -> tuple[str, str, str]:
    """Generate a new bearer token for ``secret_id``.

    Returns ``(bearer_token, secret_prefix, secret_hash)``:

    * ``bearer_token`` -- the plaintext credential to hand to the caller.
      Callers MUST display this once and discard; we do not persist it.
    * ``secret_prefix`` -- first 18 chars (``lsk_`` + first chunk of the
      secret id). Safe to log/store for rotation UX without disclosing
      the secret material.
    * ``secret_hash`` -- Argon2id hash of the full bearer token; this is
      what gets persisted in the auth store.
    """
    secret_material = token_urlsafe(32)
    bearer_token = f"lsk_{secret_id}_{secret_material}"
    secret_hash = password_hash.hash(bearer_token)
    return bearer_token, bearer_token[:18], secret_hash


def parse_bearer_token(token: str) -> tuple[str, str] | None:
    """Split ``lsk_<secret_id>_<material>`` back into its components.

    Returns ``(secret_id, material)`` or ``None`` when the token does not
    match the expected shape. Centralizing this parser keeps Task 4's
    middleware honest -- there's exactly one place that knows the wire
    format.
    """
    if not token.startswith("lsk_"):
        return None
    rest = token[len("lsk_") :]
    sep = rest.find("_")
    if sep <= 0 or sep == len(rest) - 1:
        return None
    secret_id = rest[:sep]
    material = rest[sep + 1 :]
    if not secret_id or not material:
        return None
    return secret_id, material
