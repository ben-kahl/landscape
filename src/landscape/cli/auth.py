"""``landscape auth`` -- local CLI for API client + bearer-secret administration.

Every command operates directly on the local SQLite auth DB at
``settings.auth_db_path``. There is no network path here: the very first
admin credential must be minted on the same host, by an operator with
filesystem access. This is the intentional bootstrap story for Task 5.
"""
from __future__ import annotations

import argparse
import asyncio

from landscape.storage import auth_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("auth", help="Manage API clients and bearer secrets")
    auth_sub = parser.add_subparsers(dest="auth_command", required=True)

    create_client = auth_sub.add_parser("create-client", help="Create a named API client")
    create_client.add_argument("--name", required=True)
    create_client.add_argument(
        "--scope", action="append", required=True, dest="scopes", metavar="SCOPE"
    )
    create_client.add_argument("--description", default=None)
    create_client.set_defaults(func=handle_create_client)

    list_clients = auth_sub.add_parser("list-clients", help="List API clients")
    list_clients.set_defaults(func=handle_list_clients)

    list_secrets = auth_sub.add_parser(
        "list-secrets", help="List secret metadata for a client (no plaintext)"
    )
    list_secrets.add_argument("--client-id", required=True)
    list_secrets.set_defaults(func=handle_list_secrets)

    create_secret = auth_sub.add_parser(
        "create-secret", help="Mint an additional bearer secret on an existing client"
    )
    create_secret.add_argument("--client-id", required=True)
    create_secret.set_defaults(func=handle_create_secret)

    rotate_secret = auth_sub.add_parser(
        "rotate-secret",
        help="Mint a new bearer secret; old secrets stay live until revoked",
    )
    rotate_secret.add_argument("--client-id", required=True)
    rotate_secret.add_argument(
        "--revoke-after",
        type=int,
        default=None,
        metavar="SECONDS",
        help="If set, revoke prior live secrets after this overlap window.",
    )
    rotate_secret.set_defaults(func=handle_rotate_secret)

    revoke_secret = auth_sub.add_parser("revoke-secret", help="Revoke a single bearer secret")
    revoke_secret.add_argument("--secret-id", required=True)
    revoke_secret.set_defaults(func=handle_revoke_secret)

    disable_client = auth_sub.add_parser("disable-client", help="Disable a client")
    disable_client.add_argument("--client-id", required=True)
    disable_client.set_defaults(func=handle_disable_client)

    enable_client = auth_sub.add_parser("enable-client", help="Re-enable a disabled client")
    enable_client.add_argument("--client-id", required=True)
    enable_client.set_defaults(func=handle_enable_client)


async def _ensure_schema() -> None:
    await auth_store.ensure_schema()


def _print_created(created: auth_store.CreatedClientSecret, header: str) -> None:
    """Print a created/rotated secret. The bearer token appears here once and
    nowhere else -- never logged, never re-fetchable from storage."""
    print(header)
    print(f"  client_id:     {created.client_id}")
    print(f"  client_name:   {created.client_name}")
    print(f"  secret_id:     {created.secret_id}")
    print(f"  secret_prefix: {created.secret_prefix}")
    print(f"  scopes:        {', '.join(created.scopes) if created.scopes else '(none)'}")
    print(f"  bearer_token:  {created.bearer_token}")
    print("WARNING: the bearer token will not be shown again. Store it now.")


async def handle_create_client(args: argparse.Namespace) -> int:
    await _ensure_schema()
    created = await auth_store.create_api_client(
        name=args.name,
        scopes=list(args.scopes),
        description=args.description,
    )
    _print_created(created, "Client created.")
    return 0


async def handle_list_clients(_args: argparse.Namespace) -> int:
    await _ensure_schema()
    clients = await auth_store.list_api_clients()
    if not clients:
        print("(no clients)")
        return 0
    for c in clients:
        print(f"client_id:     {c['client_id']}")
        print(f"  name:        {c['name']}")
        if c.get("description"):
            print(f"  description: {c['description']}")
        print(f"  scopes:      {', '.join(c['scopes']) if c['scopes'] else '(none)'}")
        print(f"  status:      {c['status']}")
        print(f"  created_at:  {c['created_at']}")
        print(f"  last_used:   {c['last_used_at'] or '(never)'}")
        print(f"  live secrets: {c['live_secret_count']}")
    return 0


async def handle_list_secrets(args: argparse.Namespace) -> int:
    await _ensure_schema()
    secrets = await auth_store.list_client_secrets(args.client_id)
    if not secrets:
        print(f"(no secrets for client_id={args.client_id})")
        return 0
    for s in secrets:
        print(f"secret_id:     {s['secret_id']}")
        print(f"  prefix:      {s['secret_prefix']}")
        print(f"  created_at:  {s['created_at']}")
        print(f"  expires_at:  {s['expires_at'] or '(never)'}")
        print(f"  revoked_at:  {s['revoked_at'] or '(live)'}")
    return 0


async def handle_create_secret(args: argparse.Namespace) -> int:
    await _ensure_schema()
    created = await auth_store.create_client_secret(args.client_id)
    _print_created(created, "Secret created.")
    return 0


async def handle_rotate_secret(args: argparse.Namespace) -> int:
    await _ensure_schema()
    existing = await auth_store.list_client_secrets(args.client_id)
    prior_live = [s["secret_id"] for s in existing if s["revoked_at"] is None]
    created = await auth_store.rotate_client_secret(args.client_id)
    _print_created(created, "Secret rotated.")

    if args.revoke_after is None:
        if prior_live:
            print(
                f"  prior live secrets: {len(prior_live)} (still live; "
                "revoke explicitly with `auth revoke-secret`)"
            )
        return 0

    if args.revoke_after < 0:
        print("Error: --revoke-after must be >= 0", flush=True)
        return 2
    if args.revoke_after > 0:
        print(f"Waiting {args.revoke_after}s before revoking prior secrets...")
        await asyncio.sleep(args.revoke_after)
    for secret_id in prior_live:
        if secret_id == created.secret_id:
            continue
        await auth_store.revoke_client_secret(secret_id)
        print(f"  revoked prior secret_id: {secret_id}")
    return 0


async def handle_revoke_secret(args: argparse.Namespace) -> int:
    await _ensure_schema()
    await auth_store.revoke_client_secret(args.secret_id)
    print(f"Revoked secret_id={args.secret_id}")
    return 0


async def handle_disable_client(args: argparse.Namespace) -> int:
    await _ensure_schema()
    await auth_store.disable_client(args.client_id)
    print(f"Disabled client_id={args.client_id}")
    return 0


async def handle_enable_client(args: argparse.Namespace) -> int:
    await _ensure_schema()
    await auth_store.enable_client(args.client_id)
    print(f"Enabled client_id={args.client_id}")
    return 0
