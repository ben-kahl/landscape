"""``landscape auth`` -- CLI for OAuth client administration.

Clients self-register via the OAuth dynamic registration endpoint when an
MCP client first connects. These commands let the operator inspect and
manage registered clients.
"""
from __future__ import annotations

import argparse

from landscape.storage import auth_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "auth",
        help="Manage registered OAuth clients",
    )
    auth_sub = parser.add_subparsers(dest="auth_command", required=True)

    list_p = auth_sub.add_parser("list-clients", help="List registered OAuth clients")
    list_p.set_defaults(func=handle_list_clients)

    disable_p = auth_sub.add_parser("disable-client", help="Disable a client (revokes access)")
    disable_p.add_argument("--client-id", required=True)
    disable_p.set_defaults(func=handle_disable_client)

    enable_p = auth_sub.add_parser("enable-client", help="Re-enable a disabled client")
    enable_p.add_argument("--client-id", required=True)
    enable_p.set_defaults(func=handle_enable_client)


async def _ensure_schema() -> None:
    await auth_store.ensure_schema()


async def handle_list_clients(_args: argparse.Namespace) -> int:
    from landscape.config import settings
    from pathlib import Path
    db_path = Path(settings.auth_db_path).expanduser().resolve()
    print(f"# auth_db: {db_path}")
    print("# For Docker deployments run: docker exec <container> uv run landscape auth list-clients")
    print()
    await _ensure_schema()
    clients = await auth_store.list_api_clients()
    if not clients:
        print("(no registered clients)")
        print("Clients appear here after connecting an MCP client (e.g. Claude Code).")
        return 0
    for c in clients:
        print(f"client_id:   {c['client_id']}")
        print(f"  name:      {c['name']}")
        print(f"  scopes:    {', '.join(c['scopes']) if c['scopes'] else '(none)'}")
        print(f"  status:    {c['status']}")
        print(f"  created:   {c['created_at']}")
        print(f"  last_used: {c['last_used_at'] or '(never)'}")
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
