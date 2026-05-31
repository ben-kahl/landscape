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

    issue_p = auth_sub.add_parser(
        "issue-token",
        help="Mint a bearer token for non-interactive clients (hooks, scripts)",
    )
    issue_p.add_argument(
        "--name", default="local-hook", help="Label for the token's client"
    )
    issue_p.add_argument(
        "--scope",
        action="append",
        dest="scopes",
        help="Scope to grant (repeatable). Default: agent",
    )
    issue_p.add_argument(
        "--expires-days",
        type=float,
        default=None,
        help="Token lifetime in days. Default: no expiry",
    )
    issue_p.set_defaults(func=handle_issue_token)


async def _ensure_schema() -> None:
    await auth_store.ensure_schema()


async def handle_list_clients(_args: argparse.Namespace) -> int:
    from pathlib import Path

    from landscape.config import settings
    db_path = Path(settings.auth_db_path).expanduser().resolve()
    print(f"# auth_db: {db_path}")
    print("# For Docker deployments run: docker exec <container> python3 -m " \
        "landscape.cli auth list-clients")
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


async def handle_issue_token(args: argparse.Namespace) -> int:
    import time

    await _ensure_schema()
    scopes = args.scopes or ["agent"]
    expires_at = (
        time.time() + args.expires_days * 86400.0
        if args.expires_days is not None
        else None
    )
    client_id, access_token = await auth_store.issue_local_token(
        name=args.name, scopes=scopes, expires_at=expires_at
    )
    expiry = "never" if expires_at is None else f"in {args.expires_days:g} day(s)"
    print(f"client_id: {client_id}")
    print(f"scopes:    {', '.join(scopes)}")
    print(f"expires:   {expiry}")
    print()
    print("Access token (shown once — store it now, it cannot be recovered):")
    print(f'  export LANDSCAPE_API_TOKEN="{access_token}"')
    print()
    print(f"Revoke later with: landscape auth disable-client --client-id {client_id}")
    return 0
