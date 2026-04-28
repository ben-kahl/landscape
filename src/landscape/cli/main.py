from __future__ import annotations

import argparse
import asyncio
import sys

from landscape.cli.runtime import _apply_cli_process_defaults


def _build_parser() -> argparse.ArgumentParser:
    _apply_cli_process_defaults()
    from landscape.cli import auth, graph, ingest, query, seed, status, wipe

    parser = argparse.ArgumentParser(
        prog="landscape",
        description="Landscape local memory CLI",
        epilog=(
            "Examples:\n"
            "  landscape ingest notes.md\n"
            "  landscape ingest-dir ./docs --glob '*.md'\n"
            "  landscape query 'Who leads the project using PostgreSQL?'\n"
            "  landscape status --verbose\n"
            "  landscape auth list-clients    (shows OAuth clients that have connected)\n"
            "  landscape auth disable-client --client-id <id>\n"
            "\nRun `landscape <command> --help` for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest.register(subparsers)
    query.register(subparsers)
    graph.register(subparsers)
    status.register(subparsers)
    seed.register(subparsers)
    wipe.register(subparsers)
    auth.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = args.func(args)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
