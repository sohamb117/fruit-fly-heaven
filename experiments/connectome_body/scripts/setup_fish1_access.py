"""Configure Fish1 CAVE access from a private, interactive Terminal prompt.

Run with: uv run --no-sync --with caveclient==8.2.1 python scripts/setup_fish1_access.py
Use --check later to verify existing credentials without prompting or exporting data.
Use --token-file PATH to import an existing private token file without a prompt.
This helper is outside the frozen experiment implementation.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import warnings
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

SERVER = "https://global.brain-wire-test.org/"
DATASTACK = "fish1_full"
TOKEN_PAGE = SERVER + "sticky_auth/settings/tokens"
NEW_TOKEN_PAGE = SERVER + "auth/api/v1/create_token"
PROJECT = Path(__file__).resolve().parents[1]
ACCESS_REPORT = PROJECT / "data/raw/fish1-access-check.json"


def read_token_file(path):
    """Accept a CAVE JSON download or a plain token without exposing its contents."""
    value = Path(path).read_text().strip()
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("token")
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError("Expected a plain token or a JSON object containing one token")
    return value


def verify_access(client):
    """Read one row per required table, without printing neuron IDs or credentials."""
    print("Checking Fish1 materialization versions...", flush=True)
    materialization = int(client.materialize.most_recent_version())
    required = ("somas", "synapses_axde_label")
    print(f"Checking tables at materialization {materialization}...", flush=True)
    available = client.materialize.get_tables(version=materialization)
    missing = set(required) - set(available)
    if missing:
        raise RuntimeError("Required Fish1 tables are unavailable")
    for table in required:
        print(f"Checking read access to {table}...", flush=True)
        client.materialize.query_table(
            table,
            materialization_version=materialization,
            limit=1,
        )
    return {
        "schema": "fish1-access-check-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "server": SERVER,
        "datastack": DATASTACK,
        "materialization_version": materialization,
        "readable_tables": list(required),
        "scope": "Authentication and one-row read checks only; no full export performed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify locally saved access only")
    mode.add_argument("--token-file", type=Path, help="Import an existing private token file")
    parser.add_argument("--no-browser", action="store_true", help="Use an already open token page")
    args = parser.parse_args()

    from caveclient import CAVEclient
    from caveclient.auth import AuthClient, server_token_filename

    if not args.check:
        if args.token_file:
            token = read_token_file(args.token_file)
        else:
            if not sys.stdin.isatty():
                parser.error("Run setup in your own Terminal; token entry must be interactive")
            print("Sign in with Google at: " + TOKEN_PAGE)
            print("Copy only the value named token. Paste it at the hidden prompt below.")
            print("If the list is empty, create your first token at: " + NEW_TOKEN_PAGE)
            if not args.no_browser:
                webbrowser.open(TOKEN_PAGE)
            with warnings.catch_warnings():
                # Refuse getpass's echoing fallback if a private terminal is unavailable.
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("Fish1 token (hidden; never paste it in chat): ").strip()
        if not token or any(char.isspace() for char in token) or token[:1] in ("'", '"'):
            parser.error("Enter only the token value, without quotes or JSON")
        destination = Path(server_token_filename(SERVER))
        previous_mask = os.umask(0o077)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if destination.exists():
                destination.chmod(0o600)
            auth = AuthClient(server_address=SERVER)
            auth.save_token(
                token=token,
                token_file=str(destination),
                overwrite=True,
                write_to_server_file=False,
                local_server=False,
                ignore_readonly=False,
            )
            destination.chmod(0o600)
        finally:
            os.umask(previous_mask)
        del token
        print("Saved a private credential file for the Fish1 server.", flush=True)

    auth = AuthClient(server_address=SERVER)
    if not auth.token:
        print("No local CAVE credential found. Run this script without --check.", file=sys.stderr)
        return 2
    print("Connecting to Fish1...", flush=True)
    client = CAVEclient(datastack_name=DATASTACK, server_address=SERVER)
    report = verify_access(client)
    ACCESS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACCESS_REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, ACCESS_REPORT)
    print("Fish1 access verified: somas and synapses are readable.")
    print(f"Current materialization: {report['materialization_version']}")
    print(f"Non-secret verification record: {ACCESS_REPORT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        # Remote response bodies and request objects can contain credentials.
        # Return only the exception class and HTTP status for diagnosis.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        description = type(exc).__name__ + (f", HTTP {status}" if status else "")
        print(f"Fish1 setup/check failed ({description}).", file=sys.stderr)
        if status in (401, 403):
            print("The token or account does not yet have Fish1 read access.", file=sys.stderr)
        print("You can share this error status in chat; do not share the token.", file=sys.stderr)
        raise SystemExit(1) from None
