"""Make a consistent SQLite online backup; never copy the live WAL main file."""
import argparse
import os
from pathlib import Path
import sqlite3
import tempfile


def backup(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination).absolute()
    if destination.exists():
        raise ValueError("Refusing to replace an existing backup")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".sqlite-backup-", dir=destination.parent)
    os.close(descriptor)
    try:
        incoming = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        outgoing = sqlite3.connect(temporary)
        try:
            incoming.backup(outgoing)
            outgoing.execute("PRAGMA journal_mode=DELETE")
            if outgoing.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite backup failed its consistency check")
        finally:
            outgoing.close()
            incoming.close()
        os.chmod(temporary, 0o600)
        # Hard linking, instead of rename, preserves no-overwrite semantics.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    backup(args.source, args.destination)
    print(args.destination)
