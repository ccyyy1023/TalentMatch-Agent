from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import UserCreate  # noqa: E402
from app.services.auth import AuthService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a TalentMatch internal user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", choices=("admin", "recruiter"), default="recruiter")
    args = parser.parse_args()
    password = os.getenv("TALENTMATCH_INITIAL_PASSWORD") or getpass.getpass("Password (at least 10 characters): ")
    try:
        user = AuthService().create_user(UserCreate(username=args.username, password=password, role=args.role))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Created {user.role} user: {user.username}")


if __name__ == "__main__":
    main()
