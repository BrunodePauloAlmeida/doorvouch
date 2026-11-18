"""Allow ``python -m doorvouch``."""

from doorvouch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
