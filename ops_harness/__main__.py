"""Allow `python -m ops_harness`."""

import sys

from ops_harness.cli import main

if __name__ == "__main__":
    sys.exit(main())
