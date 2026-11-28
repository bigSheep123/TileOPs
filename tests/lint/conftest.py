"""Put ``scripts/lint/`` on the import path, as running one of its scripts does."""

import sys

from tests.lint.lint_probe import LINT_DIR

if str(LINT_DIR) not in sys.path:
    sys.path.insert(0, str(LINT_DIR))
