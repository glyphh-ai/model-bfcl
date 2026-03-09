"""Clear cached local modules between test file collections.

Each class dir has its own intent.py / encoder.py / test_model.py.
Without this, Python's module cache loads the first one for all classes.
"""

import sys


def pytest_collectstart(collector):
    for name in ["intent", "encoder", "test_model"]:
        sys.modules.pop(name, None)
