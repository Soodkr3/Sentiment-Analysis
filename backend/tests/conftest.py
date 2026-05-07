import os
import sys
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def set_working_directory():
    """Change to backend/ so relative model paths resolve correctly."""
    original = os.getcwd()
    os.chdir(BACKEND_DIR)
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    yield
    os.chdir(original)


@pytest.fixture(scope="session")
def client(set_working_directory):
    from enhanced_app import app  # noqa: PLC0415
    with TestClient(app) as c:
        yield c
