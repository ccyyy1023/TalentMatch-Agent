import json
import os
import tempfile
from pathlib import Path

import pytest

# Tests must never depend on, lock, or mutate the developer/production data
# file. Set the database URL before importing any app module because settings
# are evaluated at import time.
_test_database = Path(tempfile.gettempdir()) / f"talentmatch-tests-{os.getpid()}.db"
os.environ["TALENTMATCH_DATABASE_URL"] = f"sqlite:///{_test_database.as_posix()}"
os.environ["TALENTMATCH_TASK_BACKEND"] = "memory"

from app.schemas import AnalysisRequest


@pytest.fixture(scope="session")
def sample_payload():
    path = Path(__file__).resolve().parents[2] / "data" / "sample_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_request(sample_payload):
    return AnalysisRequest.model_validate(sample_payload)
