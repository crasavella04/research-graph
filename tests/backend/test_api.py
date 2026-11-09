import json
import shutil
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_graph.backend.app import create_app
from research_graph.backend.config import APISettings
from research_graph.backend.service import Backend
from research_graph.ml.prepare import read_csv, write_csv
from research_graph.ml.train import train


@pytest.fixture
def initial(tmp_path: Path) -> Path:
    texts = {
        "no_issue": "Great accessible application works",
        "low": "Minor font annoyance",
        "medium": "Hard reading small text",
        "high": "Screen reader inaccessible login",
    }
    rows = [
        {
            "id": f"{label}-{i}",
            "group_id": f"{label}-{i}",
            "text": f"{text} distinct{i}",
            "label": label,
        }
        for label, text in texts.items()
        for i in range(6)
    ]
    source = tmp_path / "initial.csv"
    write_csv(source, ("id", "group_id", "text", "label"), rows)
    return source


@pytest.fixture
def settings(tmp_path: Path, initial: Path) -> APISettings:
    return APISettings(artifacts_dir=tmp_path / "artifacts", initial_dataset=initial)


@pytest.fixture
def client(tmp_path: Path, settings: APISettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, tmp_path / "data")) as connection:
        yield connection


@pytest.fixture
def trained(settings: APISettings, initial: Path) -> Path:
    model = settings.artifacts_dir / "baseline-v1"
    train(initial, model)
    return model


def backend(client: TestClient) -> Backend:
    assert isinstance(client.app, FastAPI)
    service: Backend = client.app.state.backend
    return service


def test_no_model_validation_and_cors(client: TestClient) -> None:
    assert client.get("/health").status_code == 503
    assert client.get("/models").json() == []
    assert client.post("/predict", json={"text": "Review"}).status_code == 503
    for text in ("", "   ", "!!!", "x" * 10001):
        assert client.post("/predict", json={"text": text}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert (
        client.post(
            "/feedback", json={"text": "Review", "label": "positive"}
        ).status_code
        == 422
    )
    assert client.post("/train", json={"max_iter": 0}).status_code == 422
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert client.get("/openapi.json").status_code == 200


def test_load_once_and_restart(
    tmp_path: Path, client: TestClient, initial: Path, settings: APISettings
) -> None:
    bundle = tmp_path / "external-model"
    train(initial, bundle)
    files = {
        "model": ("model.joblib", (bundle / "model.joblib").read_bytes()),
        "metadata": ("metadata.json", (bundle / "metadata.json").read_bytes()),
    }
    assert (
        client.post("/load", files={**files, "model": ("bad", b"bad")}).status_code
        == 422
    )
    assert client.get("/health").status_code == 503
    assert client.post("/load", files=files).status_code == 201
    assert client.post("/load", files=files).status_code == 409
    prediction = client.post(
        "/predict", json={"text": "Great accessible application works"}
    )
    assert prediction.status_code == 200
    assert prediction.headers["x-model-version"] == "external-model"
    assert prediction.json()["criticality"] is None
    assert 0 <= prediction.json()["confidence"] <= 1
    with TestClient(create_app(settings, tmp_path / "data")) as restarted:
        assert restarted.get("/health").json()["model_version"] == "external-model"


def test_feedback_upsert_and_synchronous_train(
    client: TestClient, initial: Path
) -> None:
    original = initial.read_bytes()
    first = client.post(
        "/feedback", json={"text": "Brand new inaccessible checkout", "label": "medium"}
    )
    assert first.json()["action"] == "added"
    correction = client.post(
        "/feedback",
        json={"text": "BRAND NEW inaccessible checkout!!!", "label": "high"},
    )
    assert correction.json()["action"] == "updated"
    assert correction.json()["pending_count"] == 1
    service = backend(client)
    parent = service.store.dataset()
    base = (parent / "reviews.csv").read_bytes()
    # Existing text correction overrides its original label, not an appended duplicate.
    client.post(
        "/feedback",
        json={"text": "Great accessible application works distinct0", "label": "high"},
    )
    response = client.post("/train")
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "completed" and "job_id" not in result
    assert sum(result["split_sizes"].values()) == 25
    assert set(result["metrics"]) == {"validation", "test"}
    assert len(result["predictions"]["test"]) == result["split_sizes"]["test"]
    assert isinstance(result["predictions"]["test"][0]["correct"], bool)
    current = service.store.dataset()
    assert current != parent
    assert service.store.feedback(current) == []
    assert (parent / "reviews.csv").read_bytes() == base
    assert initial.read_bytes() == original
    merged = read_csv(current / "reviews.csv")[1]
    assert len(merged) == 25
    assert sum(row["label"] == "high" for row in merged) == 8
    assert client.get("/health").json()["model_version"] == result["model_version"]
    assert client.get("/models").json()[0]["metrics"] is not None
    assert client.post("/predict", json={"text": "Hello"}).status_code == 200


def test_failure_preserves_model_dataset_feedback_and_allows_retry(
    client: TestClient, trained: Path
) -> None:
    service = backend(client)
    # This fixture creates the model after app startup; emulate the next startup.
    from research_graph.backend.predictor import Predictor

    service.predictor = Predictor(trained)
    client.post("/feedback", json={"text": "New feedback", "label": "low"})
    parent = service.store.dataset()
    pending = (parent / "feedback.csv").read_bytes()
    with patch(
        "research_graph.backend.service.evaluate",
        side_effect=RuntimeError("evaluation failed"),
    ):
        assert client.post("/train").status_code == 500
    assert service.predictor.version == "baseline-v1"
    assert service.store.dataset() == parent
    assert (parent / "feedback.csv").read_bytes() == pending
    assert not service.training_lock.locked()
    assert len(service.store.model_directories()) == 1
    assert client.post("/train").status_code == 201


def test_feedback_during_training_carried_forward(
    client: TestClient, trained: Path
) -> None:
    from research_graph.backend.predictor import Predictor
    from research_graph.ml.train import train as original_train

    service = backend(client)
    service.predictor = Predictor(trained)
    client.post("/feedback", json={"text": "Fresh feedback", "label": "medium"})
    entered, resume = threading.Event(), threading.Event()

    def delayed(*args: Any, **kwargs: Any) -> dict[str, object]:
        entered.set()
        assert resume.wait(15)
        return original_train(*args, **kwargs)

    with (
        patch("research_graph.backend.service.train", side_effect=delayed),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = pool.submit(client.post, "/train")
        try:
            assert entered.wait(10)
            assert client.get("/health").json()["training"] is True
            assert client.get("/models").status_code == 200
            assert (
                client.post("/predict", json={"text": "Great app"}).status_code == 200
            )
            assert (
                client.post(
                    "/feedback", json={"text": "Fresh feedback", "label": "high"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/feedback", json={"text": "Another new review", "label": "low"}
                ).status_code
                == 200
            )
        finally:
            resume.set()
        assert future.result(timeout=15).status_code == 201
    pending = service.store.feedback(service.store.dataset())
    assert {row["label"] for row in pending} == {"high", "low"}


def test_rollback_after_publication_error(client: TestClient) -> None:
    service = backend(client)
    client.post("/feedback", json={"text": "New review", "label": "low"})
    parent = service.store.dataset()
    with patch.object(service.store, "commit", side_effect=OSError("disk error")):
        assert client.post("/train").status_code == 500
    assert service.store.dataset() == parent
    assert service.store.model_directories() == []
    assert list(service.store.datasets.iterdir()) == [parent]
    assert len(service.store.feedback(parent)) == 1


def test_missing_initial_and_upload_limit(tmp_path: Path) -> None:
    config = APISettings(artifacts_dir=tmp_path / "models", max_model_bytes=2)
    with TestClient(create_app(config, tmp_path / "data")) as connection:
        assert (
            connection.post(
                "/feedback", json={"text": "Text", "label": "low"}
            ).status_code
            == 404
        )
        assert connection.post("/train").status_code == 404
        assert (
            connection.post(
                "/load", files={"model": ("model", b"123"), "metadata": ("meta", b"{}")}
            ).status_code
            == 413
        )


def test_startup_latest_model(
    tmp_path: Path, initial: Path, settings: APISettings, trained: Path
) -> None:
    latest = settings.artifacts_dir / "latest-model"
    shutil.copytree(trained, latest)
    metadata = json.loads((latest / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_version"] = "latest-model"
    (latest / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with TestClient(create_app(settings, tmp_path / "data")) as connection:
        assert connection.get("/health").json()["model_version"] == "latest-model"


def test_upload_then_train_without_local_initial_file(
    client: TestClient, initial: Path
) -> None:
    service = backend(client)
    service.store.initial = initial.with_name("missing.csv")
    content = initial.read_bytes()
    uploaded = client.post(
        "/upload-dataset", files={"file": ("reviews.csv", content, "text/csv")}
    )
    assert uploaded.status_code == 201, uploaded.text
    info = uploaded.json()
    assert info["rows"] == 24
    assert info["labels"] == {
        label: 6 for label in ("no_issue", "low", "medium", "high")
    }
    assert (
        client.get("/health").json()["dataset"]["dataset_version"]
        == info["dataset_version"]
    )
    parent = service.store.dataset()
    assert (parent / "source.csv").read_bytes() == content
    response = client.post("/train")
    assert response.status_code == 201, response.text
    assert sum(response.json()["split_sizes"].values()) == 24
    assert service.store.dataset() != parent
    assert client.get("/health").json()["dataset"]["rows"] == 24


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"text,class\nhello,high\n",
        b"text,label\nhello,positive\n",
        b"text,label\n,high\n",
        b"text,label\nhello,high\nHELLO!,low\n",
        b'text,label\n"unclosed,high\n',
        b"text,label\nhello\n",
        b"\xff\xfe\x00",
    ],
)
def test_bad_dataset_upload_preserves_current_data(
    client: TestClient, content: bytes
) -> None:
    service = backend(client)
    client.post("/feedback", json={"text": "Pending feedback", "label": "high"})
    parent = service.store.dataset()
    pending = (parent / "feedback.csv").read_bytes()
    assert (
        client.post("/upload-dataset", files={"file": ("bad.csv", content)}).status_code
        == 422
    )
    assert service.store.dataset() == parent
    assert (parent / "feedback.csv").read_bytes() == pending
    assert list(service.store.datasets.iterdir()) == [parent]


def test_upload_supports_bom_deduplicates_and_defers_class_minimum_to_training(
    client: TestClient,
) -> None:
    content = '\ufefftext,label,extra\n"Hello, world", high ,one\n"HELLO, WORLD!",high,two\n'.encode(
        "utf-8"
    )
    result = client.post("/upload-dataset", files={"file": ("reviews.csv", content)})
    assert result.status_code == 201, result.text
    assert result.json()["rows"] == 1
    assert result.json()["labels"] == {"high": 1}
    assert client.post("/train").status_code == 422
    assert client.get("/health").json()["dataset"]["rows"] == 1


def test_upload_keeps_pending_feedback_and_active_model(
    client: TestClient, initial: Path, trained: Path
) -> None:
    from research_graph.backend.predictor import Predictor

    service = backend(client)
    service.predictor = Predictor(trained)
    client.post(
        "/feedback", json={"text": "Brand new screen reader review", "label": "high"}
    )
    old = service.store.dataset()
    pending = (old / "feedback.csv").read_bytes()
    response = client.post(
        "/upload-dataset", files={"file": ("reviews.csv", initial.read_bytes())}
    )
    assert response.status_code == 201
    assert response.json()["pending_count"] == 1
    assert service.predictor.version == "baseline-v1"
    assert (service.store.dataset() / "feedback.csv").read_bytes() == pending
    assert (old / "feedback.csv").read_bytes() == pending
    result = client.post("/train")
    assert result.status_code == 201, result.text
    assert sum(result.json()["split_sizes"].values()) == 25


def test_dataset_upload_limit_and_failed_commit(
    client: TestClient, initial: Path
) -> None:
    service = backend(client)
    parent = service.store.dataset()
    with patch.object(service.store, "commit", side_effect=OSError("disk error")):
        assert (
            client.post(
                "/upload-dataset", files={"file": ("reviews.csv", initial.read_bytes())}
            ).status_code
            == 500
        )
    assert service.store.dataset() == parent
    assert list(service.store.datasets.iterdir()) == [parent]
    service.settings.max_dataset_bytes = 2
    assert (
        client.post(
            "/upload-dataset", files={"file": ("large.csv", b"123")}
        ).status_code
        == 413
    )
    assert service.store.dataset() == parent


def test_dataset_uploaded_during_training_is_not_overwritten(
    client: TestClient, initial: Path
) -> None:
    from research_graph.ml.train import train as original_train

    entered, resume = threading.Event(), threading.Event()

    def delayed(*args: Any, **kwargs: Any) -> dict[str, object]:
        entered.set()
        assert resume.wait(15)
        return original_train(*args, **kwargs)

    with (
        patch("research_graph.backend.service.train", side_effect=delayed),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = pool.submit(client.post, "/train")
        try:
            assert entered.wait(10)
            uploaded = client.post(
                "/upload-dataset", files={"file": ("reviews.csv", initial.read_bytes())}
            )
            assert uploaded.status_code == 201
        finally:
            resume.set()
        assert future.result(timeout=15).status_code == 409
    service = backend(client)
    assert service.store.dataset().name == uploaded.json()["dataset_version"]
    assert service.store.model_directories() == []
