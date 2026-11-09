"""FastAPI entry point. Run one Uvicorn worker for this single-user service."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from research_graph.backend.config import APISettings
from research_graph.backend.schemas import (
    DatasetUploadResult,
    FeedbackRequest,
    FeedbackResult,
    Health,
    Prediction,
    TextRequest,
    TrainingRequest,
    TrainingResult,
)
from research_graph.backend.service import (
    Backend,
    DatasetChangedError,
    ModelExistsError,
    TrainingBusyError,
)
from research_graph.config import get_settings
from research_graph.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def create_app(
    settings: APISettings | None = None, data_dir: Path | None = None
) -> FastAPI:
    config = settings or APISettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        general = get_settings()
        setup_logging(general.log_level, general.log_format)
        application.state.backend = Backend(data_dir or general.data_dir, config)
        yield

    api = FastAPI(
        title="Accessibility Criticality API", version="0.1.0", lifespan=lifespan
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Model-Version"],
    )

    def backend() -> Backend:
        service: Backend = api.state.backend
        return service

    @api.get("/health", response_model=Health)
    def health(response: Response) -> dict[str, Any]:
        result = backend().health()
        if not result["model_loaded"]:
            response.status_code = 503
        return result

    @api.post("/predict", response_model=Prediction)
    def predict(payload: TextRequest, response: Response) -> dict[str, Any]:
        predictor = backend().predictor
        if predictor is None:
            raise HTTPException(503, "Model is not loaded")
        try:
            prediction = predictor.predict(payload.text)
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(500, "Prediction failed") from exc
        response.headers["X-Model-Version"] = predictor.version
        return prediction

    @api.get("/models")
    def models() -> list[dict[str, Any]]:
        return backend().models()

    @api.post("/feedback", response_model=FeedbackResult)
    def feedback(payload: FeedbackRequest) -> dict[str, Any]:
        try:
            return backend().feedback(payload.text, payload.label)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            logger.exception("Could not save feedback")
            raise HTTPException(500, "Could not save feedback") from exc

    @api.post("/load", status_code=201)
    def load(
        model: Annotated[UploadFile, File()],
        metadata: Annotated[UploadFile, File()],
    ) -> dict[str, str]:
        model_bytes = model.file.read(config.max_model_bytes + 1)
        metadata_bytes = metadata.file.read(1024 * 1024 + 1)
        if (
            len(model_bytes) > config.max_model_bytes
            or len(metadata_bytes) > 1024 * 1024
        ):
            raise HTTPException(413, "Uploaded files exceed size limits")
        try:
            return backend().load(model_bytes, metadata_bytes)
        except ModelExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            logger.exception("Could not import model bundle")
            raise HTTPException(422, "Invalid or incompatible model bundle") from exc

    @api.post("/upload-dataset", status_code=201, response_model=DatasetUploadResult)
    def upload_dataset(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        content = file.file.read(config.max_dataset_bytes + 1)
        if len(content) > config.max_dataset_bytes:
            raise HTTPException(413, "Dataset exceeds the upload size limit")
        try:
            return backend().upload_dataset(content, file.filename)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            logger.exception("Could not save uploaded dataset")
            raise HTTPException(
                500, "Could not save dataset; previous version preserved"
            ) from exc

    @api.post("/train", response_model=TrainingResult, status_code=201)
    def retrain(payload: TrainingRequest | None = None) -> dict[str, Any]:
        # A normal sync endpoint executes in Starlette's thread pool. The HTTP
        # response waits for the complete result; other endpoints are not locked.
        try:
            return backend().retrain(payload or TrainingRequest())
        except (TrainingBusyError, DatasetChangedError) as exc:
            raise HTTPException(409, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("Training failed; previous model and feedback preserved")
            raise HTTPException(
                500, "Training failed; previous model and feedback preserved"
            ) from exc

    return api


app = create_app()
