from __future__ import annotations

import copy
import io
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.requests import Request
from torchvision import transforms

from model import SimpleCNN
from model_store import (
    BASE_MODEL_NAME,
    ModelRecord,
    find_model,
    latest_model,
    list_model_choices,
    load_model_from_record,
    migrate_legacy_checkpoint,
    next_checkpoint_path,
    update_model_training_stats,
)
from train import train as run_training


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class ActiveModelState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model: Optional[SimpleCNN] = None
        self.record: Optional[ModelRecord] = None

    def load(self, record: ModelRecord) -> None:
        loaded_model = load_model_from_record(record)
        with self.lock:
            self.model = loaded_model
            self.record = record

    def snapshot(self) -> tuple[Optional[SimpleCNN], Optional[ModelRecord]]:
        with self.lock:
            return self.model, self.record


active_model = ActiveModelState()
training_lock = threading.RLock()
training_thread: Optional[threading.Thread] = None
training_state: Dict[str, Any] = {
    "status": "idle",
    "message": "Training has not started.",
    "epoch": None,
    "epochs": None,
    "batch": None,
    "total_batches": None,
    "images_trained": None,
    "total_training_images": None,
    "batch_loss": None,
    "train_loss": None,
    "test_loss": None,
    "test_accuracy": None,
    "best_accuracy": None,
    "saved_model": None,
    "source_model": None,
    "source_model_epochs": 0,
    "total_epochs": 0,
    "batch_image": None,
    "training_prediction": None,
    "training_visualization": None,
    "history": [],
}


app = FastAPI(
    title="Handwritten Digit Recognizer API",
    version="1.1.0",
    description=(
        "API for predicting the digit shown in a 28x28 pixel handwritten digit "
        "image. Clients upload a single image file and receive the model's "
        "predicted digit with confidence."
    ),
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TopPrediction(BaseModel):
    digit: int = Field(..., ge=0, le=9, description="Predicted digit class.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for this digit class.",
    )


class VisualizationLayer(BaseModel):
    id: str = Field(..., description="Stable layer identifier.")
    label: str = Field(..., description="Human-readable layer label.")
    kind: str = Field(..., description="Layer type or display group.")
    detail: str = Field(..., description="Short description of what the layer represents.")
    activation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized activation intensity for this layer in the current inference.",
    )


class VisualizationNode(BaseModel):
    id: str = Field(..., description="Stable node identifier.")
    layer: str = Field(..., description="Layer identifier this node belongs to.")
    label: str = Field(..., description="Node display label.")
    kind: str = Field(..., description="Node type.")
    activation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized activation or confidence intensity for this node.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Output confidence for digit nodes.",
    )
    source_index: Optional[int] = Field(
        default=None,
        description="Original model channel, unit, or digit index represented by this sampled node.",
    )


class VisualizationConnection(BaseModel):
    id: str = Field(..., description="Stable connection identifier.")
    source: str = Field(..., description="Source node identifier.")
    target: str = Field(..., description="Target node identifier.")
    weight: float = Field(..., description="Checkpoint-derived sampled or aggregated weight value.")
    weight_label: str = Field(..., description="Short formatted weight label for display.")
    weight_kind: str = Field(
        ...,
        description="Whether the visible weight is exact, sampled, or aggregated.",
    )
    intensity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized activation intensity for lighting this connection.",
    )


class NetworkVisualization(BaseModel):
    layers: List[VisualizationLayer] = Field(..., description="Visible neural network layer groups.")
    nodes: List[VisualizationNode] = Field(..., description="Visible sampled neural network nodes.")
    connections: List[VisualizationConnection] = Field(
        ...,
        description="Visible sampled or aggregated weighted graph connections.",
    )
    note: str = Field(..., description="Clarifies how sampled and aggregated values are displayed.")


class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=9, description="Predicted digit class.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the predicted class.",
    )
    model_name: str = Field(..., description="Model version used for this prediction.")
    top_predictions: List[TopPrediction] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="Top five digit classes ordered from highest confidence to lowest.",
    )
    visualization: NetworkVisualization = Field(
        ...,
        description="Bounded graph metadata for rendering model weights and activation lighting.",
    )


class ModelMetadata(BaseModel):
    name: str = Field(..., description="Model version name.")
    filename: str = Field(..., description="Checkpoint filename.")
    version: Optional[int] = Field(default=None, ge=1, description="Integer model version.")
    active: bool = Field(..., description="Whether this model is active for prediction.")
    epochs: int = Field(default=0, ge=0, description="Cumulative trained epochs.")
    train_loss: Optional[float] = Field(default=None, ge=0.0, description="Latest persisted train loss.")
    test_loss: Optional[float] = Field(default=None, ge=0.0, description="Latest persisted MNIST test loss.")
    test_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Latest persisted MNIST test accuracy.",
    )
    best_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Best persisted MNIST test accuracy for this model.",
    )
    is_base: bool = Field(default=False, description="Whether this is the untrained base model.")


class ModelListResponse(BaseModel):
    models: List[ModelMetadata]
    active_model: Optional[str] = None


class SelectModelRequest(BaseModel):
    name: str = Field(..., description="Model version name to activate.")


class SelectModelResponse(BaseModel):
    active_model: str
    models: List[ModelMetadata]


class TrainingRequest(BaseModel):
    epochs: int = Field(default=1, ge=1, le=100, description="Number of epochs to train.")
    model_name: Optional[str] = Field(
        default=None,
        description="Model name to continue training from. Defaults to the active model.",
    )


class TrainingStatusResponse(BaseModel):
    status: str
    message: str
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    batch: Optional[int] = None
    total_batches: Optional[int] = None
    images_trained: Optional[int] = None
    total_training_images: Optional[int] = None
    batch_loss: Optional[float] = None
    train_loss: Optional[float] = None
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    best_accuracy: Optional[float] = None
    saved_model: Optional[str] = None
    source_model: Optional[str] = None
    source_model_epochs: int = 0
    total_epochs: int = 0
    batch_image: Optional[Dict[str, Any]] = None
    training_prediction: Optional[Dict[str, Any]] = None
    training_visualization: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    message: str = Field(..., description="Human-readable error message.")


def model_metadata(record: ModelRecord, active_name: Optional[str]) -> ModelMetadata:
    return ModelMetadata(
        name=record.name,
        filename=record.filename,
        version=record.version,
        active=record.name == active_name,
        epochs=record.epochs,
        train_loss=record.train_loss,
        test_loss=record.test_loss,
        test_accuracy=record.test_accuracy,
        best_accuracy=record.best_accuracy,
        is_base=record.is_base,
    )


def model_list_response() -> ModelListResponse:
    _, record = active_model.snapshot()
    active_name = record.name if record else None
    return ModelListResponse(
        models=[model_metadata(item, active_name) for item in list_model_choices()],
        active_model=active_name,
    )


def initialize_active_model() -> None:
    migrate_legacy_checkpoint()
    record = latest_model()
    active_model.load(record if record else find_model(BASE_MODEL_NAME))


@app.on_event("startup")
async def startup() -> None:
    initialize_active_model()


def is_valid_image_file(content: bytes) -> bool:
    signatures = (
        b"\xff\xd8\xff",
        b"\x89PNG\r\n\x1a\n",
        b"GIF87a",
        b"GIF89a",
        b"BM",
        b"RIFF",
        b"II*\x00",
        b"MM\x00*",
    )

    if not content.startswith(signatures):
        return False

    if content.startswith(b"RIFF") and content[8:12] != b"WEBP":
        return False

    return True


def error_response(status_code: int, message: str) -> JSONResponse:
    error = ErrorResponse(message=message)
    content = error.model_dump() if hasattr(error, "model_dump") else error.dict()
    return JSONResponse(
        status_code=status_code,
        content=content,
    )


def normalized_values(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().abs()
    max_value = values.max()
    if float(max_value.item()) <= 0:
        return torch.zeros_like(values)
    return torch.clamp(values / max_value, 0, 1)


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def weight_label(value: float) -> str:
    return f"{value:+.3f}"


def node_activation(activations: Dict[str, float], node_id: str) -> float:
    return clamp_unit(activations.get(node_id, 0.0))


def build_network_visualization(
    inference_model: SimpleCNN,
    tensor: torch.Tensor,
    conv1_out: torch.Tensor,
    conv2_out: torch.Tensor,
    fc1_out: torch.Tensor,
    probabilities: torch.Tensor,
) -> NetworkVisualization:
    conv1_channels = list(range(10))
    conv2_channels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    dense_units = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    input_activation = clamp_unit(float(tensor.abs().mean().item()) / 2.0)
    conv1_activation = normalized_values(conv1_out[0].mean(dim=(1, 2)))
    conv2_activation = normalized_values(conv2_out[0].mean(dim=(1, 2)))
    dense_activation = normalized_values(fc1_out[0])
    output_confidence = probabilities[0].detach().float()

    layers = [
        VisualizationLayer(
            id="input",
            label="Input",
            kind="image",
            detail="28x28 normalized pixels",
            activation=input_activation,
        ),
        VisualizationLayer(
            id="conv1",
            label="Conv1",
            kind="convolution",
            detail="10 filters, 5x5, ReLU + max pool",
            activation=clamp_unit(float(conv1_activation.mean().item())),
        ),
        VisualizationLayer(
            id="conv2",
            label="Conv2",
            kind="convolution",
            detail="20 filters sampled to 10 nodes",
            activation=clamp_unit(float(conv2_activation.mean().item())),
        ),
        VisualizationLayer(
            id="dense",
            label="FC1",
            kind="dense",
            detail="50 hidden units sampled to 10 nodes",
            activation=clamp_unit(float(dense_activation.mean().item())),
        ),
        VisualizationLayer(
            id="output",
            label="Output",
            kind="digits",
            detail="10 digit probabilities",
            activation=clamp_unit(float(output_confidence.max().item())),
        ),
    ]

    nodes: List[VisualizationNode] = [
        VisualizationNode(
            id="input-pixels",
            layer="input",
            label="28x28",
            kind="input",
            activation=input_activation,
            source_index=0,
        )
    ]

    activations: Dict[str, float] = {"input-pixels": input_activation}

    for channel in conv1_channels:
        node_id = f"conv1-{channel}"
        activation = clamp_unit(float(conv1_activation[channel].item()))
        activations[node_id] = activation
        nodes.append(
            VisualizationNode(
                id=node_id,
                layer="conv1",
                label=f"C1.{channel}",
                kind="filter",
                activation=activation,
                source_index=channel,
            )
        )

    for channel in conv2_channels:
        node_id = f"conv2-{channel}"
        activation = clamp_unit(float(conv2_activation[channel].item()))
        activations[node_id] = activation
        nodes.append(
            VisualizationNode(
                id=node_id,
                layer="conv2",
                label=f"C2.{channel}",
                kind="filter",
                activation=activation,
                source_index=channel,
            )
        )

    for unit in dense_units:
        node_id = f"dense-{unit}"
        activation = clamp_unit(float(dense_activation[unit].item()))
        activations[node_id] = activation
        nodes.append(
            VisualizationNode(
                id=node_id,
                layer="dense",
                label=f"H{unit}",
                kind="hidden",
                activation=activation,
                source_index=unit,
            )
        )

    for digit in range(10):
        node_id = f"digit-{digit}"
        confidence = clamp_unit(float(output_confidence[digit].item()))
        activations[node_id] = confidence
        nodes.append(
            VisualizationNode(
                id=node_id,
                layer="output",
                label=str(digit),
                kind="digit",
                activation=confidence,
                confidence=confidence,
                source_index=digit,
            )
        )

    connections: List[VisualizationConnection] = []

    def add_connection(
        source: str,
        target: str,
        weight: float,
        weight_kind: str,
        intensity_scale: float = 1.0,
    ) -> None:
        intensity = clamp_unit(
            (node_activation(activations, source) + node_activation(activations, target))
            * 0.5
            * intensity_scale
        )
        connections.append(
            VisualizationConnection(
                id=f"{source}--{target}",
                source=source,
                target=target,
                weight=float(weight),
                weight_label=weight_label(float(weight)),
                weight_kind=weight_kind,
                intensity=intensity,
            )
        )

    for channel in conv1_channels:
        filter_weight = float(inference_model.conv1.weight[channel].mean().item())
        add_connection("input-pixels", f"conv1-{channel}", filter_weight, "aggregated")

    conv2_abs = inference_model.conv2.weight.detach().abs().mean(dim=(2, 3))
    for target_channel in conv2_channels:
        top_sources = torch.topk(conv2_abs[target_channel], k=2).indices.tolist()
        for source_channel in top_sources:
            connection_weight = float(
                inference_model.conv2.weight[target_channel, source_channel].mean().item()
            )
            add_connection(
                f"conv1-{source_channel}",
                f"conv2-{target_channel}",
                connection_weight,
                "sampled aggregate",
            )

    fc1_weights = inference_model.fc1.weight.detach().view(50, 20, 4, 4)
    fc1_abs = fc1_weights.abs().mean(dim=(2, 3))
    for unit in dense_units:
        sampled_scores = fc1_abs[unit, conv2_channels]
        top_positions = torch.topk(sampled_scores, k=2).indices.tolist()
        for position in top_positions:
            source_channel = conv2_channels[position]
            connection_weight = float(fc1_weights[unit, source_channel].mean().item())
            add_connection(
                f"conv2-{source_channel}",
                f"dense-{unit}",
                connection_weight,
                "sampled aggregate",
            )

    fc2_weights = inference_model.fc2.weight.detach()
    sampled_dense_indexes = torch.tensor(dense_units)
    for digit in range(10):
        sampled_scores = fc2_weights[digit, sampled_dense_indexes].abs()
        top_positions = torch.topk(sampled_scores, k=2).indices.tolist()
        for position in top_positions:
            unit = dense_units[position]
            connection_weight = float(fc2_weights[digit, unit].item())
            add_connection(
                f"dense-{unit}",
                f"digit-{digit}",
                connection_weight,
                "exact sampled",
                intensity_scale=1.25,
            )

    return NetworkVisualization(
        layers=layers,
        nodes=nodes,
        connections=connections,
        note=(
            "Weights shown are real checkpoint values. Convolution and hidden-layer "
            "connections are sampled or aggregated to keep the graph readable; output "
            "connections are exact values for visible sampled hidden units."
        ),
    )


def run_inference_with_visualization(
    inference_model: SimpleCNN,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, NetworkVisualization]:
    conv1_out = F.relu(F.max_pool2d(inference_model.conv1(tensor), 2))
    conv2_out = F.relu(F.max_pool2d(inference_model.conv2_drop(inference_model.conv2(conv1_out)), 2))
    flat = conv2_out.view(-1, 320)
    fc1_out = F.relu(inference_model.fc1(flat))
    dropped = F.dropout(fc1_out, training=inference_model.training)
    logits = inference_model.fc2(dropped)
    output = F.log_softmax(logits, dim=1)
    probabilities = torch.exp(output)
    visualization = build_network_visualization(
        inference_model,
        tensor,
        conv1_out,
        conv2_out,
        fc1_out,
        probabilities,
    )
    return output, visualization


def update_training_state(updates: Dict[str, Any]) -> None:
    with training_lock:
        state_updates = dict(updates)
        event = state_updates.pop("event", None)
        if event == "batch":
            training_state["message"] = "Training batch in progress."
        elif event == "epoch":
            training_state["message"] = "Epoch completed."
            training_state["history"].append(
                {
                    "epoch": state_updates.get("epoch"),
                    "total_epochs": state_updates.get("total_epochs"),
                    "train_loss": state_updates.get("train_loss"),
                    "test_loss": state_updates.get("test_loss"),
                    "test_accuracy": state_updates.get("test_accuracy"),
                    "best_accuracy": state_updates.get("best_accuracy"),
                    "improved": state_updates.get("improved"),
                    "saved_model": state_updates.get("saved_model"),
                }
            )
            training_state["history"] = training_state["history"][-20:]
        elif event == "complete":
            training_state["message"] = "Training complete."
        elif event == "started":
            training_state["message"] = "Training started."
        training_state.update(state_updates)


def training_worker(checkpoint_path: Path, source_record: Optional[ModelRecord], epochs: int) -> None:
    try:
        initial_checkpoint_path = source_record.path if source_record and not source_record.is_base else None
        starting_epoch_count = source_record.epochs if source_record else 0
        result = run_training(
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            initial_checkpoint_path=initial_checkpoint_path,
            starting_epoch_count=starting_epoch_count,
            progress_callback=update_training_state,
        )
        saved_name = result.get("saved_model")
        total_epochs = int(result.get("total_epochs") or starting_epoch_count + epochs)
        if saved_name:
            update_model_training_stats(
                str(saved_name),
                epochs=total_epochs,
                train_loss=optional_float(result.get("train_loss")),
                test_loss=optional_float(result.get("test_loss")),
                test_accuracy=optional_float(result.get("test_accuracy")),
                best_accuracy=optional_float(result.get("best_accuracy")),
            )
            saved_record = find_model(str(saved_name))
            if saved_record:
                active_model.load(saved_record)
        with training_lock:
            training_state.update(
                {
                    "status": "complete",
                    "message": "Training complete.",
                    "best_accuracy": result.get("best_accuracy"),
                    "train_loss": result.get("train_loss"),
                    "test_loss": result.get("test_loss"),
                    "test_accuracy": result.get("test_accuracy"),
                    "saved_model": saved_name,
                    "total_epochs": total_epochs,
                }
            )
    except Exception as exc:
        with training_lock:
            training_state.update(
                {
                    "status": "error",
                    "message": str(exc) or "Training failed.",
                }
            )
        traceback.print_exc()


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reset_training_state(checkpoint_path: Path, source_record: Optional[ModelRecord], epochs: int) -> None:
    starting_epoch_count = source_record.epochs if source_record else 0
    training_state.clear()
    training_state.update(
        {
            "status": "running",
            "message": "Training starting.",
            "epoch": None,
            "epochs": epochs,
            "batch": None,
            "total_batches": None,
            "images_trained": None,
            "total_training_images": None,
            "batch_loss": None,
            "train_loss": source_record.train_loss if source_record else None,
            "test_loss": source_record.test_loss if source_record else None,
            "test_accuracy": source_record.test_accuracy if source_record else None,
            "best_accuracy": source_record.best_accuracy if source_record and source_record.best_accuracy is not None else 0.0,
            "saved_model": checkpoint_path.stem,
            "source_model": source_record.name if source_record else None,
            "source_model_epochs": starting_epoch_count,
            "total_epochs": starting_epoch_count,
            "batch_image": None,
            "training_prediction": None,
            "training_visualization": None,
            "history": [],
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail: Any = exc.detail
    message = detail.get("message") if isinstance(detail, dict) else str(detail)
    return error_response(exc.status_code, message)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return error_response(
        status.HTTP_400_BAD_REQUEST,
        "The request is missing required fields or contains invalid values.",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    traceback.print_exc()
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "The server failed while processing the request.",
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/models", response_model=ModelListResponse, operation_id="listModels")
async def list_available_models() -> ModelListResponse:
    return model_list_response()


@app.post("/models/active", response_model=SelectModelResponse, operation_id="selectModel")
async def select_model(request: SelectModelRequest) -> SelectModelResponse:
    record = find_model(request.name)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.name}' was not found.",
        )
    active_model.load(record)
    response = model_list_response()
    return SelectModelResponse(active_model=record.name, models=response.models)


@app.post("/train", response_model=TrainingStatusResponse, operation_id="startTraining")
async def start_training(request: Optional[TrainingRequest] = Body(default=None)) -> TrainingStatusResponse:
    global training_thread
    training_request = request or TrainingRequest()
    with training_lock:
        if training_state.get("status") == "running" and training_thread and training_thread.is_alive():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Training is already in progress.",
            )
        checkpoint_path = next_checkpoint_path()
        if training_request.model_name:
            source_record = find_model(training_request.model_name)
            if source_record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{training_request.model_name}' was not found.",
                )
        else:
            _, source_record = active_model.snapshot()
            if source_record is None:
                source_record = find_model(BASE_MODEL_NAME)
        reset_training_state(checkpoint_path, source_record, training_request.epochs)
        training_thread = threading.Thread(
            target=training_worker,
            args=(checkpoint_path, source_record, training_request.epochs),
            daemon=True,
        )
        training_thread.start()
        return TrainingStatusResponse(**copy.deepcopy(training_state))


@app.get("/train/status", response_model=TrainingStatusResponse, operation_id="getTrainingStatus")
async def get_training_status() -> TrainingStatusResponse:
    with training_lock:
        return TrainingStatusResponse(**copy.deepcopy(training_state))


@app.post(
    "/predict",
    response_model=PredictResponse,
    operation_id="predictDigit",
    summary="Predict a handwritten digit",
    description=(
        "Accepts a 28x28 pixel image file containing a single handwritten digit "
        "and returns the predicted digit with the model's confidence score."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict_digit(image: UploadFile = File(...)) -> PredictResponse:
    content = await image.read()

    if not content or not is_valid_image_file(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is missing, empty, or not a valid image format.",
        )

    inference_model, record = active_model.snapshot()
    if inference_model is None or record is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model is available. Train a model before predicting.",
        )

    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(content))
        tensor = transform(pil_image).unsqueeze(0)

        with active_model.lock:
            with torch.no_grad():
                output, visualization = run_inference_with_visualization(inference_model, tensor)
                probabilities = torch.exp(output)
                top_confidences, top_digits = torch.topk(probabilities, k=5, dim=1)

        top_predictions = [
            {
                "digit": int(digit.item()),
                "confidence": float(confidence.item()),
            }
            for digit, confidence in zip(top_digits[0], top_confidences[0])
        ]

        return PredictResponse(
            prediction=top_predictions[0]["digit"],
            confidence=top_predictions[0]["confidence"],
            model_name=record.name,
            top_predictions=top_predictions,
            visualization=visualization,
        )

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is missing, empty, or not a valid image format.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The server failed while processing the image or running model inference.",
        ) from exc
