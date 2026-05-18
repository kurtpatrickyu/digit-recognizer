from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request

from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
from model import SimpleCNN

import traceback

# Load model globally
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(
    title="Handwritten Digit Recognizer API",
    version="1.0.0",
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


class ErrorResponse(BaseModel):
    message: str = Field(..., description="Human-readable error message.")


def is_valid_image_file(content: bytes) -> bool:
    signatures = (
        b"\xff\xd8\xff",  # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"GIF87a",
        b"GIF89a",
        b"BM",  # BMP
        b"RIFF",  # WEBP starts with RIFF and includes WEBP later in the header.
        b"II*\x00",  # TIFF little-endian
        b"MM\x00*",  # TIFF big-endian
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
        filter_weight = float(model.conv1.weight[channel].mean().item())
        add_connection("input-pixels", f"conv1-{channel}", filter_weight, "aggregated")

    conv2_abs = model.conv2.weight.detach().abs().mean(dim=(2, 3))
    for target_channel in conv2_channels:
        top_sources = torch.topk(conv2_abs[target_channel], k=2).indices.tolist()
        for source_channel in top_sources:
            connection_weight = float(
                model.conv2.weight[target_channel, source_channel].mean().item()
            )
            add_connection(
                f"conv1-{source_channel}",
                f"conv2-{target_channel}",
                connection_weight,
                "sampled aggregate",
            )

    fc1_weights = model.fc1.weight.detach().view(50, 20, 4, 4)
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

    fc2_weights = model.fc2.weight.detach()
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


def run_inference_with_visualization(tensor: torch.Tensor) -> tuple[torch.Tensor, NetworkVisualization]:
    conv1_out = F.relu(F.max_pool2d(model.conv1(tensor), 2))
    conv2_out = F.relu(F.max_pool2d(model.conv2_drop(model.conv2(conv1_out)), 2))
    flat = conv2_out.view(-1, 320)
    fc1_out = F.relu(model.fc1(flat))
    dropped = F.dropout(fc1_out, training=model.training)
    logits = model.fc2(dropped)
    output = F.log_softmax(logits, dim=1)
    probabilities = torch.exp(output)
    visualization = build_network_visualization(
        tensor,
        conv1_out,
        conv2_out,
        fc1_out,
        probabilities,
    )
    return output, visualization


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
        "The uploaded file is missing, empty, or not a valid image format.",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "The server failed while processing the image or running model inference.",
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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

    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(content))
        
        # Apply transforms and add batch dimension
        tensor = transform(pil_image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output, visualization = run_inference_with_visualization(tensor)
            probabilities = torch.exp(output)
            top_confidences, top_digits = torch.topk(probabilities, k=5, dim=1)

        top_predictions = [
            {
                "digit": int(digit.item()),
                "confidence": float(confidence.item()),
            }
            for digit, confidence in zip(top_digits[0], top_confidences[0])
        ]
            
        return {
            "prediction": top_predictions[0]["digit"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "visualization": visualization,
        }
    
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
