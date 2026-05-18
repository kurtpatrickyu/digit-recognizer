from typing import Any, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request

from pathlib import Path
import torch
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
            output = model(tensor)
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
