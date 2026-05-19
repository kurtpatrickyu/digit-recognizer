from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from model import SimpleCNN


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
LEGACY_MODEL_PATH = BASE_DIR / "mnist_cnn.pth"
MODEL_PREFIX = "digit-recognizer-v"
MODEL_SUFFIX = ".pth"
MODEL_PATTERN = re.compile(r"^digit-recognizer-v(\d+)\.pth$")
BASE_MODEL_NAME = "none"
METADATA_PATH = MODEL_DIR / "metadata.json"


@dataclass(frozen=True)
class ModelRecord:
    name: str
    filename: str
    version: int | None
    path: Path | None
    epochs: int = 0
    train_loss: float | None = None
    test_loss: float | None = None
    test_accuracy: float | None = None
    best_accuracy: float | None = None
    is_base: bool = False


def ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR


def model_name(version: int) -> str:
    return f"{MODEL_PREFIX}{version}"


def checkpoint_filename(version: int) -> str:
    return f"{model_name(version)}{MODEL_SUFFIX}"


def parse_version(path: Path) -> int | None:
    match = MODEL_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_models() -> list[ModelRecord]:
    ensure_model_dir()
    metadata = load_metadata()
    records = []
    for path in MODEL_DIR.glob(f"{MODEL_PREFIX}*{MODEL_SUFFIX}"):
        version = parse_version(path)
        if version is None:
            continue
        name = model_name(version)
        entry = metadata.get(name, {})
        records.append(
            ModelRecord(
                name=name,
                filename=path.name,
                version=version,
                path=path,
                epochs=int(entry.get("epochs", 0)),
                train_loss=optional_float(entry.get("train_loss")),
                test_loss=optional_float(entry.get("test_loss")),
                test_accuracy=optional_float(entry.get("test_accuracy")),
                best_accuracy=optional_float(entry.get("best_accuracy")),
            )
        )
    return sorted(records, key=lambda record: record.version)


def base_model_record() -> ModelRecord:
    return ModelRecord(
        name=BASE_MODEL_NAME,
        filename="",
        version=None,
        path=None,
        epochs=0,
        is_base=True,
    )


def list_model_choices() -> list[ModelRecord]:
    return [base_model_record(), *list_models()]


def find_model(name: str) -> ModelRecord | None:
    if name == BASE_MODEL_NAME:
        return base_model_record()
    for record in list_models():
        if record.name == name:
            return record
    return None


def latest_model() -> ModelRecord | None:
    models = list_models()
    return models[-1] if models else None


def next_version() -> int:
    latest = latest_model()
    return 1 if latest is None else latest.version + 1


def next_checkpoint_path() -> Path:
    ensure_model_dir()
    return MODEL_DIR / checkpoint_filename(next_version())


def load_metadata() -> dict:
    ensure_model_dir()
    if not METADATA_PATH.exists():
        return {}
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_metadata(metadata: dict) -> None:
    ensure_model_dir()
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def update_model_epochs(name: str, epochs: int) -> None:
    update_model_training_stats(name, epochs=epochs)


def update_model_training_stats(
    name: str,
    *,
    epochs: int,
    train_loss: float | None = None,
    test_loss: float | None = None,
    test_accuracy: float | None = None,
    best_accuracy: float | None = None,
) -> None:
    if name == BASE_MODEL_NAME:
        return
    metadata = load_metadata()
    entry = metadata.get(name, {})
    entry["epochs"] = int(epochs)
    for key, value in {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_accuracy": best_accuracy,
    }.items():
        if value is not None:
            entry[key] = float(value)
    metadata[name] = entry
    save_metadata(metadata)


def migrate_legacy_checkpoint() -> ModelRecord | None:
    ensure_model_dir()
    if list_models():
        return latest_model()
    if not LEGACY_MODEL_PATH.exists():
        return None
    target = MODEL_DIR / checkpoint_filename(1)
    if not target.exists():
        shutil.copy2(LEGACY_MODEL_PATH, target)
    return find_model(model_name(1))


def load_model_from_record(record: ModelRecord) -> SimpleCNN:
    if record.is_base:
        return create_base_model()
    if record.path is None:
        raise FileNotFoundError(f"Model '{record.name}' does not have a checkpoint path.")
    model = SimpleCNN()
    state_dict = torch.load(record.path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_base_model() -> SimpleCNN:
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        model = SimpleCNN()
    finally:
        torch.random.set_rng_state(state)
    model.eval()
    return model
