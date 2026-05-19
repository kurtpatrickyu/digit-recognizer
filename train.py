from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import SimpleCNN
from model_store import next_checkpoint_path, update_model_training_stats


SEED = 42
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 20
LEARNING_RATE = 0.001
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
TRAIN_CONV1_CHANNELS = list(range(10))
TRAIN_CONV2_CHANNELS = list(range(0, 20, 2))
TRAIN_DENSE_UNITS = list(range(0, 50, 5))

ProgressCallback = Callable[[Dict[str, object]], None]


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            prediction = output.argmax(dim=1)
            correct += prediction.eq(target).sum().item()

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return average_loss, accuracy


def emit(progress_callback: Optional[ProgressCallback], event: Dict[str, object]) -> None:
    if progress_callback:
        progress_callback(event)


def save_checkpoint_safely(model: SimpleCNN, target: Path) -> None:
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(target)


def display_pixels(sample: torch.Tensor) -> list[int]:
    image = sample.detach().cpu()
    if image.ndim == 3:
        image = image[0]
    image = torch.clamp((image * MNIST_STD) + MNIST_MEAN, 0, 1)
    return [int(round(value * 255)) for value in image.flatten().tolist()]


def sample_weight_snapshot(model: SimpleCNN) -> Dict[str, float]:
    with torch.no_grad():
        snapshot = {}
        for conv1_channel in TRAIN_CONV1_CHANNELS:
            snapshot[f"input--conv1-{conv1_channel}"] = float(model.conv1.weight[conv1_channel].mean().item())
        for conv1_channel, conv2_channel in zip(TRAIN_CONV1_CHANNELS, TRAIN_CONV2_CHANNELS):
            snapshot[f"conv1-{conv1_channel}--conv2-{conv2_channel}"] = float(
                model.conv2.weight[conv2_channel, conv1_channel].mean().item()
            )
        for conv2_channel, dense_unit in zip(TRAIN_CONV2_CHANNELS, TRAIN_DENSE_UNITS):
            snapshot[f"conv2-{conv2_channel}--dense-{dense_unit}"] = float(
                model.fc1.weight[dense_unit].view(20, 4, 4)[conv2_channel].mean().item()
            )
        for dense_unit, digit in zip(TRAIN_DENSE_UNITS, range(10)):
            snapshot[f"dense-{dense_unit}--digit-{digit}"] = float(model.fc2.weight[digit, dense_unit].item())
        return snapshot


def normalized_activation(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().abs()
    max_value = values.max()
    if float(max_value.item()) <= 0:
        return torch.zeros_like(values)
    return torch.clamp(values / max_value, 0, 1)


def representative_prediction(model: SimpleCNN, sample: torch.Tensor) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        tensor = sample.unsqueeze(0)
        conv1_out = F.relu(F.max_pool2d(model.conv1(tensor), 2))
        conv2_out = F.relu(F.max_pool2d(model.conv2_drop(model.conv2(conv1_out)), 2))
        flat = conv2_out.view(-1, 320)
        fc1_out = F.relu(model.fc1(flat))
        logits = model.fc2(fc1_out)
        probabilities = torch.exp(F.log_softmax(logits, dim=1))[0].detach().cpu()
        confidence, prediction = torch.max(probabilities, dim=0)

        conv1_activation = normalized_activation(conv1_out[0].mean(dim=(1, 2))).cpu()
        conv2_activation = normalized_activation(conv2_out[0].mean(dim=(1, 2))).cpu()
        dense_activation = normalized_activation(fc1_out[0]).cpu()

    if was_training:
        model.train()

    return {
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item()),
        "input_activation": min(1.0, float(sample.detach().abs().mean().item()) / 2.0),
        "conv1_activation": conv1_activation,
        "conv2_activation": conv2_activation,
        "dense_activation": dense_activation,
        "probabilities": probabilities,
    }


def lowest_confidence_sample(model: SimpleCNN, data: torch.Tensor) -> int:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(data)
        probabilities = torch.exp(output)
        confidences, _ = torch.max(probabilities, dim=1)
        selected = int(torch.argmin(confidences).detach().cpu().item())
    if was_training:
        model.train()
    return selected


def training_visualization(
    previous: Dict[str, float],
    current: Dict[str, float],
    prediction_state: Dict[str, object],
) -> Dict[str, object]:
    deltas = {key: current[key] - previous.get(key, current[key]) for key in current}
    max_delta = max((abs(value) for value in deltas.values()), default=0.0) or 1.0
    conv1_activation = prediction_state["conv1_activation"]
    conv2_activation = prediction_state["conv2_activation"]
    dense_activation = prediction_state["dense_activation"]
    probabilities = prediction_state["probabilities"]
    input_activation = float(prediction_state["input_activation"])
    layers = [
        {
            "id": "input",
            "label": "Input",
            "kind": "image",
            "detail": "28x28 training image",
            "activation": input_activation,
        },
        {
            "id": "conv1",
            "label": "Conv1",
            "kind": "convolution",
            "detail": "sampled filters",
            "activation": float(conv1_activation[TRAIN_CONV1_CHANNELS].mean().item()),
        },
        {
            "id": "conv2",
            "label": "Conv2",
            "kind": "convolution",
            "detail": "sampled filters",
            "activation": float(conv2_activation[TRAIN_CONV2_CHANNELS].mean().item()),
        },
        {
            "id": "dense",
            "label": "FC1",
            "kind": "dense",
            "detail": "sampled hidden units",
            "activation": float(dense_activation[TRAIN_DENSE_UNITS].mean().item()),
        },
        {
            "id": "output",
            "label": "Output",
            "kind": "digits",
            "detail": "current prediction",
            "activation": float(probabilities.max().item()),
        },
    ]
    nodes = [
        {
            "id": "input-pixels",
            "layer": "input",
            "label": "28x28",
            "kind": "input",
            "activation": input_activation,
            "source_index": 0,
        },
    ]
    for conv1_channel in TRAIN_CONV1_CHANNELS:
        nodes.append(
            {
                "id": f"conv1-{conv1_channel}",
                "layer": "conv1",
                "label": f"C1.{conv1_channel}",
                "kind": "filter",
                "activation": float(conv1_activation[conv1_channel].item()),
                "source_index": conv1_channel,
            }
        )
    for conv2_channel in TRAIN_CONV2_CHANNELS:
        nodes.append(
            {
                "id": f"conv2-{conv2_channel}",
                "layer": "conv2",
                "label": f"C2.{conv2_channel}",
                "kind": "filter",
                "activation": float(conv2_activation[conv2_channel].item()),
                "source_index": conv2_channel,
            }
        )
    for dense_unit in TRAIN_DENSE_UNITS:
        nodes.append(
            {
                "id": f"dense-{dense_unit}",
                "layer": "dense",
                "label": f"H{dense_unit}",
                "kind": "hidden",
                "activation": float(dense_activation[dense_unit].item()),
                "source_index": dense_unit,
            }
        )
    for digit in range(10):
        confidence = float(probabilities[digit].item())
        nodes.append(
            {
                "id": f"digit-{digit}",
                "layer": "output",
                "label": str(digit),
                "kind": "digit",
                "activation": confidence,
                "confidence": confidence,
                "source_index": digit,
            }
        )
    source_target = {}
    for conv1_channel in TRAIN_CONV1_CHANNELS:
        source_target[f"input--conv1-{conv1_channel}"] = ("input-pixels", f"conv1-{conv1_channel}")
    for conv1_channel, conv2_channel in zip(TRAIN_CONV1_CHANNELS, TRAIN_CONV2_CHANNELS):
        source_target[f"conv1-{conv1_channel}--conv2-{conv2_channel}"] = (
            f"conv1-{conv1_channel}",
            f"conv2-{conv2_channel}",
        )
    for conv2_channel, dense_unit in zip(TRAIN_CONV2_CHANNELS, TRAIN_DENSE_UNITS):
        source_target[f"conv2-{conv2_channel}--dense-{dense_unit}"] = (
            f"conv2-{conv2_channel}",
            f"dense-{dense_unit}",
        )
    for dense_unit, digit in zip(TRAIN_DENSE_UNITS, range(10)):
        source_target[f"dense-{dense_unit}--digit-{digit}"] = (f"dense-{dense_unit}", f"digit-{digit}")
    connections = []
    for key, value in current.items():
        source, target = source_target[key]
        delta = deltas[key]
        adjustment = min(1.0, abs(delta) / max_delta)
        connections.append(
            {
                "id": key,
                "source": source,
                "target": target,
                "weight": value,
                "weight_label": f"{value:+.3f}",
                "weight_kind": "sampled",
                "previous_weight": previous.get(key, value),
                "delta": delta,
                "delta_label": f"{delta:+.5f}",
                "adjustment": adjustment,
                "direction": "up" if delta >= 0 else "down",
                "intensity": min(1.0, 0.18 + adjustment * 0.82),
            }
        )
    secondary_pairs = []
    for index, conv1_channel in enumerate(TRAIN_CONV1_CHANNELS):
        conv2_channel = TRAIN_CONV2_CHANNELS[(index + 1) % len(TRAIN_CONV2_CHANNELS)]
        secondary_pairs.append((f"conv1-{conv1_channel}", f"conv2-{conv2_channel}"))
    for index, conv2_channel in enumerate(TRAIN_CONV2_CHANNELS):
        dense_unit = TRAIN_DENSE_UNITS[(index + 1) % len(TRAIN_DENSE_UNITS)]
        secondary_pairs.append((f"conv2-{conv2_channel}", f"dense-{dense_unit}"))
    for index, dense_unit in enumerate(TRAIN_DENSE_UNITS):
        secondary_pairs.append((f"dense-{dense_unit}", f"digit-{(index + 1) % 10}"))
    for index, (source, target) in enumerate(secondary_pairs):
        connections.append(
            {
                "id": f"secondary-{index}-{source}--{target}",
                "source": source,
                "target": target,
                "weight": 0.0,
                "weight_label": "",
                "weight_kind": "secondary context",
                "previous_weight": 0.0,
                "delta": 0.0,
                "delta_label": "",
                "adjustment": 0.0,
                "direction": "context",
                "intensity": 0.18,
                "is_secondary": True,
            }
        )
    return {
        "layers": layers,
        "nodes": nodes,
        "connections": connections,
        "note": "Training graph shows bounded sampled weights, activation/confidence values, and optimizer-step deltas for the current batch.",
    }


def train(
    *,
    epochs: int = EPOCHS,
    checkpoint_path: Optional[Path] = None,
    initial_checkpoint_path: Optional[Path] = None,
    starting_epoch_count: int = 0,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, object]:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = checkpoint_path or next_checkpoint_path()
    emit(
        progress_callback,
        {
            "event": "started",
            "status": "running",
            "device": str(device),
            "epochs": epochs,
            "total_epochs": starting_epoch_count,
            "saved_model": output_path.stem,
        },
    )
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    if initial_checkpoint_path is not None:
        state_dict = torch.load(initial_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    print("Downloading dataset...")
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
    )

    best_accuracy = 0.0
    saved_checkpoint = False
    total_batches = len(train_loader)
    latest_train_loss: float | None = None
    latest_test_loss: float | None = None
    latest_test_accuracy: float | None = None

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        images_trained = 0

        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)
            before_weights = sample_weight_snapshot(model)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            after_weights = sample_weight_snapshot(model)
            sample_index = lowest_confidence_sample(model, data)
            selected_sample = data[sample_index]
            sample_image = display_pixels(selected_sample)
            prediction_state = representative_prediction(model, selected_sample)
            running_loss += loss.item() * len(data)
            images_trained = min(images_trained + len(data), len(train_loader.dataset))
            running_train_loss = running_loss / images_trained
            sample_label = int(target[sample_index].detach().cpu().item())
            sample_prediction = int(prediction_state["prediction"])
            is_correct = sample_prediction == sample_label

            should_print = batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches
            if should_print:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{images_trained}/{len(train_loader.dataset)}] "
                    f"Loss: {loss.item():.6f}"
                )
            emit(
                progress_callback,
                {
                    "event": "batch",
                    "status": "running",
                    "epoch": epoch,
                    "epochs": epochs,
                    "total_epochs": starting_epoch_count + epoch - 1,
                    "batch": batch_idx,
                    "total_batches": total_batches,
                    "images_trained": images_trained,
                    "total_training_images": len(train_loader.dataset),
                    "batch_loss": float(loss.item()),
                    "train_loss": float(running_train_loss),
                    "test_loss": latest_test_loss,
                    "test_accuracy": latest_test_accuracy,
                    "best_accuracy": best_accuracy,
                    "batch_image": {
                        "width": 28,
                        "height": 28,
                        "pixels": sample_image,
                        "label": sample_label,
                        "prediction": sample_prediction,
                        "confidence": prediction_state["confidence"],
                        "is_correct": is_correct,
                    },
                    "training_prediction": {
                        "prediction": sample_prediction,
                        "confidence": prediction_state["confidence"],
                        "is_correct": is_correct,
                    },
                    "training_visualization": training_visualization(
                        before_weights,
                        after_weights,
                        prediction_state,
                    ),
                    "saved_model": output_path.stem,
                },
            )

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        latest_train_loss = train_loss
        latest_test_loss = test_loss
        latest_test_accuracy = test_accuracy
        improved = test_accuracy > best_accuracy
        if improved:
            best_accuracy = test_accuracy
            save_checkpoint_safely(model, output_path)
            saved_checkpoint = True

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} test_accuracy={test_accuracy * 100:.2f}%"
        )
        if improved:
            print(f"Saved new best checkpoint to {output_path} ({best_accuracy * 100:.2f}%)")
        else:
            print(f"Best checkpoint unchanged ({best_accuracy * 100:.2f}%)")

        emit(
            progress_callback,
            {
                "event": "epoch",
                "status": "running",
                "epoch": epoch,
                "epochs": epochs,
                "total_epochs": starting_epoch_count + epoch,
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "best_accuracy": float(best_accuracy),
                "improved": improved,
                "saved_model": output_path.stem if saved_checkpoint else None,
            },
        )

    result = {
        "status": "complete",
        "best_accuracy": float(best_accuracy),
        "train_loss": float(latest_train_loss) if latest_train_loss is not None else None,
        "test_loss": float(latest_test_loss) if latest_test_loss is not None else None,
        "test_accuracy": float(latest_test_accuracy) if latest_test_accuracy is not None else None,
        "saved_model": output_path.stem if saved_checkpoint else None,
        "saved_path": str(output_path) if saved_checkpoint else None,
        "total_epochs": starting_epoch_count + epochs,
    }
    if saved_checkpoint:
        update_model_training_stats(
            output_path.stem,
            epochs=starting_epoch_count + epochs,
            train_loss=latest_train_loss,
            test_loss=latest_test_loss,
            test_accuracy=latest_test_accuracy,
            best_accuracy=best_accuracy,
        )
    emit(progress_callback, {"event": "complete", **result})
    print(f"Execution complete. Best test accuracy: {best_accuracy * 100:.2f}%")
    return result


if __name__ == "__main__":
    train()
