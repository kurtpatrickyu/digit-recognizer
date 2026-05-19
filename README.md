# digit-recognizer

## API Contract

The Handwritten Digit Recognizer API is described in [openapi.yaml](openapi.yaml).

The API exposes a single `POST /predict` endpoint. Clients send a
`multipart/form-data` request with one required binary file field named
`image`, representing a single 28x28 pixel handwritten digit image.

Successful prediction responses return JSON with:

- `prediction`: integer digit from 0 to 9
- `confidence`: model confidence from 0.0 to 1.0
- `model_name`: selected model used for inference, including `none` for the untrained baseline
- `top_predictions`: five ranked digit-confidence pairs from highest to lowest confidence

Error responses use a standard JSON object with a required `message` string.
The API returns `400 Bad Request` for missing, empty, or invalid image uploads,
and `500 Internal Server Error` for server-side processing or PyTorch inference
failures.

## Setup

Create and activate a virtual environment from the project root:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Install the runtime dependencies:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running The API

You can start the API before training and use the Train tab in the browser to
create checkpoints. If you already have a compatible root `mnist_cnn.pth` file,
the API makes it available as `model/digit-recognizer-v1.pth` on startup when no
versioned checkpoint exists. New UI and terminal training runs store versioned
checkpoints in `model/` using names like `digit-recognizer-v1.pth`.

Start the server with:

```cmd
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` to choose Predict or Train. The Predict tab lets
you pick `none` or a versioned model, draw a digit on the canvas, and get live
predictions from the `/predict` API, including the top five confidence scores.
The API documentation remains available at `http://127.0.0.1:8000/docs`.

The drawing UI predicts automatically after you pause drawing. The 28x28 preview
defaults to dark digit pixels on a white background and includes an invert toggle
for light pixels on black. The image sent to `/predict` matches the displayed
preview polarity. The page also has tabs for prediction details and a responsive
`SimpleCNN` graph with visible sampled nodes, checkpoint-derived sampled or
aggregated weights, and activation lighting after each successful prediction.
Weight labels are intentionally bounded so the UI shows real model values
without dumping every raw checkpoint parameter.

The `none` model is a deterministic untrained `SimpleCNN` baseline. It is useful
for comparison and is not saved as a checkpoint file.

The Train tab lets you choose the source model and epoch count for a background
MNIST training job. Selecting a model updates the tab with that checkpoint's
latest persisted training stats, including cumulative epochs and any saved loss
or accuracy metrics. During a run, the tab shows live loss and accuracy metrics,
the lowest-confidence representative image from each batch, a correct/incorrect
prediction indicator between the sample label and prediction, `Epoch X/Y`
progress, and the source model being extended. Its graph uses the same
layer-oriented style as the Predict visualization while showing 10 sampled nodes
per eligible layer, sampled current-stage weights, activations/confidences, and
optimizer step deltas. Each completed run saves to the next unused
`model/digit-recognizer-vX.pth` version without overwriting older models.

## Optional Terminal Training

The Train tab is the normal training workflow because it lets you choose the
source model, set the epoch count, and watch live metrics in the UI.

For a terminal-only diagnostic run or a longer unattended run, use:

```cmd
python train.py
```

Training prints train loss, MNIST test loss, and MNIST test accuracy after each
epoch. The best checkpoint from the run is saved to the next available
`model/digit-recognizer-vX.pth` path, so previous versioned checkpoints remain
available for comparison in the UI. Terminal training still uses the default
multi-epoch setting; the UI Train tab uses the epoch count you enter.
