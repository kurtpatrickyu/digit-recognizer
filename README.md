# digit-recognizer

## API Contract

The Handwritten Digit Recognizer API is described in [openapi.yaml](openapi.yaml).

The API exposes a single `POST /predict` endpoint. Clients send a
`multipart/form-data` request with one required binary file field named
`image`, representing a single 28x28 pixel handwritten digit image.

Successful responses return JSON with:

- `prediction`: integer digit from 0 to 9
- `confidence`: model confidence from 0.0 to 1.0
- `top_predictions`: five ranked digit-confidence pairs from highest to lowest confidence

Error responses use a standard JSON object with a required `message` string.
The API returns `400 Bad Request` for missing, empty, or invalid image uploads,
and `500 Internal Server Error` for server-side processing or PyTorch inference
failures.

## Running The API

Train the model before starting the API, or provide a compatible
`mnist_cnn.pth` file in the repository root. The API loads this file at startup
to serve predictions with the `SimpleCNN` model defined in [model.py](model.py).

Start the server with:

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` to draw a digit on the canvas and get live
predictions from the `/predict` API, including the top five confidence scores.
The API documentation remains available at `http://127.0.0.1:8000/docs`.

The drawing UI predicts automatically after you pause drawing. The 28x28 preview
shows the light-on-dark image sent to the model, matching MNIST-style input.

## Training Diagnostics

Run training with:

```bash
python train.py
```

Training prints train loss, MNIST test loss, and MNIST test accuracy after each
epoch. `mnist_cnn.pth` is updated only when test accuracy improves, so the API
serves the best checkpoint from the run rather than the final epoch by default.
