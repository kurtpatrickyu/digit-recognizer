const drawCanvas = document.getElementById("draw-canvas");
const previewCanvas = document.getElementById("preview-canvas");
const clearButton = document.getElementById("clear-button");
const predictionValue = document.getElementById("prediction-value");
const confidenceValue = document.getElementById("confidence-value");
const confidenceMeter = document.getElementById("confidence-meter");
const statusText = document.getElementById("status-text");
const errorText = document.getElementById("error-text");
const topPredictions = document.getElementById("top-predictions");

const drawCtx = drawCanvas.getContext("2d");
const previewCtx = previewCanvas.getContext("2d");
const modelCanvas = document.createElement("canvas");
modelCanvas.width = 28;
modelCanvas.height = 28;
const modelCtx = modelCanvas.getContext("2d", { willReadFrequently: true });

let drawing = false;
let hasInk = false;
let lastPoint = null;
let debounceTimer = null;
let requestId = 0;

function resetCanvas(ctx, canvas) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function resetModelCanvas(ctx, canvas) {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function formatPercent(value) {
  const confidence = Math.max(0, Math.min(1, Number(value)));
  return `${(confidence * 100).toFixed(2)}%`;
}

function resetState() {
  predictionValue.textContent = "-";
  confidenceValue.textContent = "0.00%";
  confidenceMeter.style.width = "0%";
  statusText.textContent = "Ready";
  errorText.textContent = "";
  resetTopPredictions();
}

function resetTopPredictions() {
  topPredictions.innerHTML = '<div class="top-row is-empty">No prediction yet</div>';
}

function renderTopPredictions(items) {
  if (!Array.isArray(items) || items.length === 0) {
    resetTopPredictions();
    return;
  }

  topPredictions.replaceChildren(
    ...items.slice(0, 5).map((item) => {
      const confidence = Math.max(0, Math.min(1, Number(item.confidence)));
      const row = document.createElement("div");
      row.className = "top-row";

      const digit = document.createElement("span");
      digit.className = "top-digit";
      digit.textContent = String(item.digit);

      const track = document.createElement("div");
      track.className = "top-bar-track";

      const bar = document.createElement("div");
      bar.className = "top-bar";
      bar.style.width = `${confidence * 100}%`;
      track.appendChild(bar);

      const score = document.createElement("span");
      score.className = "top-score";
      score.textContent = formatPercent(confidence);

      row.append(digit, track, score);
      return row;
    })
  );
}

function clearAll() {
  resetCanvas(drawCtx, drawCanvas);
  resetModelCanvas(previewCtx, previewCanvas);
  resetModelCanvas(modelCtx, modelCanvas);
  hasInk = false;
  lastPoint = null;
  window.clearTimeout(debounceTimer);
  requestId += 1;
  resetState();
}

function canvasPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function drawStroke(from, to) {
  drawCtx.strokeStyle = "#111820";
  drawCtx.lineWidth = 28;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.beginPath();
  drawCtx.moveTo(from.x, from.y);
  drawCtx.lineTo(to.x, to.y);
  drawCtx.stroke();
}

function updateModelCanvas() {
  resetModelCanvas(modelCtx, modelCanvas);
  modelCtx.globalCompositeOperation = "source-over";
  modelCtx.filter = "invert(1)";
  modelCtx.drawImage(drawCanvas, 0, 0, modelCanvas.width, modelCanvas.height);
  modelCtx.filter = "none";
  resetModelCanvas(previewCtx, previewCanvas);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.drawImage(modelCanvas, 0, 0);
}

function setLoading(isLoading) {
  statusText.textContent = isLoading ? "Predicting" : "Ready";
}

function showEmptyState() {
  resetState();
  statusText.textContent = "Draw first";
}

function schedulePrediction() {
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(() => {
    void predictDigit();
  }, 350);
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/png");
  });
}

async function predictDigit() {
  if (!hasInk) {
    showEmptyState();
    return;
  }

  updateModelCanvas();
  const currentRequest = requestId + 1;
  requestId = currentRequest;
  setLoading(true);
  errorText.textContent = "";

  const blob = await canvasToBlob(modelCanvas);
  if (!blob || currentRequest !== requestId) {
    return;
  }

  const formData = new FormData();
  formData.append("image", blob, "digit.png");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (currentRequest !== requestId) {
      return;
    }

    if (!response.ok) {
      throw new Error(payload.message || "Prediction failed.");
    }

    const confidence = Math.max(0, Math.min(1, Number(payload.confidence)));
    predictionValue.textContent = String(payload.prediction);
    confidenceValue.textContent = formatPercent(confidence);
    confidenceMeter.style.width = `${confidence * 100}%`;
    renderTopPredictions(payload.top_predictions);
    statusText.textContent = "Ready";
  } catch (error) {
    if (currentRequest !== requestId) {
      return;
    }
    predictionValue.textContent = "-";
    confidenceValue.textContent = "0.00%";
    confidenceMeter.style.width = "0%";
    resetTopPredictions();
    statusText.textContent = "Error";
    errorText.textContent = error instanceof Error ? error.message : "Prediction failed.";
  } finally {
    if (currentRequest === requestId && statusText.textContent === "Predicting") {
      setLoading(false);
    }
  }
}

drawCanvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  drawing = true;
  hasInk = true;
  drawCanvas.setPointerCapture(event.pointerId);
  lastPoint = canvasPoint(event);
  drawStroke(lastPoint, lastPoint);
  updateModelCanvas();
  schedulePrediction();
});

drawCanvas.addEventListener("pointermove", (event) => {
  if (!drawing || !lastPoint) {
    return;
  }
  event.preventDefault();
  const nextPoint = canvasPoint(event);
  drawStroke(lastPoint, nextPoint);
  lastPoint = nextPoint;
  updateModelCanvas();
  schedulePrediction();
});

function stopDrawing(event) {
  if (!drawing) {
    return;
  }
  drawing = false;
  lastPoint = null;
  try {
    drawCanvas.releasePointerCapture(event.pointerId);
  } catch (error) {
    // Pointer capture may already be released by the browser.
  }
  schedulePrediction();
}

drawCanvas.addEventListener("pointerup", stopDrawing);
drawCanvas.addEventListener("pointercancel", stopDrawing);
drawCanvas.addEventListener("pointerleave", stopDrawing);

clearButton.addEventListener("click", clearAll);

clearAll();
