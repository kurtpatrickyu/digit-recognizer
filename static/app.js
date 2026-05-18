const drawCanvas = document.getElementById("draw-canvas");
const previewCanvas = document.getElementById("preview-canvas");
const clearButton = document.getElementById("clear-button");
const predictionValue = document.getElementById("prediction-value");
const confidenceValue = document.getElementById("confidence-value");
const confidenceMeter = document.getElementById("confidence-meter");
const statusText = document.getElementById("status-text");
const errorText = document.getElementById("error-text");
const topPredictions = document.getElementById("top-predictions");
const invertPreviewToggle = document.getElementById("invert-preview-toggle");
const workbench = document.querySelector(".workbench");
const previewBlock = document.querySelector(".preview-block");
const networkStatus = document.getElementById("network-status");
const networkGraph = document.getElementById("network-graph");
const networkNote = document.getElementById("network-note");
const tabButtons = Array.from(document.querySelectorAll("[role='tab']"));
const tabPanels = Array.from(document.querySelectorAll("[role='tabpanel']"));

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
let previewInverted = false;
let currentVisualization = null;
let currentPrediction = null;
let selectedNodeId = null;
const svgNamespace = "http://www.w3.org/2000/svg";
const fallbackVisualization = {
  layers: [
    { id: "input", label: "Input", detail: "28x28 pixels", activation: 0 },
    { id: "conv1", label: "Conv1", detail: "10 filters", activation: 0 },
    { id: "conv2", label: "Conv2", detail: "20 filters sampled", activation: 0 },
    { id: "dense", label: "FC1", detail: "50 units sampled", activation: 0 },
    { id: "output", label: "Output", detail: "10 digits", activation: 0 },
  ],
  nodes: [
    { id: "input-pixels", layer: "input", label: "28x28", kind: "input", activation: 0 },
    ...Array.from({ length: 10 }, (_, index) => ({
      id: `conv1-${index}`,
      layer: "conv1",
      label: `C1.${index}`,
      kind: "filter",
      activation: 0,
    })),
    ...Array.from({ length: 10 }, (_, index) => ({
      id: `conv2-${index * 2}`,
      layer: "conv2",
      label: `C2.${index * 2}`,
      kind: "filter",
      activation: 0,
    })),
    ...Array.from({ length: 10 }, (_, index) => ({
      id: `dense-${index * 5}`,
      layer: "dense",
      label: `H${index * 5}`,
      kind: "hidden",
      activation: 0,
    })),
    ...Array.from({ length: 10 }, (_, digit) => ({
      id: `digit-${digit}`,
      layer: "output",
      label: String(digit),
      kind: "digit",
      activation: 0,
      confidence: 0,
      source_index: digit,
    })),
  ],
  connections: [],
  note: "Draw a digit to load real sampled weights and activation lighting.",
};

function isUsableVisualization(visualization) {
  return (
    visualization &&
    Array.isArray(visualization.layers) &&
    visualization.layers.length > 0 &&
    Array.isArray(visualization.nodes) &&
    visualization.nodes.length > 0 &&
    Array.isArray(visualization.connections) &&
    typeof visualization.note === "string"
  );
}

function resetCanvas(ctx, canvas) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function resetPreviewCanvas() {
  previewCtx.fillStyle = previewInverted ? "#000000" : "#ffffff";
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function updatePreviewPolarityState() {
  previewBlock?.classList.toggle("is-inverted", previewInverted);
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
  resetNetworkVisualization("Awaiting digit");
}

function resetTopPredictions() {
  topPredictions.innerHTML = '<div class="top-row is-empty">No prediction yet</div>';
}

function activateTab(tabId) {
  const isVisualizationActive = tabId === "visualization-tab";
  workbench?.classList.toggle("is-visualization-active", isVisualizationActive);
  workbench?.classList.toggle("is-prediction-active", !isVisualizationActive);

  tabButtons.forEach((button) => {
    const isActive = button.id === tabId;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  tabPanels.forEach((panel) => {
    const tab = tabButtons.find((button) => button.getAttribute("aria-controls") === panel.id);
    const isActive = tab?.id === tabId;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });
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

function resetNetworkVisualization(message) {
  networkStatus.textContent = message;
  currentVisualization = null;
  currentPrediction = null;
  selectedNodeId = null;
  try {
    renderNetworkGraph(fallbackVisualization, null);
  } catch (error) {
    networkStatus.textContent = "Graph reset failed";
    networkNote.textContent = error instanceof Error ? error.message : "Unable to reset graph.";
  }
}

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function clampUnit(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function layerPositions(layers) {
  const positions = new Map();
  const left = 78;
  const width = 704;
  layers.forEach((layer, index) => {
    const x = layers.length === 1 ? 430 : left + (width * index) / (layers.length - 1);
    positions.set(layer.id, x);
  });
  return positions;
}

function nodePositions(visualization) {
  const layers = visualization.layers || [];
  const nodes = visualization.nodes || [];
  const xPositions = layerPositions(layers);
  const byLayer = new Map(layers.map((layer) => [layer.id, []]));
  nodes.forEach((node) => {
    if (!byLayer.has(node.layer)) {
      byLayer.set(node.layer, []);
    }
    byLayer.get(node.layer).push(node);
  });

  const positions = new Map();
  byLayer.forEach((layerNodes, layerId) => {
    const x = xPositions.get(layerId) || 430;
    const count = layerNodes.length;
    const top = count <= 1 ? 210 : 72;
    const bottom = count <= 1 ? 210 : 348;
    layerNodes.forEach((node, index) => {
      const y = count <= 1 ? 210 : top + ((bottom - top) * index) / (count - 1);
      positions.set(node.id, { x, y });
    });
  });
  return positions;
}

function appendTitle(element, text) {
  const title = createSvgElement("title");
  title.textContent = text;
  element.appendChild(title);
}

function selectedPathIds(visualization, selectedId) {
  if (!selectedId || !visualization?.connections) {
    return { nodeIds: new Set(), connectionIds: new Set() };
  }

  const upstream = new Map();
  const downstream = new Map();
  visualization.connections.forEach((connection) => {
    if (!upstream.has(connection.target)) {
      upstream.set(connection.target, []);
    }
    if (!downstream.has(connection.source)) {
      downstream.set(connection.source, []);
    }
    upstream.get(connection.target).push(connection);
    downstream.get(connection.source).push(connection);
  });

  const nodeIds = new Set([selectedId]);
  const connectionIds = new Set();

  function walk(map, nodeId, direction) {
    (map.get(nodeId) || []).forEach((connection) => {
      if (connectionIds.has(connection.id)) {
        return;
      }
      connectionIds.add(connection.id);
      const nextId = direction === "up" ? connection.source : connection.target;
      nodeIds.add(nextId);
      walk(map, nextId, direction);
    });
  }

  walk(upstream, selectedId, "up");
  walk(downstream, selectedId, "down");
  return { nodeIds, connectionIds };
}

function selectGraphNode(nodeId) {
  selectedNodeId = selectedNodeId === nodeId ? null : nodeId;
  renderNetworkGraph(currentVisualization || fallbackVisualization, currentPrediction);
}

function renderNetworkGraph(visualization, prediction) {
  const graph = visualization || fallbackVisualization;
  currentVisualization = graph;
  const layers = graph.layers || [];
  const nodes = graph.nodes || [];
  const connections = graph.connections || [];
  const positions = nodePositions(graph);
  const selectedPath = selectedPathIds(graph, selectedNodeId);
  const strongestConnections = new Set(
    connections
      .slice()
      .sort((a, b) => Math.abs(Number(b.weight)) - Math.abs(Number(a.weight)))
      .slice(0, 18)
      .map((connection) => connection.id)
  );

  networkGraph.replaceChildren();

  const defs = createSvgElement("defs");
  defs.appendChild(
    createSvgElement("marker", {
      id: "arrow-head",
      viewBox: "0 0 10 10",
      refX: 9,
      refY: 5,
      markerWidth: 5,
      markerHeight: 5,
      orient: "auto-start-reverse",
    })
  );
  defs.querySelector("marker").appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z" }));
  networkGraph.appendChild(defs);

  const layerX = layerPositions(layers);
  layers.forEach((layer) => {
    const x = layerX.get(layer.id) || 430;
    const layerGroup = createSvgElement("g", { class: "graph-layer" });
    const activation = clampUnit(layer.activation);
    layerGroup.appendChild(
      createSvgElement("rect", {
        x: x - 54,
        y: 34,
        width: 108,
        height: 344,
        rx: 10,
        class: activation > 0 ? "layer-band is-lit" : "layer-band",
        style: `--intensity: ${activation.toFixed(3)}`,
        "data-activation": activation.toFixed(3),
      })
    );
    const label = createSvgElement("text", { x, y: 24, class: "layer-label", "text-anchor": "middle" });
    label.textContent = layer.label;
    const detail = createSvgElement("text", { x, y: 396, class: "layer-detail", "text-anchor": "middle" });
    detail.textContent = layer.detail;
    layerGroup.append(label, detail);
    networkGraph.appendChild(layerGroup);
  });

  connections.forEach((connection) => {
    const source = positions.get(connection.source);
    const target = positions.get(connection.target);
    if (!source || !target) {
      return;
    }

    const intensity = clampUnit(connection.intensity);
    const weight = Math.abs(Number(connection.weight) || 0);
    const strokeWidth = 0.55 + Math.min(2.6, weight * 9);
    const edge = createSvgElement("line", {
      x1: source.x + 16,
      y1: source.y,
      x2: target.x - 16,
      y2: target.y,
      class: `graph-edge ${intensity > 0.18 ? "is-lit" : ""} ${selectedPath.connectionIds.has(connection.id) ? "is-selected-path" : ""}`,
      style: `--intensity: ${intensity.toFixed(3)}`,
      "stroke-width": (selectedPath.connectionIds.has(connection.id) ? strokeWidth + 0.8 : strokeWidth).toFixed(2),
      "data-intensity": intensity.toFixed(3),
      "marker-end": "url(#arrow-head)",
    });
    appendTitle(
      edge,
      `${connection.source} -> ${connection.target}: ${connection.weight_label} (${connection.weight_kind})`
    );
    networkGraph.appendChild(edge);

    if (strongestConnections.has(connection.id)) {
      const weightText = createSvgElement("text", {
        x: (source.x + target.x) / 2,
        y: (source.y + target.y) / 2 - 4,
        class: "edge-label",
        "text-anchor": "middle",
      });
      weightText.textContent = connection.weight_label;
      networkGraph.appendChild(weightText);
    }
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    if (!position) {
      return;
    }
    const activation = clampUnit(node.activation);
    const isPrediction = node.kind === "digit" && String(node.source_index) === String(prediction);
    const isSelected = node.id === selectedNodeId;
    const isSelectedPath = selectedPath.nodeIds.has(node.id);
    const nodeGroup = createSvgElement("g", {
      class: `graph-node ${activation > 0 ? "is-lit" : ""} ${isPrediction ? "is-prediction" : ""} ${isSelected ? "is-selected" : ""} ${isSelectedPath ? "is-selected-path" : ""}`,
      transform: `translate(${position.x} ${position.y})`,
      style: `--intensity: ${activation.toFixed(3)}`,
      "data-activation": activation.toFixed(3),
      tabindex: "0",
      role: "button",
      "data-node-id": node.id,
      "aria-label": `Select ${node.label} neuron path`,
    });
    nodeGroup.addEventListener("click", () => selectGraphNode(node.id));
    nodeGroup.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectGraphNode(node.id);
      }
    });
    const radius = node.kind === "digit" ? 17 : 14;
    nodeGroup.appendChild(
      createSvgElement("circle", {
        r: radius,
        class: "node-core",
        "data-intensity": activation.toFixed(3),
      })
    );
    const label = createSvgElement("text", {
      y: 4,
      class: "node-label",
      "text-anchor": "middle",
    });
    label.textContent = node.label;
    nodeGroup.appendChild(label);

    appendTitle(
      nodeGroup,
      `${node.label}: ${formatPercent(activation)} activation${node.confidence != null ? `, ${formatPercent(node.confidence)} confidence` : ""}`
    );
    networkGraph.appendChild(nodeGroup);
  });

  networkNote.textContent = graph.note || "Weights are sampled or aggregated for readability.";
}

function renderNetworkPrediction(prediction, confidence, visualization) {
  const predictionSummary = `Digit ${prediction} at ${formatPercent(confidence)}`;
  networkStatus.textContent = "Rendering graph";
  selectedNodeId = null;
  currentPrediction = prediction;

  if (!isUsableVisualization(visualization)) {
    renderNetworkGraph(fallbackVisualization, null);
    networkStatus.textContent = "Visualization unavailable";
    networkNote.textContent = "Prediction succeeded, but the response did not include usable graph data.";
    return;
  }

  try {
    renderNetworkGraph(visualization, prediction);
    networkStatus.textContent = predictionSummary;
  } catch (error) {
    renderNetworkGraph(fallbackVisualization, null);
    networkStatus.textContent = "Visualization error";
    networkNote.textContent = error instanceof Error ? error.message : "Graph rendering failed.";
  }
}

function clearAll() {
  resetCanvas(drawCtx, drawCanvas);
  resetCanvas(modelCtx, modelCanvas);
  resetPreviewCanvas();
  hasInk = false;
  lastPoint = null;
  window.clearTimeout(debounceTimer);
  requestId += 1;
  selectedNodeId = null;
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
  resetCanvas(modelCtx, modelCanvas);
  modelCtx.globalCompositeOperation = "source-over";
  modelCtx.filter = "none";
  modelCtx.drawImage(drawCanvas, 0, 0, modelCanvas.width, modelCanvas.height);
  if (previewInverted) {
    const imageData = modelCtx.getImageData(0, 0, modelCanvas.width, modelCanvas.height);
    for (let index = 0; index < imageData.data.length; index += 4) {
      imageData.data[index] = 255 - imageData.data[index];
      imageData.data[index + 1] = 255 - imageData.data[index + 1];
      imageData.data[index + 2] = 255 - imageData.data[index + 2];
    }
    modelCtx.putImageData(imageData, 0, 0);
  }
  resetPreviewCanvas();
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.drawImage(modelCanvas, 0, 0);
}

function setLoading(isLoading) {
  statusText.textContent = isLoading ? "Predicting" : "Ready";
  if (isLoading) {
    networkStatus.textContent = "Predicting";
  }
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
    networkStatus.textContent = `Prediction ${payload.prediction} received`;
    renderNetworkPrediction(payload.prediction, confidence, payload.visualization);
    statusText.textContent = "Ready";
  } catch (error) {
    if (currentRequest !== requestId) {
      return;
    }
    predictionValue.textContent = "-";
    confidenceValue.textContent = "0.00%";
    confidenceMeter.style.width = "0%";
    resetTopPredictions();
    resetNetworkVisualization("Prediction error");
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

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button.id));
});

invertPreviewToggle.addEventListener("change", () => {
  previewInverted = invertPreviewToggle.checked;
  updatePreviewPolarityState();
  updateModelCanvas();
  if (hasInk) {
    schedulePrediction();
  }
});

activateTab("prediction-tab");
updatePreviewPolarityState();
clearAll();
