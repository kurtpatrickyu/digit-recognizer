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
const modelSelect = document.getElementById("model-select");
const modelStatus = document.getElementById("model-status");
const networkStatus = document.getElementById("network-status");
const networkGraph = document.getElementById("network-graph");
const networkNote = document.getElementById("network-note");
const workflowTabButtons = Array.from(document.querySelectorAll(".workflow-tabs [role='tab']"));
const workflowTabPanels = Array.from(document.querySelectorAll(".workflow-panel"));
const tabButtons = Array.from(document.querySelectorAll(".result-tabs [role='tab']"));
const tabPanels = Array.from(document.querySelectorAll("#prediction-panel, #visualization-panel"));
const startTrainingButton = document.getElementById("start-training-button");
const trainingModelSelect = document.getElementById("training-model-select");
const trainingEpochsInput = document.getElementById("training-epochs-input");
const trainingStatusText = document.getElementById("training-status-text");
const trainingEpoch = document.getElementById("training-epoch");
const trainingBatchLoss = document.getElementById("training-batch-loss");
const trainingTrainLoss = document.getElementById("training-train-loss");
const trainingTestLoss = document.getElementById("training-test-loss");
const trainingTestAccuracy = document.getElementById("training-test-accuracy");
const trainingBestAccuracy = document.getElementById("training-best-accuracy");
const trainingTotalEpochs = document.getElementById("training-total-epochs");
const trainingMessage = document.getElementById("training-message");
const trainingSourceModel = document.getElementById("training-source-model");
const trainingSavedModel = document.getElementById("training-saved-model");
const trainingImageCanvas = document.getElementById("training-image-canvas");
const trainingImageCtx = trainingImageCanvas.getContext("2d");
const trainingImageLabel = document.getElementById("training-image-label");
const trainingGraph = document.getElementById("training-graph");

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
let activeModelName = null;
let modelRecords = [];
let selectedTrainingModelName = null;
let currentVisualization = null;
let currentPrediction = null;
let selectedNodeId = null;
let trainingPollTimer = null;
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
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "-";
  }
  const confidence = Math.max(0, Math.min(1, numericValue));
  return `${(confidence * 100).toFixed(2)}%`;
}

function modelOptionLabel(model) {
  if (!model) {
    return "Unknown model";
  }
  const suffix = model.is_base ? "baseline" : `${Number(model.epochs) || 0} epochs`;
  return `${model.name} (${suffix})`;
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

function activateWorkflowTab(tabId) {
  workflowTabButtons.forEach((button) => {
    const isActive = button.id === tabId;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  workflowTabPanels.forEach((panel) => {
    const tab = workflowTabButtons.find((button) => button.getAttribute("aria-controls") === panel.id);
    const isActive = tab?.id === tabId;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });
}

function updateModelControls(models, activeModel) {
  activeModelName = activeModel || null;
  modelRecords = Array.isArray(models) ? models : [];
  modelSelect.replaceChildren();
  trainingModelSelect.replaceChildren();

  if (modelRecords.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No trained model";
    option.value = "";
    modelSelect.appendChild(option);
    trainingModelSelect.appendChild(option.cloneNode(true));
    modelSelect.disabled = true;
    trainingModelSelect.disabled = true;
    modelStatus.textContent = "";
    trainingMessage.textContent = "No models are available.";
    return;
  }

  if (!selectedTrainingModelName || !modelRecords.some((model) => model.name === selectedTrainingModelName)) {
    selectedTrainingModelName = activeModel || modelRecords[0]?.name || null;
  }

  modelRecords.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = modelOptionLabel(model);
    option.selected = model.name === activeModel;
    modelSelect.appendChild(option);

    const trainingOption = document.createElement("option");
    trainingOption.value = model.name;
    trainingOption.textContent = modelOptionLabel(model);
    trainingOption.selected = model.name === selectedTrainingModelName;
    trainingModelSelect.appendChild(trainingOption);
  });
  modelSelect.disabled = false;
  trainingModelSelect.disabled = false;
  modelStatus.textContent = "";
  renderSelectedTrainingModelStats();
}

async function loadModels() {
  try {
    const response = await fetch("/models");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Failed to load models.");
    }
    updateModelControls(payload.models, payload.active_model);
  } catch (error) {
    modelSelect.disabled = true;
    trainingModelSelect.disabled = true;
    modelStatus.textContent = error instanceof Error ? error.message : "Model list unavailable.";
    trainingMessage.textContent = error instanceof Error ? error.message : "Model list unavailable.";
  }
}

async function selectActiveModel(modelName) {
  if (!modelName) {
    return;
  }
  modelSelect.disabled = true;
  modelStatus.textContent = "";
  try {
    const response = await fetch("/models/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: modelName }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Failed to select model.");
    }
    updateModelControls(payload.models, payload.active_model);
    if (hasInk) {
      schedulePrediction();
    }
  } catch (error) {
    modelStatus.textContent = error instanceof Error ? error.message : "Model selection failed.";
    await loadModels();
  }
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

function formatMetric(value, digits = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

function formatImageProgress(status = {}) {
  const trained = Number(status.images_trained);
  const total = Number(status.total_training_images);
  if (Number.isFinite(trained) && Number.isFinite(total) && total > 0) {
    return `${trained}/${total}`;
  }
  return "-";
}

function formatEpochProgress(status = {}) {
  const epoch = Number(status.epoch);
  const epochs = Number(status.epochs);
  if (Number.isFinite(epoch) && Number.isFinite(epochs) && epochs > 0) {
    return `Epoch ${Math.max(0, epoch)}/${epochs}`;
  }
  return "-";
}

function selectedTrainingModel() {
  return modelRecords.find((model) => model.name === selectedTrainingModelName) || null;
}

function renderSelectedTrainingModelStats() {
  if (!trainingModelSelect) {
    return;
  }
  const model = selectedTrainingModel();
  if (!model) {
    trainingStatusText.textContent = "idle";
    trainingEpoch.textContent = "-";
    trainingBatchLoss.textContent = "-";
    trainingTrainLoss.textContent = "-";
    trainingTestLoss.textContent = "-";
    trainingTestAccuracy.textContent = "-";
    trainingBestAccuracy.textContent = "-";
    trainingTotalEpochs.textContent = "0";
    trainingSourceModel.textContent = "Source: none";
    trainingSavedModel.textContent = "No checkpoint yet";
    return;
  }
  trainingStatusText.textContent = "idle";
  trainingEpoch.textContent = "-";
  trainingBatchLoss.textContent = "-";
  trainingTrainLoss.textContent = formatMetric(model.train_loss);
  trainingTestLoss.textContent = formatMetric(model.test_loss);
  trainingTestAccuracy.textContent = model.test_accuracy == null ? "-" : formatPercent(model.test_accuracy);
  trainingBestAccuracy.textContent = model.best_accuracy == null ? "-" : formatPercent(model.best_accuracy);
  trainingTotalEpochs.textContent = String(Number(model.epochs) || 0);
  trainingSourceModel.textContent = `Source: ${model.name}${model.is_base ? " baseline" : ""}`;
  trainingSavedModel.textContent = model.is_base ? "Baseline" : model.name;
  trainingModelSelect.disabled = modelRecords.length === 0;
  trainingEpochsInput.disabled = false;
}

function renderTrainingGraph(status = {}) {
  if (status.training_visualization) {
    renderTrainingAdjustmentGraph(status.training_visualization, status);
    return;
  }
  const best = clampUnit(status.best_accuracy || 0);
  const test = clampUnit(status.test_accuracy || 0);
  const epoch = Number(status.epoch) || 0;
  const epochs = Number(status.epochs) || 1;
  const progress = clampUnit(epoch / epochs);
  const isRunning = status.status === "running";
  const isComplete = status.status === "complete";
  const isError = status.status === "error";
  const stateLabel = isError ? "Error" : isComplete ? "Complete" : isRunning ? "Running" : "Idle";
  trainingGraph.replaceChildren();

  const layers = [
    { label: "Input", x: 120, y: 230 },
    { label: "Conv", x: 270, y: 170 },
    { label: "FC", x: 430, y: 230 },
    { label: "Output", x: 590, y: 170 },
    { label: "Best", x: 740, y: 230 },
  ];

  layers.slice(0, -1).forEach((layer, index) => {
    const next = layers[index + 1];
    const line = createSvgElement("line", {
      x1: layer.x + 32,
      y1: layer.y,
      x2: next.x - 32,
      y2: next.y,
      class: `training-edge ${isRunning ? "is-running" : ""} ${isComplete ? "is-complete" : ""}`,
      style: `--progress: ${progress.toFixed(3)}`,
    });
    trainingGraph.appendChild(line);
  });

  layers.forEach((layer, index) => {
    const group = createSvgElement("g", {
      class: `training-node ${isRunning ? "is-running" : ""} ${isComplete ? "is-complete" : ""} ${isError ? "is-error" : ""}`,
      transform: `translate(${layer.x} ${layer.y})`,
      style: `--progress: ${index === layers.length - 1 ? best.toFixed(3) : progress.toFixed(3)}`,
    });
    group.appendChild(createSvgElement("circle", { r: 32, class: "training-node-core" }));
    const label = createSvgElement("text", { y: 5, class: "training-node-label", "text-anchor": "middle" });
    label.textContent = layer.label;
    group.appendChild(label);
    trainingGraph.appendChild(group);
  });

  const title = createSvgElement("text", { x: 430, y: 38, class: "training-graph-title", "text-anchor": "middle" });
  title.textContent = `${stateLabel} - ${formatEpochProgress(status)}`;
  const progressLabel = createSvgElement("text", { x: 430, y: 382, class: "training-graph-detail", "text-anchor": "middle" });
  progressLabel.textContent = `test ${formatPercent(test)} | best ${formatPercent(best)}`;
  trainingGraph.append(title, progressLabel);
}

function trainingLayerPositions(layers) {
  const positions = new Map();
  const left = 92;
  const width = 676;
  layers.forEach((layer, index) => {
    const x = layers.length === 1 ? 430 : left + (width * index) / (layers.length - 1);
    positions.set(layer.id, x);
  });
  return positions;
}

function trainingNodePositions(visualization) {
  const layers = visualization.layers || [];
  const nodes = visualization.nodes || [];
  const xPositions = trainingLayerPositions(layers);
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
    const top = count <= 1 ? 210 : 92;
    const bottom = count <= 1 ? 210 : 318;
    layerNodes.forEach((node, index) => {
      const y = count <= 1 ? 210 : top + ((bottom - top) * index) / (count - 1);
      positions.set(node.id, { x, y });
    });
  });
  return positions;
}

function renderTrainingAdjustmentGraph(visualization, status = {}) {
  const layers = visualization.layers || [];
  const nodes = visualization.nodes || [];
  const connections = visualization.connections || [];
  const positions = trainingNodePositions(visualization);
  const predictedDigit = status.training_prediction?.prediction ?? status.batch_image?.prediction;
  const strongestConnections = new Set(
    connections
      .slice()
      .sort((a, b) => Math.abs(Number(b.delta ?? b.weight)) - Math.abs(Number(a.delta ?? a.weight)))
      .slice(0, 10)
      .map((connection) => connection.id)
  );

  trainingGraph.replaceChildren();
  const defs = createSvgElement("defs");
  function appendMarker(id, className) {
    const marker = createSvgElement("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: 8.5,
      refY: 5,
      markerWidth: 8,
      markerHeight: 8,
      markerUnits: "userSpaceOnUse",
      orient: "auto-start-reverse",
    });
    marker.appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", class: className }));
    defs.appendChild(marker);
  }
  appendMarker("training-arrow-head", "training-arrow-head-primary");
  appendMarker("training-arrow-head-down", "training-arrow-head-down");
  appendMarker("training-arrow-head-secondary", "training-arrow-head-secondary");
  trainingGraph.appendChild(defs);

  const layerX = trainingLayerPositions(layers);
  layers.forEach((layer) => {
    const x = layerX.get(layer.id) || 430;
    const activation = clampUnit(layer.activation);
    const layerGroup = createSvgElement("g", { class: "graph-layer" });
    layerGroup.appendChild(
      createSvgElement("rect", {
        x: x - 54,
        y: 74,
        width: 108,
        height: 268,
        rx: 10,
        class: activation > 0 ? "layer-band is-lit" : "layer-band",
        style: `--intensity: ${activation.toFixed(3)}`,
      })
    );
    const label = createSvgElement("text", { x, y: 58, class: "layer-label", "text-anchor": "middle" });
    label.textContent = layer.label;
    const detail = createSvgElement("text", { x, y: 356, class: "layer-detail", "text-anchor": "middle" });
    detail.textContent = layer.detail || "";
    layerGroup.append(label, detail);
    trainingGraph.appendChild(layerGroup);
  });

  connections.forEach((connection) => {
    const source = positions.get(connection.source);
    const target = positions.get(connection.target);
    if (!source || !target) {
      return;
    }
    const isSecondary = connection.is_secondary === true;
    const adjustment = clampUnit(connection.adjustment);
    const intensity = clampUnit(connection.intensity);
    const weight = Math.abs(Number(connection.weight) || 0);
    const strokeWidth = isSecondary ? 1.05 : 0.55 + Math.min(2.6, weight * 9) + adjustment * 1.2;
    const markerId = isSecondary
      ? "training-arrow-head-secondary"
      : connection.direction === "down"
        ? "training-arrow-head-down"
        : "training-arrow-head";
    const edge = createSvgElement("line", {
      x1: source.x + 16,
      y1: source.y,
      x2: target.x - 16,
      y2: target.y,
      class: `graph-edge training-adjustment-edge ${isSecondary ? "is-secondary" : "is-lit"} is-${connection.direction === "down" ? "down" : "up"}`,
      style: `--intensity: ${intensity.toFixed(3)}; --adjustment: ${adjustment.toFixed(3)}`,
      "stroke-width": strokeWidth.toFixed(2),
      "marker-end": `url(#${markerId})`,
    });
    appendTitle(
      edge,
      `${connection.source} -> ${connection.target}: weight ${connection.weight_label || formatMetric(connection.weight, 4)}, delta ${connection.delta_label || formatMetric(connection.delta, 6)}`
    );
    trainingGraph.appendChild(edge);

    if (!isSecondary && strongestConnections.has(connection.id)) {
      const weightText = createSvgElement("text", {
        x: (source.x + target.x) / 2,
        y: (source.y + target.y) / 2 - 8,
        class: "edge-label training-value-label",
        "text-anchor": "middle",
      });
      weightText.textContent = `${connection.weight_label || formatMetric(connection.weight, 3)} ${connection.delta_label || formatMetric(connection.delta, 4)}`;
      trainingGraph.appendChild(weightText);
    }
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    if (!position) {
      return;
    }
    const activation = clampUnit(node.activation);
    const isPrediction = node.kind === "digit" && String(node.source_index) === String(predictedDigit);
    const group = createSvgElement("g", {
      class: `graph-node ${activation > 0 ? "is-lit" : ""} ${isPrediction ? "is-prediction" : ""}`,
      transform: `translate(${position.x} ${position.y})`,
      style: `--intensity: ${activation.toFixed(3)}`,
    });
    group.appendChild(createSvgElement("circle", { r: node.kind === "digit" ? 17 : 14, class: "node-core" }));
    const label = createSvgElement("text", { y: 4, class: "node-label", "text-anchor": "middle" });
    label.textContent = node.label;
    group.appendChild(label);
    appendTitle(
      group,
      `${node.label}: ${formatPercent(activation)} activation${node.confidence != null ? `, ${formatPercent(node.confidence)} confidence` : ""}`
    );
    trainingGraph.appendChild(group);
  });

  const title = createSvgElement("text", { x: 430, y: 28, class: "training-graph-title", "text-anchor": "middle" });
  title.textContent = `Training graph - ${formatEpochProgress(status)}`;
  const detail = createSvgElement("text", { x: 430, y: 392, class: "training-graph-detail", "text-anchor": "middle" });
  detail.textContent = `loss ${formatMetric(status.batch_loss)} | prediction ${predictedDigit ?? "-"} ${formatPercent(status.training_prediction?.confidence ?? status.batch_image?.confidence)}`;
  trainingGraph.append(title, detail);
}

function renderTrainingImage(batchImage) {
  if (!batchImage || !Array.isArray(batchImage.pixels)) {
    resetCanvas(trainingImageCtx, trainingImageCanvas);
    trainingImageLabel.textContent = "-";
    return;
  }
  const imageData = trainingImageCtx.createImageData(28, 28);
  batchImage.pixels.slice(0, 784).forEach((value, index) => {
    const offset = index * 4;
    const pixel = Math.max(0, Math.min(255, Number(value) || 0));
    imageData.data[offset] = pixel;
    imageData.data[offset + 1] = pixel;
    imageData.data[offset + 2] = pixel;
    imageData.data[offset + 3] = 255;
  });
  trainingImageCtx.imageSmoothingEnabled = false;
  trainingImageCtx.putImageData(imageData, 0, 0);
  const prediction = batchImage.prediction ?? null;
  const confidence = batchImage.confidence ?? null;
  const isCorrect = batchImage.is_correct === true;
  const hasCorrectness = typeof batchImage.is_correct === "boolean";
  const correctnessText = hasCorrectness ? (isCorrect ? "✓" : "×") : "-";
  const correctnessClass = hasCorrectness ? (isCorrect ? "is-correct" : "is-incorrect") : "";
  const correctnessLabel = hasCorrectness
    ? isCorrect
      ? "Prediction is correct"
      : "Prediction is incorrect"
    : "Prediction correctness unavailable";
  trainingImageLabel.replaceChildren();
  if (batchImage.label == null) {
    trainingImageLabel.textContent = "-";
    return;
  }
  const label = document.createElement("span");
  label.textContent = `Label ${batchImage.label}`;
  const marker = document.createElement("span");
  marker.className = `training-correctness ${correctnessClass}`;
  marker.textContent = correctnessText;
  marker.setAttribute("aria-label", correctnessLabel);
  marker.setAttribute("role", "img");
  const predictionLabel = document.createElement("span");
  predictionLabel.textContent = `Pred ${prediction ?? "-"} ${confidence == null ? "" : formatPercent(confidence)}`;
  trainingImageLabel.append(label, marker, predictionLabel);
}

function renderTrainingStatus(status) {
  const isRunning = status.status === "running";
  trainingStatusText.textContent = status.status || "idle";
  trainingEpoch.textContent = formatEpochProgress(status);
  trainingBatchLoss.textContent = formatMetric(status.batch_loss);
  trainingTrainLoss.textContent = formatMetric(status.train_loss);
  trainingTestLoss.textContent = formatMetric(status.test_loss);
  trainingTestAccuracy.textContent = status.test_accuracy == null ? "-" : formatPercent(status.test_accuracy);
  trainingBestAccuracy.textContent = status.best_accuracy == null ? "-" : formatPercent(status.best_accuracy);
  trainingTotalEpochs.textContent = String(Number(status.total_epochs) || 0);
  trainingMessage.textContent = status.message || "No training job has started.";
  trainingSavedModel.textContent = status.saved_model ? status.saved_model : "No checkpoint yet";
  trainingSourceModel.textContent = `Source: ${status.source_model || selectedTrainingModelName || "none"}`;
  startTrainingButton.disabled = isRunning;
  trainingModelSelect.disabled = isRunning || modelRecords.length === 0;
  trainingEpochsInput.disabled = isRunning;
  renderTrainingImage(status.batch_image);
  renderTrainingGraph(status);
}

async function pollTrainingStatus() {
  try {
    const response = await fetch("/train/status");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Unable to get training status.");
    }
    renderTrainingStatus(payload);
    if (payload.status === "running") {
      trainingPollTimer = window.setTimeout(pollTrainingStatus, 1500);
    } else {
      trainingPollTimer = null;
      if (payload.status === "complete") {
        if (payload.saved_model) {
          selectedTrainingModelName = payload.saved_model;
        }
        await loadModels();
      }
    }
  } catch (error) {
    trainingMessage.textContent = error instanceof Error ? error.message : "Training status unavailable.";
    startTrainingButton.disabled = false;
    trainingModelSelect.disabled = modelRecords.length === 0;
    trainingEpochsInput.disabled = false;
    trainingPollTimer = null;
  }
}

async function startTraining() {
  const epochs = Number.parseInt(trainingEpochsInput.value, 10);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 100) {
    trainingMessage.textContent = "Enter an epoch count from 1 to 100.";
    return;
  }
  const modelName = trainingModelSelect.value || selectedTrainingModelName;
  startTrainingButton.disabled = true;
  trainingModelSelect.disabled = true;
  trainingEpochsInput.disabled = true;
  trainingMessage.textContent = "Starting training.";
  try {
    const response = await fetch("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, model_name: modelName }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Training could not start.");
    }
    renderTrainingStatus(payload);
    window.clearTimeout(trainingPollTimer);
    trainingPollTimer = window.setTimeout(pollTrainingStatus, 1200);
  } catch (error) {
    trainingMessage.textContent = error instanceof Error ? error.message : "Training could not start.";
    startTrainingButton.disabled = false;
    trainingModelSelect.disabled = modelRecords.length === 0;
    trainingEpochsInput.disabled = false;
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
  if (!activeModelName) {
    showEmptyState();
    errorText.textContent = "Train or select a model before predicting.";
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
    activeModelName = payload.model_name || activeModelName;
    modelStatus.textContent = "";
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

workflowTabButtons.forEach((button) => {
  button.addEventListener("click", () => activateWorkflowTab(button.id));
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button.id));
});

modelSelect.addEventListener("change", () => {
  void selectActiveModel(modelSelect.value);
});

trainingModelSelect.addEventListener("change", () => {
  selectedTrainingModelName = trainingModelSelect.value || null;
  renderSelectedTrainingModelStats();
});

startTrainingButton.addEventListener("click", () => {
  void startTraining();
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
activateWorkflowTab("predict-workflow-tab");
updatePreviewPolarityState();
renderTrainingStatus({ status: "idle", message: "No training job has started." });
clearAll();
void loadModels();
