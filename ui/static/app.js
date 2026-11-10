/**
 * CGE — Visualizador Causal Interactivo y Copiloto (Fase 6)
 * Código 100% genérico sin tokens de dominios específicos.
 */

// Estado global de la aplicación web
let cy = null;
let currentSchema = null;
let currentDiagnostics = [];
let currentImportance = {};
let currentPaths = [];
let initialControlValues = {};

// Paleta de colores estándar por categoría causal
const CATEGORY_COLORS = {
  exogenous: "#0284c7",
  control: "#16a34a",
  state: "#d97706",
  target: "#9333ea"
};

// Inicialización tras cargar el DOM
document.addEventListener("DOMContentLoaded", () => {
  initCytoscape();
  setupEventListeners();
  loadAllData();
});

// Configuración de Cytoscape
function initCytoscape() {
  if (typeof cytoscape !== "undefined" && typeof cytoscapeDagre !== "undefined") {
    try {
      cytoscape.use(cytoscapeDagre);
    } catch (e) {
      // Ignorar si ya está registrado
    }
  }

  const container = document.getElementById("cy");
  if (!container) return;

  cy = cytoscape({
    container: container,
    boxSelectionEnabled: false,
    autounselectify: false,
    style: [
      {
        selector: "node",
        style: {
          "label": "data(label)",
          "color": "#f8fafc",
          "font-size": "11px",
          "font-weight": "bold",
          "text-valign": "center",
          "text-halign": "center",
          "text-wrap": "wrap",
          "text-max-width": "90px",
          "background-color": "data(bgColor)",
          "width": "data(nodeSize)",
          "height": "data(nodeSize)",
          "border-width": 2,
          "border-color": "#475569",
          "transition-property": "border-color, border-width, background-color, width, height",
          "transition-duration": "0.2s"
        }
      },
      {
        selector: "node:selected",
        style: {
          "border-color": "#38bdf8",
          "border-width": 4,
          "shadow-blur": 15,
          "shadow-color": "#38bdf8",
          "shadow-opacity": 0.8
        }
      },
      {
        selector: "node.highlighted",
        style: {
          "border-color": "#4ade80",
          "border-width": 5,
          "shadow-blur": 20,
          "shadow-color": "#4ade80",
          "shadow-opacity": 0.9
        }
      },
      {
        selector: "node.alerted",
        style: {
          "border-color": "#ef4444",
          "border-width": 4,
          "border-style": "dashed"
        }
      },
      {
        selector: "edge",
        style: {
          "width": "data(edgeWidth)",
          "line-color": "#64748b",
          "target-arrow-color": "#64748b",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "arrow-scale": 1.2,
          "opacity": 0.75,
          "transition-property": "line-color, target-arrow-color, width, opacity",
          "transition-duration": "0.2s"
        }
      },
      {
        selector: "edge.highlighted-edge",
        style: {
          "width": 4,
          "line-color": "#4ade80",
          "target-arrow-color": "#4ade80",
          "opacity": 1.0,
          "z-index": 10
        }
      }
    ],
    elements: []
  });

  cy.on("tap", "node", (evt) => {
    const node = evt.target;
    selectNode(node.data("id"));
  });

  cy.on("tap", (evt) => {
    if (evt.target === cy) {
      clearHighlights();
    }
  });
}

// Configuración de escuchadores de eventos DOM
function setupEventListeners() {
  // Pestañas del panel derecho
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetTab = document.getElementById(targetId);
      if (targetTab) targetTab.classList.add("active");
    });
  });

  // Botón Centrar Lienzo
  const btnFit = document.getElementById("btn-fit-view");
  if (btnFit) {
    btnFit.addEventListener("click", () => {
      if (cy) cy.fit(undefined, 30);
    });
  }

  // Botón Ingestar
  const btnIngest = document.getElementById("btn-ingest");
  const inputIngest = document.getElementById("ingest-input");
  if (btnIngest && inputIngest) {
    btnIngest.addEventListener("click", async () => {
      const dirPath = inputIngest.value.trim();
      if (!dirPath) return;
      try {
        btnIngest.disabled = true;
        btnIngest.textContent = "Ingestando...";
        const res = await fetch("/api/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dir_path: dirPath })
        });
        if (!res.ok) {
          const err = await res.json();
          alert("Error de ingesta: " + (err.detail || res.statusText));
          return;
        }
        await loadAllData();
      } catch (e) {
        alert("Error de conexión durante ingesta: " + e.message);
      } finally {
        btnIngest.disabled = false;
        btnIngest.textContent = "Ingestar";
      }
    });
  }

  // Botón Calcular Receta Óptima
  const btnCalcRecipe = document.getElementById("btn-calc-recipe");
  if (btnCalcRecipe) {
    btnCalcRecipe.addEventListener("click", handleCalculateRecipe);
  }

  // Botón Restablecer Sliders
  const btnResetSliders = document.getElementById("btn-reset-sliders");
  if (btnResetSliders) {
    btnResetSliders.addEventListener("click", () => {
      resetSlidersToNominal();
    });
  }

  // Formulario del Copiloto
  const copilotForm = document.getElementById("copilot-form");
  const copilotInput = document.getElementById("copilot-input");
  if (copilotForm && copilotInput) {
    copilotForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = copilotInput.value.trim();
      if (!q) return;
      handleCopilotQuery(q);
      copilotInput.value = "";
    });
  }

  // Chips de preguntas rápidas para el copiloto
  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      if (q) {
        // Cambiar a la pestaña del copiloto
        const copilotTabBtn = document.querySelector('[data-tab="tab-copilot"]');
        if (copilotTabBtn) copilotTabBtn.click();
        handleCopilotQuery(q);
      }
    });
  });
}

// Carga inicial y reactiva de datos desde el backend
async function loadAllData() {
  try {
    const [schemaRes, diagRes, impRes, pathsRes] = await Promise.all([
      fetch("/api/schema"),
      fetch("/api/diagnostics"),
      fetch("/api/importance"),
      fetch("/api/paths")
    ]);

    if (!schemaRes.ok) return;

    currentSchema = await schemaRes.json();
    currentDiagnostics = diagRes.ok ? (await diagRes.json()).diagnostics || [] : [];
    currentImportance = impRes.ok ? await impRes.json() : {};
    currentPaths = pathsRes.ok ? (await pathsRes.json()).paths || [] : [];

    renderMetadata();
    renderGraph();
    renderAlerts();
    renderSliders();
    renderPaths();
  } catch (err) {
    console.error("Error al cargar datos del backend CGE:", err);
  }
}

// Renderizado de metadatos en la barra superior
function renderMetadata() {
  if (!currentSchema) return;
  const sysId = document.getElementById("meta-system-id");
  const ver = document.getElementById("meta-version");
  const targetId = document.getElementById("meta-target-id");

  if (sysId) sysId.textContent = currentSchema.system_id || "CGE System";
  if (ver) ver.textContent = currentSchema.version || "1.0.0";

  const targetNode = currentSchema.nodes.find((n) => n.category === "target");
  if (targetId) targetId.textContent = targetNode ? targetNode.label || targetNode.id : "Target";

  if (targetNode && targetNode.current_value !== null) {
    const canvasTarget = document.getElementById("canvas-target-val");
    if (canvasTarget) canvasTarget.textContent = Number(targetNode.current_value).toFixed(2);
  }
}

// Renderizado del grafo en Cytoscape
function renderGraph() {
  if (!cy || !currentSchema) return;

  const elements = [];
  const elasticities = currentImportance.elasticities || {};
  const alertedNodeIds = new Set(
    currentDiagnostics.map((d) => d.node).filter(Boolean)
  );

  // Mapear nodos
  currentSchema.nodes.forEach((n) => {
    const cat = n.category || "state";
    const elast = Math.abs(elasticities[n.id] || 1.0);
    const size = Math.min(80, Math.max(38, 38 + elast * 6));
    const isAlerted = alertedNodeIds.has(n.id) || Boolean(n.is_dead);

    elements.push({
      group: "nodes",
      data: {
        id: n.id,
        label: n.label || n.id,
        category: cat,
        data_type: n.data_type || "continuous",
        current_value: n.current_value,
        bounds: n.bounds || {},
        source_reference: n.source_reference,
        bgColor: CATEGORY_COLORS[cat] || "#64748b",
        nodeSize: size
      },
      classes: isAlerted ? "alerted" : ""
    });
  });

  // Mapear aristas
  currentSchema.edges.forEach((e, idx) => {
    const elastVal = e.dimensionless_elasticity !== undefined ? Math.abs(e.dimensionless_elasticity) : (e.sensitivity_weight || 1.0);
    const width = Math.min(8, Math.max(1.5, Math.log2(elastVal + 1) * 2));

    elements.push({
      group: "edges",
      data: {
        id: `e_${e.source}_${e.target}_${idx}`,
        source: e.source,
        target: e.target,
        edgeWidth: width,
        elasticity: elastVal
      }
    });
  });

  cy.elements().remove();
  cy.add(elements);

  // Aplicar layout jerárquico Dagre con fallback a breadthfirst
  try {
    const layout = cy.layout({
      name: "dagre",
      rankDir: "TB",
      nodeDimensionsIncludeLabels: true,
      rankSep: 65,
      nodeSep: 40
    });
    layout.run();
  } catch (err) {
    cy.layout({ name: "breadthfirst", directed: true, padding: 30 }).run();
  }

  cy.fit(undefined, 35);
}

// Selección e inspección de nodo
async function selectNode(nodeId) {
  if (!currentSchema) return;
  const nodeObj = currentSchema.nodes.find((n) => n.id === nodeId);
  if (!nodeObj) return;

  const inspContainer = document.getElementById("inspector-content");
  if (!inspContainer) return;

  const boundsStr = nodeObj.bounds
    ? `[${nodeObj.bounds.min !== null ? nodeObj.bounds.min : "-inf"}, ${nodeObj.bounds.max !== null ? nodeObj.bounds.max : "+inf"}]`
    : "Sin límites";

  const elast = currentImportance.elasticities && currentImportance.elasticities[nodeId] !== undefined
    ? currentImportance.elasticities[nodeId]
    : "—";

  inspContainer.innerHTML = `
    <div class="inspector-field">
      <span class="badge badge-${nodeObj.category[0]}">${nodeObj.category}</span>
      <h3 style="margin-top:6px; font-size:14px; color:#f8fafc;">${nodeObj.label || nodeObj.id}</h3>
    </div>
    <div class="inspector-field"><strong>ID:</strong> <code>${nodeObj.id}</code></div>
    <div class="inspector-field"><strong>Tipo:</strong> ${nodeObj.data_type || "continuous"}</div>
    <div class="inspector-field"><strong>Valor Actual:</strong> ${nodeObj.current_value !== null ? Number(nodeObj.current_value).toFixed(2) : "—"}</div>
    <div class="inspector-field"><strong>Límites:</strong> ${boundsStr}</div>
    <div class="inspector-field"><strong>Elasticidad Capa 1:</strong> ${elast}</div>
  `;

  // Cargar fragmento de código fuente real vía /api/source
  await loadSourceCode(nodeId);

  // Resaltar nodo y aristas conectadas en el grafo
  highlightNodeInCanvas(nodeId);
}

// Carga del código fuente exacto
async function loadSourceCode(nodeId) {
  const codeView = document.getElementById("source-code-view");
  const codeMeta = document.getElementById("code-meta");
  if (!codeView) return;

  try {
    const res = await fetch(`/api/source?node_id=${encodeURIComponent(nodeId)}`);
    if (!res.ok) {
      codeView.textContent = "// Código fuente no disponible para este nodo.";
      if (codeMeta) codeMeta.textContent = "Archivo: — | Entidad: —";
      return;
    }
    const data = await res.json();
    codeView.textContent = data.code || "// Archivo vacío";
    if (codeMeta) {
      const range = data.line_range ? `Líneas [${data.line_range[0]}-${data.line_range[1]}]` : "";
      codeMeta.textContent = `${data.file || "N/A"} (${data.entity || "módulo"}) ${range}`;
    }
  } catch (e) {
    codeView.textContent = "// Error al recuperar el código fuente: " + e.message;
  }
}

// Resaltar nodo y sus aristas
function highlightNodeInCanvas(nodeId) {
  if (!cy) return;
  cy.elements().removeClass("highlighted highlighted-edge");
  const cyNode = cy.getElementById(nodeId);
  if (cyNode.length > 0) {
    cyNode.addClass("highlighted");
    cyNode.connectedEdges().addClass("highlighted-edge");
  }
}

// Limpiar resaltados
function clearHighlights() {
  if (!cy) return;
  cy.elements().removeClass("highlighted highlighted-edge");
}

// Renderizado de la lista de alertas T1-T7
function renderAlerts() {
  const container = document.getElementById("alerts-list");
  const countBadge = document.getElementById("alerts-count");
  if (!container) return;

  container.innerHTML = "";
  if (countBadge) countBadge.textContent = String(currentDiagnostics.length);

  if (currentDiagnostics.length === 0) {
    container.innerHTML = `<p class="placeholder-text">No se detectaron alertas críticas. El sistema está limpio.</p>`;
    return;
  }

  currentDiagnostics.forEach((diag) => {
    const card = document.createElement("div");
    card.className = "alert-card";
    const targetNodeId = diag.node || (diag.details && diag.details.threshold_variable) || "";

    card.innerHTML = `
      <div class="alert-card-title">
        <span>[${diag.id}] ${diag.code || "ALERTA"}</span>
        <code style="font-size:10px;">${targetNodeId}</code>
      </div>
      <div class="alert-card-desc">
        ${diag.details && diag.details.reason ? diag.details.reason : "Anomalía o restricción auditada en este nodo."}
      </div>
    `;

    card.addEventListener("click", () => {
      if (targetNodeId) {
        selectNode(targetNodeId);
        if (cy) {
          const cyEle = cy.getElementById(targetNodeId);
          if (cyEle.length > 0) {
            cy.animate({ center: { eles: cyEle }, zoom: 1.4 }, { duration: 300 });
          }
        }
      }
    });

    container.appendChild(card);
  });
}

// Renderizado de sliders reactivos "Qué pasa si"
function renderSliders() {
  const container = document.getElementById("sliders-container");
  if (!container || !currentSchema) return;

  container.innerHTML = "";
  const controlNodes = currentSchema.nodes.filter((n) => n.category === "control");
  initialControlValues = {};

  if (controlNodes.length === 0) {
    container.innerHTML = `<p class="placeholder-text">No hay palancas de control definidas en este grafo.</p>`;
    return;
  }

  controlNodes.forEach((node) => {
    const val = node.current_value !== null ? Number(node.current_value) : 0;
    initialControlValues[node.id] = val;

    const bMin = node.bounds && node.bounds.min !== null ? Number(node.bounds.min) : 0;
    const bMax = node.bounds && node.bounds.max !== null ? Number(node.bounds.max) : Math.max(1000, val * 3);
    const step = (bMax - bMin) > 100 ? "1" : "0.1";

    const group = document.createElement("div");
    group.className = "slider-group";
    group.innerHTML = `
      <div class="slider-header">
        <strong>${node.label || node.id}</strong>
        <code>${node.id}</code>
      </div>
      <div class="slider-input-row">
        <input type="range" id="slider-${node.id}" data-node-id="${node.id}" min="${bMin}" max="${bMax}" step="${step}" value="${val}">
        <input type="number" id="num-${node.id}" class="slider-num-input" data-node-id="${node.id}" min="${bMin}" max="${bMax}" step="${step}" value="${val}">
      </div>
    `;

    const slider = group.querySelector(`#slider-${node.id}`);
    const numInput = group.querySelector(`#num-${node.id}`);

    const onValueChange = (newVal) => {
      const num = Number(newVal);
      slider.value = num;
      numInput.value = num;
      evaluateWhatIf();
    };

    slider.addEventListener("input", (e) => onValueChange(e.target.value));
    numInput.addEventListener("change", (e) => onValueChange(e.target.value));

    container.appendChild(group);
  });

  evaluateWhatIf();
}

// Evaluación en tiempo real What-If
let evalDebounceTimer = null;
function evaluateWhatIf() {
  clearTimeout(evalDebounceTimer);
  evalDebounceTimer = setTimeout(async () => {
    const payload = {};
    document.querySelectorAll(".slider-num-input").forEach((inp) => {
      const id = inp.getAttribute("data-node-id") || inp.id.replace(/^num-/, "");
      payload[id] = Number(inp.value);
    });

    try {
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ controls: payload })
      });
      if (!res.ok) return;

      const data = await res.json();
      const targetDisplay = document.getElementById("evaluate-target-display");
      const canvasTarget = document.getElementById("canvas-target-val");

      if (targetDisplay) targetDisplay.textContent = Number(data.target_value).toFixed(2);
      if (canvasTarget) canvasTarget.textContent = Number(data.target_value).toFixed(2);

      // Actualizar valores visuales en el grafo
      if (cy && data.state_values) {
        Object.keys(data.state_values).forEach((nid) => {
          const cyNode = cy.getElementById(nid);
          if (cyNode.length > 0) {
            cyNode.data("current_value", data.state_values[nid]);
          }
        });
      }
    } catch (err) {
      console.error("Error en evaluación qué-pasa-si:", err);
    }
  }, 80);
}

// Restablecer sliders
function resetSlidersToNominal() {
  Object.keys(initialControlValues).forEach((cid) => {
    const slider = document.getElementById(`slider-${cid}`);
    const numInput = document.getElementById(`num-${cid}`);
    const nomVal = initialControlValues[cid];
    if (slider) slider.value = nomVal;
    if (numInput) numInput.value = nomVal;
  });
  evaluateWhatIf();
}

// Cálculo de Receta Óptima (Neutral o CVaR)
async function handleCalculateRecipe() {
  const goalInput = document.getElementById("recipe-target-goal");
  const cvarToggle = document.getElementById("cvar-toggle");
  const outputBox = document.getElementById("recipe-output");
  const statusBadge = document.getElementById("recipe-status");
  const objVal = document.getElementById("recipe-objective-val");
  const marginVal = document.getElementById("recipe-budget-margin");
  const controlsList = document.getElementById("recipe-controls-list");
  const infeasibleBox = document.getElementById("recipe-infeasible-box");
  const blockingMsg = document.getElementById("recipe-blocking-msg");

  const targetGoal = goalInput && goalInput.value.trim() ? Number(goalInput.value.trim()) : null;
  const useCvar = cvarToggle ? cvarToggle.checked : false;

  try {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_value: targetGoal,
        use_cvar: useCvar,
        cvar_beta: 0.95,
        risk_aversion: 0.2
      })
    });
    if (!res.ok) return;

    const data = await res.json();
    outputBox.classList.remove("hidden");

    const isFeasible = data.status === "FEASIBLE";
    statusBadge.textContent = data.status || "FEASIBLE";
    statusBadge.className = `status-badge ${isFeasible ? "feasible" : "infeasible"}`;

    objVal.textContent = Number(data.objective_value || 0).toFixed(2);
    marginVal.textContent = data.budget_margin !== undefined ? Number(data.budget_margin).toFixed(2) : "N/A";

    // Mostrar valores y deltas de palancas
    controlsList.innerHTML = "";
    const controls = data.controls || {};
    const highlightedIds = [];

    Object.keys(controls).forEach((cid) => {
      highlightedIds.push(cid);
      const optVal = Number(controls[cid]);
      const baseVal = initialControlValues[cid] !== undefined ? initialControlValues[cid] : optVal;
      const delta = optVal - baseVal;
      const deltaSign = delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2);

      const item = document.createElement("div");
      item.className = "recipe-control-item";
      item.innerHTML = `
        <span><strong>${cid}</strong>: ${optVal.toFixed(2)}</span>
        <span class="delta-badge">(Δ: ${deltaSign})</span>
      `;
      controlsList.appendChild(item);
    });

    if (!isFeasible) {
      infeasibleBox.classList.remove("hidden");
      const blockStr = (data.blocking_constraints || []).join(", ") || "Capacidad o presupuesto";
      blockingMsg.textContent = `Restricciones bloqueantes: [${blockStr}]. Máximo valor alcanzable: ${data.max_achievable_J_star ? Number(data.max_achievable_J_star).toFixed(2) : "—"}`;
    } else {
      infeasibleBox.classList.add("hidden");
    }

    // Resaltar palancas de la receta en el grafo
    if (cy) {
      cy.elements().removeClass("highlighted highlighted-edge");
      highlightedIds.forEach((id) => {
        const n = cy.getElementById(id);
        if (n.length > 0) n.addClass("highlighted");
      });
    }
  } catch (err) {
    console.error("Error al calcular receta óptima:", err);
  }
}

// Renderizado de las Rutas Top-K
function renderPaths() {
  const container = document.getElementById("paths-list");
  if (!container) return;

  container.innerHTML = "";
  if (currentPaths.length === 0) {
    container.innerHTML = `<p class="placeholder-text">No hay caminos dirigidos registrados hacia el objetivo.</p>`;
    return;
  }

  currentPaths.slice(0, 10).forEach((pData) => {
    const card = document.createElement("div");
    card.className = "path-card";
    const pathNodes = pData.path || [];
    const chainStr = pathNodes.join(" → ");
    const sensVal = pData.cumulative_sensitivity !== undefined ? Number(pData.cumulative_sensitivity).toFixed(3) : "1.000";

    card.innerHTML = `
      <div class="path-chain">${chainStr}</div>
      <div class="path-stats">
        <span>Longitud: ${pData.length || pathNodes.length} pasos</span>
        <span>Sensibilidad Acumulada: <strong>${sensVal}</strong></span>
      </div>
    `;

    card.addEventListener("click", () => {
      if (cy && pathNodes.length > 0) {
        cy.elements().removeClass("highlighted highlighted-edge");
        for (let i = 0; i < pathNodes.length; i++) {
          const cyN = cy.getElementById(pathNodes[i]);
          if (cyN.length > 0) cyN.addClass("highlighted");
          if (i < pathNodes.length - 1) {
            const edge = cy.edges(`[source = "${pathNodes[i]}"][target = "${pathNodes[i + 1]}"]`);
            if (edge.length > 0) edge.addClass("highlighted-edge");
          }
        }
      }
    });

    container.appendChild(card);
  });
}

// Manejo de consultas al Copiloto Agéntico
async function handleCopilotQuery(question) {
  const messagesBox = document.getElementById("copilot-messages");
  if (!messagesBox) return;

  // Añadir mensaje del usuario
  const userMsg = document.createElement("div");
  userMsg.className = "chat-bubble chat-user";
  userMsg.textContent = question;
  messagesBox.appendChild(userMsg);
  messagesBox.scrollTop = messagesBox.scrollHeight;

  // Burbuja temporal de carga
  const loadingMsg = document.createElement("div");
  loadingMsg.className = "chat-bubble chat-agent";
  loadingMsg.textContent = "Analizando el grafo causal...";
  messagesBox.appendChild(loadingMsg);
  messagesBox.scrollTop = messagesBox.scrollHeight;

  try {
    const res = await fetch("/api/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    if (!res.ok) {
      loadingMsg.textContent = "Error al procesar la consulta con el copiloto.";
      return;
    }

    const data = await res.json();
    loadingMsg.textContent = data.answer || "Consulta procesada sin respuesta textual.";

    // Resaltar en el lienzo los nodos y restricciones citados por el copiloto
    if (cy && data.highlighted_nodes && data.highlighted_nodes.length > 0) {
      cy.elements().removeClass("highlighted highlighted-edge");
      data.highlighted_nodes.forEach((nid) => {
        const n = cy.getElementById(nid);
        if (n.length > 0) {
          n.addClass("highlighted");
          n.connectedEdges().addClass("highlighted-edge");
        }
      });
    }
  } catch (err) {
    loadingMsg.textContent = "Error de conexión con el copiloto: " + err.message;
  }
}
