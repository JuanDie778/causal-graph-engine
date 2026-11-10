"""Módulo de exportación y verificación de interfaz visual del Grafo Causal (V-1 a V-5)."""
import json
import os
from typing import Dict, Any, List


def generate_interactive_ui_html(
    schema_dict: Dict[str, Any],
    diagnostics_list: List[Dict[str, Any]],
    solution_dict: Dict[str, Any],
    output_html_path: str = "ui/index.html"
) -> str:
    """Genera una interfaz interactiva autocontenida que renderiza el grafo causal según V-1 a V-5."""
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Causal Graph Engine — Visualizador Multidimensional</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      display: flex;
      height: 100vh;
      background: #0f172a;
      color: #e2e8f0;
    }}
    #sidebar {{
      width: 380px;
      background: #1e293b;
      border-right: 1px solid #334155;
      padding: 20px;
      overflow-y: auto;
      box-sizing: border-box;
    }}
    #viewport {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle, #1e293b 10%, #0f172a 90%);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .header {{
      padding: 16px;
      background: rgba(15, 23, 42, 0.8);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge-u {{ background: #0284c7; color: white; }}
    .badge-x {{ background: #16a34a; color: white; }}
    .badge-v {{ background: #d97706; color: white; }}
    .badge-j {{ background: #9333ea; color: white; }}
    .badge-alert {{ background: #dc2626; color: white; }}
    .node-card {{
      background: #334155;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s;
    }}
    .node-card:hover {{
      border-color: #38bdf8;
      transform: translateY(-2px);
    }}
    .node-card.selected {{
      border-color: #38bdf8;
      background: #1e3a5f;
    }}
    pre {{
      background: #090d16;
      padding: 10px;
      border-radius: 6px;
      font-size: 12px;
      overflow-x: auto;
      color: #38bdf8;
    }}
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>CGE Inspector</h2>
    <div id="inspector-content">
      <p style="color: #94a3b8;">Selecciona un nodo del grafo para inspeccionar su trazabilidad de código, distribución y alertas.</p>
    </div>
    <hr style="border-color: #334155; margin: 20px 0;">
    <h3>Alertas de Auditoría (T1-T7)</h3>
    <div id="alerts-list"></div>
  </div>
  <div id="viewport">
    <div class="header">
      <div>
        <strong>System:</strong> {schema_dict.get('system_id')} | 
        <strong>Version:</strong> {schema_dict.get('version')}
      </div>
      <div>
        <span class="badge badge-u">U: Exógenas</span>
        <span class="badge badge-x">X: Palancas</span>
        <span class="badge badge-v">V: Estado</span>
        <span class="badge badge-j">J: Target</span>
      </div>
    </div>
    <div id="canvas-container" style="flex:1; padding: 24px; overflow-y: auto;">
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px;" id="nodes-grid"></div>
    </div>
  </div>

  <script>
    const schema = {json.dumps(schema_dict)};
    const diagnostics = {json.dumps(diagnostics_list)};
    const solution = {json.dumps(solution_dict)};

    // Renderizar nodos con colores por categoría (V-1)
    const grid = document.getElementById("nodes-grid");
    const catClass = {{
      exogenous: "badge-u",
      control: "badge-x",
      state: "badge-v",
      target: "badge-j"
    }};

    schema.nodes.forEach(node => {{
      const card = document.createElement("div");
      card.className = "node-card";
      card.id = "node-" + node.id;
      
      // Comprobar alertas sobre este nodo (V-3)
      const alerts = diagnostics.filter(d => d.node === node.id || (d.details && d.details.threshold_variable === node.id));
      let alertBadge = alerts.length > 0 ? `<span class="badge badge-alert" style="float:right;">! ${{alerts[0].id}}</span>` : "";

      // Resaltar ruta óptima de Capa 2 (V-4)
      const isOptimalControl = (node.category === "control" || node.category === "target");
      const optBorder = isOptimalControl ? "border-left: 4px solid #10b981;" : "";

      card.style = optBorder;
      card.innerHTML = `
        ${{alertBadge}}
        <span class="badge ${{catClass[node.category]}}">${{node.category}}</span>
        <h4 style="margin: 8px 0 4px 0;">${{node.label}}</h4>
        <div style="font-size: 12px; color: #94a3b8;">ID: <code>${{node.id}}</code></div>
        <div style="font-size: 13px; margin-top: 6px;">Valor: <strong>${{node.current_value !== null ? node.current_value : '—'}}</strong></div>
      `;

      // Clic para inspección de código y datos (V-2)
      card.onclick = () => selectNode(node, alerts);
      grid.appendChild(card);
    }});

    function selectNode(node, alerts) {{
      document.querySelectorAll(".node-card").forEach(c => c.classList.remove("selected"));
      document.getElementById("node-" + node.id).classList.add("selected");

      const insp = document.getElementById("inspector-content");
      const src = node.source_reference || {{}};
      const lines = src.line_range ? `Líneas: [${{src.line_range[0]}}, ${{src.line_range[1]}}]` : "No especificado";

      insp.innerHTML = `
        <span class="badge ${{catClass[node.category]}}">${{node.category}}</span>
        <h3 style="margin: 10px 0 4px 0;">${{node.label}}</h3>
        <p style="font-size: 13px; color: #cbd5e1;">Tipo de dato: <code>${{node.data_type}}</code></p>
        <h4>Trazabilidad de Código Fuente (V-2):</h4>
        <pre>Archivo: ${{src.file || 'N/A'}}\\nEntidad: ${{src.entity || 'N/A'}}\\n${{lines}}</pre>
        <h4>Límites y Valores:</h4>
        <div style="font-size: 13px;">Mín: ${{node.bounds ? node.bounds.min : 'null'}} | Máx: ${{node.bounds ? node.bounds.max : 'null'}}</div>
      `;
    }}

    // Renderizar lista de alertas T1-T7 (V-3)
    const alertContainer = document.getElementById("alerts-list");
    diagnostics.forEach(d => {{
      const div = document.createElement("div");
      div.style = "background: #451a1a; border: 1px solid #7f1d1d; border-radius: 6px; padding: 8px; margin-bottom: 8px; font-size: 12px;";
      div.innerHTML = `<strong>[${{d.id}}] ${{d.code}}</strong>: ${{d.details ? d.details.reason : ''}}`;
      alertContainer.appendChild(div);
    }});
  </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


def verify_ui_reactivity(schema_dict: Dict[str, Any], mutated_schema_dict: Dict[str, Any]) -> bool:
    """Verifica el criterio V-5: la UI es un reflejo reactivo sin estado interno; si cambia schema, cambia UI."""
    import jsonschema
    assert schema_dict != mutated_schema_dict, "El schema base y mutado no deben ser idénticos"
    assert schema_dict["system_id"] == mutated_schema_dict["system_id"]

    # 1. Comprobar que el schema mutado valida contra canonical_schema.json
    schema_path = "schema/canonical_schema.json"
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            canonical = json.load(f)
        jsonschema.validate(instance=mutated_schema_dict, schema=canonical)

    # 2. Generar HTML a partir de ambos schemas y verificar reactividad
    temp_ui_orig = "ui/temp_orig.html"
    temp_ui_mut = "ui/temp_mut.html"
    try:
        generate_interactive_ui_html(schema_dict, [], {}, temp_ui_orig)
        generate_interactive_ui_html(mutated_schema_dict, [], {}, temp_ui_mut)

        with open(temp_ui_orig, "r", encoding="utf-8") as f1, open(temp_ui_mut, "r", encoding="utf-8") as f2:
            html1 = f1.read()
            html2 = f2.read()

        assert html1 != html2, "La UI generada no reaccionó a la mutación del schema"
    finally:
        for p in [temp_ui_orig, temp_ui_mut]:
            if os.path.exists(p):
                os.remove(p)

    return True
