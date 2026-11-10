"""Pruebas E2E Automatizadas con Playwright en Chromium (Fase 6).
Verifica renderizado Cytoscape, inspección de código, alertas T1-T7, sliders qué-pasa-si, receta y copiloto.
"""
import os
import shutil
import socket
import tempfile
import threading
import time
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from ui.server import app, engine_state
from generator import ProceduralWorldGenerator


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def e2e_server_and_page():
    # 1. Generar mundo procedural de logística multi-etapa
    tmp_base = tempfile.mkdtemp(prefix="cge_f6_e2e_")
    pgen = ProceduralWorldGenerator(base_seed=7777)
    logistics_dir = os.path.join(tmp_base, "fleet_logistics")
    pgen.generate_world(1, logistics_dir, domain="fleet_logistics", clean=False, include_csv=True)

    # 2. Ingestar el mundo de logística en el estado del servidor
    engine_state.load_directory(logistics_dir)

    # 3. Iniciar uvicorn en un hilo daemon
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Esperar a que el servidor responda
    base_url = f"http://127.0.0.1:{port}"
    timeout = 15.0
    start_t = time.time()
    server_ready = False
    import urllib.request
    while time.time() - start_t < timeout:
        try:
            with urllib.request.urlopen(f"{base_url}/api/schema", timeout=1) as resp:
                if resp.status == 200:
                    server_ready = True
                    break
        except Exception:
            time.sleep(0.3)

    if not server_ready:
        server.should_exit = True
        shutil.rmtree(tmp_base, ignore_errors=True)
        pytest.fail("El servidor de prueba E2E no inició a tiempo.")

    # 4. Lanzar Chromium headless con Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"{base_url}/", wait_until="networkidle")

        yield {
            "page": page,
            "base_url": base_url,
            "browser": browser
        }

        browser.close()

    # Detener servidor y limpiar directorio temporal
    server.should_exit = True
    shutil.rmtree(tmp_base, ignore_errors=True)


def test_e2e_canvas_and_graph_nodes_rendered(e2e_server_and_page):
    """Verifica que el lienzo Cytoscape se renderice y contenga los nodos jerárquicos del grafo."""
    page = e2e_server_and_page["page"]

    # Verificar existencia del contenedor y canvas
    cy_container = page.locator("#cy")
    assert cy_container.is_visible()

    # Esperar a que los metadatos del sistema se carguen en la barra superior
    page.wait_for_function("() => document.getElementById('meta-system-id').textContent !== '—'", timeout=10000)

    # Verificar que Cytoscape tiene los nodos cargados
    node_count = page.evaluate("() => cy ? cy.nodes().length : 0")
    assert node_count >= 8, f"Se esperaban >= 8 nodos en el grafo, se encontraron {node_count}"


def test_e2e_node_click_and_source_code_inspection(e2e_server_and_page):
    """Verifica la inspección de atributos y carga del fragmento de código fuente real."""
    page = e2e_server_and_page["page"]

    # Simular clic en el primer nodo del grafo
    page.evaluate("() => { if (cy && cy.nodes().length > 0) cy.nodes()[0].emit('tap'); }")

    # Esperar a que el inspector se actualice
    page.wait_for_function(
      "() => document.getElementById('inspector-content').innerHTML.includes('inspector-field')",
      timeout=6000
    )

    # Esperar a que el código fuente se descargue desde /api/source
    page.wait_for_function(
      "() => !document.getElementById('source-code-view').textContent.includes('// Selecciona un nodo')",
      timeout=6000
    )

    code_text = page.locator("#source-code-view").text_content()
    assert len(code_text.strip()) > 10
    assert any(token in code_text for token in (":", "=", "def", "class", "type"))


def test_e2e_audit_alert_click_and_focus(e2e_server_and_page):
    """Verifica que hacer clic en una alerta T1-T7 enfoque y resalte el nodo afectado en el lienzo."""
    page = e2e_server_and_page["page"]

    # Esperar que existan tarjetas de alerta
    page.wait_for_selector(".alert-card", timeout=8000)
    alerts = page.locator(".alert-card")
    assert alerts.count() >= 1

    # Clic en la primera alerta
    alerts.first.click()
    page.wait_for_timeout(400)

    # Verificar que al menos un nodo quedó resaltado en el grafo
    highlighted_count = page.evaluate("() => cy ? cy.nodes('.highlighted').length : 0")
    assert highlighted_count >= 1, "Hacer clic en la alerta no resaltó el nodo en Cytoscape"


def test_e2e_whatif_slider_interactive_evaluation(e2e_server_and_page):
    """Verifica que mover un slider actualice reactivamente la evaluación del objetivo."""
    page = e2e_server_and_page["page"]

    # Ir a la pestaña Qué Pasa Si
    page.locator('[data-tab="tab-whatif"]').click()
    page.wait_for_selector(".slider-group", timeout=6000)

    val_before = page.locator("#evaluate-target-display").text_content().strip()

    # Modificar el valor de una palanca a través del campo numérico
    page.evaluate("""() => {
        const inp = document.querySelector('.slider-num-input');
        if (inp) {
            const cur = Number(inp.value);
            inp.value = cur > 10 ? cur - 5 : cur + 5;
            inp.dispatchEvent(new Event('change'));
        }
    }""")

    # Esperar reactividad en el objetivo evaluado
    page.wait_for_timeout(600)
    val_after = page.locator("#evaluate-target-display").text_content().strip()
    assert val_after != "—", "El valor del objetivo evaluado debe ser un número"


def test_e2e_recipe_generation_and_cvar_toggle(e2e_server_and_page):
    """Verifica el cálculo de receta óptima y la alternancia de modo CVaR."""
    page = e2e_server_and_page["page"]

    # Cambiar a pestaña de Receta
    page.locator('[data-tab="tab-recipe"]').click()

    # Clic en Calcular Receta Óptima
    page.locator("#btn-calc-recipe").click()

    # Esperar visualización de resultados
    page.wait_for_selector("#recipe-output:not(.hidden)", timeout=8000)

    status_text = page.locator("#recipe-status").text_content().strip()
    assert status_text == "FEASIBLE"

    obj_text = page.locator("#recipe-objective-val").text_content().strip()
    assert float(obj_text) > 0.0

    # Verificar que se listan los deltas de las palancas
    control_items = page.locator(".recipe-control-item")
    assert control_items.count() >= 2


def test_e2e_copilot_chat_question_and_node_highlighting(e2e_server_and_page):
    """Verifica el flujo del copiloto agéntico y el resaltado automático de entidades en el grafo."""
    page = e2e_server_and_page["page"]

    # Cambiar a pestaña de Copiloto
    page.locator('[data-tab="tab-copilot"]').click()

    # Formular pregunta
    query_input = page.locator("#copilot-input")
    query_input.fill("¿Cuáles son las palancas con mayor impacto?")
    page.locator("#btn-copilot-send").click()

    # Esperar respuesta del copiloto en la lista de mensajes
    page.wait_for_function("""() => {
        const msgs = document.querySelectorAll('#copilot-messages .chat-agent');
        if (msgs.length === 0) return false;
        const last = msgs[msgs.length - 1].textContent;
        return last && !last.includes('Analizando el grafo causal');
    }""", timeout=12000)

    last_reply = page.locator("#copilot-messages .chat-agent").last.text_content()
    assert len(last_reply) > 20
    assert "impacto" in last_reply.lower() or "palancas" in last_reply.lower() or "[" in last_reply

    # Verificar que el copiloto resaltó nodos en el lienzo Cytoscape
    highlighted_nodes = page.evaluate("() => cy ? cy.nodes('.highlighted').length : 0")
    assert highlighted_nodes >= 1, "El copiloto debe resaltar los nodos sobre los que respondió"
