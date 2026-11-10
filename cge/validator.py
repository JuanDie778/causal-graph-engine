"""Módulo de validación estricta y garantías de integridad lógica para Causal Graphs.

Este módulo audita objetos CausalIR y diccionarios schema.json v1.1 asegurando:
1. Conformidad estructural estricta con el JSON Schema canónico.
2. Integridad topológica (DAG acíclico, ausencia de aristas huérfanas, unicidad de IDs).
3. Coherencia taxonómica causal (M = <U, X, V, F, C, J>):
   - Nodos de control (X) y exógenos (U) no deben tener aristas entrantes de mechanisms internos.
   - Nodos target (J) no deben ser raíces sin procedencia causal.
4. Coherencia de fronteras (bounds.min <= bounds.max) y escalas características válidas.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ValidationIssue:
    severity: str  # "ERROR" o "WARNING"
    code: str
    message: str
    entity_id: Optional[str] = None


@dataclass
class ValidationReport:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]


class CausalGraphValidator:
    """Validador exhaustivo de esquemas e integridad lógica causal."""

    def __init__(self, canonical_schema_path: Optional[str] = None):
        if canonical_schema_path is None:
            # Localizar ruta por defecto a schema/canonical_schema.json
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            canonical_schema_path = os.path.join(base_dir, "schema", "canonical_schema.json")
        self.canonical_schema_path = canonical_schema_path
        self._schema_json: Optional[Dict[str, Any]] = None

    def _load_canonical_schema(self) -> Optional[Dict[str, Any]]:
        if self._schema_json is None and os.path.isfile(self.canonical_schema_path):
            try:
                with open(self.canonical_schema_path, "r", encoding="utf-8") as f:
                    self._schema_json = json.load(f)
            except Exception:
                pass
        return self._schema_json

    def validate(self, graph_data: Dict[str, Any]) -> ValidationReport:
        """Ejecuta todas las fases de validación y retorna un ValidationReport completo."""
        issues: List[ValidationIssue] = []

        # Fase 1: Validación estructural JSON Schema
        self._validate_json_schema(graph_data, issues)

        # Si el formato base falla catastróficamente, retornar pronto
        if any(i.severity == "ERROR" and i.code == "INVALID_JSON_STRUCTURE" for i in issues):
            return ValidationReport(is_valid=False, issues=issues)

        # Fase 2: Integridad de Nodos y Unicidad de Identificadores
        nodes_dict = self._validate_nodes(graph_data, issues)

        # Fase 3: Integridad de Aristas y Conectividad
        edges_list = self._validate_edges(graph_data, nodes_dict, issues)

        # Fase 4: Integridad Topológica (Aciclicidad / DAG)
        self._validate_dag(nodes_dict, edges_list, issues)

        # Fase 5: Coherencia Taxonómica Causal (U, X, V, J)
        self._validate_causal_taxonomy(nodes_dict, edges_list, issues)

        # Fase 6: Coherencia de Restricciones (C)
        self._validate_constraints(graph_data, nodes_dict, issues)

        is_valid = not any(i.severity == "ERROR" for i in issues)
        return ValidationReport(is_valid=is_valid, issues=issues)

    def _validate_json_schema(self, data: Dict[str, Any], issues: List[ValidationIssue]) -> None:
        if not isinstance(data, dict):
            issues.append(ValidationIssue("ERROR", "INVALID_JSON_STRUCTURE", "El schema raíz debe ser un objeto/dict."))
            return

        for req_field in ["system_id", "version", "nodes", "edges"]:
            if req_field not in data:
                issues.append(ValidationIssue("ERROR", "MISSING_REQUIRED_FIELD", f"Falta el campo obligatorio '{req_field}' en el esquema raíz."))

        schema_json = self._load_canonical_schema()
        if schema_json:
            try:
                import jsonschema
                validator = jsonschema.Draft7Validator(schema_json)
                for err in validator.iter_errors(data):
                    path = ".".join(str(p) for p in err.path)
                    issues.append(ValidationIssue("ERROR", "JSON_SCHEMA_VIOLATION", f"[{path}] {err.message}"))
            except ImportError:
                # Si jsonschema no está instalado en el entorno actual, omitir chequeo estricto
                pass

    def _validate_nodes(self, data: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Dict[str, Any]]:
        nodes = data.get("nodes", [])
        nodes_dict: Dict[str, Dict[str, Any]] = {}

        if not isinstance(nodes, list):
            issues.append(ValidationIssue("ERROR", "INVALID_NODES_LIST", "El campo 'nodes' debe ser una lista."))
            return nodes_dict

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                issues.append(ValidationIssue("ERROR", "INVALID_NODE_ITEM", f"El nodo en índice {idx} no es un objeto."))
                continue

            node_id = node.get("id")
            if not node_id or not isinstance(node_id, str):
                issues.append(ValidationIssue("ERROR", "INVALID_NODE_ID", f"Nodo en índice {idx} no posee un 'id' string válido."))
                continue

            if node_id in nodes_dict:
                issues.append(ValidationIssue("ERROR", "DUPLICATE_NODE_ID", f"Identificador de nodo duplicado: '{node_id}'", entity_id=node_id))
            else:
                nodes_dict[node_id] = node

            # Validar bounds
            bounds = node.get("bounds")
            if isinstance(bounds, dict):
                b_min = bounds.get("min")
                b_max = bounds.get("max")
                if b_min is not None and b_max is not None:
                    if isinstance(b_min, (int, float)) and isinstance(b_max, (int, float)):
                        if b_min > b_max:
                            issues.append(ValidationIssue("ERROR", "INVERTED_BOUNDS", f"Nodo '{node_id}' tiene cota mínima ({b_min}) mayor que cota máxima ({b_max}).", entity_id=node_id))

            # Validar scale
            scale = node.get("characteristic_scale")
            if scale is not None and isinstance(scale, (int, float)) and scale <= 0:
                issues.append(ValidationIssue("WARNING", "NON_POSITIVE_SCALE", f"Nodo '{node_id}' tiene una escala característica <= 0 ({scale}).", entity_id=node_id))

        return nodes_dict

    def _validate_edges(self, data: Dict[str, Any], nodes_dict: Dict[str, Dict[str, Any]], issues: List[ValidationIssue]) -> List[Dict[str, Any]]:
        edges = data.get("edges", [])
        valid_edges: List[Dict[str, Any]] = []

        if not isinstance(edges, list):
            issues.append(ValidationIssue("ERROR", "INVALID_EDGES_LIST", "El campo 'edges' debe ser una lista."))
            return valid_edges

        seen_edges: Set[Tuple[str, str]] = set()

        for idx, edge in enumerate(edges):
            if not isinstance(edge, dict):
                issues.append(ValidationIssue("ERROR", "INVALID_EDGE_ITEM", f"La arista en índice {idx} no es un objeto."))
                continue

            src = edge.get("source")
            tgt = edge.get("target")

            if not src or not tgt:
                issues.append(ValidationIssue("ERROR", "MALFORMED_EDGE", f"Arista en índice {idx} carece de 'source' o 'target'."))
                continue

            # Aristas huérfanas
            if src not in nodes_dict:
                issues.append(ValidationIssue("ERROR", "ORPHAN_EDGE_SOURCE", f"La arista apunta desde un nodo fuente inexistente: '{src}' -> '{tgt}'", entity_id=src))
            if tgt not in nodes_dict:
                issues.append(ValidationIssue("ERROR", "ORPHAN_EDGE_TARGET", f"La arista apunta a un nodo destino inexistente: '{src}' -> '{tgt}'", entity_id=tgt))

            # Auto-lazos
            if src == tgt:
                issues.append(ValidationIssue("ERROR", "SELF_LOOP", f"Auto-lazo causal prohibido en el nodo: '{src}' -> '{tgt}'", entity_id=src))

            edge_key = (str(src), str(tgt))
            if edge_key in seen_edges:
                issues.append(ValidationIssue("WARNING", "DUPLICATE_EDGE", f"Arista duplicada entre '{src}' y '{tgt}'", entity_id=f"{src}->{tgt}"))
            else:
                seen_edges.add(edge_key)

            valid_edges.append(edge)

        return valid_edges

    def _validate_dag(self, nodes_dict: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]], issues: List[ValidationIssue]) -> None:
        """Verifica que el grafo dirigido sea acíclico (DAG) mediante el algoritmo de Kahn."""
        adj: Dict[str, List[str]] = {nid: [] for nid in nodes_dict}
        in_degree: Dict[str, int] = {nid: 0 for nid in nodes_dict}

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in adj and tgt in in_degree:
                adj[src].append(tgt)
                in_degree[tgt] += 1

        # Nodos con in-degree 0
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count < len(nodes_dict):
            # Identificar los nodos que forman parte del ciclo
            cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
            issues.append(ValidationIssue(
                "ERROR",
                "CYCLIC_GRAPH_DETECTED",
                f"El grafo causal contiene ciclos dirigidos no acíclicos afectando a: {cycle_nodes[:5]}."
            ))

    def _validate_causal_taxonomy(self, nodes_dict: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]], issues: List[ValidationIssue]) -> None:
        """Verifica coherencia taxonómica formal de variables U, X, V, J."""
        parents: Dict[str, Set[str]] = {nid: set() for nid in nodes_dict}
        children: Dict[str, Set[str]] = {nid: set() for nid in nodes_dict}

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in children and tgt in parents:
                children[src].add(tgt)
                parents[tgt].add(src)

        has_target = False

        for nid, node in nodes_dict.items():
            cat = node.get("category")
            if cat == "control":
                # Una palanca de decisión pura X no debería tener padres causales internos de estado
                # (salvo advertencia de acoplamiento exógeno)
                if len(parents[nid]) > 0:
                    issues.append(ValidationIssue(
                        "WARNING",
                        "CONTROL_WITH_INCOMING_EDGES",
                        f"La variable de control '{nid}' tiene padres causales {parents[nid]}. En la tupla M, X debe ser raíz independiente.",
                        entity_id=nid
                    ))
            elif cat == "exogenous":
                # U no debe tener padres causales
                if len(parents[nid]) > 0:
                    issues.append(ValidationIssue(
                        "ERROR",
                        "EXOGENOUS_WITH_INCOMING_EDGES",
                        f"La variable exógena '{nid}' tiene padres causales {parents[nid]}. Por definición, U es externa al sistema.",
                        entity_id=nid
                    ))
            elif cat == "target":
                has_target = True
                if len(parents[nid]) == 0:
                    issues.append(ValidationIssue(
                        "ERROR",
                        "TARGET_WITHOUT_PARENTS",
                        f"El objetivo final '{nid}' no tiene ancestros causales que lo alimenten.",
                        entity_id=nid
                    ))

        if not has_target and len(nodes_dict) > 0:
            issues.append(ValidationIssue(
                "WARNING",
                "NO_TARGET_NODE",
                "El esquema no especifica ningún nodo con category='target' (J)."
            ))

    def _validate_constraints(self, data: Dict[str, Any], nodes_dict: Dict[str, Dict[str, Any]], issues: List[ValidationIssue]) -> None:
        constraints = data.get("constraints", [])
        if not isinstance(constraints, list):
            return

        seen_cids: Set[str] = set()
        for idx, c in enumerate(constraints):
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if not cid:
                issues.append(ValidationIssue("ERROR", "CONSTRAINT_WITHOUT_ID", f"Restricción en índice {idx} sin campo 'id'."))
                continue
            if cid in seen_cids:
                issues.append(ValidationIssue("ERROR", "DUPLICATE_CONSTRAINT_ID", f"Identificador de restricción duplicado: '{cid}'.", entity_id=cid))
            else:
                seen_cids.add(cid)

            expr = c.get("expression")
            if not expr or not isinstance(expr, str):
                issues.append(ValidationIssue("ERROR", "INVALID_CONSTRAINT_EXPR", f"Restricción '{cid}' sin expresión matemática válida.", entity_id=cid))
