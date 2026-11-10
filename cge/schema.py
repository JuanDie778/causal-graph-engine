"""Modelos de datos y serialización del esquema canónico CGE."""
import json
import os
from typing import List, Dict, Any, Optional
import jsonschema


class SourceReference:
    def __init__(
        self,
        file: Optional[str] = None,
        entity: Optional[str] = None,
        line_range: Optional[List[int]] = None,
        column_name: Optional[str] = None
    ):
        self.file = file
        self.entity = entity
        self.line_range = line_range
        self.column_name = column_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "entity": self.entity,
            "line_range": self.line_range,
            "column_name": self.column_name
        }


class Node:
    def __init__(
        self,
        id: str,
        label: str,
        category: str,
        data_type: str,
        source_reference: SourceReference,
        current_value: Any = None,
        bounds: Optional[Dict[str, Optional[float]]] = None,
        distribution: Optional[Dict[str, Any]] = None,
        characteristic_scale: Optional[float] = None,
        symbolic_expr: Optional[str] = None,
        discontinuities: Optional[List[Dict[str, Any]]] = None,
        clarke_kink_index: Optional[float] = None,
        is_dead: bool = False,
        fidelity: Optional[Dict[str, Any]] = None
    ):
        assert category in ["exogenous", "control", "state", "target"], f"Categoría inválida: {category}"
        assert data_type in ["continuous", "discrete", "boolean", "categorical"], f"Tipo inválido: {data_type}"
        self.id = id
        self.label = label
        self.category = category
        self.data_type = data_type
        self.source_reference = source_reference
        self.current_value = current_value
        self.bounds = bounds or {"min": None, "max": None}
        self.distribution = distribution
        self.characteristic_scale = characteristic_scale
        self.symbolic_expr = symbolic_expr
        self.discontinuities = discontinuities or []
        self.clarke_kink_index = clarke_kink_index
        self.is_dead = bool(is_dead)
        self.fidelity = fidelity

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "data_type": self.data_type,
            "source_reference": self.source_reference.to_dict(),
            "current_value": self.current_value,
            "bounds": self.bounds
        }
        if self.distribution is not None:
            d["distribution"] = self.distribution
        if self.characteristic_scale is not None:
            d["characteristic_scale"] = self.characteristic_scale
        if self.symbolic_expr is not None:
            d["symbolic_expr"] = self.symbolic_expr
        if self.discontinuities:
            d["discontinuities"] = self.discontinuities
        if self.clarke_kink_index is not None:
            d["clarke_kink_index"] = self.clarke_kink_index
        if self.fidelity is not None:
            d["fidelity"] = self.fidelity
        return d


class Edge:
    def __init__(
        self,
        source: str,
        target: str,
        type: str,
        sensitivity_weight: Optional[float] = None,
        confidence: float = 1.0,
        exact_mechanism: Optional[Dict[str, str]] = None,
        dimensionless_elasticity: Optional[float] = None
    ):
        assert type in ["exact_code", "discrete_rule", "empirical_data"], f"Tipo de arista inválido: {type}"
        self.source = source
        self.target = target
        self.type = type
        self.sensitivity_weight = sensitivity_weight
        self.confidence = confidence
        self.exact_mechanism = exact_mechanism
        self.dimensionless_elasticity = dimensionless_elasticity

    @property
    def engine(self) -> Optional[str]:
        return (self.exact_mechanism or {}).get("engine")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "sensitivity_weight": self.sensitivity_weight,
            "confidence": self.confidence,
            "exact_mechanism": self.exact_mechanism
        }
        if self.dimensionless_elasticity is not None:
            d["dimensionless_elasticity"] = self.dimensionless_elasticity
        return d


class Constraint:
    def __init__(
        self,
        id: str,
        type: str,
        expression: str,
        description: str = "",
        is_active: bool = False
    ):
        assert type in ["hard", "soft"], f"Tipo de restricción inválido: {type}"
        self.id = id
        self.type = type
        self.expression = expression
        self.description = description
        self.is_active = is_active

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "expression": self.expression,
            "description": self.description,
            "is_active": self.is_active
        }


class CausalGraph:
    def __init__(
        self,
        system_id: str = "cge_system",
        version: str = "1.1.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.system_id = system_id
        self.version = version
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.constraints: Dict[str, Constraint] = {}
        self.metadata: Dict[str, Any] = metadata or {}
        self._blackbox_callables: Dict[str, Any] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def add_constraint(self, constraint: Constraint):
        self.constraints[constraint.id] = constraint

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "system_id": self.system_id,
            "version": self.version,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "constraints": [c.to_dict() for c in self.constraints.values()]
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        g = cls(
            system_id=data.get("system_id", "cge_system"),
            version=data.get("version", "1.1.0"),
            metadata=data.get("metadata", {})
        )
        for nd in data.get("nodes", []):
            sr = nd.get("source_reference") or {}
            g.add_node(Node(
                id=nd["id"],
                label=nd.get("label", nd["id"]),
                category=nd["category"],
                data_type=nd.get("data_type", "continuous"),
                source_reference=SourceReference(
                    file=sr.get("file"),
                    entity=sr.get("entity"),
                    line_range=sr.get("line_range"),
                    column_name=sr.get("column_name")
                ),
                current_value=nd.get("current_value"),
                bounds=nd.get("bounds"),
                distribution=nd.get("distribution"),
                characteristic_scale=nd.get("characteristic_scale"),
                symbolic_expr=nd.get("symbolic_expr"),
                discontinuities=nd.get("discontinuities"),
                clarke_kink_index=nd.get("clarke_kink_index"),
                is_dead=bool(nd.get("is_dead", False)),
                fidelity=nd.get("fidelity")
            ))
        for ed in data.get("edges", []):
            g.add_edge(Edge(
                source=ed["source"],
                target=ed["target"],
                type=ed["type"],
                sensitivity_weight=ed.get("sensitivity_weight"),
                confidence=ed.get("confidence", 1.0),
                exact_mechanism=ed.get("exact_mechanism"),
                dimensionless_elasticity=ed.get("dimensionless_elasticity")
            ))
        for cd in data.get("constraints", []):
            g.add_constraint(Constraint(
                id=cd["id"],
                type=cd["type"],
                expression=cd["expression"],
                description=cd.get("description", ""),
                is_active=cd.get("is_active", False)
            ))
        # Recomputar topológicamente is_dead para nodos de estado sin camino hacia target
        target_ids = [nid for nid, n in g.nodes.items() if n.category == "target"]
        if target_ids:
            import networkx as nx
            dag = nx.DiGraph()
            for nid in g.nodes:
                dag.add_node(nid)
            for e in g.edges:
                if e.type != "empirical_data":
                    dag.add_edge(e.source, e.target)
            t_id = target_ids[0]
            for nid, n in g.nodes.items():
                if n.category == "state" and nid != t_id:
                    n.is_dead = not nx.has_path(dag, nid, t_id)
        return g

    def validate_canonical(self, schema_path: str = "schema/canonical_schema.json") -> bool:
        """Valida este grafo contra el JSON Schema canónico."""
        with open(schema_path, "r", encoding="utf-8") as f:
            canonical_schema = json.load(f)
        jsonschema.validate(instance=self.to_dict(), schema=canonical_schema)
        return True

    def save_to_file(self, file_path: str):
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

