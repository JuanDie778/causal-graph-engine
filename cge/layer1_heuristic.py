"""Capa 1: Enrutamiento Causal en el Semianillo Tropical Max-Product y AD en Modo Inverso (< 50 ms)."""
from __future__ import annotations
import math
import time
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx


class Layer1Router:
    """Enrutador sobre el grafo causal ponderado usando programación dinámica Max-Product y AD inverso."""

    def __init__(
        self,
        causal_graph_dict: Optional[Any] = None,
        graph: Optional[Any] = None,
        source_dir: Optional[str] = None
    ):
        self.graph = nx.DiGraph()
        self.control_nodes: List[str] = []
        self.target_nodes: List[str] = []
        target_g = graph if graph is not None else causal_graph_dict
        if target_g is None and source_dir is not None:
            from .ingest import ASTIngestPipeline
            target_g = ASTIngestPipeline(source_dir).run()
        if target_g is not None:
            self.load_from_dict(target_g.to_dict() if hasattr(target_g, "to_dict") else target_g)

    def load_from_dict(self, data: Dict[str, Any]):
        self.graph.clear()
        self.control_nodes = []
        self.target_nodes = []

        for node in data.get("nodes", []):
            nid = node["id"]
            cat = node.get("category")
            self.graph.add_node(nid, **node)
            if cat == "control":
                self.control_nodes.append(nid)
            elif cat == "target":
                self.target_nodes.append(nid)

        for edge in data.get("edges", []):
            u = edge["source"]
            v = edge["target"]
            e_type = edge.get("type", "exact_code")
            # Si ya existe una arista de código exacto, no sobrescribir su peso con la arista empírica paralela
            if self.graph.has_edge(u, v) and e_type == "empirical_data":
                continue
            weight = edge.get("sensitivity_weight")
            dim_elast = edge.get("dimensionless_elasticity")
            if dim_elast is None and weight is not None:
                s_u = (self.graph.nodes[u].get("characteristic_scale") if self.graph.has_node(u) else None) or 1.0
                s_v = (self.graph.nodes[v].get("characteristic_scale") if self.graph.has_node(v) else None) or 1.0
                dim_elast = float(weight) * (float(s_u) / max(float(s_v), 1e-6))
            dim_val = float(dim_elast) if dim_elast is not None else (float(weight) if weight is not None else 1.0)
            cost = 1.0 / (abs(dim_val) + 1e-4)
            self.graph.add_edge(u, v, weight=cost, original_weight=weight, dimensionless_elasticity=dim_val, edge_data=edge)

    def _get_default_target(self) -> Optional[str]:
        if self.target_nodes:
            return self.target_nodes[0]
        sinks = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        return sinks[0] if sinks else (next(iter(self.graph.nodes)) if self.graph.nodes else None)

    def find_top_k_paths(self, source: str, target: str, k: int = 3) -> List[Dict[str, Any]]:
        """Devuelve las Top-K rutas causales entre 'source' y 'target' maximizando el producto
        multiplicativo de elasticidades adimensionales normalizadas |prod w_tilde| en el semianillo Tropical Max-Product.
        """
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []

        # Para grafos causales con pesos de sensibilidad explícitos y tamaño moderado, usar DP exacta Max-Product sobre DAG
        if self.graph.number_of_nodes() <= 500 and nx.is_directed_acyclic_graph(self.graph):
            topo_nodes = list(nx.topological_sort(self.graph))
            dp: Dict[str, List[Tuple[float, float, List[str]]]] = {nid: [] for nid in topo_nodes}
            dp[source] = [(1.0, 1.0, [source])]

            for u in topo_nodes:
                if not dp[u]:
                    continue
                for v in self.graph.successors(u):
                    ed = self.graph[u][v]
                    dim_w = ed.get("dimensionless_elasticity")
                    if dim_w is None:
                        orig_w = ed.get("original_weight")
                        dim_w = float(orig_w) if orig_w is not None else float(ed.get("weight", 1.0))
                    else:
                        dim_w = float(dim_w)
                    orig_w = ed.get("original_weight")
                    w_orig_val = float(orig_w) if orig_w is not None else dim_w
                    abs_w = max(abs(dim_w), 1e-9)
                    for abs_prod, signed_prod, p_list in dp[u]:
                        dp[v].append((abs_prod * abs_w, signed_prod * w_orig_val, p_list + [v]))
                    if len(dp[v]) > k * 2:
                        dp[v].sort(key=lambda item: item[0], reverse=True)
                        dp[v] = dp[v][:k * 2]

            target_candidates = dp.get(target, [])
            target_candidates.sort(key=lambda item: item[0], reverse=True)
            results = []
            for abs_prod, signed_prod, p in target_candidates[:k]:
                edge_details = [
                    self.graph[p[i]][p[i + 1]].get("edge_data", {})
                    for i in range(len(p) - 1)
                ]
                results.append({
                    "path": p,
                    "length": len(p),
                    "cumulative_sensitivity": signed_prod,
                    "abs_product": abs_prod,
                    "edges": edge_details
                })
            return results

        # Enrutamiento de alta velocidad con penalización sucesiva para grafos sintéticos masivos (> 500 nodos)
        paths = []
        temp_penalties = []
        for _ in range(k):
            try:
                p = nx.shortest_path(self.graph, source, target, weight="weight")
                cumulative_sensitivity = 1.0
                edge_details = []
                for i in range(len(p) - 1):
                    u_node, v_node = p[i], p[i + 1]
                    ed = self.graph[u_node][v_node]
                    orig_w = ed.get("original_weight")
                    if orig_w is not None:
                        cumulative_sensitivity *= orig_w
                    edge_details.append(ed.get("edge_data", {}))

                paths.append({
                    "path": p,
                    "length": len(p),
                    "cumulative_sensitivity": cumulative_sensitivity,
                    "edges": edge_details
                })

                for i in range(len(p) - 1):
                    u_node, v_node = p[i], p[i + 1]
                    cur_w = self.graph[u_node][v_node]["weight"]
                    self.graph[u_node][v_node]["weight"] = cur_w * 3.0
                    temp_penalties.append((u_node, v_node, cur_w))
            except nx.NetworkXNoPath:
                break

        for u_node, v_node, orig_w in reversed(temp_penalties):
            self.graph[u_node][v_node]["weight"] = orig_w

        return paths

    def get_all_control_to_target_paths(
        self,
        target: Optional[str] = None,
        k_per_control: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Obtiene las mejores rutas para cada palanca de control hacia el nodo objetivo."""
        t = target or self._get_default_target()
        if t is None:
            return {}
        result = {}
        for c in self.control_nodes:
            result[c] = self.find_top_k_paths(c, t, k=k_per_control)
        return result

    def compute_path_sensitivity(self, path: List[str]) -> float:
        """Calcula la sensibilidad causal acumulada a lo largo de un camino dirigido aplicando la regla de la cadena."""
        if len(path) < 2:
            return 0.0
        cumulative = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.graph.has_edge(u, v):
                return 0.0
            w = self.graph[u][v].get("original_weight")
            if w is None:
                return 0.0
            cumulative *= w
        return cumulative

    def compute_reverse_mode_ad_adjoints(self, target: str) -> Dict[str, float]:
        """Calcula las derivadas totales d(target)/dv_i para todos los nodos en tiempo lineal O(|V| + |E|)
        mediante diferenciación automática en modo inverso sobre el DAG.
        """
        adjoints: Dict[str, float] = {nid: 0.0 for nid in self.graph.nodes}
        if not self.graph.has_node(target):
            return adjoints
        adjoints[target] = 1.0

        if nx.is_directed_acyclic_graph(self.graph):
            topo_rev = list(reversed(list(nx.topological_sort(self.graph))))
            for u in topo_rev:
                if u == target:
                    continue
                acc = 0.0
                for v in self.graph.successors(u):
                    w = self.graph[u][v].get("original_weight")
                    if w is not None:
                        acc += adjoints[v] * float(w)
                adjoints[u] = acc
        return adjoints

    def compute_total_sensitivity(self, source: str, target: str, max_depth: int = 12) -> float:
        """Calcula la derivada total exacta de 'source' hacia 'target' en O(|V|+|E|) vía AD inverso."""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return 0.0
        if not nx.has_path(self.graph, source, target):
            return 0.0
        if nx.is_directed_acyclic_graph(self.graph):
            return self.compute_reverse_mode_ad_adjoints(target).get(source, 0.0)

        total_sens = 0.0
        for p in nx.all_simple_paths(self.graph, source, target, cutoff=max_depth):
            total_sens += self.compute_path_sensitivity(p)
        return total_sens

    def get_control_sensitivities(self, target: Optional[str] = None) -> Dict[str, float]:
        """Calcula las sensibilidades causales totales de todas las palancas y nodos hacia el nodo objetivo."""
        t = target or self._get_default_target()
        if t is None:
            return {}
        return self.compute_reverse_mode_ad_adjoints(t)

    def rank_control_variables(self, target: Optional[str] = None, k: int = 3) -> List[Dict[str, Any]]:
        """Ordena las palancas de control según la magnitud de su derivada total hacia el nodo objetivo."""
        t = target or self._get_default_target()
        if t is None:
            return []
        adjoints = self.compute_reverse_mode_ad_adjoints(t)
        ranked = []
        for cid in self.control_nodes:
            tot_d = float(adjoints.get(cid, 0.0))
            s_c = (self.graph.nodes[cid].get("characteristic_scale") if self.graph.has_node(cid) else None) or 1.0
            s_t = (self.graph.nodes[t].get("characteristic_scale") if self.graph.has_node(t) else None) or 1.0
            dim_elast = tot_d * (float(s_c) / max(float(s_t), 1e-6))
            ranked.append({
                "control": cid,
                "total_derivative": tot_d,
                "dimensionless_elasticity": round(dim_elast, 4),
                "top_paths": self.find_top_k_paths(cid, t, k=k)
            })
        ranked.sort(key=lambda item: abs(item["dimensionless_elasticity"]), reverse=True)
        return ranked

    def measure_latency_ms(self, source: str, target: str, k: int = 3) -> float:
        """Mide la latencia de búsqueda en milisegundos."""
        t0 = time.perf_counter()
        _ = self.find_top_k_paths(source, target, k=k)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0


def create_synthetic_benchmark_graph(num_nodes: int = 5000, layers: int = 10) -> nx.DiGraph:
    """Genera un grafo sintético estratificado de 5.000 nodos para el benchmark de latencia R-3."""
    import random
    G = nx.DiGraph()
    nodes_per_layer = num_nodes // layers
    rng = random.Random(42)

    for l in range(layers):
        for i in range(nodes_per_layer):
            nid = f"l{l}_n{i}"
            cat = "control" if l == 0 and i == 0 else ("target" if l == layers - 1 and i == 0 else "state")
            G.add_node(nid, category=cat)

    for l in range(layers - 1):
        for i in range(nodes_per_layer):
            src = f"l{l}_n{i}"
            if i == 0:
                G.add_edge(src, f"l{l+1}_n0", weight=1.0)
            for _ in range(3):
                dst = f"l{l+1}_n{rng.randint(0, nodes_per_layer-1)}"
                G.add_edge(src, dst, weight=rng.uniform(0.1, 1.0))

    return G
