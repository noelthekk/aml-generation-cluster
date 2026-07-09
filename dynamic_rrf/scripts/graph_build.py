"""Build a NetworkX knowledge graph from the AML clause corpus.

Node types:
  document : one per source document (mlr_2017, poca_2002, ...)
  clause   : one per clause in clauses.jsonl
  term     : one per defined term extracted from definition-type clauses

Edge types:
  IN_DOCUMENT     : clause -> document
  CROSS_REFERENCES: clause -> clause  (from extracted cross-refs)
  DEFINES         : clause -> term
"""

import json
import re
import networkx as nx
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_clauses(path: Optional[Path] = None) -> list[dict]:
    p = path or DATA_DIR / "clauses.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cross_refs(path: Optional[Path] = None) -> list[dict]:
    p = path or DATA_DIR / "cross_refs.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_Q = r'["“‘]'   # opening quote: ASCII ", Unicode left double/single
_QE = r'["”’]'  # closing quote: ASCII ", Unicode right double/single
_TERM_CONTEXT = re.compile(
    _Q + r'([^"“”‘’]{2,60})' + _QE +
    r'\s*(means|has the meaning|is defined|includes)\b',
    re.IGNORECASE,
)
_TERM_EXTRACT = re.compile(
    _Q + r'([^"“”‘’]{2,60})' + _QE
)


def build_graph(
    clauses: list[dict],
    cross_refs: Optional[list[dict]] = None,
) -> nx.DiGraph:
    """Build and return a NetworkX DiGraph from clauses and optional cross-refs."""
    G = nx.DiGraph()

    # --- Document nodes ---
    docs_seen: set[str] = set()
    for c in clauses:
        src = c["source"]
        if src not in docs_seen:
            G.add_node(
                src,
                node_type="document",
                title=c.get("document_title", src),
                url=c.get("url", ""),
            )
            docs_seen.add(src)

    # --- Clause nodes + IN_DOCUMENT edges ---
    for c in clauses:
        G.add_node(
            c["clause_id"],
            node_type="clause",
            source=c["source"],
            marker=c.get("marker", ""),
            clause_type=c.get("clause_type", "procedural"),
            part=c.get("part") or "",
            part_title=c.get("part_title") or "",
            licence=c.get("licence", ""),
            text_snippet=c["text"][:300],
        )
        G.add_edge(c["clause_id"], c["source"], rel="IN_DOCUMENT")

    # --- CROSS_REFERENCES edges ---
    if cross_refs:
        for r in cross_refs:
            src_id = r.get("source_id")
            tgt_id = r.get("target_id")
            if src_id and tgt_id and G.has_node(src_id) and G.has_node(tgt_id):
                if not G.has_edge(src_id, tgt_id):
                    G.add_edge(
                        src_id, tgt_id,
                        rel="CROSS_REFERENCES",
                        raw=r.get("target_raw", ""),
                    )

    # --- DEFINES edges (any clause containing a quoted-term + defining verb) ---
    for c in clauses:
        text = c["text"]
        for m in _TERM_EXTRACT.finditer(text):
            # Check that this term is followed by a defining verb
            surrounding = text[max(0, m.start() - 5): m.end() + 60]
            if _TERM_CONTEXT.search(surrounding):
                term = m.group(1).strip().lower()
                term_id = f"term:{term}"
                if not G.has_node(term_id):
                    G.add_node(term_id, node_type="term", name=term)
                if not G.has_edge(c["clause_id"], term_id):
                    G.add_edge(c["clause_id"], term_id, rel="DEFINES")

    return G


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------

def neighbours(G: nx.DiGraph, clause_id: str, rel: Optional[str] = None,
               hops: int = 1) -> list[str]:
    """Return clause_ids reachable from clause_id within `hops` steps.

    Filters to clause nodes only (excludes document and term nodes).
    Optionally filters by edge rel type.
    """
    if clause_id not in G:
        return []

    visited = {clause_id}
    frontier = {clause_id}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for _, tgt, data in G.out_edges(node, data=True):
                if rel and data.get("rel") != rel:
                    continue
                if tgt not in visited and G.nodes[tgt].get("node_type") == "clause":
                    next_frontier.add(tgt)
        visited.update(next_frontier)
        frontier = next_frontier

    visited.discard(clause_id)
    return list(visited)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def graph_stats(G: nx.DiGraph) -> dict:
    node_types: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        r = d.get("rel", "unknown")
        edge_types[r] = edge_types.get(r, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
    }
