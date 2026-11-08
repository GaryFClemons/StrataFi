"""Pipeline maps: a RAG pipeline's stages and the datastores and services each touches.

``build_map`` reads a resolved pipeline spec, a resolved retrieval config, or both, with the
component registry and what this deployment configures (``Deployment``), and returns:

- **nodes**, in four lanes:
  - ``ingest``: ``extract``; one ``scan:<i>`` per check; one ``parse:<route>`` per route;
    one ``clean:<i>`` per rule; ``chunk``, ``enrich``, ``embed``; ``index:dense`` and
    ``index:keyword``
  - ``query``: ``query:input``, ``query:embed``, ``query:dense``, ``query:keyword``,
    ``query:fuse``, ``query:rerank`` and ``query:final`` (with ``query:results`` for an eval
    run, which stores its results)
  - ``stores``: ``store:database`` (Postgres, schema ``stratafi``), ``store:objects`` (the
    artifact store), ``store:lab-index`` (Postgres, schema ``stratafi_index``) and
    ``store:model-files`` (in-process models' files)
  - ``services``: the model gateway, clamd, hosted providers (Tika among them), and the
    sources documents are imported from
- **edges:** ``flow`` (an artifact or list between steps), ``read`` and ``write`` (a step and a
  store), ``call`` (a step and a service). A call that sends document or query text out of the
  lab says so (``sends_text``).

What a component touches comes from its declarations: ``egress`` (the model gateway,
``provider:<name>``) and ``services``. Every ingest stage reads its inputs from
the artifact store, writes its output there, and records its tasks and artifacts in the
database.

Datastores and services are named, never addressed: a map holds no host, URL or credential, only
the kind of store and whether this deployment configures it. ``to_mermaid`` writes a map as a
Mermaid flowchart.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from stratafi_core.components import ComponentRegistry, UnknownComponent
from stratafi_core.hashing import content_hash

MAP_SCHEMA = "stratafi.pipeline-map.v1"
NodeKind = Literal["input", "stage", "query", "output", "datastore", "service"]
EdgeKind = Literal["flow", "read", "write", "call"]
Lane = Literal["services", "ingest", "query", "stores"]

DATASTORES: dict[str, tuple[str, str]] = {
    "database": (
        "Stratafi's database",
        "Postgres, schema stratafi: runs, stage tasks, artifact records, eval results",
    ),
    "objects": ("Artifact store", "Every artifact's payload, by its cache key"),
    "lab-index": (
        "Lab index schema",
        "Postgres, schema stratafi_index: pgvector and full-text index tables",
    ),
    "model-files": (
        "Model files",
        "In-process models: the Hugging Face cache, and trained models from the artifact store",
    ),
}
SERVICES: dict[str, tuple[str, str]] = {
    "model-gateway": ("Model gateway", "Embeddings, reranking and chat, by model alias"),
    "clamd": ("ClamAV", "clamd's INSTREAM malware scan"),
    "s3-source": ("S3 source", "A bucket the configuration allows imports from"),
    "web": ("The web", "URLs fetched through the SSRF guard"),
    "server-folder": ("Server folder", "A folder on the API's host the configuration names"),
}
PROVIDER_LABELS = {"tika": "Apache Tika", "openai-compatible": "OpenAI-compatible endpoint"}
INBOUND = frozenset({"s3-source", "web", "server-folder"})
NOOP_COMPONENTS = frozenset({"enrich/none", "rerank/none"})
# What a stage sends to the services it calls.
SENDS: dict[str, str] = {
    "extract": "document bytes",
    "scan": "document bytes",
    "parse": "document bytes",
    "clean": "document text",
    "chunk": "chunk text",
    "enrich": "chunk text",
    "embed": "chunk text",
    "query:embed": "query text",
    "query:rerank": "query and passage text",
}
# What flows out of each ingest stage.
ARTIFACTS: dict[str, str] = {
    "extract": "raw",
    "scan": "raw and scan_report",
    "parse": "parsed",
    "clean": "cleaned",
    "chunk": "chunks",
    "enrich": "chunks",
    "embed": "vectors",
}


@dataclass(frozen=True)
class Deployment:
    """What this deployment configures; never where. ``objects`` is ``s3`` or ``filesystem``."""

    database: bool = True
    objects: str = "filesystem"
    gateway: bool = False
    clamd: bool = False
    providers: frozenset[str] = frozenset()
    in_process: bool = True

    def configured(self, node: str) -> bool | None:
        if node.startswith("provider:"):
            return node.removeprefix("provider:") in self.providers
        return {
            "database": self.database,
            "objects": True,
            "lab-index": self.database,
            "model-files": self.in_process,
            "model-gateway": self.gateway,
            "clamd": self.clamd,
        }.get(node)


@dataclass
class MapNode:
    id: str
    kind: NodeKind
    lane: Lane
    label: str
    column: int = 0
    row: int = 0
    stage: str | None = None
    component: str | None = None
    version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    params_hash: str | None = None
    description: str | None = None
    egress: str | None = None
    lab_only: bool = False
    noop: bool = False
    configured: bool | None = None
    backend: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MapEdge:
    source: str
    target: str
    kind: EdgeKind
    label: str
    sends_text: bool = False

    @property
    def id(self) -> str:
        return f"{self.source}->{self.target}:{self.kind}"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, **self.__dict__}


class _Builder:
    def __init__(self, registry: ComponentRegistry, deployment: Deployment) -> None:
        self.registry = registry
        self.deployment = deployment
        self.nodes: dict[str, MapNode] = {}
        self.edges: dict[str, MapEdge] = {}

    # --- nodes ----------------------------------------------------------------------------

    def component_node(
        self,
        node_id: str,
        lane: Lane,
        kind: NodeKind,
        stage: str,
        ref: Mapping[str, Any],
        *,
        column: int,
        row: int = 0,
        label: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> MapNode:
        component_id = str(ref.get("component"))
        params = dict(ref.get("params") or {})
        try:
            info = self.registry.get(component_id).info
        except UnknownComponent:
            info = None
        node = MapNode(
            id=node_id,
            kind=kind,
            lane=lane,
            label=label or (info.title if info else component_id),
            column=column,
            row=row,
            stage=stage,
            component=component_id,
            version=info.version if info else None,
            params=params,
            params_hash=content_hash({"component": component_id, "params": params}),
            description=info.description if info else None,
            egress=info.egress if info else None,
            lab_only=info.lab_only if info else False,
            noop=component_id in NOOP_COMPONENTS,
            details=dict(details or {}),
        )
        self.nodes[node_id] = node
        if info is not None:
            self._touches(node, info.egress, info.services)
        return node

    def store(self, name: str) -> str:
        node_id = f"store:{name}"
        if node_id not in self.nodes:
            label, description = DATASTORES[name]
            backend = {
                "database": "postgres",
                "lab-index": "postgres",
                "objects": "s3" if self.deployment.objects == "s3" else "filesystem",
                "model-files": "files",
            }[name]
            self.nodes[node_id] = MapNode(
                id=node_id,
                kind="datastore",
                lane="stores",
                label=label,
                description=description,
                configured=self.deployment.configured(name),
                backend=backend,
            )
        return node_id

    def service(self, name: str) -> str:
        node_id = f"service:{name}"
        if node_id not in self.nodes:
            if name.startswith("provider:"):
                provider = name.removeprefix("provider:")
                label = PROVIDER_LABELS.get(provider, f"Hosted: {provider}")
                description = "A provider outside the lab, behind the allow-list"
            else:
                label, description = SERVICES[name]
            self.nodes[node_id] = MapNode(
                id=node_id,
                kind="service",
                lane="services",
                label=label,
                description=description,
                configured=self.deployment.configured(name),
            )
        return node_id

    # --- edges ----------------------------------------------------------------------------

    def edge(
        self, source: str, target: str, kind: EdgeKind, label: str, *, sends_text: bool = False
    ) -> None:
        found = MapEdge(source, target, kind, label, sends_text)
        self.edges.setdefault(found.id, found)

    def _touches(self, node: MapNode, egress: str, services: tuple[str, ...]) -> None:
        sends = SENDS.get(node.id) or SENDS.get(node.stage or "", "data")
        if egress == "model_gateway":
            self.edge(node.id, self.service("model-gateway"), "call", sends, sends_text=True)
        elif egress.startswith("provider:"):
            self.edge(node.id, self.service(egress), "call", sends, sends_text=True)
        for name in services:
            if name == "clamd":
                self.edge(node.id, self.service("clamd"), "call", "document bytes", sends_text=True)
            elif name == "lab-index":
                kind: EdgeKind = "write" if node.lane == "ingest" else "read"
                self.edge(
                    *(
                        (node.id, self.store("lab-index"))
                        if kind == "write"
                        else (self.store("lab-index"), node.id)
                    ),
                    kind,
                    "index tables",
                )
            elif name == "model-files":
                self.edge(self.store("model-files"), node.id, "read", "model weights")
            elif name in INBOUND:
                self.edge(self.service(name), node.id, "read", "documents, at intake")

    def artifacts(self, node_id: str, *, reads: bool = True) -> None:
        """An ingest step: inputs from the artifact store, its output to it, its records to the
        database."""
        objects, database = self.store("objects"), self.store("database")
        if reads:
            self.edge(objects, node_id, "read", "input artifacts")
        self.edge(node_id, objects, "write", "its artifact")
        self.edge(node_id, database, "write", "tasks and artifact records")


def _ref(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"component": value, "params": {}}
    return dict(value or {})


def _chain(builder: _Builder, sources: list[str], target: str, label: str) -> None:
    for source in sources:
        builder.edge(source, target, "flow", label)


def _ingest(builder: _Builder, stages: Mapping[str, Any]) -> dict[str, str | None]:
    """The ingest lane; returns the index nodes by kind."""
    column = 0
    extract = stages.get("extract") or {}
    unpack = [_ref(item).get("component") for item in extract.get("unpack") or []]
    builder.component_node(
        "extract",
        "ingest",
        "stage",
        "extract",
        _ref(extract.get("component") or "extract/upload"),
        column=column,
        details={"unpack": unpack},
    )
    builder.artifacts("extract")
    previous = ["extract"]
    scan = stages.get("scan") or {}
    for index, check in enumerate(scan.get("checks") or []):
        column += 1
        node = builder.component_node(
            f"scan:{index}", "ingest", "stage", "scan", _ref(check), column=column
        )
        if index == 0:
            node.details["on_outcome"] = dict(scan.get("on_outcome") or {})
        _chain(builder, previous, node.id, "raw" if index == 0 else "raw, if it passes")
        builder.artifacts(node.id)
        previous = [node.id]
    column += 1
    routes = dict((stages.get("parse") or {}).get("route") or {})
    parsers = []
    for row, (key, ref) in enumerate(sorted(routes.items(), key=lambda item: item[0] != "default")):
        node = builder.component_node(
            f"parse:{key}",
            "ingest",
            "stage",
            "parse",
            _ref(ref),
            column=column,
            row=row,
            details={"route": key},
        )
        _chain(builder, previous, node.id, "raw" if key == "default" else f"raw: {key}")
        builder.artifacts(node.id)
        parsers.append(node.id)
    previous = parsers or previous
    for index, rule in enumerate(stages.get("clean") or []):
        column += 1
        node = builder.component_node(
            f"clean:{index}", "ingest", "stage", "clean", _ref(rule), column=column
        )
        _chain(builder, previous, node.id, "parsed" if index == 0 else "cleaned")
        builder.artifacts(node.id)
        previous = [node.id]
    flow = "cleaned" if stages.get("clean") else "parsed"
    for stage, label in (("chunk", flow), ("enrich", "chunks"), ("embed", "chunks")):
        if stage not in stages:
            continue
        column += 1
        ref = _ref(stages[stage])
        details = {"index_text": ref.pop("index_text")} if "index_text" in ref else {}
        builder.component_node(stage, "ingest", "stage", stage, ref, column=column, details=details)
        _chain(builder, previous, stage, label)
        builder.artifacts(stage)
        previous = [stage]
    column += 1
    indexes: dict[str, str | None] = {"dense": None, "keyword": None}
    for row, kind in enumerate(("dense", "keyword")):
        ref = (stages.get("index") or {}).get(kind)
        if not ref:
            continue
        node_id = f"index:{kind}"
        node = builder.component_node(
            node_id, "ingest", "stage", "index", _ref(ref), column=column, row=row
        )
        _chain(builder, previous, node_id, "vectors and chunks" if kind == "dense" else "chunks")
        if "lab-index" not in _services(builder, node.component):
            builder.edge(node_id, builder.store("objects"), "write", "the index")
        builder.edge(node_id, builder.store("database"), "write", "tasks and artifact records")
        indexes[kind] = node_id
    return indexes


def _services(builder: _Builder, component: str | None) -> tuple[str, ...]:
    try:
        return builder.registry.get(str(component)).info.services
    except UnknownComponent:
        return ()


def _query(
    builder: _Builder,
    retrieval: Mapping[str, Any],
    stages: Mapping[str, Any] | None,
    indexes: Mapping[str, str | None],
    *,
    evaluation: bool,
) -> None:
    retrievers = retrieval.get("retrievers") or {}
    dense, keyword = retrievers.get("dense"), retrievers.get("keyword")
    builder.nodes["query:input"] = MapNode(
        id="query:input",
        kind="input",
        lane="query",
        label="Query",
        description="A question with its asker and as-of date",
        column=0,
    )
    fused_inputs = []
    if dense:
        embed = _ref((stages or {}).get("embed") or {"component": "embed/gateway"})
        node = builder.component_node(
            "query:embed",
            "query",
            "query",
            "embed",
            embed,
            column=1,
            label="Embed the query",
            details={"adapter": dense.get("adapter")} if dense.get("adapter") else {},
        )
        if dense.get("adapter"):
            builder.edge(builder.store("model-files"), node.id, "read", "the adapter")
        builder.edge("query:input", "query:embed", "flow", "query text")
        builder.nodes["query:dense"] = MapNode(
            id="query:dense",
            kind="query",
            lane="query",
            label="Dense search",
            column=2,
            details={"top_n": dense.get("top_n")},
        )
        builder.edge("query:embed", "query:dense", "flow", "query vector")
        fused_inputs.append("query:dense")
        _index_read(builder, "dense", indexes, stages)
    if keyword:
        builder.nodes["query:keyword"] = MapNode(
            id="query:keyword",
            kind="query",
            lane="query",
            label="Keyword search",
            column=2,
            row=1,
            details={"top_n": keyword.get("top_n")},
        )
        builder.edge("query:input", "query:keyword", "flow", "query text")
        fused_inputs.append("query:keyword")
        _index_read(builder, "keyword", indexes, stages)
    fusion = dict(retrieval.get("fusion") or {})
    builder.nodes["query:fuse"] = MapNode(
        id="query:fuse",
        kind="query",
        lane="query",
        label="Fuse and filter",
        column=3,
        details={
            "fusion": fusion,
            "candidates": retrieval.get("candidates"),
            "filters": retrieval.get("filters"),
        },
    )
    for source in fused_inputs:
        builder.edge(source, "query:fuse", "flow", "ranked list")
    builder.edge(builder.store("database"), "query:fuse", "read", "corpus entries: ACL, dates")
    builder.component_node(
        "query:rerank",
        "query",
        "query",
        "rerank",
        _ref(retrieval.get("rerank") or "rerank/none"),
        column=4,
        label="Rerank",
    )
    builder.edge("query:fuse", "query:rerank", "flow", "candidates")
    builder.nodes["query:final"] = MapNode(
        id="query:final",
        kind="output",
        lane="query",
        label=f"Top {retrieval.get('top_k', '?')}",
        description="The passages returned",
        column=5,
        details={"top_k": retrieval.get("top_k")},
    )
    builder.edge("query:rerank", "query:final", "flow", "reranked list")
    if evaluation:
        builder.nodes["query:results"] = MapNode(
            id="query:results",
            kind="output",
            lane="query",
            label="Eval results",
            description="Ranked lists and per-query scores",
            column=6,
        )
        builder.edge("query:final", "query:results", "flow", "each query's lists")
        builder.edge("query:results", builder.store("objects"), "write", "ranked lists, results")
        builder.edge("query:results", builder.store("database"), "write", "per-query scores")


def _index_read(
    builder: _Builder,
    kind: str,
    indexes: Mapping[str, str | None],
    stages: Mapping[str, Any] | None,
) -> None:
    target = f"query:{kind}"
    source = indexes.get(kind)
    if source is not None:
        builder.edge(source, target, "flow", "the index")
    component = None
    if stages is not None:
        component = _ref((stages.get("index") or {}).get(kind) or {}).get("component")
    if component and "lab-index" in _services(builder, component):
        builder.edge(builder.store("lab-index"), target, "read", "index tables")
    else:
        builder.edge(builder.store("objects"), target, "read", "the index, loaded in memory")


def build_map(
    pipeline: Mapping[str, Any] | None,
    retrieval: Mapping[str, Any] | None,
    registry: ComponentRegistry,
    deployment: Deployment,
    *,
    evaluation: bool = False,
) -> dict[str, Any]:
    """The map the module docstring describes. ``pipeline`` and ``retrieval`` are resolved
    documents (``stratafi.pipeline.v1``, ``stratafi.retrieval.v1``); either may be None."""
    if pipeline is None and retrieval is None:
        raise ValueError("a map needs a pipeline spec or a retrieval config")
    builder = _Builder(registry, deployment)
    stages = dict((pipeline or {}).get("stages") or {}) if pipeline is not None else None
    indexes: dict[str, str | None] = {"dense": None, "keyword": None}
    if stages is not None:
        indexes = _ingest(builder, stages)
    if retrieval is not None:
        _query(builder, retrieval, stages, indexes, evaluation=evaluation)
    lanes: dict[str, int] = {"services": 0, "stores": 0}
    for node in builder.nodes.values():
        if node.lane in lanes:
            node.column = lanes[node.lane]
            lanes[node.lane] += 1
    edges = list(builder.edges.values())
    return {
        "schema": MAP_SCHEMA,
        "pipeline": _ref_of(pipeline),
        "retrieval": _ref_of(retrieval),
        "nodes": [node.as_dict() for node in builder.nodes.values()],
        "edges": [edge.as_dict() for edge in edges],
        "summary": {
            "stages": sum(1 for node in builder.nodes.values() if node.kind == "stage"),
            "datastores": sorted(
                node.id.removeprefix("store:")
                for node in builder.nodes.values()
                if node.kind == "datastore"
            ),
            "services": sorted(
                node.id.removeprefix("service:")
                for node in builder.nodes.values()
                if node.kind == "service"
            ),
            "sends_text_to": sorted(
                {edge.target.removeprefix("service:") for edge in edges if edge.sends_text}
            ),
            "lab_only": sorted(
                {str(node.component) for node in builder.nodes.values() if node.lab_only}
            ),
        },
    }


def _ref_of(document: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {"name": document.get("name"), "description": document.get("description")}


# --- Mermaid ----------------------------------------------------------------------------------

_ID_RE = re.compile(r"[^A-Za-z0-9_]")


def _mermaid_id(node_id: str) -> str:
    return "n_" + _ID_RE.sub("_", node_id)


def _mermaid_label(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


SHAPES = {
    "datastore": ('[("', '")]'),
    "service": ('{{"', '"}}'),
    "input": ('(["', '"])'),
    "output": ('(["', '"])'),
}


def to_mermaid(found: Mapping[str, Any], *, stores: bool = True) -> str:
    """A map as a Mermaid flowchart, left to right. With ``stores`` false, only the steps and
    their services."""
    lines = ["flowchart LR"]
    nodes = [node for node in found["nodes"] if stores or node["kind"] != "datastore"]
    shown = {node["id"] for node in nodes}
    for lane, title in (("ingest", "Ingest"), ("query", "Query time")):
        members = [node for node in nodes if node["lane"] == lane]
        if not members:
            continue
        lines.append(f'  subgraph {lane}["{title}"]')
        for node in members:
            label = node["label"]
            if node.get("component") and node["kind"] in ("stage", "query"):
                label = f"{label}<br/>{node['component']}"
            opening, closing = SHAPES.get(node["kind"], ('["', '"]'))
            lines.append(f"    {_mermaid_id(node['id'])}{opening}{_mermaid_label(label)}{closing}")
        lines.append("  end")
    for node in nodes:
        if node["lane"] in ("ingest", "query"):
            continue
        opening, closing = SHAPES.get(node["kind"], ('["', '"]'))
        lines.append(
            f"  {_mermaid_id(node['id'])}{opening}{_mermaid_label(node['label'])}{closing}"
        )
    for edge in found["edges"]:
        if edge["source"] not in shown or edge["target"] not in shown:
            continue
        arrow = "-->" if edge["kind"] == "flow" else "-.->"
        label = _mermaid_label(
            edge["label"] + (" (text leaves the lab)" if edge["sends_text"] else "")
        )
        lines.append(
            f'  {_mermaid_id(edge["source"])} {arrow}|"{label}"| {_mermaid_id(edge["target"])}'
        )
    return "\n".join(lines) + "\n"


def overlay_stages(tasks: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """A pipeline run's numbers per map node, from its stage tasks: each task's ``stage``,
    ``component_id``, ``route``, ``status``, ``cache_hit``, ``duration_ms`` and ``size_bytes``
    (its artifact's). Scan and clean run as one task per document, so their numbers go on each
    of their nodes."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        stage = str(task.get("stage"))
        if stage == "parse":
            key = f"parse:{task.get('route') or 'default'}"
        elif stage == "index":
            key = f"index:{task.get('index_kind') or 'dense'}"
        else:
            key = stage
        groups.setdefault(key, []).append(task)
    overlay: dict[str, dict[str, Any]] = {}
    for key, members in groups.items():
        statuses: dict[str, int] = {}
        for task in members:
            statuses[str(task.get("status"))] = statuses.get(str(task.get("status")), 0) + 1
        durations = [float(t["duration_ms"]) for t in members if t.get("duration_ms") is not None]
        overlay[key] = {
            "tasks": len(members),
            "statuses": dict(sorted(statuses.items())),
            "cache_hits": sum(1 for task in members if task.get("cache_hit")),
            "duration_ms": round(sum(durations), 3),
            "bytes": int(sum(int(task.get("size_bytes") or 0) for task in members)),
        }
    return overlay


# Which latency step and which checkpoints belong to each query node (an eval run's overlay).
QUERY_OVERLAY: dict[str, tuple[str, tuple[str, ...]]] = {
    "query:keyword": ("keyword_ms", ("keyword",)),
    "query:embed": ("embed_ms", ()),
    "query:dense": ("dense_ms", ("dense",)),
    "query:fuse": ("fusion_ms", ("fused", "candidates")),
    "query:rerank": ("rerank_ms", ("reranked",)),
    "query:final": ("total_ms", ("final",)),
}


def eval_overlay(summary: Mapping[str, Any] | None, rule: str = "knowledge-hit-v1") -> dict:
    """An eval run's numbers per query node: the step's latency (p50, p95), and for each list
    it makes, the share of queries with a relevant chunk anywhere in it (``hit@all``)."""
    summary = summary or {}
    latency = summary.get("latency_ms") or {}
    metrics = (summary.get("metrics") or {}).get(rule) or {}
    overlay: dict[str, dict[str, Any]] = {}
    for node_id, (step, checkpoints) in QUERY_OVERLAY.items():
        entry: dict[str, Any] = {}
        if step in latency:
            entry["latency_ms"] = {
                stat: latency[step].get(stat) for stat in ("p50", "p95") if stat in latency[step]
            }
        reaching = {
            checkpoint: (metrics.get(checkpoint, {}).get("hit@all") or {}).get("mean")
            for checkpoint in checkpoints
            if checkpoint in metrics
        }
        if reaching:
            entry["evidence_reaches"] = reaching
        if entry:
            overlay[node_id] = entry
    return overlay


def attach_overlay(
    found: dict[str, Any],
    stages: Mapping[str, Mapping[str, Any]] | None = None,
    queries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Put a run's numbers on the map's nodes, as each node's ``overlay``: ``stages`` from
    ``overlay_stages`` (parse by route, index by kind, the other stages by stage), ``queries``
    from ``eval_overlay``."""
    for node in found["nodes"]:
        node_id, stage = node["id"], node.get("stage")
        overlay = None
        if stages:
            if node_id in stages:
                overlay = stages[node_id]
            elif node["lane"] == "ingest" and stage in stages:
                overlay = stages[stage]
        if queries and node_id in queries:
            overlay = {**(overlay or {}), **queries[node_id]}
        if overlay is not None:
            node["overlay"] = overlay
    return found
