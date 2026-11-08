// Pipeline maps: lay a map out in lanes (services, ingest, query
// time, datastores), choose which edges to draw, and diff two maps node by
// node. Pure functions; the PipelineMap component draws the result as SVG.

export interface MapNodeLike {
  id: string;
  kind: string;
  lane: string;
  label: string;
  column: number;
  row: number;
  stage?: string | null;
  component?: string | null;
  version?: string | null;
  params_hash?: string | null;
  params?: Record<string, unknown>;
  description?: string | null;
  egress?: string | null;
  lab_only?: boolean;
  noop?: boolean;
  configured?: boolean | null;
  backend?: string | null;
  details?: Record<string, unknown>;
  overlay?: Record<string, unknown> | null;
}

export interface MapEdgeLike {
  id: string;
  source: string;
  target: string;
  kind: string;
  label: string;
  sends_text: boolean;
}

export interface MapLike {
  nodes: MapNodeLike[];
  edges: MapEdgeLike[];
}

export const NODE_WIDTH = 172;
export const NODE_HEIGHT = 70;
export const COLUMN_GAP = 48;
export const ROW_GAP = 18;
export const LANE_GAP = 44;
export const LANE_PAD = 30;
export const MARGIN = 16;

export const LANES = [
  { lane: "services", label: "Services it calls" },
  { lane: "ingest", label: "Ingest" },
  { lane: "query", label: "Query time" },
  { lane: "stores", label: "Datastores" },
] as const;

export interface Box {
  id: string;
  node: MapNodeLike;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LaneBox {
  lane: string;
  label: string;
  y: number;
  height: number;
}

export interface LaidEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  label: string;
  sendsText: boolean;
  bundle: boolean;
  highlighted: boolean;
  path: string;
  labelX: number;
  labelY: number;
}

export interface Layout {
  boxes: Box[];
  lanes: LaneBox[];
  edges: LaidEdge[];
  width: number;
  height: number;
}

function isFlowLane(lane: string): boolean {
  return lane === "ingest" || lane === "query";
}

/** Which edges to draw. Ingest steps' reads and writes of a datastore are one
 * bundle per store unless every edge is asked for or the step is selected;
 * every other edge is drawn. */
export function visibleEdges(
  map: MapLike,
  {
    selected = null,
    showAll = false,
  }: { selected?: string | null; showAll?: boolean } = {},
): { edges: MapEdgeLike[]; bundles: { store: string; count: number }[] } {
  const lanes = new Map(map.nodes.map((node) => [node.id, node.lane]));
  const edges: MapEdgeLike[] = [];
  const bundled = new Map<string, number>();
  for (const edge of map.edges) {
    const sourceLane = lanes.get(edge.source);
    const targetLane = lanes.get(edge.target);
    const store =
      sourceLane === "stores"
        ? edge.source
        : targetLane === "stores"
          ? edge.target
          : null;
    const other = store === edge.source ? edge.target : edge.source;
    if (
      showAll ||
      store === null ||
      lanes.get(other) !== "ingest" ||
      edge.source === selected ||
      edge.target === selected
    ) {
      edges.push(edge);
    } else {
      bundled.set(store, (bundled.get(store) ?? 0) + 1);
    }
  }
  return {
    edges,
    bundles: [...bundled.entries()].map(([store, count]) => ({ store, count })),
  };
}

function curve(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  vertical: boolean,
) {
  if (vertical) {
    const middle = (y1 + y2) / 2;
    return `M ${x1} ${y1} C ${x1} ${middle}, ${x2} ${middle}, ${x2} ${y2}`;
  }
  const middle = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2} ${y2}`;
}

/** The path between two boxes: side to side within a lane, top to bottom
 * between lanes. */
export function edgePath(
  a: Box,
  b: Box,
  offset = 0,
): { path: string; x: number; y: number } {
  if (a.node.lane === b.node.lane && isFlowLane(a.node.lane)) {
    const forward = b.x >= a.x + a.width;
    const x1 = forward ? a.x + a.width : a.x + a.width / 2;
    const y1 = forward ? a.y + a.height / 2 : a.y + a.height;
    const x2 = forward ? b.x : b.x + b.width / 2;
    const y2 = forward ? b.y + b.height / 2 : b.y;
    return {
      path: curve(x1, y1, x2, y2, !forward),
      x: (x1 + x2) / 2,
      y: (y1 + y2) / 2,
    };
  }
  const down = a.y < b.y;
  const x1 = a.x + a.width / 2 + offset;
  const x2 = b.x + b.width / 2 + offset;
  const y1 = down ? a.y + a.height : a.y;
  const y2 = down ? b.y : b.y + b.height;
  return {
    path: curve(x1, y1, x2, y2, true),
    x: (x1 + x2) / 2,
    // Two edges between one pair of boxes (a read and a write) keep their
    // labels apart.
    y: (y1 + y2) / 2 + Math.sign(offset) * 10,
  };
}

/** Place the map's nodes in lanes; services and datastores go under (or
 * over) the steps they connect to. */
export function layoutMap(
  map: MapLike,
  options: { selected?: string | null; showAll?: boolean } = {},
): Layout {
  const boxes = new Map<string, Box>();
  const lanes: LaneBox[] = [];
  let top = MARGIN;
  const neighbours = new Map<string, string[]>();
  for (const edge of map.edges) {
    neighbours.set(edge.source, [
      ...(neighbours.get(edge.source) ?? []),
      edge.target,
    ]);
    neighbours.set(edge.target, [
      ...(neighbours.get(edge.target) ?? []),
      edge.source,
    ]);
  }
  const columnX = (column: number) =>
    MARGIN + column * (NODE_WIDTH + COLUMN_GAP);
  for (const { lane, label } of LANES) {
    const members = map.nodes.filter((node) => node.lane === lane);
    if (!members.length) continue;
    if (isFlowLane(lane)) {
      const rows = Math.max(...members.map((node) => node.row)) + 1;
      for (const node of members) {
        boxes.set(node.id, {
          id: node.id,
          node,
          x: columnX(node.column),
          y: top + LANE_PAD + node.row * (NODE_HEIGHT + ROW_GAP),
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
        });
      }
      const height = LANE_PAD + rows * (NODE_HEIGHT + ROW_GAP) - ROW_GAP;
      lanes.push({ lane, label, y: top, height });
      top += height + LANE_GAP;
      continue;
    }
    // A service or store: at the mean x of the steps it touches, then pushed
    // right past its left neighbour.
    const wanted = members.map((node) => {
      const xs = (neighbours.get(node.id) ?? [])
        .map((id) => map.nodes.find((other) => other.id === id))
        .filter(
          (other): other is MapNodeLike => !!other && isFlowLane(other.lane),
        )
        .map((other) => columnX(other.column) + NODE_WIDTH / 2);
      const centre = xs.length
        ? xs.reduce((sum, x) => sum + x, 0) / xs.length
        : MARGIN;
      return { node, x: centre - NODE_WIDTH / 2 };
    });
    wanted.sort((a, b) => a.x - b.x || a.node.id.localeCompare(b.node.id));
    let right = MARGIN - COLUMN_GAP;
    for (const { node, x } of wanted) {
      const placed = Math.max(x, right + COLUMN_GAP, MARGIN);
      boxes.set(node.id, {
        id: node.id,
        node,
        x: placed,
        y: top + LANE_PAD,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      });
      right = placed + NODE_WIDTH;
    }
    const height = LANE_PAD + NODE_HEIGHT;
    lanes.push({ lane, label, y: top, height });
    top += height + LANE_GAP;
  }
  const visible = visibleEdges(map, options);
  const pairs = new Set(
    visible.edges.map((edge) => `${edge.source}|${edge.target}`),
  );
  const edges: LaidEdge[] = [];
  for (const edge of visible.edges) {
    const a = boxes.get(edge.source);
    const b = boxes.get(edge.target);
    if (!a || !b) continue;
    const both = pairs.has(`${edge.target}|${edge.source}`);
    const offset = both ? (edge.kind === "read" ? -8 : 8) : 0;
    const { path, x, y } = edgePath(a, b, offset);
    edges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      kind: edge.kind,
      label: edge.label,
      sendsText: edge.sends_text,
      bundle: false,
      highlighted:
        !!options.selected &&
        (edge.source === options.selected || edge.target === options.selected),
      path,
      labelX: x,
      labelY: y,
    });
  }
  const ingest = lanes.find((lane) => lane.lane === "ingest");
  for (const { store, count } of visible.bundles) {
    const box = boxes.get(store);
    if (!box || !ingest) continue;
    const x = box.x + box.width / 2;
    const y1 = ingest.y + ingest.height + 4;
    const y2 = box.y;
    edges.push({
      id: `bundle:${store}`,
      source: "lane:ingest",
      target: store,
      kind: "bundle",
      label: `${count} reads and writes by ingest steps`,
      sendsText: false,
      bundle: true,
      highlighted: false,
      path: `M ${x} ${y1} L ${x} ${y2}`,
      labelX: x,
      labelY: (y1 + y2) / 2,
    });
  }
  const width =
    Math.max(MARGIN, ...[...boxes.values()].map((box) => box.x + box.width)) +
    MARGIN;
  return {
    boxes: [...boxes.values()],
    lanes,
    edges,
    width,
    height: top - LANE_GAP + MARGIN,
  };
}

export type Change = "added" | "removed" | "changed" | "same";

/** How each node of ``other`` differs from ``base``'s node with its ID:
 * another component, version or parameters is a change. */
export function diffMaps(
  base: MapLike,
  other: MapLike,
): Record<string, Change> {
  const before = new Map(base.nodes.map((node) => [node.id, node]));
  const after = new Map(other.nodes.map((node) => [node.id, node]));
  const changes: Record<string, Change> = {};
  for (const [id, node] of after) {
    const old = before.get(id);
    if (!old) changes[id] = "added";
    else if (
      old.component !== node.component ||
      old.version !== node.version ||
      old.params_hash !== node.params_hash ||
      old.label !== node.label
    ) {
      changes[id] = "changed";
    } else changes[id] = "same";
  }
  for (const id of before.keys()) if (!after.has(id)) changes[id] = "removed";
  return changes;
}

function formatMs(value: unknown): string {
  return typeof value === "number"
    ? `${value.toFixed(value < 10 ? 1 : 0)} ms`
    : "–";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

/** A node's numbers on one line: a stage's tasks, cache hits, time and bytes;
 * a query step's latency and how often the evidence reaches it. */
export function overlaySummary(
  overlay: Record<string, unknown> | null | undefined,
): string {
  return overlayParts(overlay).join(" · ");
}

/** The same numbers on at most two lines, for a node on the map. */
export function overlayLines(
  overlay: Record<string, unknown> | null | undefined,
): string[] {
  const parts = overlayParts(overlay);
  return [parts.slice(0, 2).join(" · "), parts.slice(2).join(" · ")].filter(
    Boolean,
  );
}

function overlayParts(
  overlay: Record<string, unknown> | null | undefined,
): string[] {
  if (!overlay) return [];
  const parts: string[] = [];
  if (typeof overlay.tasks === "number") {
    parts.push(`${overlay.tasks} tasks`);
    if (typeof overlay.cache_hits === "number" && overlay.cache_hits > 0) {
      parts.push(`${overlay.cache_hits} cached`);
    }
    if (typeof overlay.duration_ms === "number")
      parts.push(formatMs(overlay.duration_ms));
    if (typeof overlay.bytes === "number" && overlay.bytes > 0) {
      parts.push(formatBytes(overlay.bytes));
    }
  }
  const latency = overlay.latency_ms as Record<string, number> | undefined;
  if (latency)
    parts.push(`p50 ${formatMs(latency.p50)}, p95 ${formatMs(latency.p95)}`);
  const reaches = overlay.evidence_reaches as
    Record<string, number | null> | undefined;
  if (reaches) {
    const shares = Object.values(reaches).filter(
      (value): value is number => typeof value === "number",
    );
    if (shares.length) {
      parts.push(`evidence ${Math.round(Math.min(...shares) * 100)}%`);
    }
  }
  return parts;
}

/** A node's second line: its component, a datastore's backend, or whether a
 * service is configured here. */
export function nodeSubtitle(node: MapNodeLike): string {
  if (node.kind === "datastore") {
    return [node.backend, node.configured === false ? "not configured" : null]
      .filter(Boolean)
      .join(" · ");
  }
  if (node.kind === "service") {
    return node.configured === false ? "not configured here" : "configured";
  }
  return node.component ?? "";
}

export const EDGE_KINDS: Record<string, string> = {
  flow: "Data flows to the next step",
  read: "Reads from",
  write: "Writes to",
  call: "Calls a service",
  bundle: "Every ingest step reads and writes",
};

/** The map as rows for its table view: every node and edge, in order. */
export function mapRows(map: MapLike): {
  nodes: string[][];
  edges: string[][];
} {
  const names = new Map(map.nodes.map((node) => [node.id, node.label]));
  return {
    nodes: map.nodes.map((node) => [
      node.label,
      node.kind,
      node.lane,
      node.component ?? node.backend ?? "",
      overlaySummary(node.overlay),
    ]),
    edges: map.edges.map((edge) => [
      names.get(edge.source) ?? edge.source,
      edge.kind,
      names.get(edge.target) ?? edge.target,
      edge.label + (edge.sends_text ? " (text leaves the lab)" : ""),
    ]),
  };
}
