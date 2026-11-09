// A RAG pipeline's map, drawn as SVG: lanes for the services it
// calls, its ingest steps, its query steps and its datastores; arrows for
// what flows, what's read and written, and what's sent out. Select a step to
// see its reads and writes; the table below holds every node and edge.

import { Button } from "@heroui/react";
import { useMemo, useRef, useState, type KeyboardEvent } from "react";
import {
  EDGE_KINDS,
  layoutMap,
  mapRows,
  nodeSubtitle,
  overlayLines,
  overlaySummary,
  type Box,
  type Change,
  type MapLike,
  type MapNodeLike,
} from "../maps";
import { DataTable } from "./ui";

const KIND_CLASS: Record<string, string> = {
  flow: "text-foreground",
  read: "text-muted",
  write: "text-muted",
  bundle: "text-muted",
  call: "text-accent",
};

const CHANGE_CLASS: Record<string, string> = {
  added: "stroke-success",
  changed: "stroke-warning",
};

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function Shape({
  box,
  selected,
  change,
}: {
  box: Box;
  selected: boolean;
  change?: Change;
}) {
  const { node, width: w, height: h } = box;
  const outline =
    change && CHANGE_CLASS[change]
      ? `${CHANGE_CLASS[change]} [stroke-width:3]`
      : selected
        ? "stroke-accent [stroke-width:2.5]"
        : "stroke-separator";
  const fill = node.noop ? "fill-background" : "fill-default";
  if (node.kind === "datastore") {
    return (
      <g className={outline}>
        <path
          d={`M 0 8 v ${h - 16} a ${w / 2} 8 0 0 0 ${w} 0 v ${-(h - 16)}`}
          className={`${fill} [stroke-width:inherit]`}
        />
        <ellipse
          cx={w / 2}
          cy={8}
          rx={w / 2}
          ry={8}
          className={`${fill} [stroke-width:inherit]`}
        />
      </g>
    );
  }
  if (node.kind === "service") {
    return (
      <polygon
        points={`12,0 ${w - 12},0 ${w},${h / 2} ${w - 12},${h} 12,${h} 0,${h / 2}`}
        className={`${fill} ${outline}`}
      />
    );
  }
  const radius = node.kind === "input" || node.kind === "output" ? h / 2 : 8;
  return (
    <rect
      width={w}
      height={h}
      rx={radius}
      className={`${fill} ${outline}`}
      strokeDasharray={node.noop ? "4 3" : undefined}
    />
  );
}

function NodeView({
  box,
  selected,
  change,
  onSelect,
}: {
  box: Box;
  selected: boolean;
  change?: Change;
  onSelect: (id: string) => void;
}) {
  const { node } = box;
  const lines = overlayLines(node.overlay);
  const top = node.kind === "datastore" ? 6 : 0;
  const onKey = (event: KeyboardEvent<SVGGElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(node.id);
    }
  };
  return (
    <g
      transform={`translate(${box.x} ${box.y})`}
      role="button"
      tabIndex={0}
      aria-label={`${node.label}: ${node.kind}${node.component ? `, ${node.component}` : ""}`}
      aria-pressed={selected}
      onClick={() => onSelect(node.id)}
      onKeyDown={onKey}
      className="cursor-pointer outline-none"
      data-node={node.id}
      data-change={change}
    >
      <title>{node.description ?? node.label}</title>
      <Shape box={box} selected={selected} change={change} />
      <text
        x={12}
        y={top + 20}
        className="fill-foreground text-[12px] font-semibold"
      >
        {truncate(node.label, 24)}
        {node.lab_only ? " · lab" : ""}
      </text>
      <text x={12} y={top + 35} className="fill-current text-[10px] text-muted">
        {truncate(nodeSubtitle(node), 28)}
      </text>
      {lines.map((line, index) => (
        <text
          key={index}
          x={12}
          y={top + 50 + index * 12}
          className="fill-current text-[10px] text-accent"
        >
          {truncate(line, 32)}
        </text>
      ))}
      {change === "added" || change === "changed" ? (
        <text
          x={box.width - 16}
          y={14}
          className="fill-foreground text-[11px] font-bold"
        >
          {change === "added" ? "+" : "Δ"}
        </text>
      ) : null}
    </g>
  );
}

/** Parse routes that share a parser make identical rows: each shows once. */
function uniqueRows(rows: string[][]): string[][] {
  return [...new Map(rows.map((row) => [row.join("|"), row])).values()];
}

/** The table view: every node and edge as text. */
export function MapTable({ map }: { map: MapLike }) {
  const rows = mapRows(map);
  return (
    <details className="text-sm">
      <summary className="cursor-pointer text-link">
        Nodes and edges as a table
      </summary>
      <div className="mt-2 grid gap-3">
        <DataTable<string[]>
          label="Map nodes"
          rows={uniqueRows(rows.nodes)}
          rowKey={(row) => row.join("|")}
          columns={[
            "Node",
            "Kind",
            "Lane",
            "Component or store",
            "Numbers",
          ].map((header, index) => ({
            header,
            cell: (row: string[]) => row[index],
          }))}
        />
        <DataTable<string[]>
          label="Map edges"
          rows={uniqueRows(rows.edges)}
          rowKey={(row) => row.join("|")}
          columns={["From", "Kind", "To", "What"].map((header, index) => ({
            header,
            cell: (row: string[]) => row[index],
          }))}
        />
      </div>
    </details>
  );
}

function NodeDetail({ node, map }: { node: MapNodeLike; map: MapLike }) {
  const names = new Map(map.nodes.map((item) => [item.id, item.label]));
  const edges = map.edges.filter(
    (edge) => edge.source === node.id || edge.target === node.id,
  );
  return (
    <div className="grid gap-2 text-sm" data-testid="map-node-detail">
      <div>
        <span className="font-semibold">{node.label}</span>{" "}
        <span className="text-muted">
          {node.kind}
          {node.component ? ` · ${node.component} ${node.version ?? ""}` : ""}
        </span>
      </div>
      {node.description ? (
        <p className="text-muted">{node.description}</p>
      ) : null}
      {overlaySummary(node.overlay) ? (
        <p>{overlaySummary(node.overlay)}</p>
      ) : null}
      {node.params && Object.keys(node.params).length ? (
        <pre className="max-h-48 overflow-auto rounded-md border border-separator p-2 font-mono text-xs">
          {JSON.stringify(node.params, null, 2)}
        </pre>
      ) : null}
      <ul className="grid gap-0.5">
        {edges.map((edge) => (
          <li key={edge.id}>
            {edge.source === node.id ? "→" : "←"}{" "}
            {names.get(edge.source === node.id ? edge.target : edge.source)}:{" "}
            {edge.label}
            {edge.sends_text ? (
              <strong className="text-warning"> · text leaves the lab</strong>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A map, its legend, a selected node's detail, exports and the table view. */
export function PipelineMap({
  map,
  mermaid,
  changes,
  label = "Pipeline map",
}: {
  map: MapLike;
  mermaid?: string;
  changes?: Record<string, Change>;
  label?: string;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const svg = useRef<SVGSVGElement>(null);
  const layout = useMemo(
    () => layoutMap(map, { selected, showAll }),
    [map, selected, showAll],
  );
  const node = map.nodes.find((item) => item.id === selected) ?? null;
  const removed = Object.entries(changes ?? {})
    .filter(([, change]) => change === "removed")
    .map(([id]) => id);

  const download = () => {
    if (!svg.current) return;
    const text = new XMLSerializer().serializeToString(svg.current);
    const url = URL.createObjectURL(
      new Blob([text], { type: "image/svg+xml" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "pipeline-map.svg";
    link.click();
    URL.revokeObjectURL(url);
  };
  const copy = async () => {
    if (mermaid) await navigator.clipboard?.writeText(mermaid);
  };

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        {Object.entries(EDGE_KINDS).map(([kind, text]) => (
          <span
            key={kind}
            className={`inline-flex items-center gap-1 ${KIND_CLASS[kind] ?? ""}`}
          >
            <svg width="22" height="8" aria-hidden="true">
              <line
                x1="0"
                y1="4"
                x2="22"
                y2="4"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
            <span className="text-foreground">{text}</span>
          </span>
        ))}
        <span className="text-warning">⚠ text leaves the lab</span>
        <label className="ml-auto flex items-center gap-1">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(event) => setShowAll(event.target.checked)}
          />
          Every read and write
        </label>
        <Button size="sm" variant="secondary" onPress={download}>
          Download SVG
        </Button>
        {mermaid ? (
          <Button size="sm" variant="secondary" onPress={copy}>
            Copy Mermaid
          </Button>
        ) : null}
      </div>
      <div className="overflow-x-auto rounded-md border border-separator">
        <svg
          ref={svg}
          role="group"
          aria-label={label}
          data-testid="pipeline-map"
          width={layout.width}
          height={layout.height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="text-foreground"
        >
          <defs>
            <marker
              id="map-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" />
            </marker>
          </defs>
          {layout.lanes.map((lane) => (
            <g key={lane.lane}>
              <rect
                x={4}
                y={lane.y}
                width={layout.width - 8}
                height={lane.height + 8}
                rx={10}
                className="fill-transparent stroke-separator"
              />
              <text
                x={14}
                y={lane.y + 18}
                className="fill-current text-[11px] uppercase text-muted"
              >
                {lane.label}
              </text>
            </g>
          ))}
          {layout.edges.map((edge) => (
            <g
              key={edge.id}
              className={`${edge.sendsText ? "text-warning" : (KIND_CLASS[edge.kind] ?? "text-muted")} ${
                edge.highlighted
                  ? "opacity-100"
                  : selected
                    ? "opacity-40"
                    : "opacity-90"
              }`}
              data-edge={edge.id}
            >
              <path
                d={edge.path}
                fill="none"
                stroke="currentColor"
                strokeWidth={
                  edge.highlighted || edge.kind === "flow" ? 2 : 1.25
                }
                markerEnd={edge.bundle ? undefined : "url(#map-arrow)"}
              />
              {edge.kind !== "flow" &&
              (edge.highlighted || edge.sendsText || edge.bundle) ? (
                <text
                  x={edge.labelX + 6}
                  y={edge.labelY}
                  className="fill-current text-[10px]"
                >
                  {edge.sendsText ? "⚠ " : ""}
                  {truncate(edge.label, 36)}
                </text>
              ) : null}
            </g>
          ))}
          {layout.boxes.map((box) => (
            <NodeView
              key={box.id}
              box={box}
              selected={box.id === selected}
              change={changes?.[box.id]}
              onSelect={(id) => setSelected(id === selected ? null : id)}
            />
          ))}
        </svg>
      </div>
      {removed.length ? (
        <p className="text-sm">
          Not in this map, but in the baseline's: {removed.join(", ")}
        </p>
      ) : null}
      {node ? <NodeDetail node={node} map={map} /> : null}
      <MapTable map={map} />
    </div>
  );
}
