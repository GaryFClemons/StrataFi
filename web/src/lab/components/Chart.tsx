// A chart: an ECharts instance on a div, given an option; re-drawn when the
// option changes, resized with its container, and disposed with the
// component. ECharts loads on first use (../echarts). Canvas by default, for
// large scatter plots; SVG keeps the marks in the DOM for assistive technology
// and browser tests (the experiment views).

import { useEffect, useRef } from "react";

export function Chart({
  option,
  label,
  height = 240,
  testId = "chart",
  renderer = "canvas",
  onClick,
}: {
  option: Record<string, unknown>;
  label: string;
  height?: number;
  testId?: string;
  renderer?: "canvas" | "svg";
  onClick?: (params: { dataIndex: number; seriesIndex: number }) => void;
}) {
  const node = useRef<HTMLDivElement>(null);
  const chart = useRef<{
    dispose(): void;
    resize(): void;
    setOption(o: unknown, notMerge?: boolean): void;
  } | null>(null);
  const latest = useRef(option);
  const clicks = useRef(onClick);
  useEffect(() => {
    clicks.current = onClick;
  }, [onClick]);

  useEffect(() => {
    let alive = true;
    let observer: ResizeObserver | null = null;
    void import("../echarts").then(({ echarts }) => {
      if (!alive || !node.current) return;
      const instance = echarts.init(node.current, undefined, { renderer });
      instance.on("click", (params: unknown) =>
        clicks.current?.(params as { dataIndex: number; seriesIndex: number }),
      );
      chart.current = instance;
      instance.setOption(latest.current, true);
      node.current.dataset.rendered = "true";
      if (typeof ResizeObserver !== "undefined") {
        observer = new ResizeObserver(() => instance.resize());
        observer.observe(node.current);
      }
    });
    return () => {
      alive = false;
      observer?.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, [renderer]);

  useEffect(() => {
    latest.current = option;
    chart.current?.setOption(option, true);
  }, [option]);

  return (
    <div
      ref={node}
      role="img"
      aria-label={label}
      data-testid={testId}
      className="w-full"
      style={{ height }}
    />
  );
}
