// ECharts, tree-shaken to what the lab draws: bars (token histograms,
// effects with error bars), scatter plots (projections, Pareto fronts), lines,
// heatmaps and custom series, with grids, tooltips, legends, mark lines, zoom
// and a visual map; on a canvas or as SVG; and parallel coordinates and box
// plots for the Compare workspace. Loaded on first use, so screens without
// charts don't carry it.

import {
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ParallelChart,
  ScatterChart,
} from "echarts/charts";
import {
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  ParallelComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer, SVGRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ParallelChart,
  ScatterChart,
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  ParallelComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
  SVGRenderer,
]);

export { echarts };
