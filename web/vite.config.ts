import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));
// The API's development address; `uv run stratafi-api` listens here.
const apiTarget = process.env.STRATAFI_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  root,
  build: {
    // The oldest browsers Tailwind CSS 4 supports.
    target: ["chrome111", "edge111", "firefox128", "safari16.4"],
    // ECharts is a chunk of its own, about 700 kB with both renderers
    // and every chart type the lab draws, loaded only where a chart is shown.
    chunkSizeWarningLimit: 750,
  },
  // The standalone host's base path (src/standalone/host.tsx reads it back as
  // import.meta.env.BASE_URL). Another host sets its own: STRATAFI_BASE_PATH=/a/b/.
  base: process.env.STRATAFI_BASE_PATH ?? "/stratafi/",
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5190,
    strictPort: true,
    proxy: { "/api": { target: apiTarget, changeOrigin: false } },
  },
  preview: { host: "127.0.0.1", port: 4180, strictPort: true },
});
