// pdf.js, with its parsing in a web worker. Documents come from the API as
// bytes; pdf.js 6 never evaluates code, and it is told not to run XFA forms or
// fetch anything a document points at (no font, CMap or stream requests).

import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
} from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = workerUrl;

export type PdfDocument = PDFDocumentProxy;

export async function openPdf(data: ArrayBuffer): Promise<PdfDocument> {
  return getDocument({
    data: new Uint8Array(data),
    enableXfa: false,
    disableAutoFetch: true,
    disableStream: true,
    disableFontFace: false,
    useSystemFonts: false,
    stopAtErrors: false,
  }).promise;
}

/** Render one page into a canvas at `scale`; returns the page's size in points. */
export async function renderPage(
  document: PdfDocument,
  number: number,
  canvas: HTMLCanvasElement,
  scale: number,
): Promise<{ width: number; height: number }> {
  const page = await document.getPage(number);
  const viewport = page.getViewport({ scale });
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("no canvas");
  await page.render({ canvasContext: context, viewport, canvas }).promise;
  const base = page.getViewport({ scale: 1 });
  return { width: base.width, height: base.height };
}
