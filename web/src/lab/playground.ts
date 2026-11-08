// The search playground's retrieval configs: the editor's
// state as a stratafi.retrieval.v1 document, sent unsaved with each search.

export interface PlaygroundConfig {
  dense: boolean;
  keyword: boolean;
  topN: number;
  fusion: "rrf" | "weighted" | "none";
  k: number;
  alpha: number;
  candidates: number;
  topK: number;
  rerank: string;
  rerankParams: Record<string, unknown>;
}

export const DEFAULT_CONFIG: PlaygroundConfig = {
  dense: true,
  keyword: true,
  topN: 50,
  fusion: "rrf",
  k: 60,
  alpha: 0.5,
  candidates: 50,
  topK: 8,
  rerank: "rerank/none",
  rerankParams: {},
};

/** The document a config stands for; one retriever needs no fusion. */
export function retrievalDocument(
  config: PlaygroundConfig,
  name: string,
): Record<string, unknown> {
  const both = config.dense && config.keyword;
  return {
    schema: "stratafi.retrieval.v1",
    name,
    retrievers: {
      dense: config.dense ? { top_n: config.topN } : null,
      keyword: config.keyword ? { top_n: config.topN } : null,
    },
    fusion: {
      method: both
        ? config.fusion === "none"
          ? "rrf"
          : config.fusion
        : "none",
      k: config.k,
      alpha: config.alpha,
    },
    candidates: Math.max(config.candidates, config.topK),
    filters: { current_only: true },
    rerank: { component: config.rerank, params: config.rerankParams },
    top_k: config.topK,
  };
}

/** A saved retrieval config's document as editor state. */
export function configOf(document: Record<string, unknown>): PlaygroundConfig {
  const retrievers = (document.retrievers ?? {}) as Record<
    string,
    { top_n?: number } | null
  >;
  const fusion = (document.fusion ?? {}) as {
    method?: PlaygroundConfig["fusion"];
    k?: number;
    alpha?: number;
  };
  const rerank = (document.rerank ?? {}) as {
    component?: string;
    params?: Record<string, unknown>;
  };
  return {
    dense: Boolean(retrievers.dense),
    keyword: Boolean(retrievers.keyword),
    topN: retrievers.dense?.top_n ?? retrievers.keyword?.top_n ?? 50,
    fusion: fusion.method ?? "rrf",
    k: fusion.k ?? 60,
    alpha: fusion.alpha ?? 0.5,
    candidates: Number(document.candidates ?? 50),
    topK: Number(document.top_k ?? 8),
    rerank: rerank.component ?? "rerank/none",
    rerankParams: rerank.params ?? {},
  };
}

/** The evidence phrases of an eval item's expected list. */
export function evidenceOf(expected: Record<string, unknown>[]): string[] {
  return expected
    .map((item) => item.evidence)
    .filter(
      (phrase): phrase is string => typeof phrase === "string" && phrase !== "",
    );
}
