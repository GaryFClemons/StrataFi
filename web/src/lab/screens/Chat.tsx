// Chat: ask questions about a run's documents and read cited answers as they
// stream, or compare two models on the same questions. Answers render
// Markdown's shape as elements (never HTML), and each [n] opens its passage.

import { Button, Chip } from "@heroui/react";
import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { unwrap, useLoad } from "../api/load";
import {
  applyEvent,
  conversationFor,
  emptyAnswer,
  loadChatModels,
  passageLabel,
  splitCitations,
  streamChat,
  type Answer,
  type ChatModel,
  type ChatPassage,
  type RetrievalMode,
  type Turn,
} from "../chat";
import {
  Empty,
  ErrorNote,
  Loading,
  Section,
  inputClass,
} from "../components/ui";
import { Link, useLab } from "../context";
import { formatTime, shortHash } from "../format";
import { parseMarkdown, type Inline } from "../markdown";
import { useIndexedRuns } from "./Search";

interface Settings {
  runId: string; // "" chats with the model alone
  mode: RetrievalMode;
  configId: string; // "" searches by mode
  topK: number;
  model: string;
  compareWith: string; // "" answers with one model
  temperature: number;
  maxTokens: number;
}

const DEFAULTS: Omit<Settings, "runId" | "model"> = {
  mode: "hybrid",
  configId: "",
  topK: 5,
  compareWith: "",
  temperature: 0.2,
  maxTokens: 600,
};

function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return "–";
  return value < 1000
    ? `${Math.round(value)} ms`
    : `${(value / 1000).toFixed(1)} s`;
}

function modelLabel(model: ChatModel): string {
  const details = [model.parameter_size, model.quantization].filter(Boolean);
  return details.length ? `${model.id} (${details.join(", ")})` : model.id;
}

// --- answers -------------------------------------------------------------

function Cite({
  number,
  onCite,
}: {
  number: number;
  onCite: (number: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onCite(number)}
      aria-label={`Source ${number}`}
      className="mx-0.5 rounded bg-default px-1 align-baseline font-mono text-[0.7rem] text-link hover:underline"
    >
      {number}
    </button>
  );
}

function inline(
  parts: Inline[],
  passages: number,
  onCite: (number: number) => void,
): ReactNode[] {
  return parts.flatMap((part, index): ReactNode[] => {
    if (part.kind === "strong")
      return [<strong key={index}>{part.text}</strong>];
    if (part.kind === "em") return [<em key={index}>{part.text}</em>];
    if (part.kind === "code") {
      return [
        <code key={index} className="rounded bg-default px-1 text-xs">
          {part.text}
        </code>,
      ];
    }
    return splitCitations(part.text, passages).map((piece, at) =>
      piece.kind === "cite" ? (
        <Cite key={`${index}.${at}`} number={piece.number} onCite={onCite} />
      ) : (
        <span key={`${index}.${at}`}>{piece.text}</span>
      ),
    );
  });
}

/** An answer's Markdown, drawn as elements, with clickable citations. */
function AnswerText({
  text,
  passages,
  onCite,
}: {
  text: string;
  passages: number;
  onCite: (number: number) => void;
}) {
  return (
    <div className="grid gap-2 text-sm leading-relaxed">
      {parseMarkdown(text).map((block, index) => {
        switch (block.kind) {
          case "heading":
            return (
              <p key={index} className="font-semibold">
                {inline(block.inline, passages, onCite)}
              </p>
            );
          case "paragraph":
            return <p key={index}>{inline(block.inline, passages, onCite)}</p>;
          case "code":
            return (
              <pre
                key={index}
                className="overflow-auto rounded-md border border-separator p-2 font-mono text-xs"
              >
                {block.text}
              </pre>
            );
          case "list": {
            const items = block.items.map((item, at) => (
              <li key={at}>{inline(item, passages, onCite)}</li>
            ));
            return block.ordered ? (
              <ol key={index} className="list-decimal pl-5">
                {items}
              </ol>
            ) : (
              <ul key={index} className="list-disc pl-5">
                {items}
              </ul>
            );
          }
        }
        return null;
      })}
    </div>
  );
}

function Sources({
  passages,
  cited,
  focused,
}: {
  passages: ChatPassage[];
  cited: number[];
  focused: number | null;
}) {
  const refs = useRef(new Map<number, HTMLDetailsElement>());
  useEffect(() => {
    if (focused === null) return;
    const element = refs.current.get(focused);
    if (element) {
      element.open = true;
      element.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [focused]);
  return (
    <ol className="grid gap-1" aria-label="Sources">
      {passages.map((passage) => (
        <li key={passage.number}>
          <details
            ref={(element) => {
              if (element) refs.current.set(passage.number, element);
              else refs.current.delete(passage.number);
            }}
            className={`rounded-md border p-2 text-xs ${
              focused === passage.number ? "border-accent" : "border-separator"
            }`}
          >
            <summary className="cursor-pointer select-none">
              <span className="font-mono font-semibold">
                [{passage.number}]
              </span>{" "}
              {passageLabel(passage)}{" "}
              <span className="text-muted">
                · score {passage.score.toFixed(3)}
              </span>
              {cited.includes(passage.number) ? (
                <Chip size="sm" color="success" className="ml-1">
                  cited
                </Chip>
              ) : null}
            </summary>
            <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
              {passage.text}
            </pre>
          </details>
        </li>
      ))}
    </ol>
  );
}

function Stats({ answer }: { answer: Answer }) {
  const done = answer.done;
  const retrieval = answer.retrieval;
  const items: string[] = [];
  if (retrieval) {
    items.push(
      `${retrieval.passages.length} passages in ${ms(retrieval.latency_ms)}`,
    );
  }
  if (done) {
    items.push(`first token ${ms(done.timings.ttft_ms)}`);
    if (done.timings.completion_tokens_per_second) {
      items.push(`${done.timings.completion_tokens_per_second} tokens/s`);
    }
    if (done.usage) items.push(`${done.usage.total_tokens} tokens`);
    items.push(`total ${ms(done.timings.total_ms)}`);
    if (done.finish_reason === "length") items.push("cut off at max tokens");
  }
  return items.length ? (
    <p className="text-xs text-muted" data-testid="answer-stats">
      {items.join(" · ")}
    </p>
  ) : null;
}

function AnswerCard({
  answer,
  compared,
}: {
  answer: Answer;
  compared: boolean;
}) {
  const [focused, setFocused] = useState<number | null>(null);
  const passages = answer.retrieval?.passages ?? [];
  const cited = answer.done?.citations ?? [];
  const stopped = !answer.streaming && !answer.done && !answer.error;
  return (
    <article
      className="grid content-start gap-2 rounded-md border border-separator p-3"
      aria-busy={answer.streaming}
      aria-label={`Answer from ${answer.model}`}
    >
      <header className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-xs font-semibold">{answer.model}</span>
        {answer.done?.revision ? (
          <span className="font-mono text-[0.7rem] text-muted">
            rev {shortHash(answer.done.revision, 8)}
          </span>
        ) : null}
      </header>
      {answer.error ? (
        <p role="alert" className="text-sm text-danger">
          {answer.error.detail}{" "}
          <span className="font-mono text-xs">({answer.error.code})</span>
        </p>
      ) : answer.text ? (
        <AnswerText
          text={answer.text}
          passages={passages.length}
          onCite={setFocused}
        />
      ) : answer.streaming ? (
        <Loading
          label={
            answer.retrieval ? "Writing" : compared ? "Waiting" : "Searching"
          }
        />
      ) : null}
      {stopped ? <p className="text-xs text-muted">Stopped.</p> : null}
      <Stats answer={answer} />
      {passages.length ? (
        <details open={focused !== null}>
          <summary className="cursor-pointer text-xs font-medium">
            Sources ({passages.length}
            {cited.length ? `, ${cited.length} cited` : ""})
          </summary>
          <div className="mt-2">
            <Sources passages={passages} cited={cited} focused={focused} />
          </div>
        </details>
      ) : null}
    </article>
  );
}

// --- settings ------------------------------------------------------------

function SettingsPanel({
  settings,
  onChange,
  models,
}: {
  settings: Settings;
  onChange: (change: Partial<Settings>) => void;
  models: ChatModel[];
}) {
  const { api } = useLab();
  const runs = useIndexedRuns();
  const configs = useLoad(
    async () =>
      unwrap(
        await api.GET("/retrieval-configs", {
          params: { query: { limit: 50 } },
        }),
      ).items,
    "retrieval-configs",
  );
  const label = "grid min-w-0 gap-1 text-xs font-medium";
  const control = `${inputClass} w-full min-w-0`;
  return (
    <form
      className="grid content-start gap-3"
      aria-label="Chat settings"
      onSubmit={(event) => event.preventDefault()}
    >
      <label className={label}>
        Knowledge
        <select
          className={control}
          value={settings.runId}
          onChange={(event) => onChange({ runId: event.target.value })}
        >
          <option value="">None: the model alone</option>
          {(runs.data ?? []).map((run) => (
            <option key={run.id} value={run.id}>
              Run {shortHash(run.id, 8)} · {formatTime(run.created_at)}
            </option>
          ))}
        </select>
      </label>
      {runs.data && runs.data.length === 0 ? (
        <p className="text-xs text-muted">
          No indexed runs yet. Run a pipeline to Embed from{" "}
          <Link to={{ name: "pipelines", step: "embed" }} className="text-link">
            Pipelines
          </Link>
          .
        </p>
      ) : null}
      {settings.runId ? (
        <>
          <label className={label}>
            Retrieval
            <select
              className={control}
              value={settings.configId || `mode:${settings.mode}`}
              onChange={(event) => {
                const value = event.target.value;
                if (value.startsWith("mode:")) {
                  onChange({
                    configId: "",
                    mode: value.slice(5) as RetrievalMode,
                  });
                } else onChange({ configId: value });
              }}
            >
              <option value="mode:hybrid">Hybrid (dense + keyword, RRF)</option>
              <option value="mode:dense">Dense only</option>
              <option value="mode:keyword">Keyword (BM25) only</option>
              {(configs.data ?? []).map((config) => (
                <option key={config.id} value={config.id}>
                  {config.name} v{config.version}
                </option>
              ))}
            </select>
          </label>
          <label className={label}>
            Passages
            <input
              type="number"
              min={1}
              max={20}
              className={control}
              value={settings.topK}
              onChange={(event) =>
                onChange({
                  topK: Math.min(
                    20,
                    Math.max(1, Number(event.target.value) || 1),
                  ),
                })
              }
            />
          </label>
        </>
      ) : null}
      <label className={label}>
        Model
        <select
          className={control}
          value={settings.model}
          onChange={(event) => onChange({ model: event.target.value })}
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {modelLabel(model)}
            </option>
          ))}
        </select>
      </label>
      <label className={label}>
        Compare with
        <select
          className={control}
          value={settings.compareWith}
          onChange={(event) => onChange({ compareWith: event.target.value })}
        >
          <option value="">Nothing</option>
          {models
            .filter((model) => model.id !== settings.model)
            .map((model) => (
              <option key={model.id} value={model.id}>
                {modelLabel(model)}
              </option>
            ))}
        </select>
      </label>
      <div className="grid grid-cols-2 gap-2">
        <label className={label}>
          Temperature
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            className={control}
            value={settings.temperature}
            onChange={(event) =>
              onChange({
                temperature: Math.min(
                  2,
                  Math.max(0, Number(event.target.value) || 0),
                ),
              })
            }
          />
        </label>
        <label className={label}>
          Max tokens
          <input
            type="number"
            min={16}
            max={4000}
            step={50}
            className={control}
            value={settings.maxTokens}
            onChange={(event) =>
              onChange({
                maxTokens: Math.min(
                  4000,
                  Math.max(16, Number(event.target.value) || 16),
                ),
              })
            }
          />
        </label>
      </div>
      {settings.compareWith ? (
        <p className="text-xs text-muted">
          The two models answer one after the other, so neither slows the other
          and their timings compare fairly.
        </p>
      ) : null}
    </form>
  );
}

// --- the screen ----------------------------------------------------------

export function Chat() {
  const { host } = useLab();
  const models = useLoad(() => loadChatModels(host), "chat-models");
  const runs = useIndexedRuns();
  const [chosen, setChosen] = useState<Partial<Settings>>({});
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);

  const available = models.data?.models ?? [];
  const settings: Settings = {
    ...DEFAULTS,
    runId: runs.data?.[0]?.id ?? "",
    model: models.data?.default ?? available[0]?.id ?? "",
    ...chosen,
  };

  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length]);

  const update = (
    index: number,
    column: number,
    change: (answer: Answer) => Answer,
  ) =>
    setTurns((previous) =>
      previous.map((turn, at) =>
        at !== index
          ? turn
          : {
              ...turn,
              answers: turn.answers.map((answer, c) =>
                c === column ? change(answer) : answer,
              ),
            },
      ),
    );

  async function ask(question: string) {
    const columns = [settings.model, settings.compareWith].filter(Boolean);
    const index = turns.length;
    const history = turns;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setTurns([...history, { question, answers: columns.map(emptyAnswer) }]);
    for (const [column, model] of columns.entries()) {
      if (abort.signal.aborted) break;
      await streamChat(
        host,
        {
          messages: conversationFor(history, column, question),
          model,
          run_id: settings.runId || null,
          mode: settings.mode,
          retrieval_config_id: settings.configId || null,
          top_k: settings.topK,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
        },
        (event) => update(index, column, (answer) => applyEvent(answer, event)),
        abort.signal,
      );
    }
    // Answers a Stop cut short end as stopped.
    setTurns((previous) =>
      previous.map((turn, at) =>
        at !== index
          ? turn
          : {
              ...turn,
              answers: turn.answers.map((answer) =>
                answer.streaming ? { ...answer, streaming: false } : answer,
              ),
            },
      ),
    );
    setBusy(false);
  }

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = draft.trim();
    if (!question || busy || !settings.model) return;
    setDraft("");
    void ask(question);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      submit(event);
    }
  }

  function newChat() {
    controller.current?.abort();
    setTurns([]);
    setBusy(false);
  }

  return (
    <Section
      title="Chat"
      description="Ask about a run's documents and read cited answers as they stream, or compare two models on the same questions."
      actions={
        <Button
          size="sm"
          variant="secondary"
          onPress={newChat}
          isDisabled={!turns.length}
        >
          New chat
        </Button>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[17rem_minmax(0,1fr)]">
        <aside className="rounded-md border border-separator p-3">
          {models.data === undefined && !models.error ? (
            <Loading label="Loading models" />
          ) : (
            <SettingsPanel
              settings={settings}
              onChange={(change) =>
                setChosen((previous) => ({ ...previous, ...change }))
              }
              models={available}
            />
          )}
        </aside>
        <div className="grid min-h-[28rem] content-between gap-3">
          <div className="grid content-start gap-4" aria-live="polite">
            <ErrorNote error={models.error} />
            {models.error ? (
              <Empty>
                The model gateway didn't answer. Start Ollama and the gateway (
                <code className="text-xs">uv run stratafi-gateway</code>), then
                reload.
              </Empty>
            ) : null}
            {models.data && available.length === 0 ? (
              <Empty>
                The gateway has no chat models. Pull one with{" "}
                <code className="text-xs">ollama pull qwen2.5:3b</code>.
              </Empty>
            ) : null}
            {turns.length === 0 && available.length ? (
              <Empty>
                Pick the run whose documents you want to ask about (or none, to
                talk to the model alone) and ask a question. Answers cite their
                sources like [1]: click one to read the passage.
              </Empty>
            ) : null}
            {turns.map((turn, index) => (
              <section
                key={index}
                className="grid gap-2"
                aria-label={`Question ${index + 1}`}
              >
                <p className="justify-self-end whitespace-pre-wrap rounded-md bg-default px-3 py-2 text-sm">
                  {turn.question}
                </p>
                <div
                  className={`grid gap-3 ${turn.answers.length > 1 ? "md:grid-cols-2" : ""}`}
                >
                  {turn.answers.map((answer, column) => (
                    <AnswerCard
                      key={column}
                      answer={answer}
                      compared={column > 0}
                    />
                  ))}
                </div>
              </section>
            ))}
            <div ref={end} />
          </div>
          <form
            className="grid gap-2"
            onSubmit={submit}
            aria-label="Ask a question"
          >
            <textarea
              className={`${inputClass} min-h-[4.5rem] resize-y`}
              placeholder="Ask a question… (Enter sends, Shift+Enter adds a line)"
              aria-label="Question"
              value={draft}
              maxLength={20000}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onKeyDown}
            />
            <div className="flex items-center justify-end gap-2">
              {busy ? (
                <Button
                  size="sm"
                  variant="secondary"
                  onPress={() => controller.current?.abort()}
                >
                  Stop
                </Button>
              ) : null}
              <Button
                type="submit"
                size="sm"
                variant="primary"
                isDisabled={busy || !draft.trim() || !settings.model}
              >
                Send
              </Button>
            </div>
          </form>
        </div>
      </div>
    </Section>
  );
}

export default Chat;
