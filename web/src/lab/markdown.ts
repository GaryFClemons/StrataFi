// A run's note is Markdown, and text is shown as text: this
// parses the small part of Markdown notes use into blocks and inline spans,
// which the Note component turns into React elements. Nothing becomes HTML,
// and links stay text: nothing in a note is fetched.

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "strong"; text: string }
  | { kind: "em"; text: string }
  | { kind: "code"; text: string };

export type Block =
  | { kind: "heading"; level: number; inline: Inline[] }
  | { kind: "paragraph"; inline: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] }
  | { kind: "code"; text: string };

const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;

export function parseInline(text: string): Inline[] {
  const parts: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const token = match[0];
    const at = match.index ?? 0;
    if (at > last) parts.push({ kind: "text", text: text.slice(last, at) });
    if (token.startsWith("`"))
      parts.push({ kind: "code", text: token.slice(1, -1) });
    else if (token.startsWith("**"))
      parts.push({ kind: "strong", text: token.slice(2, -2) });
    else parts.push({ kind: "em", text: token.slice(1, -1) });
    last = at + token.length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

/** Markdown into blocks: headings, paragraphs, lists and fenced code. */
export function parseMarkdown(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: Inline[][] } | null = null;
  const flush = () => {
    if (paragraph.length) {
      blocks.push({
        kind: "paragraph",
        inline: parseInline(paragraph.join(" ")),
      });
      paragraph = [];
    }
    if (list) {
      blocks.push({ kind: "list", ...list });
      list = null;
    }
  };
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    if (line.startsWith("```")) {
      flush();
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !(lines[index] ?? "").startsWith("```")) {
        code.push(lines[index] ?? "");
        index += 1;
      }
      blocks.push({ kind: "code", text: code.join("\n") });
      continue;
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      blocks.push({
        kind: "heading",
        level: (heading[1] ?? "#").length,
        inline: parseInline((heading[2] ?? "").trim()),
      });
      continue;
    }
    const item = /^\s*([-*]|\d+\.)\s+(.*)$/.exec(line);
    if (item) {
      if (paragraph.length) flush();
      const ordered = /\d/.test(item[1] ?? "");
      if (!list || list.ordered !== ordered) {
        if (list) flush();
        list = { ordered, items: [] };
      }
      list.items.push(parseInline(item[2] ?? ""));
      continue;
    }
    if (!line.trim()) {
      flush();
      continue;
    }
    if (list) flush();
    paragraph.push(line.trim());
  }
  flush();
  return blocks;
}
