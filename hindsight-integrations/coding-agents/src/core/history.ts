/**
 * Import a harness's PAST sessions from local disk — the migration path off the older per-agent
 * plugins.
 *
 * Day to day, conversations reach a bank through live write-back, so a fresh install starts from
 * git history alone and knows nothing of what you discussed last month. The old per-agent plugins
 * stored their memory in differently-scoped banks (`claude-code::<project>` vs this package's
 * per-repo `coding-agent::{gitProject}`), and the server's bank import restores a whole bank rather
 * than merging — so those banks cannot be folded together. Re-reading the transcripts the agent
 * already wrote to disk sidesteps that entirely: the same conversations are re-extracted into
 * whichever bank is current.
 *
 * Scoped to ONE repo on purpose. This machine has ~14k Claude sessions; importing all of them would
 * cost extraction on every unrelated project. Each harness below can answer "which sessions belong
 * to this directory" cheaply.
 *
 * Only file-based harnesses are supported. opencode, Kilo, Cursor, Cline, Copilot and Devin keep
 * history in SQLite (`opencode.db`, `store.db`, …) whose schemas are internal and unversioned;
 * reading them would break on any upstream change, so they report as unsupported rather than
 * silently importing nothing.
 */
import { closeSync, existsSync, openSync, readdirSync, readSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ChatSession } from "./types";
import { readClaudeTranscript } from "./transcript";
import { readCodexTranscript } from "./transcript-codex";
import type { TransportTurn } from "./chat";

export interface HistoryImport {
  supported: boolean;
  /** Why, when unsupported — surfaced to the user rather than failing silently. */
  reason?: string;
  sessions: ChatSession[];
}

/** Claude encodes a project directory as its absolute path with separators replaced by `-`. */
export function claudeProjectDir(repoDir: string, home = homedir()): string {
  return join(home, ".claude", "projects", repoDir.replace(/[/.]/g, "-"));
}

function toSession(id: string, turns: TransportTurn[]): ChatSession | undefined {
  // `action` turns are tool-call breadcrumbs; the interchange format carries prose only.
  const prose = turns
    .filter((t) => t.role === "user" || t.role === "assistant")
    .map((t) => ({
      role: t.role,
      text: t.content,
      ...(t.timestamp ? { timestamp: t.timestamp } : {}),
    }));
  return prose.length ? { id, turns: prose } : undefined;
}

/**
 * First line of a file, read in chunks.
 *
 * Codex's `session_meta` header is a single line that carries the agent's full base instructions —
 * tens of KB. Reading a fixed prefix and splitting on newline truncated it mid-string, so every
 * rollout failed to parse and the import silently found nothing. Capped so a file with no newline
 * can't pull an unbounded amount into memory.
 */
function firstLine(path: string, cap = 1_000_000): string | undefined {
  const fd = openSync(path, "r");
  try {
    const chunk = Buffer.alloc(64 * 1024);
    let acc = "";
    while (acc.length < cap) {
      const n = readSync(fd, chunk, 0, chunk.length, null);
      if (n <= 0) break;
      acc += chunk.subarray(0, n).toString("utf8");
      const nl = acc.indexOf("\n");
      if (nl !== -1) return acc.slice(0, nl);
    }
    return acc.length && acc.length < cap ? acc : undefined;
  } finally {
    closeSync(fd);
  }
}

function jsonlFiles(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".jsonl"))
    .map((f) => join(dir, f));
}

/** Claude Code: one directory per project, one .jsonl per session — no scanning needed. */
function claudeHistory(repoDir: string, home: string): HistoryImport {
  const dir = claudeProjectDir(repoDir, home);
  const sessions: ChatSession[] = [];
  for (const file of jsonlFiles(dir)) {
    try {
      const id = file
        .split("/")
        .pop()!
        .replace(/\.jsonl$/, "");
      const s = toSession(id, readClaudeTranscript(file));
      if (s) sessions.push(s);
    } catch {
      /* a single unreadable transcript must not abort the import */
    }
  }
  return { supported: true, sessions };
}

/**
 * Codex: rollouts are partitioned by DATE, not project, so the repo is read from the `session_meta`
 * header each file opens with — cheap enough to check without parsing the whole transcript.
 */
function codexHistory(repoDir: string, home: string): HistoryImport {
  const root = join(home, ".codex", "sessions");
  if (!existsSync(root)) return { supported: true, sessions: [] };
  const files: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry);
      if (statSync(p).isDirectory()) walk(p);
      else if (entry.endsWith(".jsonl")) files.push(p);
    }
  };
  try {
    walk(root);
  } catch {
    return { supported: true, sessions: [] };
  }

  const sessions: ChatSession[] = [];
  for (const file of files) {
    try {
      const head = firstLine(file);
      if (!head) continue;
      const meta = JSON.parse(head) as { payload?: { cwd?: string; id?: string } };
      if (meta?.payload?.cwd !== repoDir) continue;
      const s = toSession(meta.payload.id ?? file, readCodexTranscript(file));
      if (s) sessions.push(s);
    } catch {
      /* skip unreadable/short files */
    }
  }
  return { supported: true, sessions };
}

const SQLITE_HISTORY =
  "keeps session history in an internal SQLite database, whose schema is unversioned and would " +
  "break on any upstream change";

/** Read a harness's past sessions for one repo. Never throws. */
export function importLocalHistory(
  harness: string,
  repoDir: string,
  home = homedir()
): HistoryImport {
  switch (harness) {
    case "claude-code":
      return claudeHistory(repoDir, home);
    case "codex":
      return codexHistory(repoDir, home);
    case "opencode":
    case "kilo":
    case "cursor-cli":
    case "cline-cli":
    case "copilot-cli":
    case "devin-cli":
      return { supported: false, reason: `${harness} ${SQLITE_HISTORY}`, sessions: [] };
    default:
      return { supported: false, reason: `no local history reader for ${harness}`, sessions: [] };
  }
}
