import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { claudeProjectDir, importLocalHistory } from "./history";

let home: string;
afterEach(() => {
  if (home) rmSync(home, { recursive: true, force: true });
});

function newHome(): string {
  home = mkdtempSync(join(tmpdir(), "hs-history-"));
  return home;
}

const claudeLine = (role: string, text: string) =>
  JSON.stringify({ type: role, message: { role, content: [{ type: "text", text }] } });

describe("local history import", () => {
  it("reads Claude sessions from the project directory for THIS repo only", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    const dir = claudeProjectDir(repo, h);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "s1.jsonl"),
      `${claudeLine("user", "why retry 429?")}\n${claudeLine("assistant", "because backpressure")}\n`
    );
    // A different project must not leak into this repo's import.
    const other = claudeProjectDir("/Users/x/dev/otherrepo", h);
    mkdirSync(other, { recursive: true });
    writeFileSync(join(other, "s2.jsonl"), `${claudeLine("user", "unrelated")}\n`);

    const r = importLocalHistory("claude-code", repo, h);
    expect(r.supported).toBe(true);
    expect(r.sessions).toHaveLength(1);
    expect(r.sessions[0].id).toBe("s1");
    expect(JSON.stringify(r.sessions[0].turns)).toContain("why retry 429?");
    expect(JSON.stringify(r.sessions)).not.toContain("unrelated");
  });

  it("matches Codex rollouts by the cwd in their session_meta header", () => {
    const h = newHome();
    const day = join(h, ".codex", "sessions", "2026", "08", "03");
    mkdirSync(day, { recursive: true });
    const rollout = (cwd: string, text: string) =>
      `${JSON.stringify({ type: "session_meta", payload: { id: "abc", cwd } })}\n` +
      `${JSON.stringify({ type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text }] } })}\n`;
    writeFileSync(join(day, "mine.jsonl"), rollout("/repo/mine", "mine"));
    writeFileSync(join(day, "theirs.jsonl"), rollout("/repo/theirs", "theirs"));

    const r = importLocalHistory("codex", "/repo/mine", h);
    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("mine");
    expect(JSON.stringify(r.sessions)).not.toContain("theirs");
  });

  it("handles a session_meta header larger than one read chunk", () => {
    const h = newHome();
    const day = join(h, ".codex", "sessions", "2026", "08", "03");
    mkdirSync(day, { recursive: true });
    // Codex embeds the agent's full base instructions in this single line — tens of KB. Reading a
    // fixed-size prefix truncated it mid-JSON, so EVERY rollout was skipped and the import
    // silently returned nothing.
    const huge = "x".repeat(200_000);
    writeFileSync(
      join(day, "big.jsonl"),
      `${JSON.stringify({ type: "session_meta", payload: { id: "big", cwd: "/repo/mine", instructions: huge } })}\n` +
        `${JSON.stringify({ type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "found me" }] } })}\n`
    );
    const r = importLocalHistory("codex", "/repo/mine", h);
    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("found me");
  });

  it("reports SQLite-backed harnesses as unsupported with a reason, not an empty success", () => {
    const h = newHome();
    for (const harness of ["opencode", "kilo", "cursor-cli", "cline-cli"]) {
      const r = importLocalHistory(harness, "/repo/mine", h);
      expect(r.supported).toBe(false);
      expect(r.reason).toMatch(/SQLite/);
      expect(r.sessions).toEqual([]);
    }
  });

  it("returns empty (not an error) when a supported harness has no history", () => {
    const h = newHome();
    expect(importLocalHistory("claude-code", "/repo/none", h)).toEqual({
      supported: true,
      sessions: [],
    });
  });
});
