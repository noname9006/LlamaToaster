import { describe, it, expect, vi } from "vitest";
import {
  makeId,
  deriveTitle,
  loadDialogues,
  saveDialogues,
  loadActiveDialogueId,
  saveActiveDialogueId,
  loadMemory,
  saveMemory,
  type Dialogue,
  type DisplayMessage,
} from "./aiStorage";

const DIALOGUES_KEY = "llamatoaster:ai-dialogues";
const ACTIVE_KEY = "llamatoaster:ai-active-dialogue";
const MEMORY_KEY = "llamatoaster:ai-memory";
const LEGACY_KEY = "llamatoaster:ai-chat-history";

function msg(role: DisplayMessage["role"], content: string): DisplayMessage {
  return { id: makeId(), role, content };
}

function dialogue(patch: Partial<Dialogue> = {}): Dialogue {
  return { id: makeId(), title: "t", messages: [], createdAt: 1, updatedAt: 1, ...patch };
}

describe("makeId", () => {
  it("never repeats within a session", () => {
    const ids = new Set(Array.from({ length: 500 }, () => makeId()));
    expect(ids.size).toBe(500);
  });
});

describe("deriveTitle", () => {
  it("uses the first user message, ignoring assistant messages before it", () => {
    expect(deriveTitle([msg("assistant", "hi there"), msg("user", "what is a GGUF")])).toBe("what is a GGUF");
  });

  it("collapses internal whitespace and newlines to single spaces", () => {
    expect(deriveTitle([msg("user", "  what   is\n\na  GGUF  ")])).toBe("what is a GGUF");
  });

  it("falls back to 'New chat' with no user message or an empty one", () => {
    expect(deriveTitle([])).toBe("New chat");
    expect(deriveTitle([msg("assistant", "hello")])).toBe("New chat");
    expect(deriveTitle([msg("user", "   ")])).toBe("New chat");
  });

  it("truncates a long title with an ellipsis at 48 characters", () => {
    const title = deriveTitle([msg("user", "x".repeat(100))]);
    expect(title).toHaveLength(49); // 48 chars + the ellipsis
    expect(title.endsWith("…")).toBe(true);
  });

  it("leaves a title of exactly the limit untruncated", () => {
    expect(deriveTitle([msg("user", "y".repeat(48))])).toBe("y".repeat(48));
  });
});

describe("loadDialogues / saveDialogues", () => {
  it("round-trips through localStorage", () => {
    const d = dialogue({ id: "d1", title: "hello", messages: [msg("user", "hi")] });
    saveDialogues([d]);
    expect(loadDialogues()).toEqual([d]);
  });

  it("returns an empty list when nothing is stored", () => {
    expect(loadDialogues()).toEqual([]);
  });

  it("returns an empty list instead of throwing on a corrupt blob", () => {
    localStorage.setItem(DIALOGUES_KEY, "{not json");
    expect(loadDialogues()).toEqual([]);
  });

  it("caps stored dialogues at 50, keeping the newest-first order it was given", () => {
    const many = Array.from({ length: 60 }, (_, i) => dialogue({ id: `d${i}` }));
    saveDialogues(many);
    const loaded = loadDialogues();
    expect(loaded).toHaveLength(50);
    expect(loaded[0].id).toBe("d0");
    expect(loaded[49].id).toBe("d49");
  });

  it("caps messages per dialogue at 200, keeping the most recent ones", () => {
    const messages = Array.from({ length: 250 }, (_, i) => msg("user", `m${i}`));
    saveDialogues([dialogue({ id: "d", messages })]);
    const loaded = loadDialogues();
    expect(loaded[0].messages).toHaveLength(200);
    // slice(-200) keeps the tail: the oldest 50 are dropped.
    expect(loaded[0].messages[0].content).toBe("m50");
    expect(loaded[0].messages[199].content).toBe("m249");
  });

  it("leaves a dialogue under the message cap untrimmed", () => {
    const messages = Array.from({ length: 3 }, (_, i) => msg("user", `m${i}`));
    saveDialogues([dialogue({ id: "d", messages })]);
    expect(loadDialogues()[0].messages).toHaveLength(3);
  });

  it("swallows a quota-exceeded write rather than breaking the panel", () => {
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    expect(() => saveDialogues([dialogue()])).not.toThrow();
    spy.mockRestore();
  });
});

describe("legacy single-thread history migration", () => {
  it("folds an old flat message list into one dialogue and persists it", () => {
    const legacy = [msg("user", "old question"), msg("assistant", "old answer")];
    localStorage.setItem(LEGACY_KEY, JSON.stringify(legacy));

    const loaded = loadDialogues();

    expect(loaded).toHaveLength(1);
    expect(loaded[0].messages).toHaveLength(2);
    expect(loaded[0].title).toBe("old question");
    // Persisted, so the next load does not depend on the legacy key surviving.
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull();
    expect(loadDialogues()).toHaveLength(1);
  });

  it("consumes the legacy key even when it holds an empty list", () => {
    localStorage.setItem(LEGACY_KEY, JSON.stringify([]));
    expect(loadDialogues()).toEqual([]);
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull();
  });

  it("never runs once real dialogues exist, so it cannot resurrect old history", () => {
    saveDialogues([dialogue({ id: "current" })]);
    localStorage.setItem(LEGACY_KEY, JSON.stringify([msg("user", "should not appear")]));

    const loaded = loadDialogues();

    expect(loaded).toHaveLength(1);
    expect(loaded[0].id).toBe("current");
    // The legacy key is left alone in this branch.
    expect(localStorage.getItem(LEGACY_KEY)).not.toBeNull();
  });

  it("ignores a corrupt legacy blob", () => {
    localStorage.setItem(LEGACY_KEY, "garbage");
    expect(loadDialogues()).toEqual([]);
  });
});

describe("active dialogue id", () => {
  it("round-trips an id", () => {
    saveActiveDialogueId("d1");
    expect(loadActiveDialogueId()).toBe("d1");
  });

  it("is null when never set", () => {
    expect(loadActiveDialogueId()).toBeNull();
  });

  it("clears the key when passed null", () => {
    saveActiveDialogueId("d1");
    saveActiveDialogueId(null);
    expect(loadActiveDialogueId()).toBeNull();
    expect(localStorage.getItem(ACTIVE_KEY)).toBeNull();
  });
});

describe("memory items", () => {
  it("round-trips", () => {
    const items = [{ id: "1", text: "prefers Q4_K_M", createdAt: 5 }];
    saveMemory(items);
    expect(loadMemory()).toEqual(items);
  });

  it("returns an empty list when unset or corrupt", () => {
    expect(loadMemory()).toEqual([]);
    localStorage.setItem(MEMORY_KEY, "]]not json[[");
    expect(loadMemory()).toEqual([]);
  });

  it("swallows a quota-exceeded write", () => {
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    expect(() => saveMemory([{ id: "1", text: "x", createdAt: 0 }])).not.toThrow();
    spy.mockRestore();
  });
});
