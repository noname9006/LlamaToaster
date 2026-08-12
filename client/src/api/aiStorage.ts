// Client-side persistence for the AI assistant panel -- dialogues (separate
// conversation threads, like ChatGPT/Claude/Qwen Studio's chat history) and
// memory (small persisted facts injected into every dialogue's system
// prompt). Everything here is localStorage-only, same as the assistant's
// pre-existing single-thread history was -- no server/DB involvement, so
// none of this needs a deploy to take effect.

export interface DisplayMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  /** Upstream model that produced this reply (assistant messages only). */
  model?: string;
}

export interface Dialogue {
  id: string;
  title: string;
  messages: DisplayMessage[];
  createdAt: number;
  updatedAt: number;
}

export interface MemoryItem {
  id: string;
  text: string;
  createdAt: number;
}

const DIALOGUES_KEY = "llamatoaster:ai-dialogues";
const ACTIVE_DIALOGUE_KEY = "llamatoaster:ai-active-dialogue";
const MEMORY_KEY = "llamatoaster:ai-memory";
// Pre-multi-dialogue single-history format, read once for migration below.
// Uses the new prefix, not the literal old key -- main.tsx's
// migrateLegacyStorageKeys() already renames every "llama-bench:"-prefixed
// key to "llamatoaster:" before any component (and so this module) runs.
const LEGACY_HISTORY_KEY = "llamatoaster:ai-chat-history";

const MAX_DIALOGUES = 50;
const MAX_MESSAGES_PER_DIALOGUE = 200;
const TITLE_MAX_LEN = 48;

let idCounter = 0;
export function makeId(): string {
  idCounter += 1;
  return `${Date.now()}-${idCounter}`;
}

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function deriveTitle(messages: DisplayMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  const text = firstUser?.content.trim().replace(/\s+/g, " ") ?? "";
  if (!text) return "New chat";
  return text.length > TITLE_MAX_LEN ? `${text.slice(0, TITLE_MAX_LEN)}…` : text;
}

// One-time migration: this assistant used to keep a single flat message list
// under LEGACY_HISTORY_KEY. The first time dialogues are loaded and none
// exist yet, fold that old history into a single migrated dialogue instead
// of silently discarding a user's existing chat.
function migrateLegacyHistory(): Dialogue[] {
  const legacy = safeParse<DisplayMessage[]>(localStorage.getItem(LEGACY_HISTORY_KEY), []);
  localStorage.removeItem(LEGACY_HISTORY_KEY);
  if (legacy.length === 0) return [];
  const now = Date.now();
  return [{ id: makeId(), title: deriveTitle(legacy), messages: legacy, createdAt: now, updatedAt: now }];
}

export function loadDialogues(): Dialogue[] {
  const raw = localStorage.getItem(DIALOGUES_KEY);
  if (raw === null) {
    const migrated = migrateLegacyHistory();
    if (migrated.length > 0) saveDialogues(migrated);
    return migrated;
  }
  return safeParse<Dialogue[]>(raw, []);
}

export function saveDialogues(dialogues: Dialogue[]): void {
  try {
    const trimmed = dialogues
      .slice(0, MAX_DIALOGUES)
      .map((d) => (d.messages.length > MAX_MESSAGES_PER_DIALOGUE ? { ...d, messages: d.messages.slice(-MAX_MESSAGES_PER_DIALOGUE) } : d));
    localStorage.setItem(DIALOGUES_KEY, JSON.stringify(trimmed));
  } catch {
    /* best-effort, e.g. quota exceeded */
  }
}

export function loadActiveDialogueId(): string | null {
  return localStorage.getItem(ACTIVE_DIALOGUE_KEY);
}

export function saveActiveDialogueId(id: string | null): void {
  try {
    if (id) localStorage.setItem(ACTIVE_DIALOGUE_KEY, id);
    else localStorage.removeItem(ACTIVE_DIALOGUE_KEY);
  } catch {
    /* best-effort */
  }
}

export function loadMemory(): MemoryItem[] {
  return safeParse<MemoryItem[]>(localStorage.getItem(MEMORY_KEY), []);
}

export function saveMemory(items: MemoryItem[]): void {
  try {
    localStorage.setItem(MEMORY_KEY, JSON.stringify(items));
  } catch {
    /* best-effort */
  }
}
