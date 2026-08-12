import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent as ReactMouseEvent } from "react";
import { api } from "../api/client";
import { streamChatCompletion, type ChatMessage } from "../api/aiChat";
import { buildContextSnapshot } from "../api/aiContext";
import {
  deriveTitle,
  loadActiveDialogueId,
  loadDialogues,
  loadMemory,
  makeId,
  saveActiveDialogueId,
  saveDialogues,
  saveMemory,
  type Dialogue,
  type DisplayMessage,
  type MemoryItem,
} from "../api/aiStorage";
import { formatRelativeTime } from "../utils";
import { IconBrain, IconChevronDown, IconHistory, IconMessageCircle, IconPlus, IconRefreshCw, IconTrash, IconX } from "./icons";
import { MarkdownMessage } from "./MarkdownMessage";

const OPEN_KEY = "llamatoaster:ai-panel-open";
const WIDTH_KEY = "llamatoaster:ai-panel-width";
const DEFAULT_WIDTH = 384; // matches the old fixed w-96
const MIN_WIDTH = 320;
const MAX_WIDTH = 720;
const BOTTOM_THRESHOLD_PX = 24;

const SYSTEM_PROMPT =
  "You are an assistant embedded in LlamaToaster, a local LLM inference benchmarking tool for " +
  "llama.cpp / llama-bench. Help the user choose llama.cpp parameters (threads, n_gpu_layers, " +
  "batch/ubatch size, KV cache quantization, flash attention), suggest GGUF models that fit their " +
  "hardware, and analyze the benchmark data below to say which configuration performs best on " +
  "their machine(s).\n\n" +
  "Grounding: you have search_huggingface_gguf_models and list_huggingface_gguf_files tools. Your " +
  "training data's knowledge of model releases is stale — treat any belief about whether a model " +
  "exists yet, its parameter count, or its file size as unverified until a tool confirms it this " +
  "turn. Never say a model \"isn't released yet\" or \"isn't available\" based on memory; search " +
  "first. If a tool call fails or returns nothing useful, say so plainly instead of falling back to " +
  "a guess.\n\n" +
  "Style: keep replies short — a few sentences or a short bullet list. Don't use a markdown table " +
  "unless the user is comparing multiple options side by side or explicitly asks for one. Skip " +
  "preamble and caveats; lead with the answer.";

const EXAMPLE_PROMPTS = [
  "Which of my models runs fastest, and with what settings?",
  "What n_gpu_layers should I try on my GPU?",
  "Suggest a model under 8B that suits my hardware.",
];

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function loadWidth(): number {
  const raw = Number(localStorage.getItem(WIDTH_KEY));
  return Number.isFinite(raw) && raw > 0 ? clamp(raw, MIN_WIDTH, MAX_WIDTH) : DEFAULT_WIDTH;
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 px-0.5 py-1" aria-label="Assistant is responding">
      <span className="ai-typing-dot" />
      <span className="ai-typing-dot" />
      <span className="ai-typing-dot" />
    </span>
  );
}

export function ChatPanel() {
  const [open, setOpen] = useState(() => localStorage.getItem(OPEN_KEY) === "1");
  const [width, setWidth] = useState(loadWidth);
  const [resizing, setResizing] = useState(false);
  // undefined = not checked yet, null = checked, not configured
  const [model, setModel] = useState<string | null | undefined>(undefined);
  const [checkingStatus, setCheckingStatus] = useState(false);
  const [dialogues, setDialogues] = useState<Dialogue[]>(loadDialogues);
  const [activeId, setActiveId] = useState<string | null>(loadActiveDialogueId);
  const [memory, setMemory] = useState<MemoryItem[]>(loadMemory);
  const [memoryDraft, setMemoryDraft] = useState("");
  const [overlay, setOverlay] = useState<"history" | "memory" | null>(null);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [context, setContext] = useState<string | null>(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [isAtBottom, setIsAtBottom] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const resizeStartRef = useRef({ x: 0, width: 0 });

  const configured = Boolean(model);
  const activeDialogue = dialogues.find((d) => d.id === activeId) ?? null;
  const messages = activeDialogue?.messages ?? [];

  async function checkStatus() {
    setCheckingStatus(true);
    try {
      const status = await api.getAiStatus();
      setModel(status.configured ? (status.model ?? "") : null);
    } catch {
      setModel(null);
    } finally {
      setCheckingStatus(false);
    }
  }

  // Cheap single request -- checked once up front (not gated on the panel
  // being open) so opening the panel never shows a configuration-check
  // flash before the chat or "not configured" view appears.
  useEffect(() => {
    void checkStatus();
  }, []);

  useEffect(() => {
    localStorage.setItem(OPEN_KEY, open ? "1" : "0");
  }, [open]);

  useEffect(() => {
    localStorage.setItem(WIDTH_KEY, String(width));
  }, [width]);

  useEffect(() => {
    saveDialogues(dialogues);
  }, [dialogues]);

  useEffect(() => {
    saveActiveDialogueId(activeId);
  }, [activeId]);

  useEffect(() => {
    saveMemory(memory);
  }, [memory]);

  // "Stick to bottom" like most chat UIs: while the user is already at the
  // bottom, new/streaming content keeps following it down. The moment they
  // scroll up (to re-read something while a reply is still generating, say),
  // this stops fighting them -- see handleScroll and the jump-to-bottom
  // button below instead of always forcing scrollTop to the end.
  // `configured` is in the deps even though this effect doesn't read it --
  // the scrollable list only mounts once the "Checking…" branch gives way to
  // the real chat UI, and without `configured` here, that mount (triggered by
  // *its* own state change, not messages/open/isAtBottom) wouldn't re-run
  // this effect, so a reload with existing history would silently load
  // scrolled to the top.
  useEffect(() => {
    if (!isAtBottom || overlay) return;
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages, open, isAtBottom, configured, overlay]);

  function handleScroll() {
    const el = listRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setIsAtBottom(distanceFromBottom < BOTTOM_THRESHOLD_PX);
  }

  function scrollToBottom() {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    setIsAtBottom(true);
  }

  function handleResizeStart(e: ReactMouseEvent) {
    e.preventDefault();
    resizeStartRef.current = { x: e.clientX, width };
    setResizing(true);
  }

  // Attached only while actively dragging (not on every render) -- a plain
  // window listener is simplest here since the drag can leave the handle's
  // own element while the mouse button is still down.
  useEffect(() => {
    if (!resizing) return;
    function handleMove(e: globalThis.MouseEvent) {
      const delta = resizeStartRef.current.x - e.clientX; // dragging left (toward the handle) widens the panel
      setWidth(clamp(resizeStartRef.current.width + delta, MIN_WIDTH, MAX_WIDTH));
    }
    function handleUp() {
      setResizing(false);
    }
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [resizing]);

  async function refreshContext() {
    setContextLoading(true);
    try {
      setContext(await buildContextSnapshot());
    } catch {
      setContext("(context unavailable — couldn't reach LlamaToaster's own API)");
    } finally {
      setContextLoading(false);
    }
  }

  // Pulls a fresh hardware/models/results snapshot the first time the panel
  // is actually opened (not on app load) -- cheap per open, avoids a
  // background fetch burst for a panel that's collapsed by default.
  useEffect(() => {
    if (open && configured && context === null && !contextLoading) {
      void refreshContext();
    }
    // context/contextLoading deliberately excluded: this effect only reacts
    // to the panel opening or the assistant becoming configured, not to the
    // context fetch it itself triggers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, configured]);

  function patchDialogueMessages(dialogueId: string, updater: (msgs: DisplayMessage[]) => DisplayMessage[]) {
    setDialogues((prev) =>
      prev.map((d) => (d.id === dialogueId ? { ...d, messages: updater(d.messages), updatedAt: Date.now() } : d))
    );
  }

  async function handleSend() {
    const text = draft.trim();
    if (!text || streaming) return;
    setDraft("");

    const isNewDialogue = activeId === null;
    const dialogueId = activeId ?? makeId();
    const userMsg: DisplayMessage = { id: makeId(), role: "user", content: text };
    const assistantId = makeId();
    const history = [...messages, userMsg];

    setDialogues((prev) => {
      if (isNewDialogue) {
        const newDialogue: Dialogue = {
          id: dialogueId,
          title: deriveTitle([userMsg]),
          messages: [...history, { id: assistantId, role: "assistant", content: "" }],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        return [newDialogue, ...prev];
      }
      return prev.map((d) =>
        d.id === dialogueId
          ? { ...d, messages: [...history, { id: assistantId, role: "assistant", content: "" }], updatedAt: Date.now() }
          : d
      );
    });
    if (isNewDialogue) setActiveId(dialogueId);

    setStreaming(true);
    setIsAtBottom(true); // sending a message should always jump back to the live edge, even if scrolled up

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let ctx = context;
      if (ctx === null) {
        ctx = await buildContextSnapshot().catch(() => "(context unavailable)");
        setContext(ctx);
      }
      const memoryBlock =
        memory.length > 0
          ? `\n\nThings to remember about this user, persisted across chats:\n${memory.map((m) => `- ${m.text}`).join("\n")}`
          : "";
      const outgoing: ChatMessage[] = [
        { role: "system", content: `${SYSTEM_PROMPT}${memoryBlock}\n\n${ctx}` },
        ...history
          .filter((m): m is DisplayMessage & { role: "user" | "assistant" } => m.role !== "error")
          .map((m) => ({ role: m.role, content: m.content })),
      ];
      let acc = "";
      await streamChatCompletion(
        outgoing,
        (delta) => {
          acc += delta;
          patchDialogueMessages(dialogueId, (msgs) => msgs.map((m) => (m.id === assistantId ? { ...m, content: acc } : m)));
        },
        controller.signal,
        (usedModel) => {
          patchDialogueMessages(dialogueId, (msgs) => msgs.map((m) => (m.id === assistantId ? { ...m, model: usedModel } : m)));
        }
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // user hit Stop -- keep whatever partial text streamed in, no error bubble
      } else {
        const message = err instanceof Error ? err.message : String(err);
        patchDialogueMessages(dialogueId, (msgs) => msgs.map((m) => (m.id === assistantId ? { ...m, role: "error", content: message } : m)));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function handleComposerKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  function handleNewChat() {
    setActiveId(null);
    setOverlay(null);
  }

  function handleSelectDialogue(id: string) {
    setActiveId(id);
    setOverlay(null);
  }

  function handleDeleteDialogue(id: string, e: ReactMouseEvent) {
    e.stopPropagation();
    const target = dialogues.find((d) => d.id === id);
    if (target && !window.confirm(`Delete "${target.title}"?`)) return;
    setDialogues((prev) => prev.filter((d) => d.id !== id));
    if (id === activeId) setActiveId(null);
  }

  function toggleOverlay(name: "history" | "memory") {
    setOverlay((prev) => (prev === name ? null : name));
  }

  function handleAddMemory() {
    const text = memoryDraft.trim();
    if (!text) return;
    setMemory((prev) => [...prev, { id: makeId(), text, createdAt: Date.now() }]);
    setMemoryDraft("");
  }

  function handleMemoryKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddMemory();
    }
  }

  function handleDeleteMemory(id: string) {
    setMemory((prev) => prev.filter((m) => m.id !== id));
  }

  const iconBtnCls = "rounded-md p-1.5 text-muted hover:bg-white/5 hover:text-fg disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-muted";
  const sortedDialogues = [...dialogues].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <aside
      className={`relative flex h-screen shrink-0 flex-col border-l border-border bg-surface ${
        resizing ? "" : "transition-[width] duration-150"
      } ${open ? "" : "w-11"}`}
      style={open ? { width } : undefined}
    >
      {open && (
        <div
          onMouseDown={handleResizeStart}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize AI assistant panel"
          className="absolute left-0 top-0 z-10 h-full w-1.5 -translate-x-1/2 cursor-col-resize hover:bg-accent/30 active:bg-accent/50"
        />
      )}
      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex h-full flex-col items-center justify-center gap-3 text-muted hover:text-accent"
          aria-label="Open AI assistant"
        >
          <IconMessageCircle width={18} height={18} />
          <span className="text-xs font-medium tracking-wide [writing-mode:vertical-rl]">AI Assistant</span>
        </button>
      ) : (
        <>
          <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2.5">
            <h2 className="flex items-center gap-1.5 text-sm font-semibold text-fg">
              <IconMessageCircle width={16} height={16} />
              AI Assistant
            </h2>
            <div className="flex items-center gap-0.5">
              {configured && (
                <>
                  <button
                    type="button"
                    onClick={handleNewChat}
                    disabled={streaming}
                    className={iconBtnCls}
                    aria-label="New chat"
                    title="New chat"
                  >
                    <IconPlus width={15} height={15} />
                  </button>
                  <button
                    type="button"
                    onClick={() => toggleOverlay("history")}
                    disabled={streaming}
                    className={`${iconBtnCls} ${overlay === "history" ? "bg-white/5 text-accent" : ""}`}
                    aria-label="Chat history"
                    title="Chat history"
                  >
                    <IconHistory width={15} height={15} />
                  </button>
                  <button
                    type="button"
                    onClick={() => toggleOverlay("memory")}
                    className={`${iconBtnCls} ${overlay === "memory" ? "bg-white/5 text-accent" : ""}`}
                    aria-label="Memory"
                    title="Memory"
                  >
                    <IconBrain width={15} height={15} />
                  </button>
                </>
              )}
              <button
                type="button"
                onClick={() => setOpen(false)}
                className={iconBtnCls}
                aria-label="Collapse panel"
                title="Collapse"
              >
                <IconX width={16} height={16} />
              </button>
            </div>
          </header>

          {model === undefined ? (
            <div className="flex flex-1 items-center justify-center text-sm text-muted">Checking…</div>
          ) : !configured ? (
            <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4 text-sm text-muted">
              <p>The AI Assistant isn't configured on this server yet.</p>
              <p>
                Set <code className="text-fg">AI_API_KEY</code>, <code className="text-fg">AI_BASE_URL</code>, and{" "}
                <code className="text-fg">AI_MODEL</code> in a <code className="text-fg">.env</code> file at the
                repo root (see <code className="text-fg">.env.example</code>), then restart the server.
              </p>
              <button
                type="button"
                onClick={() => void checkStatus()}
                disabled={checkingStatus}
                className="flex w-fit items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
              >
                <IconRefreshCw width={12} height={12} className={checkingStatus ? "animate-spin" : ""} />
                Check again
              </button>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-1.5 text-xs text-muted">
                <span className="truncate" title={model || undefined}>
                  {contextLoading
                    ? "Loading hardware & results…"
                    : context
                      ? `Context loaded · ${model}`
                      : `No context loaded yet · ${model}`}
                </span>
                <button
                  type="button"
                  onClick={() => void refreshContext()}
                  disabled={contextLoading}
                  className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 hover:bg-white/5 hover:text-fg disabled:opacity-40"
                  title="Refresh hardware & results context"
                >
                  <IconRefreshCw width={12} height={12} className={contextLoading ? "animate-spin" : ""} />
                  Refresh
                </button>
              </div>

              {overlay === "history" ? (
                <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-2">
                  {sortedDialogues.length === 0 ? (
                    <p className="p-2 text-sm text-muted">No previous chats yet.</p>
                  ) : (
                    <div className="flex flex-col gap-0.5">
                      {sortedDialogues.map((d) => (
                        <div
                          key={d.id}
                          role="button"
                          tabIndex={0}
                          onClick={() => handleSelectDialogue(d.id)}
                          onKeyDown={(e) => e.key === "Enter" && handleSelectDialogue(d.id)}
                          className={`group flex cursor-pointer items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-left hover:bg-white/5 ${
                            d.id === activeId ? "bg-white/5" : ""
                          }`}
                        >
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm text-fg">{d.title}</div>
                            <div className="text-xs text-muted">
                              {formatRelativeTime(new Date(d.updatedAt).toISOString()) ?? "today"} · {d.messages.length}{" "}
                              message{d.messages.length === 1 ? "" : "s"}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => handleDeleteDialogue(d.id, e)}
                            className="shrink-0 rounded-md p-1.5 text-muted opacity-0 hover:bg-white/10 hover:text-danger group-hover:opacity-100"
                            aria-label={`Delete "${d.title}"`}
                            title="Delete chat"
                          >
                            <IconTrash width={14} height={14} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : overlay === "memory" ? (
                <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
                  <p className="text-xs text-muted">
                    Facts saved here are included in every chat, so you don't have to repeat context.
                  </p>
                  <div className="flex items-center gap-2">
                    <input
                      value={memoryDraft}
                      onChange={(e) => setMemoryDraft(e.target.value)}
                      onKeyDown={handleMemoryKeyDown}
                      placeholder="e.g. I mostly run Q4_K_M quants"
                      className="w-full rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-fg outline-none focus:border-accent"
                    />
                    <button
                      type="button"
                      onClick={handleAddMemory}
                      disabled={!memoryDraft.trim()}
                      className="shrink-0 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-accent-fg disabled:opacity-50"
                    >
                      Add
                    </button>
                  </div>
                  {memory.length === 0 ? (
                    <p className="text-sm text-muted">Nothing saved yet.</p>
                  ) : (
                    <div className="flex flex-col gap-1">
                      {memory.map((item) => (
                        <div key={item.id} className="group flex items-start justify-between gap-2 rounded-lg px-2 py-1.5 hover:bg-white/5">
                          <span className="min-w-0 flex-1 whitespace-pre-wrap text-sm text-fg">{item.text}</span>
                          <button
                            type="button"
                            onClick={() => handleDeleteMemory(item.id)}
                            className="shrink-0 rounded-md p-1 text-muted opacity-0 hover:bg-white/10 hover:text-danger group-hover:opacity-100"
                            aria-label="Forget this"
                            title="Forget this"
                          >
                            <IconTrash width={13} height={13} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <div className="relative min-h-0 flex-1">
                    <div ref={listRef} onScroll={handleScroll} className="h-full overflow-y-auto px-3 py-3">
                      {messages.length === 0 ? (
                        <div className="flex h-full flex-col justify-center gap-3">
                          <p className="text-center text-sm text-muted">
                            Ask about model choices, llama.cpp parameters, or which config performs best
                            on your hardware.
                          </p>
                          <div className="flex flex-col gap-1.5">
                            {EXAMPLE_PROMPTS.map((p) => (
                              <button
                                key={p}
                                type="button"
                                onClick={() => setDraft(p)}
                                className="rounded-lg border border-border px-3 py-1.5 text-left text-xs text-muted hover:border-accent/40 hover:text-fg"
                              >
                                {p}
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col gap-2.5">
                          {messages.map((m) => (
                            <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                              <div
                                className={`max-w-[90%] rounded-lg px-3 py-2 ${
                                  m.role === "user"
                                    ? "whitespace-pre-wrap rounded-br-sm bg-accent/15 text-sm text-accent"
                                    : m.role === "error"
                                      ? "whitespace-pre-wrap rounded-bl-sm border border-danger/30 bg-danger-bg text-sm text-danger"
                                      : "rounded-bl-sm bg-surface-raised text-fg"
                                }`}
                              >
                                {m.role === "assistant" && m.content ? (
                                  <>
                                    <MarkdownMessage content={m.content} />
                                    {m.model && !(streaming && m.id === messages[messages.length - 1]?.id) && (
                                      <div className="mt-1 select-none text-[10px] text-muted/60">{m.model}</div>
                                    )}
                                  </>
                                ) : m.role === "assistant" && !m.content && streaming ? (
                                  <TypingDots />
                                ) : (
                                  m.content
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    {!isAtBottom && (
                      <button
                        type="button"
                        onClick={scrollToBottom}
                        aria-label="Scroll to latest"
                        title="Scroll to latest"
                        className="absolute bottom-3 right-3 flex h-8 w-8 items-center justify-center rounded-full border border-border bg-surface-raised text-fg shadow-lg hover:border-accent/40 hover:text-accent"
                      >
                        <IconChevronDown width={16} height={16} />
                      </button>
                    )}
                  </div>

                  <div className="flex flex-col gap-2 border-t border-border p-3">
                    <textarea
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onKeyDown={handleComposerKeyDown}
                      placeholder="Ask about parameters, models, or your results…"
                      rows={2}
                      className="w-full resize-none rounded-lg border border-border bg-surface px-3 py-2 text-sm text-fg outline-none focus:border-accent"
                    />
                    <div className="flex items-center justify-end gap-2">
                      {streaming ? (
                        <button
                          type="button"
                          onClick={handleStop}
                          className="rounded-lg border border-border px-4 py-1.5 text-sm font-semibold text-fg hover:border-danger/40 hover:text-danger"
                        >
                          Stop
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => void handleSend()}
                          disabled={!draft.trim()}
                          className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
                        >
                          Send
                        </button>
                      )}
                    </div>
                    <p className="text-center text-[10px] text-muted/70">
                      AI can make mistakes. Check important information before relying on it.
                    </p>
                  </div>
                </>
              )}
            </>
          )}
        </>
      )}
    </aside>
  );
}
