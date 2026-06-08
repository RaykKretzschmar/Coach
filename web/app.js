const els = {
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  setupText: document.querySelector("#setupText"),
  modelPill: document.querySelector("#modelPill"),
  threadId: document.querySelector("#threadId"),
  newThread: document.querySelector("#newThread"),
  refreshMemories: document.querySelector("#refreshMemories"),
  memoryList: document.querySelector("#memoryList"),
  messages: document.querySelector("#messages"),
  promptStrip: document.querySelector("#promptStrip"),
  composer: document.querySelector("#composer"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
};

const STORAGE_KEY = "coach.threadId";
let currentMemories = [];

function defaultThreadId() {
  return localStorage.getItem(STORAGE_KEY) || "default-thread";
}

function setThreadId(threadId) {
  els.threadId.value = threadId;
  localStorage.setItem(STORAGE_KEY, threadId);
}

function makeThreadId() {
  if (crypto.randomUUID) {
    return `thread-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `thread-${Date.now().toString(36)}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw Object.assign(new Error(payload.error || "Request failed."), { payload });
  }
  return payload;
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function renderSetupLines(lines) {
  clearNode(els.setupText);
  lines.filter(Boolean).forEach((line) => {
    if (line.startsWith("ollama pull")) {
      const code = document.createElement("code");
      code.textContent = line;
      els.setupText.append(code);
      return;
    }
    const div = document.createElement("div");
    div.textContent = line;
    els.setupText.append(div);
  });
}

function setStatus(kind, text, lines = []) {
  els.statusDot.classList.remove("ready", "error");
  if (kind) els.statusDot.classList.add(kind);
  els.statusText.textContent = text;
  renderSetupLines(lines);
}

async function loadStatus() {
  try {
    const status = await api("/api/status");
    els.modelPill.textContent = status.model || "gemma4:26b";
    if (status.ok) {
      setStatus("ready", "Ready");
    } else if (status.stage === "python") {
      setStatus("error", "Setup needed", [status.hint, status.error]);
    } else {
      const lines = [];
      if (status.error) lines.push("Start Ollama and refresh.");
      if (status.missing_models?.length) lines.push("Missing local model weights:");
      lines.push(...(status.pull_commands || []));
      setStatus("error", "Local model unavailable", lines);
    }
  } catch (error) {
    setStatus("error", "Status unavailable", [error.message]);
  }
}

function welcomeState() {
  clearNode(els.messages);
  const panel = document.createElement("div");
  panel.className = "welcome";
  const heading = document.createElement("h3");
  heading.textContent = "Start a conversation";
  const copy = document.createElement("p");
  copy.textContent = "Bring a decision, a goal, a fear, or a recurring pattern.";
  panel.append(heading, copy);
  els.messages.append(panel);
  els.promptStrip.hidden = false;
}

function renderMessage(message, pending = false) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${message.role || "assistant"}${pending ? " pending" : ""}`;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = message.role === "user" ? "You" : "Coach";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.content;

  wrapper.append(label, bubble);
  return wrapper;
}

function renderMessages(history) {
  clearNode(els.messages);
  if (!history.length) {
    welcomeState();
    return;
  }
  els.promptStrip.hidden = true;
  history.forEach((message) => els.messages.append(renderMessage(message)));
  els.messages.scrollTop = els.messages.scrollHeight;
}

function appendMessage(message, pending = false) {
  if (els.messages.querySelector(".welcome")) clearNode(els.messages);
  els.promptStrip.hidden = true;
  const node = renderMessage(message, pending);
  els.messages.append(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  return node;
}

function renderMemories(memories) {
  currentMemories = memories || [];
  clearNode(els.memoryList);
  if (!currentMemories.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No saved memories yet.";
    els.memoryList.append(empty);
    return;
  }
  currentMemories.forEach((memory) => {
    const item = document.createElement("article");
    item.className = "memory-item";
    item.dataset.memoryId = memory.id;

    const topRow = document.createElement("div");
    topRow.className = "memory-top-row";

    const fact = document.createElement("p");
    fact.className = "memory-text";
    fact.textContent = memory.fact;

    const actions = document.createElement("div");
    actions.className = "memory-actions";

    const editButton = document.createElement("button");
    editButton.className = "mini-button";
    editButton.type = "button";
    editButton.title = "Edit memory";
    editButton.setAttribute("aria-label", "Edit memory");
    editButton.dataset.action = "edit-memory";
    editButton.textContent = "Edit";

    const deleteButton = document.createElement("button");
    deleteButton.className = "mini-button danger";
    deleteButton.type = "button";
    deleteButton.title = "Delete memory";
    deleteButton.setAttribute("aria-label", "Delete memory");
    deleteButton.dataset.action = "delete-memory";
    deleteButton.textContent = "Del";

    actions.append(editButton, deleteButton);
    topRow.append(fact, actions);

    const time = document.createElement("time");
    time.textContent = formatDate(memory.created_at);
    item.append(topRow, time);
    els.memoryList.append(item);
  });
}

function findMemory(memoryId) {
  return currentMemories.find((memory) => memory.id === memoryId);
}

function renderMemoryEditor(item, memory) {
  clearNode(item);
  item.classList.add("editing");

  const editor = document.createElement("textarea");
  editor.className = "memory-editor";
  editor.rows = 4;
  editor.value = memory.fact;

  const actions = document.createElement("div");
  actions.className = "memory-edit-actions";

  const saveButton = document.createElement("button");
  saveButton.className = "mini-button primary";
  saveButton.type = "button";
  saveButton.dataset.action = "save-memory";
  saveButton.textContent = "Save";

  const cancelButton = document.createElement("button");
  cancelButton.className = "mini-button";
  cancelButton.type = "button";
  cancelButton.dataset.action = "cancel-memory";
  cancelButton.textContent = "Cancel";

  actions.append(saveButton, cancelButton);
  item.append(editor, actions);
  editor.focus();
}

async function saveMemory(memoryId, fact) {
  const payload = await api(`/api/memories/${encodeURIComponent(memoryId)}`, {
    method: "PUT",
    body: JSON.stringify({ fact }),
  });
  renderMemories(payload.memories || []);
}

async function deleteMemory(memoryId) {
  const payload = await api(`/api/memories/${encodeURIComponent(memoryId)}`, {
    method: "DELETE",
  });
  renderMemories(payload.memories || []);
}

async function loadHistory() {
  try {
    const threadId = encodeURIComponent(els.threadId.value.trim());
    const payload = await api(`/api/history?thread_id=${threadId}`);
    renderMessages(payload.history || []);
  } catch (error) {
    welcomeState();
  }
}

async function loadMemories() {
  try {
    const payload = await api("/api/memories?limit=12");
    renderMemories(payload.memories || []);
  } catch (error) {
    renderMemories([]);
  }
}

function setSending(isSending) {
  els.sendButton.disabled = isSending;
  els.messageInput.disabled = isSending;
  els.sendButton.textContent = isSending ? "…" : "→";
}

async function sendMessage(message) {
  const text = message.trim();
  if (!text) return;

  setSending(true);
  appendMessage({ role: "user", content: text });
  const pending = appendMessage({ role: "assistant", content: "Thinking" }, true);
  els.messageInput.value = "";
  resizeInput();

  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message: text,
        thread_id: els.threadId.value.trim(),
      }),
    });
    pending.remove();
    renderMessages(payload.history || [
      { role: "user", content: text },
      { role: "assistant", content: payload.reply },
    ]);
    renderMemories(payload.memories || []);
    await loadStatus();
  } catch (error) {
    pending.remove();
    const status = error.payload?.status;
    const details = status?.pull_commands?.length
      ? status.pull_commands.join("\n")
      : error.message;
    appendMessage({
      role: "assistant",
      content: `I could not reach the local runtime.\n\n${details}`,
    });
    await loadStatus();
  } finally {
    setSending(false);
    els.messageInput.focus();
  }
}

function resizeInput() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = `${Math.min(160, els.messageInput.scrollHeight)}px`;
}

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(els.messageInput.value);
});

els.messageInput.addEventListener("input", resizeInput);

els.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

els.threadId.addEventListener("change", async () => {
  const nextThread = els.threadId.value.trim() || "default-thread";
  setThreadId(nextThread);
  await loadHistory();
});

els.newThread.addEventListener("click", () => {
  setThreadId(makeThreadId());
  welcomeState();
});

els.refreshMemories.addEventListener("click", loadMemories);

els.memoryList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;

  const item = button.closest(".memory-item");
  const memoryId = item?.dataset.memoryId;
  const action = button.dataset.action;
  if (!item || !memoryId) return;

  try {
    if (action === "edit-memory") {
      const memory = findMemory(memoryId);
      if (memory) renderMemoryEditor(item, memory);
      return;
    }

    if (action === "cancel-memory") {
      renderMemories(currentMemories);
      return;
    }

    if (action === "save-memory") {
      const editor = item.querySelector(".memory-editor");
      await saveMemory(memoryId, editor.value);
      return;
    }

    if (action === "delete-memory") {
      if (!window.confirm("Delete this memory?")) return;
      await deleteMemory(memoryId);
    }
  } catch (error) {
    setStatus("error", "Memory update failed", [error.message]);
  }
});

els.promptStrip.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-prompt]");
  if (!button) return;
  els.messageInput.value = button.dataset.prompt;
  resizeInput();
  els.messageInput.focus();
});

setThreadId(defaultThreadId());
welcomeState();
loadStatus();
loadHistory();
loadMemories();
