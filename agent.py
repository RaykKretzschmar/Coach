"""
Personal Mentor Chatbot – LangGraph + SQLite
=============================================
Run:  python agent.py
Quit: type "exit" or "quit"

The conversation thread is persisted via SqliteSaver so you can close
the script and pick up where you left off.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import textwrap
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()  # reads local Ollama configuration from .env

DB_PATH = os.getenv("MENTOR_DB", "mentor.db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:26b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "1.0"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.95"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "64"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
SKIP_OLLAMA_MODEL_CHECK = os.getenv("SKIP_OLLAMA_MODEL_CHECK", "false").lower() in (
    "true",
    "1",
    "yes",
)
THREAD_ID = os.getenv("THREAD_ID", "default-thread")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")  # persistent vector store
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "memories_local")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "10"))  # max memories per turn
DEDUP_THRESHOLD = float(
    os.getenv("DEDUP_THRESHOLD", "0.92")
)  # similarity above this → duplicate
MAX_HISTORY_MESSAGES = int(
    os.getenv("MAX_HISTORY_MESSAGES", "20")
)  # conversation history cap
ENABLE_BACKFILL_ON_STARTUP = os.getenv(
    "ENABLE_BACKFILL_ON_STARTUP", "false"
).lower() in (
    "true",
    "1",
    "yes",
)
BACKFILL_BATCH_SIZE = int(os.getenv("BACKFILL_BATCH_SIZE", "100"))
BACKFILL_RATE_LIMIT_DELAY = float(
    os.getenv("BACKFILL_RATE_LIMIT_DELAY", "1.0")
)  # seconds between batches

# ── Lazy initialization of global objects ───────────────────────────────────
# These are initialized on first use to avoid requiring local models for --help.

_llm = None
_embeddings = None
_vector_store = None


def get_llm() -> ChatOllama:
    """Lazy initialization of LLM client."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=MODEL_TEMPERATURE,
            top_p=OLLAMA_TOP_P,
            top_k=OLLAMA_TOP_K,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    return _llm


def get_embeddings() -> OllamaEmbeddings:
    """Lazy initialization of embeddings client."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
        )
    return _embeddings


def get_vector_store() -> Chroma:
    """Lazy initialization of vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vector_store


def _ollama_api_url(path: str) -> str:
    """Build an Ollama API URL from the configured base URL."""
    return f"{OLLAMA_BASE_URL.rstrip('/')}{path}"


def _get_installed_ollama_models() -> set[str]:
    """Return model names visible to the local Ollama server."""
    request = Request(_ollama_api_url("/api/tags"), method="GET")
    with urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {model.get("name", "") for model in payload.get("models", [])}


def is_ollama_model_installed(model_name: str, installed_models: set[str]) -> bool:
    """Return True when Ollama has the model, accepting implicit :latest tags."""
    if model_name in installed_models:
        return True
    if ":" not in model_name and f"{model_name}:latest" in installed_models:
        return True
    return False


def ensure_local_models_available() -> None:
    """Fail early with local setup guidance if Ollama is not ready."""
    if SKIP_OLLAMA_MODEL_CHECK:
        return

    try:
        installed_models = _get_installed_ollama_models()
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Ollama is not reachable at "
            f"{OLLAMA_BASE_URL!r}. Start Ollama, then rerun the app.\n"
            "On macOS with Homebrew: brew install ollama && brew services start ollama"
        ) from exc

    required_models = [MODEL_NAME, EMBEDDING_MODEL_NAME]
    missing_models = [
        name
        for name in required_models
        if not is_ollama_model_installed(name, installed_models)
    ]
    if missing_models:
        pull_commands = "\n".join(f"  ollama pull {name}" for name in missing_models)
        raise RuntimeError(
            "Missing local Ollama model(s): "
            f"{', '.join(missing_models)}\n"
            "Install them with:\n"
            f"{pull_commands}"
        )


# ── Database helpers ─────────────────────────────────────────────────────────


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the mentor database (creates it if needed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(run_backfill: bool = False) -> None:
    """Create the long-term *memories* table if it doesn't exist yet.

    Args:
        run_backfill: If True, run the backfill process. Otherwise, only run
                      if ENABLE_BACKFILL_ON_STARTUP is set.
    """
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                fact        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            """
        )
        conn.commit()
    print("[db] memories table ready.")

    if run_backfill or ENABLE_BACKFILL_ON_STARTUP:
        _backfill_vector_store()
    else:
        print(
            "[db] backfill skipped (use --backfill flag or set ENABLE_BACKFILL_ON_STARTUP=true)"
        )


def _get_existing_vector_ids(batch_size: int = 1000) -> set[str]:
    """Return all existing ids from the Chroma collection using pagination.

    This avoids loading the entire collection into memory in a single call.
    """
    vector_store = get_vector_store()
    existing_ids: set[str] = set()
    offset = 0
    while True:
        # Only request ids to minimize data transferred/loaded.
        batch = vector_store.get(limit=batch_size, offset=offset, include=[])
        ids = batch.get("ids") or []
        if not ids:
            break
        existing_ids.update(ids)
        # If we received fewer ids than requested, we've reached the end.
        if len(ids) < batch_size:
            break
        offset += batch_size
    return existing_ids


def _backfill_vector_store() -> None:
    """Ensure every row in the SQLite memories table is also in Chroma.

    Performs backfill with batching and rate limiting to avoid overwhelming
    the local embedding model.
    """
    vector_store = get_vector_store()
    print("[vector] starting backfill process...")
    with _get_connection() as conn:
        rows = conn.execute("SELECT id, fact, created_at FROM memories;").fetchall()

    if not rows:
        print("[vector] no memories found in SQLite, backfill complete.")
        return

    print(f"[vector] found {len(rows)} total memories in SQLite")
    existing_ids = _get_existing_vector_ids()
    print(f"[vector] found {len(existing_ids)} existing memories in Chroma")

    new_rows = [(rid, fact, ts) for rid, fact, ts in rows if rid not in existing_ids]
    if not new_rows:
        print("[vector] all memories already in Chroma, backfill complete.")
        return

    total_to_backfill = len(new_rows)
    print(
        f"[vector] backfilling {total_to_backfill} missing memories in batches of {BACKFILL_BATCH_SIZE}..."
    )

    # Process in batches to avoid overwhelming the local embedding model.
    for i in range(0, total_to_backfill, BACKFILL_BATCH_SIZE):
        batch = new_rows[i : i + BACKFILL_BATCH_SIZE]
        batch_num = (i // BACKFILL_BATCH_SIZE) + 1
        total_batches = (
            total_to_backfill + BACKFILL_BATCH_SIZE - 1
        ) // BACKFILL_BATCH_SIZE

        print(
            f"[vector] processing batch {batch_num}/{total_batches} ({len(batch)} memories)..."
        )

        vector_store.add_texts(
            texts=[fact for _, fact, _ in batch],
            ids=[rid for rid, _, _ in batch],
            metadatas=[{"created_at": ts} for _, _, ts in batch],
        )

        # Rate limiting between batches (except for the last batch)
        if (
            i + BACKFILL_BATCH_SIZE < total_to_backfill
            and BACKFILL_RATE_LIMIT_DELAY > 0
        ):
            print(
                f"[vector] rate limiting: waiting {BACKFILL_RATE_LIMIT_DELAY}s before next batch..."
            )
            time.sleep(BACKFILL_RATE_LIMIT_DELAY)

    print(
        f"[vector] ✓ backfill complete: added {total_to_backfill} memories to Chroma."
    )


# ── Memory tools ─────────────────────────────────────────────────────────────


def save_memory(fact: str) -> Optional[str]:
    """Insert a fact into SQLite + Chroma, skipping near-duplicates."""
    vector_store = get_vector_store()
    # ── Deduplication check via vector similarity ────────────────────────
    hits = vector_store.similarity_search_with_relevance_scores(fact, k=1)
    if hits:
        _doc, score = hits[0]
        if score >= DEDUP_THRESHOLD:
            print(f"[memory] ⏭  duplicate (score={score:.2f}), skipped: {fact!r}")
            return None  # near-duplicate already exists

    row_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    # SQLite (source of truth)
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO memories (id, fact, created_at) VALUES (?, ?, ?);",
            (row_id, fact, now),
        )
        conn.commit()

    # Chroma (vector index)
    vector_store.add_texts(
        texts=[fact],
        ids=[row_id],
        metadatas=[{"created_at": now}],
    )

    print(f"[memory] 💾  saved: {fact!r}")
    return row_id


def search_memories(query: str, k: int = MEMORY_TOP_K) -> str:
    """Return the top-k memories most relevant to *query*."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    if not results:
        return "(no memories stored yet)"
    lines = [f"  {i}. {doc.page_content}" for i, doc in enumerate(results, 1)]
    return "\n".join(lines)


def list_recent_memories(limit: int = 12) -> list[dict[str, str]]:
    """Return recent memories from SQLite for the web UI."""
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, fact, created_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
    return [
        {"id": row_id, "fact": fact, "created_at": created_at}
        for row_id, fact, created_at in rows
    ]


def update_memory(memory_id: str, fact: str) -> dict[str, str]:
    """Update a memory in SQLite and refresh its Chroma vector."""
    normalized_fact = fact.strip()
    if not normalized_fact:
        raise ValueError("Memory text is required.")

    with _get_connection() as conn:
        row = conn.execute(
            "SELECT created_at FROM memories WHERE id = ?;",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory not found: {memory_id}")

        created_at = row[0]
        conn.execute(
            "UPDATE memories SET fact = ? WHERE id = ?;",
            (normalized_fact, memory_id),
        )
        conn.commit()

    vector_store = get_vector_store()
    vector_store.delete(ids=[memory_id])
    vector_store.add_texts(
        texts=[normalized_fact],
        ids=[memory_id],
        metadatas=[{"created_at": created_at}],
    )

    print(f"[memory] updated: {memory_id}")
    return {"id": memory_id, "fact": normalized_fact, "created_at": created_at}


def delete_memory(memory_id: str) -> bool:
    """Delete a memory from SQLite and Chroma."""
    vector_store = get_vector_store()
    vector_store.delete(ids=[memory_id])

    with _get_connection() as conn:
        cursor = conn.execute("DELETE FROM memories WHERE id = ?;", (memory_id,))
        conn.commit()

    deleted = cursor.rowcount > 0
    if deleted:
        print(f"[memory] deleted: {memory_id}")
    return deleted


# ── LangGraph state ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: list[BaseMessage]
    current_memories: str


# ── Graph nodes ──────────────────────────────────────────────────────────────

GEMMA_THOUGHT_BLOCK_RE = re.compile(
    r"<\|channel\>thought.*?(?:<channel\|>|<\|channel\|>)",
    re.DOTALL,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
GEMMA_CONTROL_TOKEN_RE = re.compile(
    r"</?think>|<\|think\|>|<\|channel\>[^<\n]*|<channel\|>|<\|channel\|>"
)


def clean_model_text(content: object) -> str:
    """Remove model-internal thinking markers from local Gemma output."""
    text = content if isinstance(content, str) else str(content)
    text = GEMMA_THOUGHT_BLOCK_RE.sub("", text)
    text = THINK_BLOCK_RE.sub("", text)
    text = GEMMA_CONTROL_TOKEN_RE.sub("", text)
    return text.strip()


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a warm, insightful Personal Mentor.
    Your role is to help the user reflect, set goals, overcome fears,
    and grow — personally and professionally.

    Below are things you already know about this person from previous
    conversations.  Use them to give personalised advice, but NEVER
    repeat the list back verbatim.

    === Known facts ===
    {memories}
    ===================

    Guidelines:
    • Ask thoughtful follow-up questions.
    • Be encouraging but honest.
    • Keep responses concise (2-4 paragraphs max).
"""
)

REFLECTOR_PROMPT = textwrap.dedent(
    """\
    You are an internal analysis module (the user never sees your output).

    Given the latest exchange between a user and a mentor chatbot,
    decide whether the user revealed a NEW core goal, fear, value,
    preference, or important life fact that is worth permanently saving.

    Rules:
    • Only extract genuinely important, long-term facts.
    • Do NOT extract trivial or transient information.
    • Return exactly ONE short sentence describing the fact,
      or the single word NONE if nothing should be saved.

    Respond with ONLY the fact or NONE — no explanation.
"""
)


def load_memories(state: AgentState) -> dict:
    """Node 1 – retrieve only the memories relevant to the latest message."""
    # Use the most recent user message as the search query.
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    query = user_msgs[-1].content if user_msgs else ""
    memories = search_memories(query)
    print(f"[node] load_memories → {len(memories)} chars (top-{MEMORY_TOP_K})")
    return {"current_memories": memories}


def agent(state: AgentState) -> dict:
    """Node 2 – the mentor chatbot. Produces a reply for the user."""
    llm = get_llm()
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(memories=state["current_memories"])
    )
    response: AIMessage = llm.invoke([system] + state["messages"])
    response.content = clean_model_text(response.content)
    print(f"[node] agent → generated {len(response.content)} chars")
    return {"messages": state["messages"] + [response]}


def memory_reflector(state: AgentState) -> dict:
    """Node 3 – separate LLM call that decides whether to save a new memory."""
    llm = get_llm()
    # Build a mini-context with just the last user msg + assistant reply.
    recent = state["messages"][-2:]  # [HumanMessage, AIMessage]
    exchange = "\n".join(f"{m.type.upper()}: {m.content}" for m in recent)

    analysis: AIMessage = llm.invoke(
        [
            SystemMessage(content=REFLECTOR_PROMPT),
            HumanMessage(content=exchange),
        ]
    )
    fact = clean_model_text(analysis.content)
    print(f"[node] memory_reflector → {fact!r}")

    if fact.upper() != "NONE" and len(fact) > 2:
        save_memory(fact)

    return {}  # no state mutation needed


# ── Build the graph ──────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_memories", load_memories)
    graph.add_node("agent", agent)
    graph.add_node("memory_reflector", memory_reflector)

    graph.set_entry_point("load_memories")
    graph.add_edge("load_memories", "agent")
    graph.add_edge("agent", "memory_reflector")
    graph.add_edge("memory_reflector", END)

    return graph


def _thread_config(thread_id: str) -> dict:
    """Build LangGraph config for a conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


def get_thread_messages(checkpointer, thread_id: str) -> list[BaseMessage]:
    """Load checkpointed conversation messages for a thread."""
    checkpoint = checkpointer.get(_thread_config(thread_id))
    if checkpoint and "channel_values" in checkpoint:
        return checkpoint["channel_values"].get("messages", [])
    return []


def run_chat_turn(
    user_input: str,
    app,
    checkpointer,
    thread_id: str = THREAD_ID,
) -> str:
    """Run one mentor turn against a compiled LangGraph app."""
    past_messages = get_thread_messages(checkpointer, thread_id)

    # Cap conversation history to avoid unbounded token growth.
    if len(past_messages) > MAX_HISTORY_MESSAGES:
        past_messages = past_messages[-MAX_HISTORY_MESSAGES:]

    messages = past_messages + [HumanMessage(content=user_input)]
    result = app.invoke(
        {"messages": messages, "current_memories": ""},
        config=_thread_config(thread_id),
    )

    return clean_model_text(result["messages"][-1].content)


def message_to_dict(message: BaseMessage) -> dict[str, str]:
    """Serialize a LangChain message for the web UI."""
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = message.type
    return {"role": role, "content": clean_model_text(message.content)}


class MentorRuntime:
    """Long-lived compiled graph runtime shared by CLI or web UI."""

    def __init__(self, default_thread_id: str = THREAD_ID) -> None:
        self.default_thread_id = default_thread_id
        self._checkpointer_context = None
        self.checkpointer = None
        self.app = None

    def __enter__(self) -> "MentorRuntime":
        self._checkpointer_context = SqliteSaver.from_conn_string(DB_PATH)
        self.checkpointer = self._checkpointer_context.__enter__()
        self.app = build_graph().compile(checkpointer=self.checkpointer)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._checkpointer_context is not None:
            self._checkpointer_context.__exit__(exc_type, exc, traceback)

    def reply(self, user_input: str, thread_id: str | None = None) -> str:
        if self.app is None or self.checkpointer is None:
            raise RuntimeError("MentorRuntime must be used as a context manager.")
        return run_chat_turn(
            user_input=user_input,
            app=self.app,
            checkpointer=self.checkpointer,
            thread_id=thread_id or self.default_thread_id,
        )

    def history(self, thread_id: str | None = None) -> list[dict[str, str]]:
        if self.checkpointer is None:
            raise RuntimeError("MentorRuntime must be used as a context manager.")
        messages = get_thread_messages(
            self.checkpointer,
            thread_id or self.default_thread_id,
        )
        return [message_to_dict(message) for message in messages]


# ── Main loop ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personal Mentor Chatbot with LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python agent.py              # Start the chatbot (backfill only if ENABLE_BACKFILL_ON_STARTUP=true)
              python agent.py --backfill   # Run backfill maintenance and start chatbot
              
            Environment variables:
              MODEL_NAME                 - Ollama chat model (default: gemma4:26b)
              EMBEDDING_MODEL_NAME       - Ollama embedding model (default: nomic-embed-text)
              OLLAMA_BASE_URL            - Ollama server URL (default: http://localhost:11434)
              OLLAMA_NUM_CTX             - Context window for local inference (default: 8192)
              ENABLE_BACKFILL_ON_STARTUP - Auto-run backfill on startup (default: false)
              BACKFILL_BATCH_SIZE        - Number of memories per batch (default: 100)
              BACKFILL_RATE_LIMIT_DELAY  - Seconds between batches (default: 1.0)
            """
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Force backfill of vector store from SQLite on startup",
    )
    args = parser.parse_args()

    try:
        ensure_local_models_available()
    except RuntimeError as exc:
        print(f"[ollama] {exc}", file=sys.stderr)
        sys.exit(1)

    init_db(run_backfill=args.backfill)

    with MentorRuntime() as mentor:
        print("\n🎓  Personal Mentor is ready.  Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye! 👋")
                break

            assistant_reply = mentor.reply(user_input)
            print(f"\nMentor: {assistant_reply}\n")


if __name__ == "__main__":
    main()
