"""
Personal Mentor Chatbot – LangGraph + SQLite
=============================================
Run:  python agent.py
Quit: type "exit" or "quit"

The conversation thread is persisted via SqliteSaver so you can close
the script and pick up where you left off.
"""

from __future__ import annotations

import os
import sqlite3
import textwrap
import uuid
from datetime import datetime, timezone
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()  # reads OPENAI_API_KEY (and optionally MODEL_NAME) from .env

DB_PATH = os.getenv("MENTOR_DB", "mentor.db")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
THREAD_ID = os.getenv("THREAD_ID", "default-thread")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")  # persistent vector store
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "10"))  # max memories per turn
DEDUP_THRESHOLD = float(
    os.getenv("DEDUP_THRESHOLD", "0.92")
)  # similarity above this → duplicate
MAX_HISTORY_MESSAGES = int(
    os.getenv("MAX_HISTORY_MESSAGES", "20")
)  # conversation history cap

llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="memories",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


# ── Database helpers ─────────────────────────────────────────────────────────


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the mentor database (creates it if needed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Create the long-term *memories* table if it doesn't exist yet."""
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
    _backfill_vector_store()


def _get_existing_vector_ids(batch_size: int = 1000) -> set[str]:
    """Return all existing ids from the Chroma collection using pagination.

    This avoids loading the entire collection into memory in a single call.
    """
    existing_ids: set[str] = set()
    offset = 0
    while True:
        # Only request ids to minimize data transferred/loaded.
        batch = vector_store.get(limit=batch_size, offset=offset, include=["ids"])
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
    """Ensure every row in the SQLite memories table is also in Chroma."""
    with _get_connection() as conn:
        rows = conn.execute("SELECT id, fact, created_at FROM memories;").fetchall()
    if not rows:
        return
    existing_ids = _get_existing_vector_ids()
    new_rows = [(rid, fact, ts) for rid, fact, ts in rows if rid not in existing_ids]
    if not new_rows:
        return
    vector_store.add_texts(
        texts=[fact for _, fact, _ in new_rows],
        ids=[rid for rid, _, _ in new_rows],
        metadatas=[{"created_at": ts} for _, _, ts in new_rows],
    )
    print(f"[vector] back-filled {len(new_rows)} memories into Chroma.")


# ── Memory tools ─────────────────────────────────────────────────────────────


def save_memory(fact: str) -> Optional[str]:
    """Insert a fact into SQLite + Chroma, skipping near-duplicates."""
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
    results = vector_store.similarity_search(query, k=k)
    if not results:
        return "(no memories stored yet)"
    lines = [f"  {i}. {doc.page_content}" for i, doc in enumerate(results, 1)]
    return "\n".join(lines)


# ── LangGraph state ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: list[BaseMessage]
    current_memories: str


# ── Graph nodes ──────────────────────────────────────────────────────────────

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
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(memories=state["current_memories"])
    )
    response: AIMessage = llm.invoke([system] + state["messages"])
    print(f"[node] agent → generated {len(response.content)} chars")
    return {"messages": state["messages"] + [response]}


def memory_reflector(state: AgentState) -> dict:
    """Node 3 – separate LLM call that decides whether to save a new memory."""
    # Build a mini-context with just the last user msg + assistant reply.
    recent = state["messages"][-2:]  # [HumanMessage, AIMessage]
    exchange = "\n".join(f"{m.type.upper()}: {m.content}" for m in recent)

    analysis: AIMessage = llm.invoke(
        [
            SystemMessage(content=REFLECTOR_PROMPT),
            HumanMessage(content=exchange),
        ]
    )
    fact = analysis.content.strip()
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


# ── Main loop ────────────────────────────────────────────────────────────────


def main() -> None:
    init_db()

    graph = build_graph()

    # SqliteSaver stores LangGraph checkpoints (conversation thread state).
    # We reuse the same DB file so everything lives in one place.
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": THREAD_ID}}

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

            # Retrieve the latest checkpoint so we carry forward history.
            checkpoint = checkpointer.get(config)
            past_messages: list[BaseMessage] = []
            if checkpoint and "channel_values" in checkpoint:
                past_messages = checkpoint["channel_values"].get("messages", [])

            # Cap conversation history to avoid unbounded token growth.
            if len(past_messages) > MAX_HISTORY_MESSAGES:
                past_messages = past_messages[-MAX_HISTORY_MESSAGES:]

            # Append the new human message.
            messages = past_messages + [HumanMessage(content=user_input)]

            result = app.invoke(
                {"messages": messages, "current_memories": ""},
                config=config,
            )

            # The last message in the result is the assistant reply.
            assistant_reply = result["messages"][-1].content
            print(f"\nMentor: {assistant_reply}\n")


if __name__ == "__main__":
    main()
