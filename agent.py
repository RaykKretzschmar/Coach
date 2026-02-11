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
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()  # reads OPENAI_API_KEY (and optionally MODEL_NAME) from .env

DB_PATH = os.getenv("MENTOR_DB", "mentor.db")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
THREAD_ID = os.getenv("THREAD_ID", "default-thread")

llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)


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


# ── Memory tools ─────────────────────────────────────────────────────────────


def save_memory(fact: str) -> str:
    """Insert a single fact into the memories table. Returns the new row id."""
    row_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO memories (id, fact, created_at) VALUES (?, ?, ?);",
            (row_id, fact, now),
        )
        conn.commit()
    print(f"[memory] 💾  saved: {fact!r}")
    return row_id


def get_all_memories() -> str:
    """Return every stored fact as a numbered, human-readable string."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT fact, created_at FROM memories ORDER BY created_at;"
        ).fetchall()
    if not rows:
        return "(no memories stored yet)"
    lines = [f"  {i}. {fact}  (saved {ts})" for i, (fact, ts) in enumerate(rows, 1)]
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
    """Node 1 – fetch all stored memories and put them in state."""
    memories = get_all_memories()
    print(f"[node] load_memories → {len(memories)} chars")
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
