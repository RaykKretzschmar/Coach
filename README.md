# Coach — Personal Mentor Chatbot

An AI-powered personal mentor chatbot built with **LangGraph** and **OpenAI**. Coach helps you reflect, set goals, overcome fears, and grow — personally and professionally. It remembers important facts about you across sessions using a dual-storage memory system (SQLite + ChromaDB).

## Features

- **Persistent conversations** — pick up where you left off, powered by LangGraph's SQLite checkpointer
- **Long-term memory** — automatically extracts and stores key facts (goals, values, fears) from your conversations
- **Semantic recall** — retrieves only the most relevant memories each turn via ChromaDB vector search
- **Deduplication** — skips near-duplicate memories using cosine similarity thresholds
- **Backfill support** — syncs SQLite memories into the vector store with batching and rate limiting

## Architecture

The agent runs a three-node LangGraph pipeline per turn:

1. **load_memories** — searches ChromaDB for memories relevant to the latest user message
2. **agent** — generates a mentor response using GPT, informed by retrieved memories
3. **memory_reflector** — a separate LLM call that decides whether the exchange revealed a new fact worth saving

## Quickstart

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

Optional environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI chat model to use |
| `MENTOR_DB` | `mentor.db` | Path to the SQLite database |
| `THREAD_ID` | `default-thread` | Conversation thread identifier |
| `CHROMA_DIR` | `chroma_db` | ChromaDB persistent storage directory |
| `MEMORY_TOP_K` | `10` | Max memories retrieved per turn |
| `DEDUP_THRESHOLD` | `0.92` | Similarity score above which a memory is considered a duplicate |
| `MAX_HISTORY_MESSAGES` | `20` | Conversation history cap (prevents unbounded token growth) |
| `ENABLE_BACKFILL_ON_STARTUP` | `false` | Auto-run vector store backfill on startup |
| `BACKFILL_BATCH_SIZE` | `100` | Memories per backfill batch |
| `BACKFILL_RATE_LIMIT_DELAY` | `1.0` | Seconds between backfill batches |

### Usage

```bash
python agent.py
```

To force a vector store backfill from SQLite on startup:

```bash
python agent.py --backfill
```

Type `exit`, `quit`, or `q` to end the session.
