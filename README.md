# Coach — Personal Mentor Chatbot

An AI-powered personal mentor chatbot built with **LangGraph** and local **Ollama** models. Coach helps you reflect, set goals, overcome fears, and grow — personally and professionally. It remembers important facts about you across sessions using a dual-storage memory system (SQLite + ChromaDB).

## Features

- **Persistent conversations** — pick up where you left off, powered by LangGraph's SQLite checkpointer
- **Long-term memory** — automatically extracts and stores key facts (goals, values, fears) from your conversations
- **Semantic recall** — retrieves only the most relevant memories each turn via ChromaDB vector search
- **Deduplication** — skips near-duplicate memories using cosine similarity thresholds
- **Backfill support** — syncs SQLite memories into the vector store with batching and rate limiting

## Architecture

The agent runs a three-node LangGraph pipeline per turn:

1. **load_memories** — searches ChromaDB for memories relevant to the latest user message
2. **agent** — generates a mentor response using local Gemma 4 26B-A4B, informed by retrieved memories
3. **memory_reflector** — a separate LLM call that decides whether the exchange revealed a new fact worth saving

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- Local model weights:

```bash
ollama pull gemma4:26b
ollama pull nomic-embed-text
```

`gemma4:26b` is the default chat model. It is the Gemma 4 26B-A4B mixture-of-experts model and is a more practical default than the dense 31B model on a MacBook Pro with 48 GB unified memory, especially with an 8k context window and a local embedding model running alongside it.

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

No API key is required. Optional `.env` configuration:

```env
MODEL_NAME=gemma4:26b
EMBEDDING_MODEL_NAME=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=2048
```

Optional environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gemma4:26b` | Ollama chat model to use |
| `EMBEDDING_MODEL_NAME` | `nomic-embed-text` | Ollama embedding model for ChromaDB memory search |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server URL |
| `MODEL_TEMPERATURE` | `1.0` | Chat sampling temperature |
| `OLLAMA_TOP_P` | `0.95` | Chat nucleus sampling value |
| `OLLAMA_TOP_K` | `64` | Chat top-k sampling value |
| `OLLAMA_NUM_CTX` | `8192` | Context window used for local inference |
| `OLLAMA_NUM_PREDICT` | `2048` | Max generated tokens per response before continuation handling |
| `OLLAMA_CONTINUATION_ATTEMPTS` | `2` | Extra generation passes when Ollama reports that an answer hit the length limit |
| `OLLAMA_KEEP_ALIVE` | `10m` | How long Ollama should keep the model loaded after use |
| `SKIP_OLLAMA_MODEL_CHECK` | `false` | Skip startup validation for installed local models |
| `MENTOR_DB` | `mentor.db` | Path to the SQLite database |
| `THREAD_ID` | `default-thread` | Conversation thread identifier |
| `CHROMA_DIR` | `chroma_db` | ChromaDB persistent storage directory |
| `CHROMA_COLLECTION` | `memories_local` | Chroma collection for local embeddings |
| `MEMORY_TOP_K` | `10` | Max memories retrieved per turn |
| `DEDUP_THRESHOLD` | `0.92` | Similarity score above which a memory is considered a duplicate |
| `MAX_HISTORY_MESSAGES` | `20` | Recent conversation messages sent to the model each turn; full app history is preserved |
| `ENABLE_BACKFILL_ON_STARTUP` | `false` | Auto-run vector store backfill on startup |
| `BACKFILL_BATCH_SIZE` | `100` | Memories per backfill batch |
| `BACKFILL_RATE_LIMIT_DELAY` | `1.0` | Seconds between backfill batches |

### Usage

Start the web UI:

```bash
python web_app.py
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The original terminal UI is still available:

```bash
python agent.py
```

To force a vector store backfill from SQLite on startup:

```bash
python agent.py --backfill
```

Run `--backfill` once after migrating from the old OpenAI version. The app now uses a separate `memories_local` Chroma collection because local embeddings have a different vector dimension than OpenAI embeddings.

Type `exit`, `quit`, or `q` to end the session.
