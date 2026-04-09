# Deep Research Agent

A production-ready, memory-constrained multi-step research agent built on Claude (Anthropic).  The agent answers complex, multi-part research questions while enforcing explicit token budgets at every stage of processing.

---

## Quick Start

```bash
# 1. Clone & enter project
git clone <repo-url>
cd deep-research-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the API server
uvicorn src.api.app:app --reload --port 8000

# 5. Run tests
pytest tests/ -v

# 6. Submit a research query
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare transformer and LSTM architectures for NLP tasks: what are the trade-offs in performance, training cost, and deployment complexity?"}'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST Layer                           │
│   POST /research   GET /research/{id}   GET /health   GET /metrics  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   DeepResearchAgent   │
                    │   (Orchestrator)      │
                    └───┬──────┬───────┬───┘
                        │      │       │
           ┌────────────▼─┐ ┌──▼───┐ ┌▼────────────────┐
           │QueryDecomposer│ │Memory│ │AnswerSynthesizer│
           │  (LLM → JSON)│ │Mgr   │ │  (Final merge)  │
           └──────────────┘ └──┬───┘ └─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Memory Hierarchy  │
                    │  Layer 1: Buffer    │
                    │  Layer 2: Episodes  │
                    │  Layer 3: Summary   │
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    ToolRegistry     │
                    │  web_search         │
                    │  wikipedia          │
                    │  calculator         │
                    │  doc_reader         │
                    └─────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| `DeepResearchAgent` | `src/agent/research_agent.py` | Top-level orchestrator; manages the agentic loop |
| `QueryDecomposer` | `src/agent/query_decomposer.py` | Breaks complex queries into ≤5 atomic sub-questions |
| `AnswerSynthesizer` | `src/agent/answer_synthesizer.py` | Merges sub-answers into a coherent final response |
| `MemoryManager` | `src/memory/memory_manager.py` | Episodic storage, relevance retrieval, LLM compression |
| `TokenBudget` | `src/memory/token_budget.py` | Tracks cumulative token consumption across a session |
| `ToolRegistry` | `src/tools/tool_registry.py` | Dispatches web_search, wikipedia, calculator, doc_reader |
| FastAPI app | `src/api/app.py` | REST endpoints: sync research, async jobs, metrics |

---

## Memory Strategy

The agent uses a **3-layer hierarchical episodic memory**:

```
Layer 3: Rolling Narrative (LLM-compressed)
          ↑ Generated when episodic store grows beyond threshold
Layer 2: Episodic Store (structured Episode objects)
          ↑ Written after each completed sub-query
Layer 1: Working Buffer (raw message list for current LLM call)
          ↑ Compressed on the fly when approaching token limit
```

**Retrieval** uses Jaccard similarity on keyword sets — zero latency, no embedding overhead, sufficient for ≤10 episodes per session.

**Compression** is triggered when the working buffer exceeds 75% of the per-call token limit.  The compressor uses `claude-haiku` (fast, cheap) to summarise older messages, preserving the most recent turn intact.

---

## Token Constraints

| Constraint | Default | Config Key |
|---|---|---|
| Max tokens per LLM call | 2,000 | `max_context_tokens` |
| Session token budget | 10,000 | `max_session_tokens` |
| Compression trigger | 75% of per-call limit | `summarization_threshold` |
| Max sub-queries | 5 | `max_sub_queries` |
| Max iterations per sub-query | 3 | `max_iterations_per_query` |

All constraints are configurable via `ResearchConfig` or the API request body.

---

## API Reference

### `POST /research`

Synchronous research endpoint.

```json
// Request
{
  "query": "What are the trade-offs between RAG and fine-tuning for LLM specialisation?",
  "max_context_tokens": 2000,
  "max_session_tokens": 10000,
  "max_sub_queries": 4
}

// Response
{
  "session_id": "a1b2c3d4",
  "original_query": "...",
  "sub_queries": ["What is RAG?", "What is fine-tuning?", "..."],
  "final_answer": "...",
  "sources": [{"title": "...", "url": "..."}],
  "token_usage": {"consumed": 1847, "limit": 10000, "remaining": 8153, "utilization_pct": 18.5},
  "elapsed_seconds": 3.2,
  "memory_strategy": "hierarchical_episodic",
  "success": true
}
```

### `POST /research/async`

Returns a `job_id`.  Poll `GET /research/{job_id}` for status and result.

### `GET /metrics`

Returns aggregate token usage across all sessions in the current process.

---

## Testing

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Run specific test class
pytest tests/test_suite.py::TestMemoryManager -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

Test categories:
- **Unit** — `TokenBudget`, `MemoryManager`, `QueryDecomposer`, `AnswerSynthesizer`, `ToolRegistry`
- **Integration** — `DeepResearchAgent` end-to-end with mocked Anthropic client
- **Edge cases** — Empty inputs, budget exhaustion, API failures, malformed LLM outputs

---

## Project Structure

```
deep-research-agent/
├── src/
│   ├── agent/
│   │   ├── research_agent.py      # Core orchestrator
│   │   ├── query_decomposer.py    # LLM-based decomposition
│   │   └── answer_synthesizer.py  # Final answer merging
│   ├── memory/
│   │   ├── memory_manager.py      # Episodic memory + compression
│   │   └── token_budget.py        # Session token accounting
│   ├── tools/
│   │   └── tool_registry.py       # Tool dispatch (web, wiki, calc, doc)
│   └── api/
│       └── app.py                 # FastAPI REST layer
├── tests/
│   └── test_suite.py              # 38 unit + integration tests
├── docs/
│   └── evaluation.md              # Architecture trade-off analysis
├── config/
│   └── settings.yaml              # Default configuration
├── requirements.txt
└── README.md
```

---

## Configuration

Override defaults via environment variables or `config/settings.yaml`:

```yaml
agent:
  max_context_tokens: 2000
  max_session_tokens: 10000
  max_sub_queries: 5
  max_iterations_per_query: 3
  summarization_threshold: 0.75
  model: "claude-sonnet-4-20250514"

memory:
  summarize_after: 3  # Episodes before rolling summary kicks in

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

---

## Production Deployment

For multi-user scale:

1. **Replace in-memory job store** → Redis + Celery
2. **Replace episodic store** → Pinecone / Weaviate for cross-session persistence
3. **Add rate limiting** → FastAPI middleware or NGINX
4. **Monitor token budgets** → per-user quotas backed by PostgreSQL
5. **Cache decompositions** → Redis with query hash as key, 1-hour TTL

See `docs/evaluation.md` for a full discussion of scalability trade-offs.
