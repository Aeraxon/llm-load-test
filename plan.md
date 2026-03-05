# Implementation Plan: Realistic Load Testing

## Decisions

| Topic | Decision |
|---|---|
| Language | All code, comments, docs in English |
| Ollama API | Uses `/api/chat` (ndjson), not `/v1/chat/completions` |
| Mode flag | `--mode multi-turn` (default) / `--mode single-turn` (legacy) |
| User profiles | `--profile-mix 40:40:20` (Power:Normal:Occasional), configurable |
| System prompts | Loaded from file via `--system-prompts`, optional |
| New metric | TPOT (inter-token latency) alongside TTFT |

## New CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--mode` | `multi-turn` | `multi-turn` or `single-turn` (legacy behavior) |
| `--system-prompts` | None | Path to system prompts file (one per line) |
| `--turns-min` | `3` | Minimum turns per conversation session |
| `--turns-max` | `7` | Maximum turns per conversation session |
| `--profile-mix` | `40:40:20` | Power:Normal:Occasional user split (must sum to 100) |

## User Profiles

| Profile | Pause | Turns |
|---|---|---|
| Power | 2–5s | up to `--turns-max` |
| Normal | 15–45s | `--turns-min` to `--turns-max` |
| Occasional | 60–120s | `--turns-min` |

## New Metric: TPOT

Time Per Output Token = `(total_time - ttft) / (token_count - 1)`

- Ollama: use `eval_count` from the final done-message
- OpenAI-compatible: use `usage.completion_tokens` if present, else count SSE chunks
- Added to: TestResult dataclass, results table, CSV export, summary.md

## Multi-Turn Session Logic

```
session_start:
  history = []
  if system_prompts: history.append({role: system, content: random_system_prompt})
  turns = random(turns_min, turns_max)  # adjusted by profile

  for turn in range(turns):
    prompt = random choice from prompts
    history.append({role: user, content: prompt})
    response = adapter.make_chat_request(model, history)
    history.append({role: assistant, content: response})
    measure: response_time, ttft, tpot

  pause (per profile)
  goto session_start  (until test_duration exceeded)
```

## Work Streams & Dependencies

```
Stream A (Task #1): api_adapters.py
  - Add make_chat_request() to base class
  - OllamaAdapter: /api/chat + TPOT via eval_count
  - OpenAICompatibleAdapter: /v1/chat/completions + TPOT
  - Keep make_request() intact (single-turn mode)

Stream B (Task #2): Prompt data files            [parallel with A]
  - system_prompts.txt (15-20 enterprise personas)
  - prompts_long_context.txt (8-12 doc-paste prompts, 800-2000 words inline)
  - Extend prompts_english.txt (add longer-output prompts)

Stream C (Task #3): llm_load_test.py             [after A + B]
  - New CLI params (--mode, --system-prompts, --turns-min/max, --profile-mix)
  - User profile assignment at process spawn
  - Multi-turn session loop (replaces llm_chat_continuous for multi-turn mode)
  - TPOT shared list, reset_counters, TestResult, table, CSV, markdown
```

## Expected Impact on Measured Capacity

| Factor | Capacity reduction |
|---|---|
| Multi-Turn (3–5 turns) | −20–30% |
| Long input contexts | −10–15% |
| Mixed response lengths | −5–10% |
| **Total** | **~60–70% of single-turn result** |

The recommendation output should note this correction factor when running in multi-turn mode.
