# LLM Load Test Tool

A realistic load testing tool for LLM APIs that simulates real multi-turn user behavior and helps determine the capacity limits of your LLM chat environment.

**Supports multiple backends**: Ollama, vLLM, LM Studio, llama.cpp, OpenAI, and any OpenAI-compatible API.

## Table of Contents

- [Quickstart](#quickstart)
- [Supported Backends](#supported-backends)
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Parameters](#parameters)
- [Examples](#examples)
- [Output Files](#output-files)
- [Prompt Files](#prompt-files)
- [Test Duration and Timing Behavior](#test-duration-and-timing-behavior)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure via .env

```bash
cp .env.example .env
# Edit .env: set MODEL, LLM_PROVIDER, USERS, API_TYPE, etc.
```

### 3. Run

```bash
# All config from .env — no CLI flags needed
python llm_load_test.py

# Or override individual values on the command line
python llm_load_test.py --model llama2 --users 20 --llm-provider "Ollama"

# Workload mix (20% single-turn, 60% multi-turn, 20% long-context)
python llm_load_test.py \
  --workload-mix 20:60:20 \
  --long-context-prompts prompts_long_context.txt \
  --users 50 --model llama2 --llm-provider "Ollama"
```

Results are saved to `results/<timestamp>/` — a CSV file and a Markdown summary.

## Supported Backends

| Backend | `--api-type` | Default Port |
|---------|-------------|--------------|
| Ollama | `ollama` | 11434 |
| vLLM | `vllm` | 8000 |
| LM Studio | `lmstudio` | 1234 |
| llama.cpp | `llamacpp` | 8080 |
| OpenAI | `openai` | — |
| Any OpenAI-compatible API | `openai` | — |

## Overview

The tool runs a step-by-step load test: it starts with `--step-size` users and increases in increments until `--users` is reached. Each step runs for `--test-duration` seconds and collects detailed metrics. At the end, a results table and Markdown report help identify the optimal operating point.

**Key Features:**
- **Workload Mix**: Split users across single-turn, multi-turn, and long-context slices in one run
- **Multi-Turn Sessions**: Simulates real chat conversations with growing message history (default mode)
- **User Profiles**: Power / Normal / Occasional users with different pacing and session depth
- **TPOT Measurement**: Tracks inter-token latency (Time Per Output Token) alongside TTFT
- **System Prompts**: Load enterprise assistant personas from a file to simulate varied user contexts
- **Full .env Config**: All parameters readable from `.env` — no required CLI flags
- **Multi-Backend Support**: Ollama, vLLM, LM Studio, llama.cpp, OpenAI, and more
- **Multi-Model Support**: Compare multiple models in one run
- **Automatic Overload Detection**: Aborts and skips higher user counts when error rate exceeds 30%
- **CSV + Markdown Export**: Timestamped results folder per run

## Features

### Workload Mix

The `--workload-mix` flag splits simulated users into three concurrent slices:

```
--workload-mix 20:60:20
                 │  │  └── 20% long-context users  (use --long-context-prompts, capped turns)
                 │  └───── 60% multi-turn users     (realistic chat sessions)
                 └──────── 20% single-turn users    (stateless requests)
```

When `--workload-mix` is set, it overrides `--mode`. All three slices run simultaneously within the same test step. This is the recommended mode for testing real production workloads where traffic is always mixed.

If `--workload-mix` is not set, `--mode` applies to all users.

### Multi-Turn Session Simulation (Default)

In `--mode multi-turn` each simulated user:
1. Starts a new conversation session and optionally receives a random system prompt
2. Sends multiple turns (controlled by `--turns-min` / `--turns-max`), accumulating message history
3. Pauses between sessions according to their user profile
4. Repeats until the test duration expires

Growing context windows increase per-request compute cost, so multi-turn results reflect ~60–70% of single-turn benchmark capacity by design.

Use `--mode single-turn` for a stateless throughput baseline.

### Long-Context Slice

When `--workload-mix` includes a long-context percentage, the long-context users draw prompts from `--long-context-prompts` instead of the main prompts file. Each prompt in that file contains a full document (600–900 words) inline, followed by a task instruction. The turn count for this slice is capped at `--lc-turns-max` (default: 2) to avoid excessive context accumulation.

### User Profiles

Three behavioral profiles are mixed proportionally via `--profile-mix` (Power:Normal:Occasional, default `40:40:20`). Profiles apply to all slices in a workload mix.

**Power user** — someone who uses the assistant as a core work tool all day: they fire off the next message as soon as they've read the reply (2–5 s pause), always run the maximum number of turns, and keep multiple long conversations going in parallel. Heaviest load per user.

**Normal user** — typical knowledge worker who turns to the assistant a few times per hour: reads the reply, does something else, comes back (15–45 s pause). Turn count varies — sometimes a quick one-shot question, sometimes a longer back-and-forth.

**Occasional user** — someone who drops in once in a while for a specific task (e.g. drafting an email): long pauses between sessions (60–120 s), only the minimum number of turns. Lightest load per user.

| Profile | Pause Between Sessions | Turns per Session |
|---------|----------------------|-------------------|
| Power | 2–5 s | `--turns-max` (fixed) |
| Normal | 15–45 s | random between `--turns-min` and `--turns-max` |
| Occasional | 60–120 s | `--turns-min` (fixed) |

The default mix of `40:40:20` (Power:Normal:Occasional) reflects a typical enterprise chat deployment where a significant share of users are heavy adopters. Adjust with `--profile-mix` to match your actual user distribution — e.g. `10:60:30` for a broader rollout where most users are light adopters.

### Metrics

- **TTFT (Time-to-First-Token)**: Time until first token — the primary UX metric, used for recommendations
- **TPOT (Time Per Output Token)**: `(total_time - ttft) / (token_count - 1)` — measures generation throughput
- **Response Time**: Average, maximum, minimum of complete response time per request
- **Error Rate**: Percentage of failed requests
- **CPU / Memory**: System resource usage during the test

### Automated Test Execution

- User count increases in configurable steps from `--step-size` up to `--users`
- Error rate is checked every 30 seconds; steps abort if it exceeds 30%
- When a model hits the 30% threshold, higher user counts are skipped automatically

## Installation

### Prerequisites
- Python 3.8 or higher
- A running LLM API server (Ollama, vLLM, LM Studio, etc.)

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

All parameters can be set in `.env`, on the command line, or both. **CLI arguments override `.env` values, which override built-in defaults.**

### Finding the Right Model Name

The value for `MODEL` must exactly match what the backend reports — not a human-readable name, but the internal identifier. Here's how to look it up for each backend before filling in `.env`.

#### Ollama

```bash
curl http://127.0.0.1:11434/api/tags
```

Example response (shortened):

```json
{
  "models": [
    { "name": "llama3.2:latest",  "size": 2019393189 },
    { "name": "mistral:7b-q4",    "size": 4109854720 },
    { "name": "codellama:latest",  "size": 3825819519 }
  ]
}
```

Use the `name` field exactly as shown — including the tag (`:latest`, `:7b-q4`, etc.):

```env
API_TYPE=ollama
API_BASE_URL=http://127.0.0.1:11434
MODEL=llama3.2:latest
LLM_PROVIDER=Ollama
```

If you omit the tag, Ollama defaults to `:latest` and that usually works too (`MODEL=llama3.2`).

#### vLLM

```bash
curl http://127.0.0.1:8000/v1/models
```

Example response (shortened):

```json
{
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model"
    }
  ]
}
```

Use the `id` field exactly as shown — this is the HuggingFace model path that vLLM was started with:

```env
API_TYPE=vllm
API_BASE_URL=http://127.0.0.1:8000
MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_PROVIDER=vLLM
```

vLLM usually serves exactly one model, so there is only one entry in `data`. Copy the `id` value directly.

#### LM Studio

```bash
curl http://127.0.0.1:1234/v1/models
```

Same OpenAI-compatible format as vLLM. Copy the `id` from the response — it typically looks like `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` or similar.

#### llama.cpp server

llama.cpp does not have a `/v1/models` endpoint. The model name is whatever you pass via `--model` — it can be anything you like and is used only as a label in the results file. Use the filename without path, e.g. `llama-3.1-8b-instruct.Q4_K_M.gguf`.

---

### .env File (Recommended)

```bash
cp .env.example .env
```

**Ollama example:**

```env
# === API Connection ===
API_TYPE=ollama
API_BASE_URL=http://127.0.0.1:11434

# === Model ===
MODEL=llama3.2:latest
LLM_PROVIDER=Ollama
GPU=RTX A2000

# === Prompt Files ===
PROMPTS_FILE=prompts_english.txt
SYSTEM_PROMPTS_FILE=system_prompts.txt
LONG_CONTEXT_PROMPTS_FILE=prompts_long_context.txt

# === Workload Mix (single-turn:multi-turn:long-context, must sum to 100) ===
# Overrides MODE when set
WORKLOAD_MIX=20:60:20

# === Multi-Turn Settings ===
PROFILE_MIX=40:40:20
TURNS_MIN=3
TURNS_MAX=7
LC_TURNS_MAX=2

# === Test Parameters ===
USERS=50
STEP_SIZE=5
TEST_DURATION=300
PAUSE_MIN=3
PAUSE_MAX=30
```

**vLLM example:**

```env
# === API Connection ===
API_TYPE=vllm
API_BASE_URL=http://127.0.0.1:8000

# === Model (exact id from GET /v1/models) ===
MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_PROVIDER=vLLM
GPU=A100

# === Prompt Files ===
PROMPTS_FILE=prompts_english.txt
SYSTEM_PROMPTS_FILE=system_prompts.txt
LONG_CONTEXT_PROMPTS_FILE=prompts_long_context.txt

# === Workload Mix ===
WORKLOAD_MIX=20:60:20

# === Multi-Turn Settings ===
PROFILE_MIX=40:40:20
TURNS_MIN=3
TURNS_MAX=7
LC_TURNS_MAX=2

# === Test Parameters ===
USERS=100
STEP_SIZE=10
TEST_DURATION=300
PAUSE_MIN=3
PAUSE_MAX=30
```

With a fully populated `.env`, the tool can be launched with no CLI arguments:

```bash
python llm_load_test.py
```

### CLI Arguments

Any `.env` value can be overridden on the command line:

```bash
python llm_load_test.py --model mistral --users 30 --workload-mix 0:100:0
```

## Usage

### Basic Syntax
```bash
python llm_load_test.py [OPTIONS]
```

All parameters have defaults or can be read from `.env`. The only hard requirement is that `--prompts` / `PROMPTS_FILE`, `--users` / `USERS`, `--model` / `MODEL`, and `--llm-provider` / `LLM_PROVIDER` must be specified via one of those two sources.

### Minimal Examples

```bash
# Everything from .env
python llm_load_test.py

# Override model and user count, rest from .env
python llm_load_test.py --model mistral --users 30

# Explicit CLI-only invocation (no .env needed)
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 25 \
  --model llama2 \
  --llm-provider "Ollama"
```

## Parameters

### Parameters That Must Be Set (via CLI or .env)

| CLI Flag | .env Key | Description |
|----------|----------|-------------|
| `--prompts` | `PROMPTS_FILE` | Path to main prompts file |
| `--users` | `USERS` | Maximum number of simulated users |
| `--model` | `MODEL` | Model(s), comma-separated for multiple |
| `--llm-provider` | `LLM_PROVIDER` | Provider name for documentation (e.g. `Ollama`) |

### All Parameters

| CLI Flag | .env Key | Default | Description |
|----------|----------|---------|-------------|
| `--prompts` | `PROMPTS_FILE` | — | Path to main prompts file |
| `--users` | `USERS` | — | Maximum concurrent users |
| `--model` | `MODEL` | — | Model(s), comma-separated |
| `--llm-provider` | `LLM_PROVIDER` | — | Provider label for reports |
| `--gpu` | `GPU` | `Unknown` | GPU label for reports |
| `--mode` | `MODE` | `multi-turn` | `multi-turn` or `single-turn`; ignored when `--workload-mix` is set |
| `--workload-mix` | `WORKLOAD_MIX` | None | `single:multi:lc` percentages summing to 100; overrides `--mode` |
| `--long-context-prompts` | `LONG_CONTEXT_PROMPTS_FILE` | None | Prompts file for the long-context slice |
| `--lc-turns-max` | `LC_TURNS_MAX` | `2` | Max turns for the long-context slice |
| `--system-prompts` | `SYSTEM_PROMPTS_FILE` | None | System prompts file; one persona per line |
| `--turns-min` | `TURNS_MIN` | `3` | Minimum turns per multi-turn session |
| `--turns-max` | `TURNS_MAX` | `7` | Maximum turns per multi-turn session |
| `--profile-mix` | `PROFILE_MIX` | `40:40:20` | Power:Normal:Occasional split, must sum to 100 |
| `--api-type` | `API_TYPE` | `ollama` | `ollama`, `vllm`, `lmstudio`, `llamacpp`, `openai` |
| `--host` | `API_BASE_URL` | `127.0.0.1:11434` | API host and port |
| `--api-key` | `API_KEY` | None | API key for authenticated backends |
| `--pause-min` | `PAUSE_MIN` | `3.0` | Minimum inter-session pause in seconds |
| `--pause-max` | `PAUSE_MAX` | `30.0` | Maximum inter-session pause in seconds |
| `--step-size` | `STEP_SIZE` | `5` | User count increment per step |
| `--test-duration` | `TEST_DURATION` | `300` | Duration per step in seconds |
| `--output` | `OUTPUT` | auto | Custom CSV filename; disables auto folder + `summary.md` |

## Examples

### Mixed Workload (Recommended for Production Testing)

```bash
# 20% single-turn, 60% multi-turn, 20% long-context — mirrors real traffic
python llm_load_test.py \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --long-context-prompts prompts_long_context.txt \
  --workload-mix 20:60:20 \
  --users 50 \
  --model llama2 \
  --llm-provider "Ollama" \
  --gpu "RTX A2000"
```

### Realistic Multi-Turn Load Test

```bash
python llm_load_test.py \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --users 30 \
  --model llama2 \
  --llm-provider "Ollama" \
  --gpu "RTX A2000"
```

### Long-Context Stress Test

```bash
# All users send document Q&A prompts (600–900 words of inline content)
python llm_load_test.py \
  --prompts prompts_long_context.txt \
  --users 20 \
  --model llama2 \
  --llm-provider "Ollama" \
  --turns-min 1 \
  --turns-max 2
```

### Single-Turn Throughput Benchmark

```bash
# Stateless requests — measures peak throughput, no conversation history
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 40 \
  --model llama2 \
  --llm-provider "Ollama" \
  --mode single-turn \
  --pause-min 1 \
  --pause-max 5
```

### Custom Profile Mix (Power-User Skew)

```bash
# 60% power users, 30% normal, 10% occasional
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 25 \
  --model llama2 \
  --llm-provider "Ollama" \
  --profile-mix 60:30:10 \
  --turns-min 5 \
  --turns-max 10
```

### vLLM Backend

```bash
python llm_load_test.py \
  --api-type vllm \
  --host 127.0.0.1:8000 \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --long-context-prompts prompts_long_context.txt \
  --workload-mix 20:60:20 \
  --users 50 \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --llm-provider "vLLM" \
  --gpu "A100"
```

### Multi-Model Comparison

```bash
python llm_load_test.py \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --users 30 \
  --model "llama2,mistral,codellama" \
  --llm-provider "Ollama" \
  --gpu "RTX A2000"
```

### Remote Server with Custom Output

```bash
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 25 \
  --model codellama \
  --llm-provider "Ollama" \
  --host 192.168.1.100:11434 \
  --test-duration 600 \
  --gpu "V100" \
  --output remote_test_results.csv
```

## Output Files

Each run (without `--output`) creates a timestamped directory:

```
results/
├── 20250305_143022/
│   ├── results.csv
│   └── summary.md
├── 20250305_150815/
│   ├── results.csv
│   └── summary.md
```

### results.csv

```
Users, Model, LLM_Provider, GPU,
Avg_Response_Time, Avg_TTFT, Avg_TPOT,
Max_Response_Time, Min_Response_Time,
Error_Rate, CPU_Percent, Memory_Percent,
Total_Requests, Successful_Requests, Failed_Requests,
Test_Duration, Recommendation
```

### summary.md

- **Test Configuration**: timestamp, provider, API type, base URL, mode or workload mix, profile mix, turns range
- **Results per Model**: table with Avg. Time, TTFT, TPOT, Max. Time, Error Rate, CPU, Memory, Requests, Recommendation
- **Overall Summary**: total requests, average TTFT, overall error rate
- **Recommendations**: suggested maximum concurrent users; note on correction factor for multi-turn / mixed workload results

### Manual CSV Export

```bash
python llm_load_test.py --prompts prompts_english.txt --users 20 --model llama2 --llm-provider "Ollama" \
  --output my_results.csv
```

## Prompt Files

### Main Prompts File (`--prompts` / `PROMPTS_FILE`)

One prompt per line, no blank lines. Prompts are selected randomly for each turn.

#### Included Files

| File | Prompts | Description |
|------|---------|-------------|
| `prompts_english.txt` | 200 | Business prompts covering emails, HR, finance, IT, strategy, code review, SQL, legal, compliance, and more |
| `prompts_long_context.txt` | 25 | Single-line prompts each containing 600–900 words of inline fictional document content (meeting minutes, audit reports, RFPs, contracts, post-mortems, DPIAs, etc.) followed by a concrete task |
| `prompts_deutsch.txt` | — | German-language business prompts for German-locale testing |

### System Prompts File (`--system-prompts` / `SYSTEM_PROMPTS_FILE`)

Optional. One enterprise assistant persona per line, randomly assigned at the start of each conversation session. Ensures different simulated users operate under different system-prompt prefixes, preventing KV-cache prefix sharing.

```txt
You are an IT support assistant. Help employees resolve technical issues clearly and step by step.
You are an HR policy assistant. Answer employee questions about benefits, leave, and company policies.
You are a project management assistant. Support project managers with planning, reporting, and risk tracking.
```

#### Included System Prompts File

| File | Personas | Roles Covered |
|------|----------|---------------|
| `system_prompts.txt` | 18 | IT support, HR, legal, finance, procurement, security, onboarding, training, data analysis, and more |

### Long-Context Prompts File (`--long-context-prompts` / `LONG_CONTEXT_PROMPTS_FILE`)

Used by the long-context slice of `--workload-mix`. Each prompt is a single line containing a full inline document followed by a specific task (e.g. "Summarize the key decisions", "List all action items with owners", "Identify the top three risks"). Document types include:

- Board meeting minutes, product strategy sessions
- Project status reports and post-mortems
- IT architecture decision records (ADR) and change requests
- Vendor evaluation and procurement tender reports
- Internal audit and regulatory compliance reviews
- Data governance and data protection impact assessments (DPIA)
- HR documents: performance review guidelines, engagement survey results
- SLAs, supplier contracts, M&A due diligence checklists
- IT disaster recovery plans, RFPs, budget justifications, product roadmaps

### KV-Cache Diversity at Scale

For large concurrency tests (100+ users), prompt diversity matters. With vLLM or similar KV-cache-enabled backends, repeated identical prefixes reduce measured load. The included files are sized to keep expected collisions low:

- 200 regular prompts → first-turn combinations in multi-turn: 18 × 200 = 3,600 unique prefixes
- 25 long-context prompts → 50 LC users means each prompt used ~2× on average (acceptable since task instructions differ)
- 18 system prompts → each adds a unique per-session prefix on top of the user message

## Test Duration and Timing Behavior

`--test-duration` defines the **active request phase**. After it expires, the tool waits for all in-flight requests to complete (drain phase).

```
00:00       Test starts
00:00–05:00 Active phase: users continuously send requests and start new sessions
05:00       No new requests started
05:00–06:15 Drain phase: in-flight requests finish
06:15       Step complete — metrics collected
```

### Automatic Termination on Overload

- Error rate is checked every 30 seconds during the active phase
- If error rate > 30% with at least 10 requests, the step aborts immediately
- All higher user-count steps for that model are then skipped

```
[Progress] Requests: 47, Error rate: 35.2%
⚠️ ABORT: Error rate (35.2%) exceeds 30%!
```

## Interpreting Results

### Live Output

```
[Step 2/8] Testing 10 users with llama2...
Workload mix: 20% single-turn / 60% multi-turn / 20% long-context
[User 3|turn 1] ✓ 4.23s (TTFT: 1.45s, TPOT: 0.042s) - Explain the difference between...
[User 7|turn 2] ✗ Timeout - restarting session
[User 1|turn 3] ✓ 2.45s (TTFT: 0.89s, TPOT: 0.031s) - What is Python?...
```

### Results Table

```
LOAD TEST RESULTS
Users  Model    LLM Provider  GPU    Avg. Time  TTFT    TPOT    Max. Time  Error Rate  Recommendation
5      llama2   Ollama        A2000  2.34       1.12    0.038   4.12       0.0         ✅ Optimal
10     llama2   Ollama        A2000  3.78       1.89    0.051   7.45       2.1         ⚠️ Unstable
15     llama2   Ollama        A2000  5.23       3.45    0.078   12.34      5.4         ❌ Overloaded
```

### Metrics Reference

**TTFT (Time-to-First-Token)** — primary recommendation metric:

| TTFT | Assessment |
|------|-----------|
| < 2 s | Optimal |
| 2–5 s | Good |
| 5–10 s | Acceptable |
| 10–20 s | Slow |
| > 20 s | Unacceptable |

**TPOT (Time Per Output Token)** — generation throughput indicator:
- Typical range: 0.02–0.10 s/token
- Lower is better; rising TPOT under load indicates GPU saturation

**Error Rate** — stability indicator:

| Error Rate | Assessment |
|------------|-----------|
| 0–2% | Production-ready |
| 2–5% | Unstable |
| 5–10% | Overloaded |
| > 10% | Critical |

### Workload Mode Comparison

| Mode | What It Measures | Correction Factor |
|------|-----------------|------------------|
| `--mode single-turn` | Peak stateless throughput | 1× (baseline) |
| `--mode multi-turn` | Realistic chat capacity | ~0.6–0.7× vs. single-turn |
| `--workload-mix` | Mixed production traffic | ~0.6–0.7× for multi-turn/LC slices |

## Best Practices

### Start with a Baseline

```bash
# Quick single-turn baseline to confirm the model responds correctly
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 5 --step-size 5 \
  --model llama2 --llm-provider "Ollama" \
  --mode single-turn --test-duration 120
```

### Realistic Production Test with .env

Set everything in `.env` once and just run:

```bash
python llm_load_test.py
```

Example `.env` for a vLLM production test:

```env
API_TYPE=vllm
API_BASE_URL=http://127.0.0.1:8000
MODEL=meta-llama/Llama-3-8B-Instruct
LLM_PROVIDER=vLLM
GPU=A100
PROMPTS_FILE=prompts_english.txt
SYSTEM_PROMPTS_FILE=system_prompts.txt
LONG_CONTEXT_PROMPTS_FILE=prompts_long_context.txt
WORKLOAD_MIX=20:60:20
USERS=100
STEP_SIZE=10
TEST_DURATION=300
PROFILE_MIX=40:40:20
```

### Multi-Model Comparison Under Identical Conditions

```bash
python llm_load_test.py \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --users 25 \
  --model "llama2,mistral,codellama" \
  --llm-provider "Ollama" \
  --gpu "RTX A2000"
```

### Server-Side Monitoring

```bash
htop                      # CPU and memory
nvidia-smi -l 1           # GPU utilization and VRAM
journalctl -u ollama -f   # Ollama logs
```

### Optimization Based on Results

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| High TPOT, low TTFT | GPU throughput bottleneck | Smaller model or tensor parallelism |
| High TTFT, low TPOT | Scheduling or batch size issue | Tune `max_num_batched_tokens` / `OLLAMA_NUM_PARALLEL` |
| Rising error rate | Concurrency limit exceeded | Reduce user count or add backend capacity |
| TPOT flat, TTFT rising | Context window growth | Expected in multi-turn; reduce `--turns-max` or `--lc-turns-max` |

```bash
# Ollama tuning
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

# vLLM: tensor parallelism for large models
python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --tensor-parallel-size 2
```

---

## Disclaimer

**USE AT YOUR OWN RISK**

This software is provided "as is" without warranty of any kind. The author(s) assume **NO LIABILITY** for any damages, losses, or issues arising from its use.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

- **Non-commercial use** (personal, education, research, open source): free
- **Commercial use**: open an issue or contact via GitHub

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
