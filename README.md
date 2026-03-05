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

Get started in 3 steps:

### 1. Install Dependencies
```bash
pip install requests psutil python-dotenv
```

### 2. Create a Prompts File
Create a file `prompts.txt` with one prompt per line:
```txt
What is Python?
Explain machine learning in simple terms
Write a hello world function in JavaScript
```

### 3. Run Your First Test
```bash
# Multi-turn mode (default) with Ollama
python llm_load_test.py --prompts prompts.txt --users 20 --model llama2 --llm-provider "Ollama"

# With system prompts for more realistic persona simulation
python llm_load_test.py --prompts prompts.txt --users 20 --model llama2 --llm-provider "Ollama" \
  --system-prompts system_prompts.txt

# For vLLM
python llm_load_test.py --api-type vllm --host 127.0.0.1:8000 --prompts prompts.txt --users 20 \
  --model "meta-llama/Llama-2-7b-chat-hf" --llm-provider "vLLM"

# Single-turn mode (original behavior)
python llm_load_test.py --prompts prompts.txt --users 20 --model llama2 --llm-provider "Ollama" \
  --mode single-turn
```

That's it! The tool automatically tests with 5, 10, 15, and 20 users, and saves results to `results/<timestamp>/` including a CSV file and a Markdown summary.

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

This tool performs automated load tests by gradually increasing the number of simulated users (e.g., 5, 10, 15, 20 users). Each step runs for a defined duration (default 5 minutes) and collects detailed metrics. At the end, an evaluation table helps identify optimal capacity.

**Key Features:**
- **Multi-Turn Sessions**: Simulates real chat conversations with growing message history (default mode)
- **User Profiles**: Power / Normal / Occasional users with different pacing and session depth
- **TPOT Measurement**: Tracks inter-token latency (Time Per Output Token) alongside TTFT
- **System Prompts**: Load enterprise assistant personas from a file to simulate varied user contexts
- **Multi-Backend Support**: Works with Ollama, vLLM, LM Studio, llama.cpp, OpenAI, and more
- **Gradual User Increase**: Automatic tests from step-size up to the desired maximum
- **Multi-Model Support**: Compare multiple models in one run
- **Automatic Evaluation**: Results table with all important metrics
- **System Monitoring**: CPU and memory monitoring during tests
- **CSV + Markdown Export**: Structured results for further analysis

## Features

### Multi-Turn Session Simulation (Default)

By default the tool runs in `--mode multi-turn`. Each simulated user:
1. Starts a new conversation session and optionally receives a random system prompt
2. Sends multiple turns within that session (3–7 by default), accumulating message history
3. Pauses between sessions according to their user profile
4. Repeats until the test duration is up

This models real chat application behavior where growing context windows increase per-request compute cost. Results in multi-turn mode reflect ~60–70% of single-turn benchmark capacity by design.

Use `--mode single-turn` to reproduce the original single-request behavior for pure throughput measurement.

### User Profiles

Three behavioral profiles are mixed into the simulated user pool via `--profile-mix` (Power:Normal:Occasional, default `40:40:20`):

| Profile | Pause Between Sessions | Turns per Session |
|---------|----------------------|-------------------|
| Power | 2–5 s | `--turns-max` (fixed) |
| Normal | 15–45 s | random between `--turns-min` and `--turns-max` |
| Occasional | 60–120 s | `--turns-min` (fixed) |

### Comprehensive Metrics

- **Response Time**: Average, maximum, minimum of complete response time per request
- **TTFT (Time-to-First-Token)**: Time until first token — the primary UX metric
- **TPOT (Time Per Output Token)**: `(total_time - ttft) / (token_count - 1)` — measures generation throughput
- **Error Rate**: Percentage of failed requests
- **System Monitoring**: CPU and memory usage during tests
- **Request Statistics**: Successful vs. failed requests

### Automated Test Execution

- **Gradual Increase**: User count is automatically increased in configurable steps
- **Automatic Termination**: Tests abort early if error rate exceeds 30%
- **Skip-on-Overload**: When a model hits the 30% threshold, higher user counts are skipped automatically

## Installation

### Prerequisites
- Python 3.8 or higher
- A running LLM API server (Ollama, vLLM, LM Studio, etc.)

### Installing Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install requests psutil python-dotenv
```

### Download
```bash
git clone <repository-url>
cd llm-load-test
```

## Configuration

Configure via a `.env` file or command-line arguments. CLI arguments override `.env`.

### Option 1: Using .env File (Recommended)

```bash
cp .env.example .env
```

```env
# For Ollama (default)
API_TYPE=ollama
API_BASE_URL=http://127.0.0.1:11434

# For vLLM
# API_TYPE=vllm
# API_BASE_URL=http://127.0.0.1:8000

# For LM Studio
# API_TYPE=lmstudio
# API_BASE_URL=http://127.0.0.1:1234

# For llama.cpp
# API_TYPE=llamacpp
# API_BASE_URL=http://127.0.0.1:8080

# For OpenAI
# API_TYPE=openai
# API_BASE_URL=https://api.openai.com
# API_KEY=your-api-key-here
```

### Option 2: Command-Line Arguments

```bash
python llm_load_test.py --api-type vllm --host 127.0.0.1:8000 --prompts prompts.txt --users 25 --model llama2 --llm-provider "vLLM"
```

## Usage

### Basic Syntax
```bash
python llm_load_test.py --prompts PROMPTS_FILE --users MAX_USERS --model MODEL(S) --llm-provider NAME [OPTIONS]
```

### Minimal Example
```bash
# Multi-turn (default): tests 5, 10, 15, 20, 25 users
python llm_load_test.py --prompts prompts_english.txt --users 25 --model llama2 --llm-provider "Ollama"

# Single-turn: original throughput benchmark behavior
python llm_load_test.py --prompts prompts_english.txt --users 25 --model llama2 --llm-provider "Ollama" --mode single-turn

# Multi-model comparison
python llm_load_test.py --prompts prompts_english.txt --users 25 --model "llama2,mistral" --llm-provider "Ollama"
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--prompts` | Path to prompts file | `--prompts prompts_english.txt` |
| `--users` | Maximum number of users (reached gradually) | `--users 50` |
| `--model` | Model(s), comma-separated for multiple | `--model "llama2,mistral"` |
| `--llm-provider` | Provider name for documentation | `--llm-provider "Ollama"` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `multi-turn` | `multi-turn` (realistic chat) or `single-turn` (throughput benchmark) |
| `--system-prompts` | None | Path to system prompts file (one per line); randomly assigned per session |
| `--turns-min` | `3` | Minimum turns per multi-turn session |
| `--turns-max` | `7` | Maximum turns per multi-turn session |
| `--profile-mix` | `40:40:20` | Power:Normal:Occasional user split, must sum to 100 |
| `--api-type` | `ollama` | API type: `ollama`, `vllm`, `lmstudio`, `llamacpp`, `openai` |
| `--host` | from .env or `127.0.0.1:11434` | API host and port |
| `--api-key` | from .env | API key for authentication |
| `--gpu` | `Unknown` | GPU label for documentation |
| `--pause-min` | `3.0` | Minimum inter-session pause in seconds (single-turn: inter-request) |
| `--pause-max` | `30.0` | Maximum inter-session pause in seconds |
| `--step-size` | `5` | User count increment per step |
| `--test-duration` | `300` | Duration per step in seconds (default: 5 minutes) |
| `--output` | auto | Custom CSV filename; disables folder + summary.md creation |

## Examples

### Realistic Multi-Turn Load Test (Default)

```bash
# With enterprise system prompts for varied user personas
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
# Use long-context prompts to simulate document Q&A workloads
python llm_load_test.py \
  --prompts prompts_long_context.txt \
  --system-prompts system_prompts.txt \
  --users 20 \
  --model llama2 \
  --llm-provider "Ollama" \
  --turns-min 2 \
  --turns-max 4
```

### Custom Profile Mix

```bash
# Heavy power-user skew (60% power, 30% normal, 10% occasional)
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 25 \
  --model llama2 \
  --llm-provider "Ollama" \
  --profile-mix 60:30:10 \
  --turns-min 5 \
  --turns-max 10
```

### Single-Turn Throughput Benchmark

```bash
# Original behavior — pure throughput, no conversation history
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 40 \
  --model llama2 \
  --llm-provider "Ollama" \
  --mode single-turn \
  --pause-min 1 \
  --pause-max 5
```

### vLLM Backend

```bash
python llm_load_test.py \
  --api-type vllm \
  --host 127.0.0.1:8000 \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --users 30 \
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

### Remote Server with Longer Tests

```bash
python llm_load_test.py \
  --prompts prompts_english.txt \
  --users 25 \
  --model codellama \
  --llm-provider "Ollama" \
  --host 192.168.x.x:11434 \
  --test-duration 600 \
  --gpu "V100" \
  --output remote_test_results.csv
```

## Output Files

The tool automatically creates a timestamped folder for each test run.

### Directory Structure

```
results/
├── 20250105_143022/
│   ├── results.csv
│   └── summary.md
├── 20250105_150815/
│   ├── results.csv
│   └── summary.md
```

### results.csv

Columns (in order):

```
Users, Model, LLM_Provider, GPU,
Avg_Response_Time, Avg_TTFT, Avg_TPOT,
Max_Response_Time, Min_Response_Time,
Error_Rate, CPU_Percent, Memory_Percent,
Total_Requests, Successful_Requests, Failed_Requests,
Test_Duration, Recommendation
```

### summary.md

A human-readable Markdown report containing:

- **Test Configuration**: timestamp, provider, API type, base URL, mode, profile mix, turns range
- **Results per Model**: table with Avg. Time, TTFT, TPOT, Max. Time, Error Rate, CPU, Memory, Requests, Recommendation
- **Overall Summary**: total requests, average TTFT, overall error rate
- **Recommendations**: suggested maximum concurrent users; in multi-turn mode, a note on the ~0.6–0.7× correction factor vs. single-turn benchmarks

### Manual CSV Export

```bash
# Save only CSV, no folder or summary.md
python llm_load_test.py --prompts prompts_english.txt --users 20 --model llama2 --llm-provider "Ollama" --output my_results.csv
```

## Prompt Files

### Main Prompts File (`--prompts`)

One prompt per line, no blank lines. Prompts are selected randomly for each turn.

```txt
Explain the difference between Machine Learning and Deep Learning
Write Python code for a simple to-do list
What are the pros and cons of microservices?
```

#### Included Prompt Files

| File | Description |
|------|-------------|
| `prompts_english.txt` | ~108 business prompts (emails, HR, finance, IT, strategy) |
| `prompts_long_context.txt` | 8 prompts with 600–900 words of inline document content — tests long-context handling |

### System Prompts File (`--system-prompts`)

Optional. One enterprise assistant persona per line, assigned randomly at the start of each conversation session. This ensures different simulated users operate under different contexts, preventing KV-cache prefix sharing.

```txt
You are an IT support assistant. Help employees resolve technical issues...
You are an HR policy assistant. Answer employee questions about benefits...
You are a project management assistant. Support project managers with planning...
```

#### Included System Prompts File

| File | Description |
|------|-------------|
| `system_prompts.txt` | 18 enterprise personas (IT, HR, legal, finance, procurement, security, etc.) |

### Best Practices for Prompts

- Collect realistic prompts from your own use case or chat logs
- Mix short factual questions with longer generation tasks
- Use `prompts_long_context.txt` to simulate document Q&A workloads
- Keep each prompt on a single line

## Test Duration and Timing Behavior

The `--test-duration` defines the **active request phase**, not the total wall-clock time.

### Test Phases

**Phase 1 – Active Phase** (e.g., 5 minutes): Users continuously start new sessions and turns until the duration expires.

**Phase 2 – Drain Phase** (variable): No new requests are started; the tool waits for all in-flight requests to complete.

```
00:00       Test starts, first sessions begin
00:00–05:00 Active phase: new turns started continuously
05:00       No new requests
05:00–06:15 Drain phase: running requests finish
06:15       Test done (total: 6m 15s instead of 5m)
```

### Automatic Termination on Overload

- Error rate is checked every 30 seconds
- If error rate > 30% with at least 10 requests, the step is aborted
- Further steps with higher user counts for that model are skipped

```
[Progress] Requests: 15, Error rate: 35.2%
⚠️ ABORT: Error rate (35.2%) exceeds 30%!
```

## Interpreting Results

### Live Output

```
TESTING MODEL: llama2
[Step 2/8] Testing 10 users with llama2...
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

**TTFT (Time-to-First-Token)** — primary UX metric:

| TTFT | Assessment |
|------|-----------|
| < 2 s | Optimal |
| 2–5 s | Good |
| 5–10 s | Acceptable |
| 10–20 s | Slow |
| > 20 s | Unacceptable |

**TPOT (Time Per Output Token)** — generation throughput:
- Typical range: 0.02–0.10 s/token
- Lower is better; rising TPOT under load indicates GPU saturation

**Error Rate** — stability indicator:

| Error Rate | Assessment |
|------------|-----------|
| 0–2% | Production-ready |
| 2–5% | Unstable, tolerable |
| 5–10% | Overloaded |
| > 10% | Critical |

### Multi-Turn vs. Single-Turn Results

Multi-turn mode measures capacity under realistic chat load, where each turn adds to the model's input context. Expect ~30–40% lower throughput compared to single-turn benchmarks. The `summary.md` includes a note with the applicable correction factor.

To compare directly:
```bash
# Realistic capacity
python llm_load_test.py --mode multi-turn --users 30 --model llama2 --llm-provider "Ollama" --prompts prompts_english.txt

# Peak throughput
python llm_load_test.py --mode single-turn --users 30 --model llama2 --llm-provider "Ollama" --prompts prompts_english.txt
```

## Best Practices

### Start with a Baseline

```bash
# Quick single-turn baseline (5 min, 1 step)
python llm_load_test.py --prompts prompts_english.txt --users 5 --model llama2 --llm-provider "Ollama" \
  --mode single-turn --step-size 5 --test-duration 120
```

### Realistic Production Test

```bash
python llm_load_test.py \
  --prompts prompts_english.txt \
  --system-prompts system_prompts.txt \
  --users 30 \
  --model llama2 \
  --llm-provider "Ollama" \
  --gpu "RTX A2000" \
  --profile-mix 40:40:20 \
  --turns-min 3 \
  --turns-max 7
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
htop                  # CPU and memory
nvidia-smi -l 1       # GPU utilization and VRAM
journalctl -u ollama -f   # Ollama logs
```

### Optimization Based on Results

| Symptom | Action |
|---------|--------|
| High TPOT, low TTFT | GPU throughput bottleneck — consider smaller model or tensor parallelism |
| High TTFT, low TPOT | Scheduling or batch size issue |
| Rising error rate | Reduce concurrency, tune `OLLAMA_NUM_PARALLEL`, or add a second backend |
| TPOT stays flat, TTFT rises | Context window growth — expected in multi-turn; reduce `--turns-max` |

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

This software is provided "as is" without warranty of any kind. The author(s) assume **NO LIABILITY** for any damages, losses, or issues arising from its use, including but not limited to: data loss, hardware damage, system failures, or any other consequences.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

- **Non-commercial use** (personal, education, research, open source): free
- **Commercial use**: open an issue or contact via GitHub

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
