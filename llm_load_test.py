import requests
import multiprocessing
import time
import random
import argparse
import signal
import json
import psutil
import threading
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from api_adapters import create_adapter

# Load environment variables from .env file
load_dotenv()

def resolve_arg(cli_value, env_key, default=None, cast=None):
    """Return CLI value if given, else .env value, else default. cast applied to env string."""
    if cli_value is not None:
        return cli_value
    env_val = os.getenv(env_key)
    if env_val is not None:
        return cast(env_val) if cast else env_val
    return default

@dataclass
class TestResult:
    """Data class for test results"""
    users: int
    model: str
    llm_provider: str
    gpu: str
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    avg_ttft: float
    avg_tpot: float
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    cpu_usage: float
    memory_usage: float
    test_duration: float
    recommendation: str

class ResultCollector:
    """Collects and manages test results"""
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()

    def add_result(self, result: TestResult):
        with self.lock:
            self.results.append(result)

    def get_results(self) -> List[TestResult]:
        with self.lock:
            return self.results.copy()

class SystemMonitor:
    """Monitors system resources during the test"""
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
            except:
                pass

    def get_average_cpu(self):
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0

    def get_average_memory(self):
        return sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0

# Global shared variables for result collection across processes
response_times = multiprocessing.Manager().list()
ttft_times = multiprocessing.Manager().list()
tpot_times = multiprocessing.Manager().list()
error_count = multiprocessing.Manager().Value('i', 0)
success_count = multiprocessing.Manager().Value('i', 0)

def reset_counters():
    """Resets global counters before each test step"""
    global response_times, ttft_times, tpot_times, error_count, success_count
    response_times[:] = []
    ttft_times[:] = []
    tpot_times[:] = []
    error_count.value = 0
    success_count.value = 0

def terminate_processes(processes):
    """Terminates all child processes"""
    for p in processes:
        if p.is_alive():
            p.terminate()
    print("All processes terminated.")

def load_prompts(file_path):
    """Reads prompts from a text file, one prompt per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def assign_profiles(user_count, profile_mix_str, turns_min, turns_max):
    """
    Assigns a user profile to each simulated user index.

    Profile mix is applied proportionally; remainder goes to Normal.
    Returns a list of profile dicts, one per user (length == user_count).
    """
    parts = profile_mix_str.split(':')
    power_pct, normal_pct, occasional_pct = int(parts[0]), int(parts[1]), int(parts[2])

    power_count = int(user_count * power_pct / 100)
    occasional_count = int(user_count * occasional_pct / 100)
    normal_count = user_count - power_count - occasional_count  # remainder to Normal

    profiles = []
    for _ in range(power_count):
        profiles.append({"name": "power", "turns": turns_max})
    for _ in range(normal_count):
        profiles.append({"name": "normal", "turns": (turns_min, turns_max)})
    for _ in range(occasional_count):
        profiles.append({"name": "occasional", "turns": turns_min})

    return profiles

def llm_chat_continuous(model, prompts, user_id, pause_min, pause_max, api_type, base_url, api_key, test_duration):
    """Simulates a single-turn user for a fixed test duration (--mode single-turn)"""
    global response_times, ttft_times, error_count, success_count

    adapter = create_adapter(api_type, base_url, api_key)

    end_time = time.time() + test_duration

    while time.time() < end_time:
        prompt = random.choice(prompts)

        success, elapsed_time, first_token_time, error_msg = adapter.make_request(model, prompt, timeout=120)

        if success:
            response_times.append(elapsed_time)
            ttft_times.append(first_token_time)
            success_count.value += 1
            print(f"[User {user_id}] ✓ {elapsed_time:.2f}s (TTFT: {first_token_time:.2f}s) - {prompt[:30]}...")

            if time.time() < end_time:
                pause_time = random.uniform(pause_min, pause_max)
                time.sleep(min(pause_time, end_time - time.time()))
        else:
            error_count.value += 1
            print(f"[User {user_id}] ✗ {error_msg} - retrying immediately...")
            continue

def llm_chat_multiturn(model, prompts, system_prompts, user_id, profile,
                       turns_min, turns_max, api_type, base_url, api_key, test_duration):
    """Simulates a multi-turn chat user for a fixed test duration (--mode multi-turn)"""
    global response_times, ttft_times, tpot_times, error_count, success_count

    adapter = create_adapter(api_type, base_url, api_key)
    end_time = time.time() + test_duration

    while time.time() < end_time:
        # Start a new conversation session
        messages = []
        if system_prompts:
            messages.append({"role": "system", "content": random.choice(system_prompts)})

        # Determine turn count for this session
        if isinstance(profile["turns"], int):
            turns = profile["turns"]
        else:
            turns = random.randint(profile["turns"][0], profile["turns"][1])

        for turn in range(turns):
            if time.time() >= end_time:
                break

            prompt = random.choice(prompts)
            messages.append({"role": "user", "content": prompt})

            success, elapsed, ttft, tpot, error = adapter.make_chat_request(model, messages, timeout=120)

            if success:
                response_times.append(elapsed)
                ttft_times.append(ttft)
                tpot_times.append(tpot)
                success_count.value += 1
                messages.append({"role": "assistant", "content": "[response]"})
                print(f"[User {user_id}|turn {turn+1}] ✓ {elapsed:.2f}s (TTFT: {ttft:.2f}s, TPOT: {tpot:.3f}s) - {prompt[:30]}...")
            else:
                error_count.value += 1
                print(f"[User {user_id}|turn {turn+1}] ✗ {error} - restarting session")
                break  # abandon this session on error, start fresh next loop

def get_recommendation(avg_time, max_time, error_rate, cpu_usage, avg_ttft):
    """Generates a recommendation based on TTFT and error rate"""
    if error_rate > 10:
        return "❌ Critical"
    elif error_rate > 5:
        return "❌ Overloaded"
    elif error_rate > 2:
        return "⚠️ Unstable"
    elif avg_ttft > 30:
        return "❌ Unacceptable"
    elif avg_ttft > 20:
        return "⚠️ Very slow"
    elif avg_ttft > 10:
        return "⚠️ Slow"
    elif avg_ttft > 5:
        return "✅ Acceptable"
    elif avg_ttft > 2:
        return "✅ Good"
    else:
        return "✅ Optimal"

def check_api_connection(adapter):
    """Checks if the API is reachable"""
    return adapter.check_connection()

def run_load_test(model, prompts, user_count, pause_min, pause_max, test_duration,
                  api_type, base_url, api_key, gpu_name, llm_provider,
                  mode='multi-turn', system_prompts_list=None,
                  turns_min=3, turns_max=7, profiles=None,
                  workload_mix_tuple=None, lc_prompts=None, lc_turns_max=2):
    """Runs a load test with a given number of simulated users"""
    reset_counters()

    print(f"\n{'='*60}")
    print(f"Starting test with {user_count} users...")
    print(f"Test duration: {test_duration/60:.1f} minutes")
    if workload_mix_tuple:
        s, m, lc = workload_mix_tuple
        print(f"Workload mix: {s}% single-turn / {m}% multi-turn / {lc}% long-context")
    else:
        print(f"Mode: {mode}")
    print(f"{'='*60}")

    monitor = SystemMonitor()
    monitor.start_monitoring()

    processes = []
    start_time = time.time()

    # Determine slice sizes
    if workload_mix_tuple:
        single_pct, multi_pct, lc_pct = workload_mix_tuple
        n_single = int(user_count * single_pct / 100)
        n_lc     = int(user_count * lc_pct   / 100)
        n_multi  = user_count - n_single - n_lc
    else:
        n_single = user_count if mode == 'single-turn' else 0
        n_multi  = user_count if mode == 'multi-turn'  else 0
        n_lc     = 0

    # Profiles are always assigned so single-turn users also get profile-based pauses
    if profiles is None:
        profiles = assign_profiles(user_count, '40:40:20', turns_min, turns_max)

    try:
        for user_id in range(user_count):
            profile = profiles[user_id]
            if user_id < n_single:
                p = multiprocessing.Process(
                    target=llm_chat_continuous,
                    args=(model, prompts, user_id,
                          pause_min, pause_max,
                          api_type, base_url, api_key, test_duration)
                )
            elif user_id < n_single + n_multi:
                p = multiprocessing.Process(
                    target=llm_chat_multiturn,
                    args=(model, prompts, system_prompts_list, user_id, profile,
                          turns_min, turns_max, api_type, base_url, api_key, test_duration)
                )
            else:
                # Long-context slice: use lc_prompts pool, capped turn count
                lc_profile = {**profile, 'turns': (1, lc_turns_max)}
                p = multiprocessing.Process(
                    target=llm_chat_multiturn,
                    args=(model, lc_prompts or prompts, system_prompts_list, user_id,
                          lc_profile, 1, lc_turns_max, api_type, base_url, api_key,
                          test_duration)
                )
            p.start()
            processes.append(p)
            time.sleep(0.1)

        print(f"All {user_count} users started. Waiting {test_duration/60:.1f} minutes...")

        check_interval = 30
        next_check = time.time() + check_interval

        while any(p.is_alive() for p in processes):
            time.sleep(1)

            if time.time() >= next_check:
                total_requests = success_count.value + error_count.value
                if total_requests >= 10:
                    timeout_rate = (error_count.value / total_requests) * 100
                    print(f"[Progress] Requests: {total_requests}, Error rate: {timeout_rate:.1f}%")

                    if timeout_rate > 30:
                        print(f"\n⚠️ ABORT: Error rate ({timeout_rate:.1f}%) exceeds 30%!")
                        print("System is overloaded - aborting test.")
                        break

                next_check = time.time() + check_interval

        for p in processes:
            if p.is_alive():
                p.terminate()

        time.sleep(2)

    except KeyboardInterrupt:
        print("\nTest aborted...")
        terminate_processes(processes)
        return None
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()

    monitor.stop_monitoring()
    actual_duration = time.time() - start_time

    times = list(response_times)
    ttft_list = list(ttft_times)
    tpot_list = list(tpot_times)
    total_requests = success_count.value + error_count.value

    if not times:
        print(f"No successful requests in {user_count}-user test!")
        return None

    recommendation = get_recommendation(
        sum(times) / len(times),
        max(times),
        (error_count.value / total_requests * 100) if total_requests > 0 else 0,
        monitor.get_average_cpu(),
        sum(ttft_list) / len(ttft_list) if ttft_list else 0
    )

    result = TestResult(
        users=user_count,
        model=model,
        llm_provider=llm_provider,
        gpu=gpu_name,
        avg_response_time=sum(times) / len(times),
        max_response_time=max(times),
        min_response_time=min(times),
        avg_ttft=sum(ttft_list) / len(ttft_list) if ttft_list else 0.0,
        avg_tpot=sum(tpot_list) / len(tpot_list) if tpot_list else 0.0,
        error_rate=(error_count.value / total_requests * 100) if total_requests > 0 else 0,
        total_requests=total_requests,
        successful_requests=success_count.value,
        failed_requests=error_count.value,
        cpu_usage=monitor.get_average_cpu(),
        memory_usage=monitor.get_average_memory(),
        test_duration=actual_duration,
        recommendation=recommendation
    )

    print(f"\nTest completed:")
    print(f"  Successful requests: {result.successful_requests}")
    print(f"  Failed requests: {result.failed_requests}")
    print(f"  Average response time: {result.avg_response_time:.2f}s")
    print(f"  Average TTFT: {result.avg_ttft:.2f}s")
    print(f"  Average TPOT: {result.avg_tpot:.3f}s")
    print(f"  Max response time: {result.max_response_time:.2f}s")
    print(f"  Error rate: {result.error_rate:.1f}%")
    print(f"  CPU usage: {result.cpu_usage:.1f}%")

    return result

def print_results_table(results: List[TestResult]):
    """Prints the results table to stdout"""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*165}")
    print("LOAD TEST RESULTS")
    print(f"{'='*165}")

    print(f"{'Users':<8} {'Model':<15} {'LLM Provider':<15} {'GPU':<12} {'Avg. Time':<10} {'TTFT':<8} {'TPOT':<8} {'Max. Time':<10} {'Min. Time':<10} {'Error Rate':<11} {'CPU %':<8} {'Memory %':<10} {'Requests':<10} {'Recommendation':<15}")
    print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*11} {'-'*8} {'-'*10} {'-'*10} {'-'*15}")

    for result in results:
        print(f"{result.users:<8} {result.model:<15} {result.llm_provider:<15} {result.gpu:<12} {result.avg_response_time:<10.2f} {result.avg_ttft:<8.2f} {result.avg_tpot:<8.3f} {result.max_response_time:<10.2f} {result.min_response_time:<10.2f} {result.error_rate:<11.1f} {result.cpu_usage:<8.1f} {result.memory_usage:<10.1f} {result.total_requests:<10} {result.recommendation:<15}")

    print(f"{'-'*165}")

def save_results_to_file(results: List[TestResult], filename: str):
    """Saves results to a CSV file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Users,Model,LLM_Provider,GPU,Avg_Response_Time,Avg_TTFT,Avg_TPOT,Max_Response_Time,Min_Response_Time,Error_Rate,CPU_Percent,Memory_Percent,Total_Requests,Successful_Requests,Failed_Requests,Test_Duration,Recommendation\n")

            for result in results:
                f.write(f"{result.users},{result.model},{result.llm_provider},{result.gpu},"
                        f"{result.avg_response_time:.3f},{result.avg_ttft:.3f},{result.avg_tpot:.4f},"
                        f"{result.max_response_time:.3f},{result.min_response_time:.3f},"
                        f"{result.error_rate:.2f},{result.cpu_usage:.2f},{result.memory_usage:.2f},"
                        f"{result.total_requests},{result.successful_requests},{result.failed_requests},"
                        f"{result.test_duration:.1f},{result.recommendation}\n")

        print(f"\nCSV saved: {filename}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def save_results_to_markdown(results: List[TestResult], filename: str, test_config: dict):
    """Saves results as a Markdown summary"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# LLM Load Test - Summary\n\n")

            f.write("## Test Configuration\n\n")
            f.write(f"- **Date/Time**: {test_config['timestamp']}\n")
            f.write(f"- **LLM Provider**: {test_config['llm_provider']}\n")
            f.write(f"- **API Type**: {test_config['api_type']}\n")
            f.write(f"- **Base URL**: {test_config['base_url']}\n")
            f.write(f"- **Models**: {test_config['models']}\n")
            f.write(f"- **GPU**: {test_config['gpu']}\n")
            f.write(f"- **Mode**: {test_config['mode']}\n")
            f.write(f"- **Test Duration per Step**: {test_config['test_duration']/60:.1f} minutes\n")
            f.write(f"- **Pause Times**: {test_config['pause_min']}–{test_config['pause_max']} seconds\n")
            f.write(f"- **User Steps**: {test_config['user_steps']}\n")
            if test_config['mode'] == 'multi-turn':
                f.write(f"- **Profile Mix (Power:Normal:Occasional)**: {test_config['profile_mix']}\n")
                f.write(f"- **Turns per Session**: {test_config['turns_min']}–{test_config['turns_max']}\n")
            f.write("\n")

            models = list(set([r.model for r in results]))

            for model in models:
                f.write(f"## Results: {model}\n\n")
                model_results = [r for r in results if r.model == model]

                f.write("| Users | Avg. Time (s) | TTFT (s) | TPOT (s) | Max. Time (s) | Error Rate (%) | CPU (%) | Memory (%) | Requests | Recommendation |\n")
                f.write("|-------|---------------|----------|----------|---------------|----------------|---------|------------|----------|----------------|\n")

                for result in model_results:
                    f.write(f"| {result.users} | {result.avg_response_time:.2f} | {result.avg_ttft:.2f} | {result.avg_tpot:.3f} | {result.max_response_time:.2f} | {result.error_rate:.1f} | {result.cpu_usage:.1f} | {result.memory_usage:.1f} | {result.total_requests} | {result.recommendation} |\n")

                f.write("\n")

                best_result = max(model_results, key=lambda r: r.users if r.error_rate < 10 else 0)
                f.write("### Summary\n\n")
                f.write(f"- **Best performance**: {best_result.users} concurrent users\n")
                f.write(f"- **Average TTFT**: {best_result.avg_ttft:.2f}s\n")
                f.write(f"- **Average TPOT**: {best_result.avg_tpot:.3f}s\n")
                f.write(f"- **Average response time**: {best_result.avg_response_time:.2f}s\n")
                f.write(f"- **Error rate**: {best_result.error_rate:.1f}%\n\n")

            f.write("## Overall Summary\n\n")
            total_requests = sum(r.total_requests for r in results)
            total_successful = sum(r.successful_requests for r in results)
            total_failed = sum(r.failed_requests for r in results)
            avg_ttft_all = sum(r.avg_ttft for r in results) / len(results) if results else 0

            f.write(f"- **Total requests**: {total_requests}\n")
            f.write(f"- **Successful requests**: {total_successful}\n")
            f.write(f"- **Failed requests**: {total_failed}\n")
            f.write(f"- **Average TTFT (all tests)**: {avg_ttft_all:.2f}s\n")
            f.write(f"- **Overall error rate**: {(total_failed/total_requests*100) if total_requests > 0 else 0:.1f}%\n\n")

            f.write("## Recommendations\n\n")

            good_results = [r for r in results if r.error_rate < 10]
            if good_results:
                best = max(good_results, key=lambda r: r.users)
                f.write(f"- Recommended maximum concurrent users: **{best.users}**\n")
                f.write(f"- At this load: TTFT {best.avg_ttft:.2f}s, error rate {best.error_rate:.1f}%\n")
            else:
                f.write("- ⚠️ All tests showed high error rates (>10%). System is overloaded.\n")

            if test_config.get('workload_mix'):
                f.write("\n> **Note:** Mixed workload (single-turn / multi-turn / long-context). Multi-turn and long-context results reflect realistic chat load; apply a 0.6–0.7x factor vs. single-turn benchmarks.\n")
            elif test_config.get('mode') == 'multi-turn':
                f.write("\n> **Note:** Results reflect realistic multi-turn chat load. Apply a 0.6–0.7 correction factor compared to single-turn benchmarks.\n")

            f.write("\n")

        print(f"Markdown saved: {filename}")
    except Exception as e:
        print(f"Error saving Markdown file: {e}")

def create_results_directory():
    """Creates a timestamped results directory"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir, timestamp

def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step load testing for LLM APIs (Ollama, vLLM, LM Studio, llama.cpp, etc.)"
    )
    parser.add_argument("--prompts", type=str, required=False, default=None,
                        help="Path to prompts file (env: PROMPTS_FILE)")
    parser.add_argument("--users", type=int, required=False, default=None,
                        help="Maximum number of users (env: USERS)")
    parser.add_argument("--model", type=str, required=False, default=None,
                        help="Model(s), comma-separated for multiple models (env: MODEL)")
    parser.add_argument("--llm-provider", type=str, required=False, default=None,
                        help="LLM provider name, e.g. 'Ollama', 'vLLM' (env: LLM_PROVIDER)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU label for documentation (default: Unknown, env: GPU)")
    parser.add_argument("--pause-min", type=float, default=None,
                        help="Minimum pause between messages in seconds (default: 3.0, env: PAUSE_MIN)")
    parser.add_argument("--pause-max", type=float, default=None,
                        help="Maximum pause between messages in seconds (default: 30.0, env: PAUSE_MAX)")
    parser.add_argument("--step-size", type=int, default=None,
                        help="User count increment per step (default: 5, env: STEP_SIZE)")
    parser.add_argument("--test-duration", type=int, default=None,
                        help="Test duration per step in seconds (default: 300, env: TEST_DURATION)")
    parser.add_argument("--host", type=str, default=None,
                        help="API host and port (default: from .env or 127.0.0.1:11434)")
    parser.add_argument("--api-type", type=str, default=None,
                        help="API type (ollama, vllm, lmstudio, llamacpp, openai)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for authentication (optional, from .env if not provided)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom CSV output filename (optional)")
    parser.add_argument("--mode", type=str, choices=["multi-turn", "single-turn"], default=None,
                        help="Test mode: multi-turn (default) or single-turn (env: MODE)")
    parser.add_argument("--system-prompts", type=str, default=None,
                        help="Path to system prompts file (one per line)")
    parser.add_argument("--turns-min", type=int, default=None,
                        help="Minimum turns per multi-turn session (default: 3, env: TURNS_MIN)")
    parser.add_argument("--turns-max", type=int, default=None,
                        help="Maximum turns per multi-turn session (default: 7, env: TURNS_MAX)")
    parser.add_argument("--profile-mix", type=str, default=None,
                        help="Power:Normal:Occasional user mix, must sum to 100 (default: 40:40:20, env: PROFILE_MIX)")
    parser.add_argument("--workload-mix", type=str, default=None,
                        help="single-turn:multi-turn:long-context percentages summing to 100 (env: WORKLOAD_MIX). Overrides --mode when set.")
    parser.add_argument("--long-context-prompts", type=str, default=None,
                        help="Path to long-context prompts file (env: LONG_CONTEXT_PROMPTS_FILE)")
    parser.add_argument("--lc-turns-max", type=int, default=None,
                        help="Max turns for long-context slice (default: 2, env: LC_TURNS_MAX)")

    args = parser.parse_args()

    # Resolve all args: CLI > .env > default
    prompts_file  = resolve_arg(args.prompts,       'PROMPTS_FILE')
    users         = resolve_arg(args.users,          'USERS',          cast=int)
    model_str     = resolve_arg(args.model,          'MODEL')
    llm_provider  = resolve_arg(args.llm_provider,  'LLM_PROVIDER')
    gpu           = resolve_arg(args.gpu,            'GPU',            default='Unknown')
    mode          = resolve_arg(args.mode,           'MODE',           default='multi-turn')
    system_prompts_path = resolve_arg(args.system_prompts, 'SYSTEM_PROMPTS_FILE')
    turns_min     = resolve_arg(args.turns_min,     'TURNS_MIN',      default=3,    cast=int)
    turns_max     = resolve_arg(args.turns_max,     'TURNS_MAX',      default=7,    cast=int)
    profile_mix   = resolve_arg(args.profile_mix,   'PROFILE_MIX',    default='40:40:20')
    workload_mix  = resolve_arg(args.workload_mix,  'WORKLOAD_MIX')
    lc_prompts_file = resolve_arg(args.long_context_prompts, 'LONG_CONTEXT_PROMPTS_FILE')
    lc_turns_max  = resolve_arg(args.lc_turns_max,  'LC_TURNS_MAX',   default=2,    cast=int)
    pause_min     = resolve_arg(args.pause_min,     'PAUSE_MIN',      default=3.0,  cast=float)
    pause_max     = resolve_arg(args.pause_max,     'PAUSE_MAX',      default=30.0, cast=float)
    step_size     = resolve_arg(args.step_size,     'STEP_SIZE',      default=5,    cast=int)
    test_duration = resolve_arg(args.test_duration, 'TEST_DURATION',  default=300,  cast=int)
    output        = resolve_arg(args.output,        'OUTPUT')

    # Resolve API configuration
    api_type = args.api_type or os.getenv('API_TYPE', 'ollama')
    api_key  = args.api_key  or os.getenv('API_KEY')
    if args.host:
        base_url = args.host if args.host.startswith(('http://', 'https://')) else f"http://{args.host}"
    else:
        base_url = os.getenv('API_BASE_URL', 'http://127.0.0.1:11434')

    # Validate required fields
    missing = []
    if not prompts_file:  missing.append('--prompts / PROMPTS_FILE')
    if users is None:     missing.append('--users / USERS')
    if not model_str:     missing.append('--model / MODEL')
    if not llm_provider:  missing.append('--llm-provider / LLM_PROVIDER')
    if missing:
        print("Error: the following required arguments are missing (set via CLI or .env):")
        for m in missing:
            print(f"  {m}")
        return

    # Validate --profile-mix
    try:
        profile_parts = [int(x) for x in profile_mix.split(':')]
        if len(profile_parts) != 3 or sum(profile_parts) != 100:
            print(f"Error: profile-mix must be three integers summing to 100, got '{profile_mix}'")
            return
    except ValueError:
        print(f"Error: profile-mix must be three colon-separated integers, got '{profile_mix}'")
        return

    # Validate turns range
    if turns_min > turns_max:
        print(f"Error: turns-min ({turns_min}) must be <= turns-max ({turns_max})")
        return

    # Parse --workload-mix
    workload_mix_tuple = None
    if workload_mix:
        try:
            wm_parts = [int(x) for x in workload_mix.split(':')]
            if len(wm_parts) != 3 or sum(wm_parts) != 100:
                print(f"Error: workload-mix must be three integers summing to 100, got '{workload_mix}'")
                return
            workload_mix_tuple = tuple(wm_parts)
        except ValueError:
            print(f"Error: workload-mix must be three colon-separated integers, got '{workload_mix}'")
            return

    models = [m.strip() for m in model_str.split(',') if m.strip()]
    if not models:
        print("Error: No valid models specified!")
        return

    if pause_min > pause_max:
        print("Error: pause-min must not be greater than pause-max!")
        return

    if users <= 0 or step_size <= 0:
        print("Error: users and step-size must be greater than 0!")
        return

    try:
        adapter = create_adapter(api_type, base_url, api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Checking connection to {api_type.upper()} API ({base_url})...")
    if not check_api_connection(adapter):
        print(f"Error: Cannot connect to API at {base_url}!")
        print(f"Make sure the {api_type.upper()} server is running.")
        return

    print(f"✓ Connected to {api_type.upper()} API")

    try:
        prompts = load_prompts(prompts_file)
        print(f"✓ Loaded {len(prompts)} prompts from {prompts_file}")
    except FileNotFoundError:
        print(f"Error: Prompts file {prompts_file} not found!")
        return

    if len(prompts) == 0:
        print("Error: No prompts found in file!")
        return

    # Load system prompts if provided
    system_prompts_list = None
    if system_prompts_path:
        try:
            system_prompts_list = load_prompts(system_prompts_path)
            print(f"✓ Loaded {len(system_prompts_list)} system prompts from {system_prompts_path}")
        except FileNotFoundError:
            print(f"Error: System prompts file {system_prompts_path} not found!")
            return

    # Load long-context prompts if needed
    lc_prompts = None
    if workload_mix_tuple and workload_mix_tuple[2] > 0:
        if lc_prompts_file:
            try:
                lc_prompts = load_prompts(lc_prompts_file)
                print(f"✓ Loaded {len(lc_prompts)} long-context prompts from {lc_prompts_file}")
            except FileNotFoundError:
                print(f"Error: Long-context prompts file {lc_prompts_file} not found!")
                return
        else:
            print("Warning: workload-mix has long-context% > 0 but no --long-context-prompts file set. Using main prompts for long-context slice.")

    print(f"\nSTARTING STEP-BY-STEP LOAD TEST")
    print(f"API Type: {api_type.upper()}")
    print(f"Base URL: {base_url}")
    print(f"Models: {', '.join(models)}")
    print(f"GPU: {gpu}")
    if workload_mix_tuple:
        s, m, lc = workload_mix_tuple
        print(f"Workload mix: {s}% single-turn / {m}% multi-turn / {lc}% long-context")
    else:
        print(f"Mode: {mode}")
    print(f"Max users: {users}")
    print(f"Step size: {step_size}")
    print(f"Test duration per step: {test_duration/60:.1f} minutes")
    print(f"Pause times: {pause_min}–{pause_max} seconds")
    if workload_mix_tuple or mode == 'multi-turn':
        print(f"Profile mix (Power:Normal:Occasional): {profile_mix}")
        print(f"Turns per session: {turns_min}–{turns_max}")

    results = []
    user_steps = list(range(step_size, users + 1, step_size))

    if users not in user_steps:
        user_steps.append(users)

    total_steps = len(user_steps) * len(models)
    estimated_total_time = total_steps * test_duration / 60

    print(f"Planned steps: {user_steps}")
    print(f"Estimated total duration: {estimated_total_time:.1f} minutes")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    try:
        step_counter = 0

        for model in models:
            print(f"\n{'='*80}")
            print(f"TESTING MODEL: {model}")
            print(f"{'='*80}")

            model_overloaded = False

            for user_count in user_steps:
                if model_overloaded:
                    step_counter += 1
                    print(f"\n[Step {step_counter}/{total_steps}] Skipping {user_count} users for {model} (already overloaded at lower count)")
                    continue

                step_counter += 1
                print(f"\n[Step {step_counter}/{total_steps}] Testing {user_count} users with {model}...")

                profiles = assign_profiles(user_count, profile_mix, turns_min, turns_max)

                result = run_load_test(
                    model, prompts, user_count,
                    pause_min, pause_max,
                    test_duration, api_type, base_url, api_key, gpu, llm_provider,
                    mode=mode,
                    system_prompts_list=system_prompts_list,
                    turns_min=turns_min,
                    turns_max=turns_max,
                    profiles=profiles,
                    workload_mix_tuple=workload_mix_tuple,
                    lc_prompts=lc_prompts,
                    lc_turns_max=lc_turns_max,
                )

                if result:
                    results.append(result)

                    if result.error_rate > 30:
                        model_overloaded = True
                        print(f"\n⚠️ Model {model} is overloaded at {user_count} users.")
                        print(f"Skipping further tests with more users for this model.\n")

                if step_counter < total_steps:
                    print("Pause between tests (10 seconds)...")
                    time.sleep(10)


        print_results_table(results)

        if output:
            save_results_to_file(results, output)
        else:
            result_dir, timestamp_str = create_results_directory()

            csv_filename = os.path.join(result_dir, "results.csv")
            save_results_to_file(results, csv_filename)

            md_filename = os.path.join(result_dir, "summary.md")
            test_config = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'llm_provider': llm_provider,
                'api_type': api_type.upper(),
                'base_url': base_url,
                'models': ', '.join(models),
                'gpu': gpu,
                'mode': mode,
                'workload_mix': workload_mix,
                'test_duration': test_duration,
                'pause_min': pause_min,
                'pause_max': pause_max,
                'user_steps': user_steps,
                'profile_mix': profile_mix,
                'turns_min': turns_min,
                'turns_max': turns_max,
            }
            save_results_to_markdown(results, md_filename, test_config)

            print(f"\n📁 Results saved to: {result_dir}")

        print(f"\nLoad test completed at {datetime.now().strftime('%H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\nLoad test aborted!")
        if results:
            print("Results so far:")
            print_results_table(results)

if __name__ == "__main__":
    main()
