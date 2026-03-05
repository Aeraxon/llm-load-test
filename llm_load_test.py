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
        profiles.append({"name": "power", "pause_min": 2, "pause_max": 5, "turns": turns_max})
    for _ in range(normal_count):
        profiles.append({"name": "normal", "pause_min": 15, "pause_max": 45, "turns": (turns_min, turns_max)})
    for _ in range(occasional_count):
        profiles.append({"name": "occasional", "pause_min": 60, "pause_max": 120, "turns": turns_min})

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

        # Pause between sessions according to profile
        remaining = end_time - time.time()
        if remaining > 0:
            pause = random.uniform(profile["pause_min"], profile["pause_max"])
            time.sleep(min(pause, remaining))

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
                  turns_min=3, turns_max=7, profiles=None):
    """Runs a load test with a given number of simulated users"""
    reset_counters()

    print(f"\n{'='*60}")
    print(f"Starting test with {user_count} users...")
    print(f"Test duration: {test_duration/60:.1f} minutes")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    monitor = SystemMonitor()
    monitor.start_monitoring()

    processes = []
    start_time = time.time()

    try:
        for user_id in range(user_count):
            if mode == 'multi-turn':
                profile = profiles[user_id] if profiles else {
                    "name": "normal", "pause_min": pause_min, "pause_max": pause_max,
                    "turns": (turns_min, turns_max)
                }
                p = multiprocessing.Process(
                    target=llm_chat_multiturn,
                    args=(model, prompts, system_prompts_list, user_id, profile,
                          turns_min, turns_max, api_type, base_url, api_key, test_duration)
                )
            else:
                p = multiprocessing.Process(
                    target=llm_chat_continuous,
                    args=(model, prompts, user_id, pause_min, pause_max,
                          api_type, base_url, api_key, test_duration)
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

            if test_config.get('mode') == 'multi-turn':
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
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompts file")
    parser.add_argument("--users", type=int, required=True,
                        help="Maximum number of users (reached incrementally)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model(s), comma-separated for multiple models")
    parser.add_argument("--llm-provider", type=str, required=True,
                        help="LLM provider name (e.g. 'Ollama', 'vLLM', 'LM Studio')")
    parser.add_argument("--gpu", type=str, default="Unknown",
                        help="GPU label for documentation (default: Unknown)")
    parser.add_argument("--pause-min", type=float, default=3.0,
                        help="Minimum pause between messages in seconds (default: 3.0)")
    parser.add_argument("--pause-max", type=float, default=30.0,
                        help="Maximum pause between messages in seconds (default: 30.0)")
    parser.add_argument("--step-size", type=int, default=5,
                        help="User count increment per step (default: 5)")
    parser.add_argument("--test-duration", type=int, default=300,
                        help="Test duration per step in seconds (default: 300 = 5 minutes)")
    parser.add_argument("--host", type=str, default=None,
                        help="API host and port (default: from .env or 127.0.0.1:11434)")
    parser.add_argument("--api-type", type=str, default=None,
                        help="API type (ollama, vllm, lmstudio, llamacpp, openai)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for authentication (optional, from .env if not provided)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom CSV output filename (optional)")
    parser.add_argument("--mode", type=str, choices=["multi-turn", "single-turn"], default="multi-turn",
                        help="Test mode: multi-turn (default) or single-turn")
    parser.add_argument("--system-prompts", type=str, default=None,
                        help="Path to system prompts file (one per line)")
    parser.add_argument("--turns-min", type=int, default=3,
                        help="Minimum turns per multi-turn session (default: 3)")
    parser.add_argument("--turns-max", type=int, default=7,
                        help="Maximum turns per multi-turn session (default: 7)")
    parser.add_argument("--profile-mix", type=str, default="40:40:20",
                        help="Power:Normal:Occasional user mix, must sum to 100 (default: 40:40:20)")

    args = parser.parse_args()

    # Validate --profile-mix
    try:
        profile_parts = [int(x) for x in args.profile_mix.split(':')]
        if len(profile_parts) != 3 or sum(profile_parts) != 100:
            print(f"Error: --profile-mix must be three integers summing to 100 (e.g. '40:40:20'), got '{args.profile_mix}'")
            return
    except ValueError:
        print(f"Error: --profile-mix must be three colon-separated integers, got '{args.profile_mix}'")
        return

    # Validate turns range
    if args.turns_min > args.turns_max:
        print(f"Error: --turns-min ({args.turns_min}) must be <= --turns-max ({args.turns_max})")
        return

    # Resolve API configuration
    api_type = args.api_type or os.getenv('API_TYPE', 'ollama')
    api_key = args.api_key or os.getenv('API_KEY')

    if args.host:
        base_url = args.host if args.host.startswith(('http://', 'https://')) else f"http://{args.host}"
    else:
        env_url = os.getenv('API_BASE_URL')
        base_url = env_url if env_url else "http://127.0.0.1:11434"

    models = [model.strip() for model in args.model.split(',') if model.strip()]

    if not models:
        print("Error: No valid models specified!")
        return

    if args.pause_min > args.pause_max:
        print("Error: --pause-min must not be greater than --pause-max!")
        return

    if args.users <= 0 or args.step_size <= 0:
        print("Error: --users and --step-size must be greater than 0!")
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
        prompts = load_prompts(args.prompts)
        print(f"✓ Loaded {len(prompts)} prompts from {args.prompts}")
    except FileNotFoundError:
        print(f"Error: Prompts file {args.prompts} not found!")
        return

    if len(prompts) == 0:
        print("Error: No prompts found in file!")
        return

    # Load system prompts if provided
    system_prompts_list = None
    if args.system_prompts:
        try:
            system_prompts_list = load_prompts(args.system_prompts)
            print(f"✓ Loaded {len(system_prompts_list)} system prompts from {args.system_prompts}")
        except FileNotFoundError:
            print(f"Error: System prompts file {args.system_prompts} not found!")
            return

    print(f"\nSTARTING STEP-BY-STEP LOAD TEST")
    print(f"API Type: {api_type.upper()}")
    print(f"Base URL: {base_url}")
    print(f"Models: {', '.join(models)}")
    print(f"GPU: {args.gpu}")
    print(f"Mode: {args.mode}")
    print(f"Max users: {args.users}")
    print(f"Step size: {args.step_size}")
    print(f"Test duration per step: {args.test_duration/60:.1f} minutes")
    print(f"Pause times: {args.pause_min}–{args.pause_max} seconds")
    if args.mode == 'multi-turn':
        print(f"Profile mix (Power:Normal:Occasional): {args.profile_mix}")
        print(f"Turns per session: {args.turns_min}–{args.turns_max}")

    results = []
    user_steps = list(range(args.step_size, args.users + 1, args.step_size))

    if args.users not in user_steps:
        user_steps.append(args.users)

    total_steps = len(user_steps) * len(models)
    estimated_total_time = total_steps * args.test_duration / 60

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

                profiles = None
                if args.mode == 'multi-turn':
                    profiles = assign_profiles(user_count, args.profile_mix, args.turns_min, args.turns_max)

                result = run_load_test(
                    model, prompts, user_count,
                    args.pause_min, args.pause_max,
                    args.test_duration, api_type, base_url, api_key, args.gpu, args.llm_provider,
                    mode=args.mode,
                    system_prompts_list=system_prompts_list,
                    turns_min=args.turns_min,
                    turns_max=args.turns_max,
                    profiles=profiles
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

        if args.output:
            save_results_to_file(results, args.output)
        else:
            result_dir, timestamp_str = create_results_directory()

            csv_filename = os.path.join(result_dir, "results.csv")
            save_results_to_file(results, csv_filename)

            md_filename = os.path.join(result_dir, "summary.md")
            test_config = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'llm_provider': args.llm_provider,
                'api_type': api_type.upper(),
                'base_url': base_url,
                'models': ', '.join(models),
                'gpu': args.gpu,
                'mode': args.mode,
                'test_duration': args.test_duration,
                'pause_min': args.pause_min,
                'pause_max': args.pause_max,
                'user_steps': user_steps,
                'profile_mix': args.profile_mix,
                'turns_min': args.turns_min,
                'turns_max': args.turns_max,
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
