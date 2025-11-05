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
    """Datenklasse für Testergebnisse"""
    users: int
    model: str
    llm_provider: str
    gpu: str
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    avg_ttft: float
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    cpu_usage: float
    memory_usage: float
    test_duration: float
    recommendation: str

class ResultCollector:
    """Sammelt und verwaltet Testergebnisse"""
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
    """Überwacht Systemressourcen während des Tests"""
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

# Globale Variablen für Ergebnissammlung
response_times = multiprocessing.Manager().list()
ttft_times = multiprocessing.Manager().list()  # Time to First Token
error_count = multiprocessing.Manager().Value('i', 0)
success_count = multiprocessing.Manager().Value('i', 0)

def reset_counters():
    """Setzt die globalen Zähler zurück"""
    global response_times, ttft_times, error_count, success_count
    response_times[:] = []
    ttft_times[:] = []
    error_count.value = 0
    success_count.value = 0

def terminate_processes(processes):
    """Signal-Handler für kontrollierten Abbruch"""
    for p in processes:
        if p.is_alive():
            p.terminate()
    print("Alle Prozesse beendet.")

def load_prompts(file_path):
    """Liest Prompts aus einer Textdatei ein."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def llm_chat_continuous(model, prompts, user_id, pause_min, pause_max, api_type, base_url, api_key, test_duration):
    """Simuliert einen Benutzer für eine bestimmte Testdauer"""
    global response_times, ttft_times, error_count, success_count

    # Create adapter for this process
    adapter = create_adapter(api_type, base_url, api_key)

    end_time = time.time() + test_duration

    while time.time() < end_time:
        # Zufälligen Prompt auswählen
        prompt = random.choice(prompts)

        success, elapsed_time, first_token_time, error_msg = adapter.make_request(model, prompt, timeout=120)

        if success:
            response_times.append(elapsed_time)
            ttft_times.append(first_token_time)
            success_count.value += 1
            print(f"[User {user_id}] ✓ {elapsed_time:.2f}s (TTFT: {first_token_time:.2f}s) - {prompt[:30]}...")

            # Pause zwischen erfolgreichen Requests (nur wenn noch Zeit bleibt)
            if time.time() < end_time:
                pause_time = random.uniform(pause_min, pause_max)
                time.sleep(min(pause_time, end_time - time.time()))
        else:
            error_count.value += 1
            print(f"[User {user_id}] ✗ {error_msg} - Retry sofort...")
            # Bei Fehler/Timeout: Sofort neuen Versuch ohne Pause
            continue

def get_recommendation(avg_time, max_time, error_rate, cpu_usage, avg_ttft):
    """Erstellt eine Empfehlung basierend auf TTFT und anderen Metriken"""
    # Fehlerrate hat höchste Priorität
    if error_rate > 10:
        return "❌ Kritisch"
    elif error_rate > 5:
        return "❌ Überlastet"
    elif error_rate > 2:
        return "⚠️ Instabil"
    # Dann TTFT-basierte Bewertung
    elif avg_ttft > 30:
        return "❌ Inakzeptabel"
    elif avg_ttft > 20:
        return "⚠️ Sehr langsam"
    elif avg_ttft > 10:
        return "⚠️ Langsam"
    elif avg_ttft > 5:
        return "✅ Akzeptabel"
    elif avg_ttft > 2:
        return "✅ Gut"
    else:
        return "✅ Optimal"

def check_api_connection(adapter):
    """Prüft ob die API erreichbar ist"""
    return adapter.check_connection()

def run_load_test(model, prompts, user_count, pause_min, pause_max, test_duration, api_type, base_url, api_key, gpu_name, llm_provider):
    """Führt einen Load-Test mit einer bestimmten Anzahl von Benutzern durch"""
    reset_counters()

    print(f"\n{'='*60}")
    print(f"Test mit {user_count} Benutzern gestartet...")
    print(f"Testdauer: {test_duration/60:.1f} Minuten")
    print(f"{'='*60}")

    # System-Monitoring starten
    monitor = SystemMonitor()
    monitor.start_monitoring()

    processes = []
    start_time = time.time()

    try:
        # Alle Benutzer gleichzeitig starten
        for user_id in range(user_count):
            p = multiprocessing.Process(
                target=llm_chat_continuous,
                args=(model, prompts, user_id, pause_min, pause_max, api_type, base_url, api_key, test_duration)
            )
            p.start()
            processes.append(p)

            # Kleine Verzögerung zwischen Starts zur Verteilung
            time.sleep(0.1)

        print(f"Alle {user_count} Benutzer gestartet. Warte {test_duration/60:.1f} Minuten...")

        # Überwachungsschleife mit Abbruchkriterium
        check_interval = 30  # Prüfe alle 30 Sekunden
        next_check = time.time() + check_interval

        while any(p.is_alive() for p in processes):
            time.sleep(1)

            # Alle 30 Sekunden Timeout-Rate prüfen
            if time.time() >= next_check:
                total_requests = success_count.value + error_count.value
                if total_requests >= 10:  # Mindestens 10 Requests für aussagekräftige Statistik
                    timeout_rate = (error_count.value / total_requests) * 100
                    print(f"[Zwischenstand] Requests: {total_requests}, Fehlerrate: {timeout_rate:.1f}%")

                    if timeout_rate > 30:
                        print(f"\n⚠️ ABBRUCH: Fehlerrate ({timeout_rate:.1f}%) überschreitet 30%!")
                        print("System ist überlastet - Test wird abgebrochen.")
                        break

                next_check = time.time() + check_interval

        # Alle Prozesse beenden
        for p in processes:
            if p.is_alive():
                p.terminate()

        # Kurz warten, damit Prozesse sauber beenden
        time.sleep(2)

    except KeyboardInterrupt:
        print("\nTest abgebrochen...")
        terminate_processes(processes)
        return None
    finally:
        # Sicherstellen, dass alle Prozesse beendet sind
        for p in processes:
            if p.is_alive():
                p.terminate()

    # System-Monitoring stoppen
    monitor.stop_monitoring()
    actual_duration = time.time() - start_time

    # Ergebnisse auswerten
    times = list(response_times)
    ttft_list = list(ttft_times)
    total_requests = success_count.value + error_count.value

    if not times:
        print(f"Keine erfolgreichen Requests in {user_count}-Benutzer-Test!")
        return None

    # Empfehlung generieren (jetzt basierend auf TTFT)
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
        avg_ttft=sum(ttft_list) / len(ttft_list) if ttft_list else 0,
        error_rate=(error_count.value / total_requests * 100) if total_requests > 0 else 0,
        total_requests=total_requests,
        successful_requests=success_count.value,
        failed_requests=error_count.value,
        cpu_usage=monitor.get_average_cpu(),
        memory_usage=monitor.get_average_memory(),
        test_duration=actual_duration,
        recommendation=recommendation
    )

    print(f"\nTest abgeschlossen:")
    print(f"  Erfolgreiche Requests: {result.successful_requests}")
    print(f"  Fehlgeschlagene Requests: {result.failed_requests}")
    print(f"  Durchschnittliche Antwortzeit: {result.avg_response_time:.2f}s")
    print(f"  Durchschnittliche TTFT: {result.avg_ttft:.2f}s")
    print(f"  Maximale Antwortzeit: {result.max_response_time:.2f}s")
    print(f"  Fehlerrate: {result.error_rate:.1f}%")
    print(f"  CPU-Auslastung: {result.cpu_usage:.1f}%")

    return result

def print_results_table(results: List[TestResult]):
    """Gibt die Ergebnistabelle aus"""
    if not results:
        print("Keine Ergebnisse zum Anzeigen.")
        return

    print(f"\n{'='*153}")
    print("LOAD TEST ERGEBNISSE")
    print(f"{'='*153}")

    # Header
    print(f"{'Benutzer':<8} {'Modell':<15} {'LLM Provider':<15} {'GPU':<12} {'Avg. Zeit':<10} {'TTFT':<8} {'Max. Zeit':<10} {'Min. Zeit':<10} {'Fehlerrate':<11} {'CPU %':<8} {'Memory %':<10} {'Requests':<10} {'Empfehlung':<12}")
    print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*11} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    # Datenzeilen
    for result in results:
        print(f"{result.users:<8} {result.model:<15} {result.llm_provider:<15} {result.gpu:<12} {result.avg_response_time:<10.2f} {result.avg_ttft:<8.2f} {result.max_response_time:<10.2f} {result.min_response_time:<10.2f} {result.error_rate:<11.1f} {result.cpu_usage:<8.1f} {result.memory_usage:<10.1f} {result.total_requests:<10} {result.recommendation:<12}")

    print(f"{'-'*153}")

def save_results_to_file(results: List[TestResult], filename: str):
    """Speichert Ergebnisse in eine CSV-Datei"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # CSV-Header
            f.write("Benutzer,Modell,LLM_Provider,GPU,Avg_Antwortzeit,Avg_TTFT,Max_Antwortzeit,Min_Antwortzeit,Fehlerrate,CPU_Prozent,Memory_Prozent,Total_Requests,Erfolgreiche_Requests,Fehlgeschlagene_Requests,Testdauer,Empfehlung\n")

            # Datenzeilen
            for result in results:
                f.write(f"{result.users},{result.model},{result.llm_provider},{result.gpu},{result.avg_response_time:.3f},{result.avg_ttft:.3f},{result.max_response_time:.3f},{result.min_response_time:.3f},{result.error_rate:.2f},{result.cpu_usage:.2f},{result.memory_usage:.2f},{result.total_requests},{result.successful_requests},{result.failed_requests},{result.test_duration:.1f},{result.recommendation}\n")

        print(f"\nCSV gespeichert: {filename}")
    except Exception as e:
        print(f"Fehler beim Speichern der CSV: {e}")

def save_results_to_markdown(results: List[TestResult], filename: str, test_config: dict):
    """Speichert Ergebnisse als Markdown-Zusammenfassung"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("# LLM Load Test - Zusammenfassung\n\n")

            # Test-Konfiguration
            f.write("## Test-Konfiguration\n\n")
            f.write(f"- **Datum/Zeit**: {test_config['timestamp']}\n")
            f.write(f"- **LLM Provider**: {test_config['llm_provider']}\n")
            f.write(f"- **API Typ**: {test_config['api_type']}\n")
            f.write(f"- **Base URL**: {test_config['base_url']}\n")
            f.write(f"- **Modelle**: {test_config['models']}\n")
            f.write(f"- **GPU**: {test_config['gpu']}\n")
            f.write(f"- **Testdauer pro Schritt**: {test_config['test_duration']/60:.1f} Minuten\n")
            f.write(f"- **Pausenzeiten**: {test_config['pause_min']}-{test_config['pause_max']} Sekunden\n")
            f.write(f"- **Benutzer-Schritte**: {test_config['user_steps']}\n\n")

            # Ergebnisse gruppiert nach Modell
            models = list(set([r.model for r in results]))

            for model in models:
                f.write(f"## Ergebnisse: {model}\n\n")
                model_results = [r for r in results if r.model == model]

                # Tabelle
                f.write("| Benutzer | Avg. Zeit (s) | TTFT (s) | Max. Zeit (s) | Fehlerrate (%) | CPU (%) | Memory (%) | Requests | Empfehlung |\n")
                f.write("|----------|---------------|----------|---------------|----------------|---------|------------|----------|------------|\n")

                for result in model_results:
                    f.write(f"| {result.users} | {result.avg_response_time:.2f} | {result.avg_ttft:.2f} | {result.max_response_time:.2f} | {result.error_rate:.1f} | {result.cpu_usage:.1f} | {result.memory_usage:.1f} | {result.total_requests} | {result.recommendation} |\n")

                f.write("\n")

                # Zusammenfassung für dieses Modell
                best_result = max(model_results, key=lambda r: r.users if r.error_rate < 10 else 0)
                f.write(f"### Zusammenfassung\n\n")
                f.write(f"- **Beste Performance**: {best_result.users} gleichzeitige Benutzer\n")
                f.write(f"- **Durchschnittliche TTFT**: {best_result.avg_ttft:.2f}s\n")
                f.write(f"- **Durchschnittliche Antwortzeit**: {best_result.avg_response_time:.2f}s\n")
                f.write(f"- **Fehlerrate**: {best_result.error_rate:.1f}%\n\n")

            # Gesamtzusammenfassung
            f.write("## Gesamtzusammenfassung\n\n")
            total_requests = sum(r.total_requests for r in results)
            total_successful = sum(r.successful_requests for r in results)
            total_failed = sum(r.failed_requests for r in results)
            avg_ttft_all = sum(r.avg_ttft for r in results) / len(results) if results else 0

            f.write(f"- **Gesamt Requests**: {total_requests}\n")
            f.write(f"- **Erfolgreiche Requests**: {total_successful}\n")
            f.write(f"- **Fehlgeschlagene Requests**: {total_failed}\n")
            f.write(f"- **Durchschnittliche TTFT (alle Tests)**: {avg_ttft_all:.2f}s\n")
            f.write(f"- **Gesamtfehlerrate**: {(total_failed/total_requests*100) if total_requests > 0 else 0:.1f}%\n\n")

            # Empfehlungen
            f.write("## Empfehlungen\n\n")

            # Finde besten Test (höchste Benutzeranzahl mit < 10% Fehlerrate)
            good_results = [r for r in results if r.error_rate < 10]
            if good_results:
                best = max(good_results, key=lambda r: r.users)
                f.write(f"- Empfohlene maximale Benutzeranzahl: **{best.users} gleichzeitige Benutzer**\n")
                f.write(f"- Bei dieser Last: TTFT {best.avg_ttft:.2f}s, Fehlerrate {best.error_rate:.1f}%\n")
            else:
                f.write("- ⚠️ Alle Tests zeigten hohe Fehlerraten (>10%). System ist überlastet.\n")

            f.write("\n")

        print(f"Markdown gespeichert: {filename}")
    except Exception as e:
        print(f"Fehler beim Speichern der Markdown-Datei: {e}")

def create_results_directory():
    """Erstellt einen Ergebnis-Ordner mit Timestamp"""
    import os

    # Basis-Verzeichnis für Ergebnisse
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # Ordner mit Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, timestamp)

    # Verzeichnisse erstellen
    os.makedirs(result_dir, exist_ok=True)

    return result_dir, timestamp

def main():
    parser = argparse.ArgumentParser(description="Schrittweises Load Testing für LLM APIs (Ollama, vLLM, LM Studio, llama.cpp, etc.)")
    parser.add_argument("--prompts", type=str, required=True,
                       help="Pfad zur Prompts-Datei")
    parser.add_argument("--users", type=int, required=True,
                       help="Maximale Anzahl der Benutzer (wird schrittweise erreicht)")
    parser.add_argument("--model", type=str, required=True,
                       help="Modell(e), kommagetrennt für mehrere Modelle")
    parser.add_argument("--llm-provider", type=str, required=True,
                       help="LLM Provider Name (z.B. 'Ollama', 'vLLM', 'LM Studio', etc.)")
    parser.add_argument("--gpu", type=str, default="Unknown",
                       help="GPU-Bezeichnung für Dokumentation (Standard: Unknown)")
    parser.add_argument("--pause-min", type=float, default=3.0,
                       help="Minimale Pause zwischen Nachrichten in Sekunden (Standard: 3.0)")
    parser.add_argument("--pause-max", type=float, default=30.0,
                       help="Maximale Pause zwischen Nachrichten in Sekunden (Standard: 30.0)")
    parser.add_argument("--step-size", type=int, default=5,
                       help="Schrittgröße für Benutzererhöhung (Standard: 5)")
    parser.add_argument("--test-duration", type=int, default=300,
                       help="Testdauer pro Schritt in Sekunden (Standard: 300 = 5 Minuten)")
    parser.add_argument("--host", type=str, default=None,
                       help="API Host und Port (Standard: aus .env oder 127.0.0.1:11434)")
    parser.add_argument("--api-type", type=str, default=None,
                       help="API Typ (ollama, vllm, lmstudio, llamacpp, openai) (Standard: aus .env oder ollama)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API Key für Authentifizierung (optional, aus .env wenn nicht angegeben)")
    parser.add_argument("--output", type=str, default=None,
                       help="Dateiname für CSV-Export (optional)")

    args = parser.parse_args()

    # Konfiguration aus .env oder Command Line Arguments
    api_type = args.api_type or os.getenv('API_TYPE', 'ollama')
    api_key = args.api_key or os.getenv('API_KEY')

    # Host/Base URL bestimmen
    if args.host:
        base_url = args.host if args.host.startswith(('http://', 'https://')) else f"http://{args.host}"
    else:
        # Versuche aus .env zu lesen
        env_url = os.getenv('API_BASE_URL')
        if env_url:
            base_url = env_url
        else:
            # Fallback zu Standard Ollama
            base_url = "http://127.0.0.1:11434"

    # Modelle aus kommagetrenntner Liste extrahieren
    models = [model.strip() for model in args.model.split(',') if model.strip()]

    if not models:
        print("Fehler: Keine gültigen Modelle angegeben!")
        return

    # Validierung
    if args.pause_min > args.pause_max:
        print("Fehler: pause-min darf nicht größer als pause-max sein!")
        return

    if args.users <= 0 or args.step_size <= 0:
        print("Fehler: users und step-size müssen größer als 0 sein!")
        return

    # API-Adapter erstellen
    try:
        adapter = create_adapter(api_type, base_url, api_key)
    except ValueError as e:
        print(f"Fehler: {e}")
        return

    # API-Verbindung prüfen
    print(f"Prüfe Verbindung zu {api_type.upper()} API ({base_url})...")
    if not check_api_connection(adapter):
        print(f"Fehler: Kann nicht zur API unter {base_url} verbinden!")
        print(f"Stelle sicher, dass der {api_type.upper()} Server läuft.")
        return

    print(f"✓ Verbindung zur {api_type.upper()} API erfolgreich")

    # Prompts laden
    try:
        prompts = load_prompts(args.prompts)
        print(f"✓ {len(prompts)} Prompts aus {args.prompts} geladen")
    except FileNotFoundError:
        print(f"Fehler: Prompts-Datei {args.prompts} nicht gefunden!")
        return

    if len(prompts) == 0:
        print("Fehler: Keine Prompts in der Datei gefunden!")
        return

    # Test-Parameter anzeigen
    print(f"\nSTARTE SCHRITTWEISES LOAD TESTING")
    print(f"API Typ: {api_type.upper()}")
    print(f"Base URL: {base_url}")
    print(f"Modelle: {', '.join(models)}")
    print(f"GPU: {args.gpu}")
    print(f"Maximale Benutzer: {args.users}")
    print(f"Schrittgröße: {args.step_size}")
    print(f"Testdauer pro Schritt: {args.test_duration/60:.1f} Minuten")
    print(f"Pausenzeiten: {args.pause_min}-{args.pause_max} Sekunden")

    # Schrittweise Tests durchführen
    results = []
    user_steps = list(range(args.step_size, args.users + 1, args.step_size))

    # Falls die maximale Anzahl nicht durch step_size teilbar ist, hinzufügen
    if args.users not in user_steps:
        user_steps.append(args.users)

    total_steps = len(user_steps) * len(models)
    estimated_total_time = total_steps * args.test_duration / 60

    print(f"Geplante Schritte: {user_steps}")
    print(f"Geschätzte Gesamtdauer: {estimated_total_time:.1f} Minuten")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    try:
        step_counter = 0

        # Für jedes Modell alle Benutzer-Schritte durchführen
        for model in models:
            print(f"\n{'='*80}")
            print(f"TESTE MODELL: {model}")
            print(f"{'='*80}")

            for user_count in user_steps:
                step_counter += 1
                print(f"\n[Schritt {step_counter}/{total_steps}] Teste {user_count} Benutzer mit {model}...")

                result = run_load_test(
                    model, prompts, user_count,
                    args.pause_min, args.pause_max,
                    args.test_duration, api_type, base_url, api_key, args.gpu, args.llm_provider
                )

                if result:
                    results.append(result)

                # Kurze Pause zwischen Tests
                if step_counter < total_steps:
                    print("Pause zwischen Tests (10 Sekunden)...")
                    time.sleep(10)

        # Ergebnisse anzeigen
        print_results_table(results)

        # Ergebnisse speichern
        if args.output:
            # Manuell angegebener Dateiname (nur CSV, ohne Ordner)
            save_results_to_file(results, args.output)
        else:
            # Automatisches Speichern in results/ Ordner mit Timestamp
            result_dir, timestamp_str = create_results_directory()

            # CSV speichern
            csv_filename = os.path.join(result_dir, "results.csv")
            save_results_to_file(results, csv_filename)

            # Markdown-Zusammenfassung speichern
            md_filename = os.path.join(result_dir, "summary.md")
            test_config = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'llm_provider': args.llm_provider,
                'api_type': api_type.upper(),
                'base_url': base_url,
                'models': ', '.join(models),
                'gpu': args.gpu,
                'test_duration': args.test_duration,
                'pause_min': args.pause_min,
                'pause_max': args.pause_max,
                'user_steps': user_steps
            }
            save_results_to_markdown(results, md_filename, test_config)

            print(f"\n📁 Ergebnisse gespeichert in: {result_dir}")

        print(f"\nLoad Test abgeschlossen um {datetime.now().strftime('%H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\nLoad Test abgebrochen!")
        if results:
            print("Bisherige Ergebnisse:")
            print_results_table(results)

if __name__ == "__main__":
    main()
