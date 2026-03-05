# Verbesserungsvorschläge: LLM Load Test Tool

## Aktueller Stand

Das Tool simuliert Single-Turn-Anfragen mit Denkpausen – das ergibt eine **obere Grenze** der Nutzerzahl, bildet aber reales Chatverhalten nicht vollständig ab.

## Was fehlt und wie es realistischer wird

### 1. Multi-Turn-Konversationen

Aktuell sendet jeder simulierte Nutzer isolierte Einzelfragen. Echte Nutzer führen Gespräche mit 3–10 Turns, wobei der gesamte Chatverlauf bei jedem Turn mitgesendet wird. Das belastet den KV-Cache und die Prefill-Phase erheblich stärker. Das Tool sollte pro User eine `/v1/chat/completions`-Session mit wachsender `messages`-History simulieren.

### 2. Chat Completions statt Completions API

Das Tool nutzt `/v1/completions` (Legacy). Reale Anwendungen wie Open WebUI verwenden `/v1/chat/completions` mit System-Prompt, Rollen und Verlauf. Der Overhead durch Chat-Templating und längere Kontexte fehlt komplett.

### 3. Variable Prompt- und Antwortlängen

Die mitgelieferten Prompts erzeugen relativ gleichförmige, kurze bis mittellange Antworten. In der Praxis gibt es sowohl Kurzfragen ("Was heißt EBIT?") als auch Aufgaben, die 1.000+ Token Antwort produzieren ("Schreibe ein Schulungskonzept"). Eine Mischung mit realistischer Längenverteilung (z.B. lognormal) würde die Last besser abbilden.

### 4. Lange Input-Kontexte

Kein Prompt simuliert das Einfügen langer Dokumente ("Fasse dieses 10-seitige Protokoll zusammen"). Solche Anfragen mit 2.000–10.000 Input-Tokens treiben die Prefill-Phase in die Höhe und belasten den Speicher deutlich stärker als kurze Fragen.

### 5. Streaming-Messung

Reale Nutzer sehen gestreamte Tokens. Das Tool misst nur die Gesamtantwortzeit und TTFT, aber nicht die wahrgenommene Inter-Token-Latenz (TPOT). Bei hoher Last bleibt TTFT oft akzeptabel, aber das Streaming stockt – das erfasst das Tool nicht.

### 6. Gleichzeitige Sessions mit System-Prompts

Jeder Nutzer in Open WebUI hat typischerweise einen System-Prompt. 20 gleichzeitige Nutzer bedeuten 20 verschiedene System-Prompts im KV-Cache. Das Tool sollte pro simuliertem User einen individuellen System-Prompt mitsenden, um Prefix-Caching realistisch zu testen.

### 7. Realistische Nutzerprofile

Statt einheitlicher Pausen für alle Nutzer wären gemischte Profile realistischer: Power-User (2–5s Pause, viele Turns), normale Nutzer (15–45s Pause, moderate Turns) und gelegentliche Nutzer (60–120s Pause, wenige Turns). Die Verteilung sollte konfigurierbar sein.

## Geschätzte Auswirkung

| Faktor | Auswirkung auf Kapazität |
|---|---|
| Multi-Turn (3–5 Turns) | −20–30% |
| Lange Input-Kontexte | −10–15% |
| Gemischte Antwortlängen | −5–10% |
| **Gesamt** | **~60–70% der gemessenen Nutzerzahl** |

## Fazit

Für eine erste Kapazitätseinschätzung ist das Tool brauchbar. Die gemessene maximale Nutzerzahl sollte mit Faktor 0,6–0,7 multipliziert werden, um eine realistische Produktionskapazität zu erhalten. Alternativ: Nach dem Lasttest 5–10 echte Kollegen gleichzeitig in Open WebUI chatten lassen und TTFT beobachten.