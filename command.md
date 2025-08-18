### Neu aufgesetzter Mac: komplette Einrichtung (macOS)

1) Xcode Command Line Tools installieren
```bash
xcode-select --install
```

2) Homebrew installieren und zu PATH hinzufügen (Apple Silicon)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

3) Python 3.11 installieren (empfohlen für TensorFlow 2.16.x)
```bash
brew install python@3.11
python3.11 -V
```

4) FFmpeg installieren (für MP3/M4A-Decoding via librosa/audioread)
```bash
brew install ffmpeg
```

5) In das Projekt wechseln
```bash
cd /Users/milianmori/Documents/repositories/audio-sorter
```

6) Virtuelle Umgebung anlegen und aktivieren
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

7) Python-Abhängigkeiten installieren
- Basis (ohne ML)
```bash
pip install librosa soundfile numpy scipy tqdm pandas
```

- Optional: ML (YAMNet)
  - Apple Silicon (macOS)
```bash
pip install tensorflow-macos==2.16.2 tensorflow-metal tensorflow-hub
```
  - Intel/mac/Linux
```bash
pip install tensorflow==2.16.1 tensorflow-hub
```

Hinweise
- Wenn `soundfile`/`librosa` beim Lesen von MP3/M4A Probleme melden: Stelle sicher, dass FFmpeg (Schritt 4) installiert ist.
- Falls du eine andere Python-Version nutzt (z. B. 3.12), kann TensorFlow nicht verfügbar sein. Verwende dann Python 3.11 wie oben.

---

### Ausführen
Mit absoluten Pfaden (wie in diesem Repo):
```bash
python3.11 /Users/milianmori/Documents/repositories/audio-sorter/audio-sorter.py \
  --src "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" \
  --dst "/Users/milianmori/Documents/repositories/audio-sorter/test-files/dest" \
  --ml \
  --log "/Users/milianmori/Documents/repositories/audio-sorter/out.csv"
```

Oder aus dem Projektordner heraus mit relativen Pfaden:
```bash
cd /Users/milianmori/Documents/repositories/audio-sorter
source .venv/bin/activate
python audio-sorter.py \
  --src "test-files/src" \
  --dst "test-files/dest" \
  --ml \
  --log "out.csv"
```

### Run
```bash
python3 audio-sorter.py \
  --src "/Volumes/MM_DROPBOX/Dropbox/werkbank/triality" \
  --dst "/Volumes/MM_DROPBOX/Dropbox/werkbank/audio cosmos triality" \
  --ml \
  --log "/Users/milianmori/Documents/repositories/audio-sorter/out.csv"
```

Deaktivieren der virtuellen Umgebung:
```bash
deactivate
```

---

### Audio Cleanup (zweites Skript)
Dieses Skript überspringt Dateien ohne echten Audiostream oder mit reiner Stille, trimmt Stille am Anfang/Ende und findet Duplikate (Name ähnlich, Länge gleich/nahezu gleich, Audioinhalt tatsächlich identisch). Verarbeitung erfolgt jetzt **in-place**: Es wird kein Output-Ordner mehr erstellt.

Hinweis: FFmpeg/FFprobe ist erforderlich (siehe Installation oben).

My command:
python3 audio_cleanup.py "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" --delete-duplicates
python3 audio_cleanup.py "/Volumes/MM_DROPBOX/Dropbox/audio/audio cosmos triality" --delete-duplicates


Basislauf (überspringt Stille, trimmt Stille, schreibt in-place in den Quellordner):
```bash
python3.11 /Users/milianmori/Documents/repositories/audio-sorter/audio_cleanup.py \
  "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src"
```

Optionaler Output-Pfad (wird ignoriert, nur aus Kompatibilitätsgründen vorhanden), ohne Trimmen:
```bash
python3.11 /Users/milianmori/Documents/repositories/audio-sorter/audio_cleanup.py \
  "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" \
  --output "/Users/milianmori/Documents/repositories/audio-sorter/test-files/clean" \
  --no-trim
```

Nur Duplikate finden (berichten):
```bash
python3.11 /Users/milianmori/Documents/repositories/audio-sorter/audio_cleanup.py \
  "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" \
  --dedupe
```

Duplikate löschen (standardmäßig werden sie nach `tobedeleted` unterhalb des Dedupe-Zielordners verschoben):
```bash
python3.11 /Users/milianmori/Documents/repositories/audio-sorter/audio_cleanup.py \
  "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" \
  --dedupe --delete-duplicates
```
```bash
python3.11 /Users/milianmori/Documents/repositories/audio-sorter/audio_cleanup.py \
  "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" \
  --dedupe --delete-duplicates \
  --move-duplicates-to "/Users/milianmori/Documents/repositories/audio-sorter/test-files/dupes"
```

Wichtige Optionen:
- **--output**: Ignoriert; Verarbeitung erfolgt in-place (Option bleibt aus Rückwärtskompatibilität erhalten).
- **--no-skip-silent**: Reine Stille nicht überspringen.
- **--no-trim**: Stille nicht schneiden.
- **--silence-threshold-db**: dBFS-Schwelle für Stille (Standard: -50).
- **--min-silence-ms**: Mindestlänge der Stille in ms (Standard: 200).
- **--dedupe / --dedupe-path**: Duplikatsuche (optional anderer Pfad als INPUT).
- **--name-similarity**: Namensähnlichkeit 0..1 (Standard: 0.85). Datums-/Zeit-Suffixe wie `[2024-08-10 151650]` am Ende werden automatisch ignoriert.
- **--duration-tolerance-ms**: Toleranz für Längenvergleich in ms (Standard: 50).
- **--delete-duplicates**: Gefundene Duplikate entfernen (Standard: verschieben in `tobedeleted`).
- **--move-duplicates-to**: Duplikate statt Standardziel in diesen Ordner verschieben.