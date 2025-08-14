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
python3 /Users/milianmori/Documents/repositories/audio-sorter/audio-sorter.py \
  --src "/Volumes/MM_DROPBOX/Dropbox/werkbank/triality" \
  --dst "/Volumes/MM_DROPBOX/Dropbox/werkbank/audio cosmos triality" \
  --ml \
  --log "/Users/milianmori/Documents/repositories/audio-sorter/out.csv"
```

Deaktivieren der virtuellen Umgebung:
```bash
deactivate
```