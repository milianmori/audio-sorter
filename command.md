### Installation

- **Virtualenv aktivieren**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

- **Basis (ohne ML, empfohlen für Apple Silicon, aber auch allgemein ok)**
```bash
pip install librosa soundfile numpy scipy tqdm pandas
```

- **Optional: ML (YAMNet)**
  - **Apple Silicon (macOS):**
```bash
pip install tensorflow-macos==2.16.2 tensorflow-metal tensorflow-hub
```
  - **Intel/mac/Linux:**
```bash
pip install tensorflow==2.16.1 tensorflow-hub
```

---

### Run
```bash
python3 /Users/milianmori/Documents/repositories/audio-sorter/audio-sorter.py \
  --src "/Users/milianmori/Documents/repositories/audio-sorter/test-files/src" \
  --dst "/Users/milianmori/Documents/repositories/audio-sorter/test-files/dest" \
  --ml \
  --log "/Users/milianmori/Documents/repositories/audio-sorter/out.csv"
```