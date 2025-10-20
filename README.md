## Audio Tools — Cleanup + Sort (All-in-One)

One CLI to trim/skip silence, de-duplicate, and sort audio content by type.

### Requirements
- Python 3.9+
- `ffmpeg` (for cleanup/trim/hash). On macOS: `brew install ffmpeg`
- Python packages:
  - Required: `numpy`, `librosa`, `soundfile`, `tqdm`
  - Optional ML: `tensorflow-hub` + `tensorflow` (mac: `tensorflow-macos` + `tensorflow-metal`)

### Install deps (examples)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install numpy librosa soundfile tqdm
# Optional ML (macOS Apple Silicon):
pip install tensorflow-macos==2.16.2 tensorflow-metal tensorflow-hub
```

### Script
- New combined entry point: `audio_all_in_one.py`
- Legacy scripts still work: `audio_cleanup.py`, `audio-sorter.py`

### Usage
- Cleanup only (in-place):
```bash
python audio_all_in_one.py cleanup \/path\/to\/src \
  --dedupe --delete-duplicates
```
- Sort only (to destination):
```bash
python audio_all_in_one.py sort --src \/path\/to\/src --dst \/path\/to\/dst \
  --ml --log sort_log.csv
```
- Cleanup + Sort:
```bash
python audio_all_in_one.py all --src \/path\/to\/src --dst \/path\/to\/dst \
  --ml --dedupe --delete-duplicates --log sort_log.csv
```

### What happens step by step

#### cleanup
1. Verify `ffprobe`/`ffmpeg` availability.
2. Scan `src` for supported extensions.
3. For each file:
   - Probe with `ffprobe`; skip files without audio streams.
   - Measure mean volume with `volumedetect`.
   - If pure digital silence and skipping is enabled, move to `src/tobedeleted` (if `--delete-duplicates`) or skip.
   - If trimming enabled, run `silenceremove` and replace original when successful.
   - Append per-file row to `src/log.csv` (before/after durations, threshold, etc.).
4. If `--dedupe` (or `--delete-duplicates`) is set:
   - Group by quantized duration; for candidates compute a canonical PCM hash via `ffmpeg`.
   - For identical content, keep the older file and delete/move the other (to `src/tobedeleted` if requested).

#### sort
1. Collect files from `--src` by extension.
2. If file lies within a folder named `export`/`exports`, copy/normalize to `dst/exports/` without renaming and log features.
3. Otherwise for each file:
   - Decide category via priority: keywords → optional ML (YAMNet) → audio heuristics → fallback.
   - Build `dst/<Category>/<Subcategory>/` and rename as `Category_Subcategory_OriginalName.ext`.
   - Try peak normalize for `.wav/.aiff/.aif/.flac`; else copy.
   - Extract features and append to CSV log.
4. Write sorter log to `--log`.

#### all
1. Run the entire `cleanup` pipeline on `--src` (with provided cleanup flags).
2. Optionally de-duplicate.
3. Write `src/log.csv` (trim log).
4. Run the entire `sort` pipeline from `--src` to `--dst` (with `--ml`, `--exts`, `--log`).

### Notes
- Sorting supports: `.wav,.aiff,.aif,.mp3,.flac,.m4a` (configurable via `--exts`).
- Normalization target default is −1 dBFS; only for `.wav/.aif/.aiff/.flac`.
- ML is optional; without it, heuristics and filename keywords are used.



source .venv/bin/activate

python3 audio_all_in_one.py all --src /Users/milianmori/Documents/repositories/audio-sorter/test-files/src --dst /Users/milianmori/Documents/repositories/audio-sorter/test-files/dest \
  --ml --dedupe --delete-duplicates --log sort_log.csv

python3 audio_all_in_one.py all --src "/Volumes/MM_Archive/07042020 Project/" --dst "/Users/mm/Desktop/audio cosmos drone" \
  --ml --dedupe --delete-duplicates --log sort_log.csv

python3 audio_all_in_one.py all --src "/Volumes/MM_Archive/As You Were Listening" --dst "/Users/mm/Desktop/audio cosmose as you were listening" \
  --ml --dedupe --delete-duplicates --log sort_log.csv