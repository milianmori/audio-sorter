#!/usr/bin/env python3
"""
Audio Sorter (CLI)
==================

Inhaltsbasierte Sortierung großer Audio-Sammlungen (Kicks, Snares/Claps, HiHats,
Percussion, Loops, OneShots, Vocals, Field_Recordings, Resampled) mit optionaler
LUFS/TruePeak-Normalisierung über ffmpeg (EBU R128).

Hauptfunktionen:
- Klassifikation des Inhalts via YAMNet (AudioSet) über TensorFlow Hub
- Erkennung von Loops vs. One-Shots per Onset-Analyse (librosa)
- Robustes Routing in Zielordner gemäß Mapping + Heuristiken
- Optionale Lautheits-Normalisierung (ffmpeg loudnorm, 2-pass)
- Logging aller Ergebnisse (CSV)

Voraussetzungen (macOS, Linux, Windows):
- Python 3.9+
- ffmpeg im PATH (für Konvertierung/Normalisierung)
- Python-Pakete:
  * numpy, pandas, librosa, soundfile, tqdm
  * tensorflow-hub und TensorFlow (CPU reicht). macOS Apple Silicon:
      - pip install tensorflow-macos tensorflow-metal tensorflow-hub
    Intel/macOS-Linux:
      - pip install tensorflow tensorflow-hub

Installation (Beispiel macOS, Homebrew ffmpeg):
  brew install ffmpeg
  python3 -m venv .venv && source .venv/bin/activate
  pip install numpy pandas librosa soundfile tqdm tensorflow-hub
  # Apple Silicon (M1/M2/M3):
  pip install tensorflow-macos tensorflow-metal
  # Intel/macOS/Linux:
  pip install tensorflow

Beispielaufrufe:
python audio-sorter.py \
  --src "/Users/milianmori/Documents/repositories/audio-sorter/test-files" \
  --dst "/Users/milianmori/Documents/repositories/audio-sorter/test-files" \
  --normalize lufs --target-lufs -14 --true-peak -1.0 \
  --output-format wav --log out.csv

  # Nur sortieren, ohne Normalisierung:
  python audio_sorter.py --src ./unsorted --dst ./Audio_Sortiert --normalize none

Hinweise:
- YAMNet erwartet MONO@16kHz Float32. Wir resamplen intern für die Klassifikation.
- Für Normalisierung per LUFS wird immer re-encodiert. Mit --output-format kann
  ein Zielformat erzwungen werden (z.B. wav). "same" versucht, die Original-
  Erweiterung zu behalten.
- Mapping basiert auf AudioSet-Labels → Zielordner; Loops werden über
  Mehrfach-Onsets/Dauer erkannt.
- 100% perfekte Erkennung ist unrealistisch; die Pipeline ist aber robust und
  skalierbar. Du kannst am Mapping/Heuristiken feintunen.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# TensorFlow + TF Hub (YAMNet)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_OK = True
except Exception as e:
    TF_OK = False
    tf = None
    hub = None

# -------------------------------
# Utility
# -------------------------------

def run(cmd: List[str]) -> Tuple[int, str, str]:
    """Run subprocess command, return (returncode, stdout, stderr)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class FileInfo:
    src: Path
    rel: Path  # relative path from src root
    duration: float
    sr: int
    n_onsets: int


@dataclass
class ClassResult:
    top1_label: str
    top1_score: float
    topk: List[Tuple[str, float]]


# -------------------------------
# YAMNet Model Wrapper
# -------------------------------
class YamnetWrapper:
    def __init__(self):
        if not TF_OK:
            raise RuntimeError(
                "TensorFlow und tensorflow-hub sind nicht verfügbar. Bitte installiere sie (siehe Kopfzeile)."
            )
        # Lade YAMNet von TF Hub (wird beim ersten Lauf gecached)
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        # Labels laden
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path) as f:
            # CSV: index, mid, display_name
            self.labels = [line.strip().split(',')[-1] for line in f.readlines()[1:]]
        # Warmup Dummy
        _ = self.predict(np.zeros(16000, dtype=np.float32), 16000)

    def predict(self, y: np.ndarray, sr: int) -> ClassResult:
        """Return top labels and scores for a mono waveform y (float32), given sr.
        YAMNet expects 16kHz mono float32 PCM in [-1, 1]."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        y = y.astype(np.float32)
        # Model inference: returns scores [frames, 521]
        scores, embeddings, spectrogram = self.model(y)
        mean_scores = scores.numpy().mean(axis=0)
        top_idx = np.argsort(mean_scores)[::-1]
        # Top-3 labels
        topk = []
        for i in top_idx[:3]:
            topk.append((self.labels[i], float(mean_scores[i])))
        top1_label, top1_score = topk[0]
        return ClassResult(top1_label=top1_label, top1_score=top1_score, topk=topk)


# -------------------------------
# Routing / Mapping
# -------------------------------
PERC_KEYWORDS = {
    'tom', 'tom-tom', 'cymbal', 'ride cymbal', 'crash cymbal', 'cymbal crash', 'cowbell', 'wood block',
    'tambourine', 'bongo', 'conga', 'guiro', 'triangle', 'castanet', 'maraca', 'sleigh bell', 'gong', 'bell',
    'shaker', 'cabasa', 'agogo', 'clave', 'sticks', 'drumstick'
}
VOICE_KEYWORDS = {
    'speech', 'narration', 'singing', 'vocal', 'choir', 'chant', 'a capella', 'yodeling', 'vocal music'
}
ENV_KEYWORDS = {
    'environment', 'nature', 'rain', 'wind', 'thunder', 'storm', 'water', 'sea', 'ocean', 'river', 'stream',
    'fire', 'crackle', 'traffic', 'car', 'engine', 'vehicle', 'train', 'airplane', 'crowd', 'room', 'ambience',
    'bird', 'animal', 'dog', 'cat', 'insect', 'urban', 'street', 'park', 'forest', 'footsteps', 'noise'
}
MUSIC_KEYWORDS = {
    'music', 'musical', 'instrument', 'drum', 'percussion', 'guitar', 'piano', 'keyboard', 'string', 'brass', 'synth'
}


def choose_category(
    cr: ClassResult,
    duration: float,
    n_onsets: int,
    min_loop_duration: float = 2.0,
) -> str:
    """Map YAMNet top labels + simple heuristics to target category folder name."""
    labels = [l.lower() for l, _ in cr.topk]

    def has_kw(keys: set[str]) -> bool:
        return any(any(k in lbl for k in keys) for lbl in labels)

    # Direct drum mapping by specific labels
    if any('bass drum' in lbl or 'kick drum' in lbl for lbl in labels):
        return '1_Kicks'
    if any('snare drum' in lbl for lbl in labels) or any('clap' in lbl for lbl in labels):
        # Avoid crowd "applause" long clips misrouting: only treat as clap if one-shot-ish
        if duration < 3.5 and n_onsets <= 2:
            return '2_Snares_Claps'
    if any('hi-hat' in lbl or 'hihat' in lbl for lbl in labels):
        return '3_HiHats'

    # Voice
    if has_kw(VOICE_KEYWORDS):
        return '7_Vocals'

    # Loop vs OneShot decision (before generic perc):
    is_loop = (duration >= min_loop_duration and n_onsets >= 2)

    # Generic percussion set (toms, cymbals, shakers, etc.)
    if has_kw(PERC_KEYWORDS) or any('cymbal' in lbl for lbl in labels):
        return '4_Percussion' if not is_loop else '5_Loops'

    # Field recordings / environment
    if has_kw(ENV_KEYWORDS) and not has_kw(MUSIC_KEYWORDS):
        return '8_Field_Recordings'

    # Loops (music/instrument + multiple onsets)
    if is_loop and has_kw(MUSIC_KEYWORDS):
        return '5_Loops'

    # One-shots (non-drum, musical sfx/tones)
    if not is_loop:
        return '6_OneShots'

    # Fallback
    return '9_Resampled'


# -------------------------------
# FFmpeg Loudness Normalization (EBU R128, 2-pass)
# -------------------------------
LOUDNORM_RE = re.compile(r"\{[\s\S]*?\}")

def ffmpeg_loudnorm_two_pass(
    src: Path,
    dst: Path,
    target_i: float = -14.0,
    target_tp: float = -1.0,
    lra: float = 11.0,
    output_acodec: Optional[str] = None,
) -> bool:
    """Apply LUFS/TruePeak normalization using ffmpeg loudnorm 2-pass. Returns success bool."""
    # First pass: measure
    cmd1 = [
        'ffmpeg', '-hide_banner', '-y', '-i', str(src),
        '-af', f'loudnorm=I={target_i}:TP={target_tp}:LRA={lra}:print_format=json',
        '-f', 'null', '-'
    ]
    rc, out, err = run(cmd1)
    if rc != 0:
        print(f"[ffmpeg] First pass failed for {src}: {err}", file=sys.stderr)
        return False
    m = LOUDNORM_RE.search(err)
    if not m:
        print(f"[ffmpeg] Could not parse loudnorm JSON for {src}", file=sys.stderr)
        return False
    try:
        data = json.loads(m.group(0))
        measured_I = data['input_i']
        measured_TP = data['input_tp']
        measured_LRA = data['input_lra']
        measured_thresh = data['input_thresh']
        offset = data['target_offset']
    except Exception as e:
        print(f"[ffmpeg] JSON parse error for {src}: {e}", file=sys.stderr)
        return False

    # Second pass: apply
    loudnorm2 = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={lra}:"
        f"measured_I={measured_I}:measured_TP={measured_TP}:"
        f"measured_LRA={measured_LRA}:measured_thresh={measured_thresh}:"
        f"offset={offset}:linear=true:print_format=summary"
    )
    cmd2 = ['ffmpeg', '-hide_banner', '-y', '-i', str(src), '-af', loudnorm2]
    # Output codec/format selection
    if output_acodec:
        cmd2 += ['-c:a', output_acodec]
    cmd2 += [str(dst)]
    rc2, out2, err2 = run(cmd2)
    if rc2 != 0:
        print(f"[ffmpeg] Second pass failed for {src}: {err2}", file=sys.stderr)
        return False
    return True


def ffmpeg_peak_normalize(src: Path, dst: Path, peak_db: float = -1.0, output_acodec: Optional[str] = None) -> bool:
    """Simple peak normalisation using ffmpeg volume filter to reach given dBFS peak."""
    # Compute current peak via astats
    cmd1 = ['ffmpeg', '-hide_banner', '-y', '-i', str(src), '-af', 'astats=metadata=1:reset=1', '-f', 'null', '-']
    rc, out, err = run(cmd1)
    if rc != 0:
        print(f"[ffmpeg] Peak measurement failed for {src}: {err}", file=sys.stderr)
        return False
    # Parse Maximum level from stderr (assuming last printed MAXPEAK)
    max_peak_db = None
    for line in err.splitlines():
        if 'Peak level dB' in line or 'Max level dB' in line or 'max_level' in line:
            # Try to extract number
            nums = re.findall(r"-?\d+\.\d+|-?\d+", line)
            if nums:
                try:
                    val = float(nums[-1])
                    max_peak_db = val
                except:
                    pass
    if max_peak_db is None:
        # Fallback: no measurement -> just copy
        shutil.copy2(src, dst)
        return True
    gain_db = peak_db - max_peak_db
    vol_filter = f"volume={gain_db}dB"
    cmd2 = ['ffmpeg', '-hide_banner', '-y', '-i', str(src), '-af', vol_filter]
    if output_acodec:
        cmd2 += ['-c:a', output_acodec]
    cmd2 += [str(dst)]
    rc2, out2, err2 = run(cmd2)
    return rc2 == 0


# -------------------------------
# Core processing
# -------------------------------

def analyze_file(path: Path, onset_backtrack: bool = False) -> Tuple[float, int, int]:
    """Load audio quickly to compute duration/sr and number of onsets."""
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=onset_backtrack)
        n_onsets = int(len(onsets))
        return duration, sr, n_onsets
    except Exception as e:
        return 0.0, 0, 0


def classify_file(path: Path, yamnet: YamnetWrapper) -> ClassResult:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return yamnet.predict(y, sr)


def decide_output_path(
    dst_root: Path,
    category: str,
    src_path: Path,
    output_format: str = 'same'
) -> Path:
    """Build destination path in target category, keeping base name but adjusting extension if needed."""
    category_dir = dst_root / category
    ensure_dir(category_dir)
    stem = src_path.stem
    ext = src_path.suffix.lower()
    if output_format != 'same':
        ext = f'.{output_format.lower()}'
    return category_dir / f"{stem}{ext}"


def process_one(
    src_path: Path,
    src_root: Path,
    dst_root: Path,
    yamnet: YamnetWrapper,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    rel = src_path.relative_to(src_root)
    duration, sr, n_onsets = analyze_file(src_path)
    if duration <= 0:
        return None
    # Classify
    cr = classify_file(src_path, yamnet)
    # Map
    category = choose_category(cr, duration, n_onsets, min_loop_duration=args.min_loop_duration)
    # Output
    out_path = decide_output_path(dst_root, category, src_path, output_format=args.output_format)

    # Normalize / Copy
    ok = True
    norm_method = args.normalize.lower()
    if norm_method == 'lufs':
        acodec = None
        if args.output_format == 'wav':
            acodec = 'pcm_s16le'  # 16-bit WAV
        ok = ffmpeg_loudnorm_two_pass(src_path, out_path, target_i=args.target_lufs, target_tp=args.true_peak, lra=args.lra, output_acodec=acodec)
    elif norm_method == 'peak':
        acodec = None
        if args.output_format == 'wav':
            acodec = 'pcm_s16le'
        ok = ffmpeg_peak_normalize(src_path, out_path, peak_db=args.peak_db, output_acodec=acodec)
    else:
        # none
        ensure_dir(out_path.parent)
        try:
            shutil.copy2(src_path, out_path)
        except Exception as e:
            ok = False

    if not ok:
        return None

    # Build log entry
    return {
        'source_path': str(src_path),
        'rel_source': str(rel),
        'dest_path': str(out_path),
        'category': category,
        'top1_label': cr.top1_label,
        'top1_score': round(cr.top1_score, 4),
        'top3': '; '.join([f"{l}:{s:.3f}" for l, s in cr.topk]),
        'duration_s': round(duration, 3),
        'sr': sr,
        'n_onsets': n_onsets,
        'normalize': norm_method,
        'target_lufs': args.target_lufs if norm_method == 'lufs' else '',
        'true_peak': args.true_peak if norm_method == 'lufs' else '',
        'peak_db': args.peak_db if norm_method == 'peak' else '',
        'timestamp': int(time.time()),
    }


# -------------------------------
# Main CLI
# -------------------------------

def find_audio_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description='Inhaltsbasierte Audio-Sortierung + LUFS-Normalisierung')
    parser.add_argument('--src', type=str, required=True, help='Quellordner (unsortiert)')
    parser.add_argument('--dst', type=str, required=True, help='Zielordner (Audio_Sortiert)')
    parser.add_argument('--exts', type=str, default='.wav,.aiff,.aif,.mp3,.flac,.m4a', help='Kommagetrennte Erweiterungen')
    parser.add_argument('--normalize', type=str, default='lufs', choices=['none','lufs','peak'], help='Normalisierungsmethode')
    parser.add_argument('--target-lufs', type=float, default=-14.0, help='Ziel-Lautheit in LUFS (EBU R128)')
    parser.add_argument('--true-peak', type=float, default=-1.0, help='Max True Peak dBTP')
    parser.add_argument('--lra', type=float, default=11.0, help='Lautheitsbereich LRA (EBU R128)')
    parser.add_argument('--peak-db', type=float, default=-1.0, help='Ziel-Peak in dBFS (für Peak-Norm)')
    parser.add_argument('--output-format', type=str, default='wav', choices=['same','wav','flac','mp3'], help='Zielformat (Re-Encode)')
    parser.add_argument('--min-loop-duration', type=float, default=2.0, help='Mindestdauer für Loop-Erkennung (s)')
    parser.add_argument('--log', type=str, default='audio_sort_log.csv', help='Pfad zur CSV-Logdatei')
    parser.add_argument('--dry-run', action='store_true', help='Nur anzeigen, nicht schreiben')

    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    ensure_dir(dst_root)

    exts = tuple(s.strip().lower() for s in args.exts.split(','))

    files = find_audio_files(src_root, exts)
    if not files:
        print('Keine Audio-Dateien gefunden.')
        sys.exit(1)

    if not TF_OK:
        print('TensorFlow/TF-Hub nicht installiert. Bitte siehe Kopfzeile für Installation.', file=sys.stderr)
        sys.exit(2)

    yamnet = YamnetWrapper()

    logs: List[Dict[str, object]] = []

    pbar = tqdm(files, desc='Processing', unit='file')
    for f in pbar:
        try:
            if args.dry_run:
                duration, sr, n_onsets = analyze_file(f)
                cr = classify_file(f, yamnet)
                category = choose_category(cr, duration, n_onsets, min_loop_duration=args.min_loop_duration)
                out_path = decide_output_path(dst_root, category, f, output_format=args.output_format)
                pbar.set_postfix(cat=category, label=cr.top1_label[:18])
                logs.append({
                    'source_path': str(f),
                    'dest_path': str(out_path),
                    'category': category,
                    'top1_label': cr.top1_label,
                    'top1_score': round(cr.top1_score, 4),
                    'duration_s': round(duration, 3),
                    'n_onsets': n_onsets,
                    'normalize': 'DRYRUN'
                })
            else:
                entry = process_one(f, src_root, dst_root, yamnet, args)
                if entry:
                    pbar.set_postfix(cat=entry['category'], label=str(entry['top1_label'])[:18])
                    logs.append(entry)
        except KeyboardInterrupt:
            print('\nAbbruch durch Benutzer.')
            break
        except Exception as e:
            print(f"Fehler bei {f}: {e}", file=sys.stderr)
            continue

    # Write CSV log
    if logs and not args.dry_run:
        log_path = Path(args.log)
        with open(log_path, 'w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            for row in logs:
                writer.writerow(row)
        print(f"\nLog gespeichert: {log_path}")

    print("\nFertig.")


if __name__ == '__main__':
    main()
