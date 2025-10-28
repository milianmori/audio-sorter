#!/usr/bin/env python3
"""
Audio Tools — Cleanup + Sort (All-in-One CLI)
=============================================

Run audio cleanup (silence skip/trim, duplicate detection) and/or content-based sorting
with a single command.

Subcommands:
- cleanup: In-place trimming of silence, skip pure-silence files, optional dedupe
- sort: Copy/normalize and categorize into destination folders based on content
- all: First cleanup the source, then sort into destination
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Progress bar (optional)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

# -------------------------------
# Cleanup module (from audio_cleanup.py)
# -------------------------------

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
}

CODEC_FOR_EXTENSION: Dict[str, str] = {
    ".mp3": "libmp3lame",
    ".m4a": "aac",
    ".aac": "aac",
    ".wav": "pcm_s16le",
    ".flac": "flac",
    ".ogg": "libvorbis",
    ".opus": "libopus",
    ".wma": "wmav2",
    ".aiff": "pcm_s16be",
    ".aif": "pcm_s16be",
}

PROBE_TIMEOUT_S: float = 60.0
FFMPEG_TIMEOUT_S: float = 900.0
HASH_TIMEOUT_S: float = 900.0


@dataclass
class AudioFileInfo:
    path: Path
    duration_seconds: float
    has_audio_stream: bool
    mean_volume_db: Optional[float]


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def ensure_binaries_available() -> None:
    if which("ffprobe") is None or which("ffmpeg") is None:
        print(
            "Error: ffprobe/ffmpeg not found. Please install ffmpeg (e.g., `brew install ffmpeg`).",
            file=sys.stderr,
        )
        sys.exit(2)


def run_cmd(cmd: List[str], timeout_s: Optional[float] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            encoding="utf-8",  # ensure stable decoding
            errors="replace",   # avoid UnicodeDecodeError on odd bytes from tools
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="timeout")


def ffprobe_json(path: Path) -> Optional[dict]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    proc = run_cmd(cmd, timeout_s=PROBE_TIMEOUT_S)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def parse_duration_seconds(info: dict) -> Optional[float]:
    fmt = info.get("format") or {}
    dur = fmt.get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    max_dur: Optional[float] = None
    for s in info.get("streams", []) or []:
        d = s.get("duration")
        if d is None:
            continue
        try:
            f = float(d)
        except (TypeError, ValueError):
            continue
        if max_dur is None or f > max_dur:
            max_dur = f
    return max_dur


def has_audio_stream(info: dict) -> bool:
    for s in info.get("streams", []) or []:
        if s.get("codec_type") == "audio":
            return True
    return False


def measure_mean_volume_db(path: Path) -> Optional[float]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        str(path),
        "-map",
        "a:0?",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = run_cmd(cmd, timeout_s=PROBE_TIMEOUT_S)
    if proc.returncode != 0:
        return None
    mean_db: Optional[float] = None
    for line in proc.stderr.splitlines():
        line = line.strip()
        if "mean_volume:" in line:
            try:
                after = line.split("mean_volume:", 1)[1].strip()
                val = after.split(" ")[0]
                if val == "-inf":
                    return float("-inf")
                mean_db = float(val)
            except Exception:
                pass
    return mean_db


def trim_silence(
    src: Path,
    dst: Path,
    noise_threshold_db: float = -50.0,
    min_silence_ms: int = 200,
) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    threshold = f"{noise_threshold_db}dB"
    min_silence_s = max(0.0, float(min_silence_ms) / 1000.0)
    ext = dst.suffix.lower()
    codec = CODEC_FOR_EXTENSION.get(ext, "aac")
    filter_chain = (
        f"silenceremove=start_periods=1:start_threshold={threshold}:start_silence={min_silence_s}:"
        f"stop_periods=1:stop_threshold={threshold}:stop_silence={min_silence_s}"
    )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-map",
        "a:0?",
        "-af",
        filter_chain,
        "-map_metadata",
        "0",
        "-c:a",
        codec,
        str(dst),
    ]
    proc = run_cmd(cmd, timeout_s=FFMPEG_TIMEOUT_S)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def pcm_content_hash(
    path: Path,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    expected_num_bytes: Optional[int] = None,
    timeout_s: Optional[float] = None,
    progress_desc: Optional[str] = None,
) -> Optional[str]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "a:0?",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate_hz),
        "-f",
        "s16le",
        "-",
    ]
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        hasher = hashlib.sha256()
        assert proc.stdout is not None
        start_time = time.time()
        bytes_read = 0
        per_file_bar = None
        try:
            if tqdm is not None and expected_num_bytes is not None:
                per_file_bar = tqdm(total=expected_num_bytes, desc=progress_desc or f"hash {path.name}", unit="B", unit_scale=True, leave=False)
            while True:
                chunk = proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                hasher.update(chunk)
                if per_file_bar is not None:
                    per_file_bar.update(len(chunk))
                if timeout_s is not None and (time.time() - start_time) > timeout_s:
                    proc.kill()
                    proc.wait()
                    return None
            proc.wait()
            if proc.returncode != 0:
                return None
            return hasher.hexdigest()
        finally:
            if per_file_bar is not None:
                per_file_bar.close()
    except KeyboardInterrupt:
        try:
            if proc is not None:
                proc.kill()
                proc.wait()
        except Exception:
            pass
        return None
    except Exception:
        return None


def find_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def make_working_copy(src: Path, dst_root: Path) -> Path:
    """Create a full working copy of src inside dst_root and return its path.

    The working copy is created as a timestamped subdirectory to avoid collisions.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    work_dir = dst_root / f"_source_copy_{ts}"
    # Ensure destination root exists
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    # Copy entire tree (non-destructive to src)
    shutil.copytree(src, work_dir)
    return work_dir


def quantize_ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


def process_in_place(
    input_dir: Path,
    skip_silent: bool,
    trim: bool,
    silence_threshold_db: float,
    min_silence_ms: int,
    silent_move_to: Optional[Path] = None,
    trim_log_rows: Optional[List[List[str]]] = None,
) -> List[Path]:
    processed_paths: List[Path] = []
    files_list = list(find_audio_files(input_dir))
    overall_bar = None
    if tqdm is not None:
        overall_bar = tqdm(total=len(files_list), desc="process", unit="file")
    for src in files_list:
        info = ffprobe_json(src)
        if info is None:
            if overall_bar is not None:
                overall_bar.update(1)
            continue

        if not has_audio_stream(info):
            if overall_bar is not None:
                overall_bar.update(1)
            continue

        mean_db = measure_mean_volume_db(src)
        orig_duration_s: Optional[float] = parse_duration_seconds(info)
        if skip_silent and mean_db == float("-inf"):
            if silent_move_to is not None:
                try:
                    silent_move_to.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                dest = silent_move_to / src.name
                try:
                    shutil.move(str(src), str(dest))
                except Exception:
                    pass
            if overall_bar is not None:
                overall_bar.update(1)
            continue

        if trim:
            tmp_dst = src.with_name(src.stem + ".tmp" + src.suffix)
            ok = trim_silence(
                src,
                tmp_dst,
                noise_threshold_db=silence_threshold_db,
                min_silence_ms=min_silence_ms,
            )
            if ok and tmp_dst.exists():
                info_after = ffprobe_json(tmp_dst)
                after_duration_s: Optional[float] = parse_duration_seconds(info_after) if info_after else None
                try:
                    os.replace(str(tmp_dst), str(src))
                    processed_paths.append(src)
                    trimmed_ms: Optional[int] = None
                    if orig_duration_s is not None and after_duration_s is not None:
                        delta_ms = int(round((orig_duration_s - after_duration_s) * 1000.0))
                        trimmed_ms = max(0, delta_ms)
                    if trim_log_rows is not None:
                        row = [
                            str(src),
                            f"{orig_duration_s:.3f}" if orig_duration_s is not None else "",
                            f"{after_duration_s:.3f}" if after_duration_s is not None else "",
                            str(trimmed_ms) if trimmed_ms is not None else "",
                            f"{silence_threshold_db}",
                            str(min_silence_ms),
                            ("-inf" if mean_db == float("-inf") else (f"{mean_db:.2f}" if mean_db is not None else "")),
                        ]
                        trim_log_rows.append(row)
                except Exception:
                    try:
                        tmp_dst.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                try:
                    tmp_dst.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            processed_paths.append(src)

        if overall_bar is not None:
            overall_bar.update(1)

    if overall_bar is not None:
        overall_bar.close()
    return processed_paths


def dedupe_similar(
    target_dir: Path,
    name_similarity_threshold: float,
    duration_tolerance_ms: int,
    delete: bool,
    move_to: Optional[Path] = None,
) -> List[Tuple[Path, Path]]:
    files = list(find_audio_files(target_dir))
    if not files:
        return []

    infos: Dict[Path, AudioFileInfo] = {}
    for p in files:
        info = ffprobe_json(p)
        if info is None:
            continue
        dur = parse_duration_seconds(info)
        if dur is None:
            continue
        mean_db = measure_mean_volume_db(p)
        infos[p] = AudioFileInfo(
            path=p,
            duration_seconds=dur,
            has_audio_stream=has_audio_stream(info),
            mean_volume_db=mean_db,
        )

    by_duration: Dict[int, List[AudioFileInfo]] = {}
    for afi in infos.values():
        q = quantize_ms(afi.duration_seconds)
        by_duration.setdefault(q, []).append(afi)

    duplicates: List[Tuple[Path, Path]] = []
    hash_cache: Dict[Path, Optional[str]] = {}
    cache_path = (target_dir / ".audio_cleanup_cache.json")
    persistent_cache: Dict[str, Dict[str, float | str | int]] = {}
    try:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
                if isinstance(loaded, dict):
                    persistent_cache = loaded  # type: ignore
    except Exception:
        persistent_cache = {}

    def file_signature(p: Path) -> Tuple[float, int]:
        st = p.stat()
        return (st.st_mtime, st.st_size)

    hashed_seen: Dict[Path, bool] = {}
    overall_hash_bar = None
    if tqdm is not None:
        overall_hash_bar = tqdm(total=len(infos), desc="hash", unit="file")

    for qdur, group in by_duration.items():
        if len(group) < 2:
            continue
        candidates: List[AudioFileInfo] = []
        for delta in range(-duration_tolerance_ms, duration_tolerance_ms + 1, 1):
            candidates.extend(by_duration.get(qdur + delta, []))
        if len(candidates) < 2:
            continue

        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                if a.path == b.path:
                    continue
                if not a.path.exists() or not b.path.exists():
                    hash_cache.pop(a.path, None)
                    hash_cache.pop(b.path, None)
                    continue

                def ensure_hash(p: Path, dur_seconds: float) -> Optional[str]:
                    existing = hash_cache.get(p)
                    if existing is not None:
                        return existing
                    key = str(p.resolve())
                    try:
                        mtime, size = file_signature(p)
                    except FileNotFoundError:
                        return None
                    cached = persistent_cache.get(key)
                    if (
                        isinstance(cached, dict)
                        and float(cached.get("mtime", -1)) == mtime
                        and int(cached.get("size", -1)) == size
                        and isinstance(cached.get("hash"), str)
                    ):
                        h = str(cached["hash"])  # type: ignore
                        hash_cache[p] = h
                        return h
                    bytes_total = int(dur_seconds * 16000 * 1 * 2)
                    h = pcm_content_hash(
                        p,
                        sample_rate_hz=16000,
                        channels=1,
                        expected_num_bytes=bytes_total,
                        timeout_s=HASH_TIMEOUT_S,
                        progress_desc=f"hash {p.name}",
                    )
                    hash_cache[p] = h
                    if h is not None:
                        persistent_cache[key] = {"mtime": mtime, "size": size, "hash": h}
                    if p not in hashed_seen and overall_hash_bar is not None:
                        overall_hash_bar.update(1)
                        hashed_seen[p] = True
                    return h

                ha = ensure_hash(a.path, a.duration_seconds)
                hb = ensure_hash(b.path, b.duration_seconds)
                if ha is None or hb is None:
                    continue
                if ha != hb:
                    continue
                if not a.path.exists() or not b.path.exists():
                    hash_cache.pop(a.path, None)
                    hash_cache.pop(b.path, None)
                    continue
                if a.path.stat().st_mtime <= b.path.stat().st_mtime:
                    keep, drop = a.path, b.path
                else:
                    keep, drop = b.path, a.path
                if delete and drop.exists():
                    if move_to is not None:
                        move_to.mkdir(parents=True, exist_ok=True)
                        dest = move_to / drop.name
                        try:
                            shutil.move(str(drop), str(dest))
                        except Exception:
                            pass
                        hash_cache.pop(drop, None)
                        persistent_cache.pop(str(drop.resolve()), None)
                    else:
                        try:
                            drop.unlink(missing_ok=True)
                        except Exception:
                            pass
                        hash_cache.pop(drop, None)
                        persistent_cache.pop(str(drop.resolve()), None)
                duplicates.append((keep, drop))

    if overall_hash_bar is not None:
        overall_hash_bar.close()
    try:
        with open(cache_path, "w", encoding="utf-8") as fp:
            json.dump(persistent_cache, fp)
    except Exception:
        pass
    return duplicates


# -------------------------------
# Sorter module (from audio-sorter.py)
# -------------------------------

import librosa

try:
    import tensorflow_hub as hub  # type: ignore
    import tensorflow as tf  # type: ignore
    TF_OK = True
except Exception:
    hub = None  # type: ignore
    tf = None  # type: ignore
    TF_OK = False

CATS = {
    "1_Kicks": ["Acoustic", "Electronic", "808", "909", "Distorted", "Layered"],
    "2_Snares_Claps": ["Snares_Acoustic", "Snares_Electronic", "Rimshot", "Claps", "Rolls"],
    "3_HiHats": ["Closed", "Open", "Shaker"],
    "4_Percussion": ["Cymbals", "Toms", "Bells", "Wood", "Metal", "Ethnic", "Foley"],
    "5_Synth": ["Bass", "Lead", "Pad", "Pluck", "Keys", "Arp", "Stabs", "Chords", "FX"],
    "6_OneShots": ["Tonal", "Atonal", "Impacts", "Risers", "Downers", "Transitions", "Textures"],
    "7_Vocals": ["OneShots", "Phrases", "Adlibs", "FX", "Spoken", "Sung"],
    "8_Field_Recordings": ["Nature", "Urban", "Transport", "Water", "Animals", "Ambience", "RoomTone"],
}

KW = {
    ("1_Kicks", "808"): [r"\b808\b"],
    ("1_Kicks", "909"): [r"\b909\b"],
    ("1_Kicks", "Distorted"): [r"distort", r"crush", r"dirty", r"drive"],
    ("1_Kicks", "Layered"): [r"layer"],
    ("1_Kicks", "Acoustic"): [r"acoustic", r"live", r"room"],
    ("1_Kicks", "Electronic"): [r"\bkick\b", r"\bbd\b", r"bass\s?drum", r"drum\s?machine"],
    ("2_Snares_Claps", "Rimshot"): [r"\brim(\s?shot)?\b"],
    ("2_Snares_Claps", "Claps"): [r"\bclap(s)?\b", r"hand\s?clap"],
    ("2_Snares_Claps", "Rolls"): [r"\broll\b", r"buzz"],
    ("2_Snares_Claps", "Snares_Acoustic"): [r"snare.*(acoustic|live|room)", r"acoustic.*snare"],
    ("2_Snares_Claps", "Snares_Electronic"): [r"\bsnare\b", r"\bsd\b"],
    ("3_HiHats", "Closed"): [r"(closed\s?hat|\bchh\b|\bhhc\b)"],
    ("3_HiHats", "Open"): [r"(open\s?hat|\bohh\b|\bhho\b)"],
    ("3_HiHats", "Shaker"): [r"\bshaker\b"],
    ("4_Percussion", "Cymbals"): [r"cymbal|crash|ride|splash|china"],
    ("4_Percussion", "Toms"): [r"\btom(s)?\b"],
    ("4_Percussion", "Bells"): [r"bell|glock|tubular"],
    ("4_Percussion", "Wood"): [r"wood(\s?block)?|clave|claves"],
    ("4_Percussion", "Metal"): [r"metal|iron|anvil|steel|sheet"],
    ("4_Percussion", "Ethnic"): [r"bongo|conga|djembe|tabla|udu|cuica|taiko|dar(b|p)uka"],
    ("4_Percussion", "Foley"): [r"foley|coin|door|paper|keys|steps|whoosh"],
    ("5_Synth", "Bass"): [r"\bbass\b", r"\bsub\b"],
    ("5_Synth", "Lead"): [r"\blead\b"],
    ("5_Synth", "Pad"): [r"\bpad\b"],
    ("5_Synth", "Pluck"): [r"pluck"],
    ("5_Synth", "Keys"): [r"keys|piano|rhodes|clav"],
    ("5_Synth", "Arp"): [r"\barp\b|arpeggio"],
    ("5_Synth", "Stabs"): [r"stab(s)?\b"],
    ("5_Synth", "Chords"): [r"chord(s)?\b"],
    ("5_Synth", "FX"): [r"\bfx\b|sfx"],
    ("6_OneShots", "Impacts"): [r"impact|boom|hit|slam|thud|explosion|exploder"],
    ("6_OneShots", "Risers"): [r"riser|rise|uplift"],
    ("6_OneShots", "Downers"): [r"downer|fall|drop"],
    ("6_OneShots", "Transitions"): [r"transition|whoosh|swoosh|swipe"],
    ("6_OneShots", "Textures"): [r"texture|drone|granular|noise"],
    ("6_OneShots", "Tonal"): [r"tonal|note|pitch"],
    ("6_OneShots", "Atonal"): [r"atonal"],
    ("7_Vocals", "Spoken"): [r"spoken|speech|talk|words"],
    ("7_Vocals", "Sung"): [r"sung|sing(ing)?|choir|vocal\s?melody"],
    ("7_Vocals", "FX"): [r"vocal\s?fx|voxfx|\bvox\b"],
    ("7_Vocals", "Adlibs"): [r"ad[- ]?lib|uh+|oh+|hey+|yeah+"],
    ("7_Vocals", "Phrases"): [r"phrase|sentence|line"],
    ("7_Vocals", "OneShots"): [r"vocal.*(stab|one[- ]?shot)"],
    ("8_Field_Recordings", "Nature"): [r"wind|storm|thunder|forest|birds?ong"],
    ("8_Field_Recordings", "Urban"): [r"street|city|urban|crowd|market"],
    ("8_Field_Recordings", "Transport"): [r"car|bus|train|plane|tram|subway|metro"],
    ("8_Field_Recordings", "Water"): [r"water|ocean|sea|river|stream|rain|rainfall|drip"],
    ("8_Field_Recordings", "Animals"): [r"bird|dog|cat|insect|cow|sheep|animal"],
    ("8_Field_Recordings", "Ambience"): [r"ambience|ambient|atmo|atmosphere|room\s?amb"],
    ("8_Field_Recordings", "RoomTone"): [r"room[- ]?tone|silence|bg\s?noise|background"],
}

PRIORITY = [
    ("1_Kicks", "808"), ("1_Kicks", "909"), ("1_Kicks", "Distorted"), ("1_Kicks", "Layered"),
    ("2_Snares_Claps", "Rimshot"), ("2_Snares_Claps", "Claps"), ("2_Snares_Claps", "Rolls"),
    ("3_HiHats", "Shaker"), ("3_HiHats", "Open"), ("3_HiHats", "Closed"),
    ("4_Percussion", "Ethnic"), ("4_Percussion", "Cymbals"), ("4_Percussion", "Toms"),
    ("4_Percussion", "Bells"), ("4_Percussion", "Metal"), ("4_Percussion", "Wood"), ("4_Percussion", "Foley"),
    ("5_Synth", "Arp"), ("5_Synth", "Pluck"), ("5_Synth", "Stabs"), ("5_Synth", "Chords"),
    ("5_Synth", "Pad"), ("5_Synth", "Lead"), ("5_Synth", "Bass"), ("5_Synth", "Keys"), ("5_Synth", "FX"),
    ("6_OneShots", "Impacts"), ("6_OneShots", "Risers"), ("6_OneShots", "Downers"), ("6_OneShots", "Transitions"), ("6_OneShots", "Textures"), ("6_OneShots", "Tonal"), ("6_OneShots", "Atonal"),
    ("7_Vocals", "Spoken"), ("7_Vocals", "Sung"), ("7_Vocals", "Adlibs"), ("7_Vocals", "FX"), ("7_Vocals", "Phrases"), ("7_Vocals", "OneShots"),
    ("8_Field_Recordings", "Nature"), ("8_Field_Recordings", "Water"), ("8_Field_Recordings", "Urban"), ("8_Field_Recordings", "Transport"), ("8_Field_Recordings", "Animals"), ("8_Field_Recordings", "Ambience"), ("8_Field_Recordings", "RoomTone"),
]

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(name: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(name)]


@dataclass
class Features:
    duration: float
    sr: int
    n_onsets: int
    perc_ratio: float
    harm_ratio: float
    zcr: float
    centroid: float
    rolloff: float
    flatness: float
    low_ratio: float
    mid_ratio: float
    high_ratio: float


def extract_features(path: Path) -> Optional[Features]:
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return None
        harm, perc = librosa.effects.hpss(y)
        energy = np.sum(y**2) + 1e-12
        perc_energy = np.sum(perc**2)
        harm_energy = np.sum(harm**2)
        perc_ratio = float(perc_energy / energy)
        harm_ratio = float(harm_energy / energy)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        n_onsets = int(len(onsets))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        S_lin = np.abs(librosa.stft(y))**2
        freqs = librosa.fft_frequencies(sr=sr)
        def band_ratio(fmin, fmax):
            mask = (freqs >= fmin) & (freqs < fmax)
            if not np.any(mask):
                return 0.0
            num = float(np.sum(S_lin[mask, :]))
            den = float(np.sum(S_lin) + 1e-12)
            return num / den
        low_ratio = band_ratio(20, 200)
        mid_ratio = band_ratio(200, 2000)
        high_ratio = band_ratio(2000, sr/2)
        return Features(duration, sr, n_onsets, perc_ratio, harm_ratio, zcr, centroid, rolloff, flatness, low_ratio, mid_ratio, high_ratio)
    except Exception:
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def category_prefix(cat: str) -> str:
    return cat.split('_', 1)[1] if '_' in cat else cat


def unique_target(base: Path) -> Path:
    if not base.exists():
        return base
    stem, ext = base.stem, base.suffix
    n = 1
    while True:
        cand = base.with_name(f"{stem}_{n}{ext}")
        if not cand.exists():
            return cand
        n += 1


REGEX_CACHE: Dict[Tuple[str, str], List[re.Pattern]] = {}
for key, pats in KW.items():
    REGEX_CACHE[key] = [re.compile(p, re.I) for p in pats]


def match_keywords(name: str) -> Optional[Tuple[str, str, str]]:
    tokens = tokenize(name)
    s = " ".join(tokens)
    for cat_sub in PRIORITY:
        pats = REGEX_CACHE.get(cat_sub, [])
        for rx in pats:
            if rx.search(s):
                return (cat_sub[0], cat_sub[1], f"kw:{rx.pattern}")
    return None


def dbfs_to_linear(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


SUPPORTED_NORMALIZE_EXTS = {".wav", ".aiff", ".aif", ".flac"}


def peak_normalize_array(y: np.ndarray, target_dbfs: float = -1.0) -> Tuple[np.ndarray, float, float]:
    if y is None or y.size == 0:
        return y, 0.0, 0.0
    peak_before = float(np.max(np.abs(y)))
    if peak_before <= 0.0:
        return y, peak_before, peak_before
    target_peak = dbfs_to_linear(target_dbfs)
    scale = target_peak / peak_before
    y_out = y * scale
    y_out = np.clip(y_out, -1.0, 1.0)
    peak_after = float(np.max(np.abs(y_out)))
    return y_out, peak_before, peak_after


def try_normalize_file_to_target(src: Path, dst: Path, target_dbfs: float = -1.0) -> Tuple[bool, str, float, float]:
    ext = src.suffix.lower()
    if ext not in SUPPORTED_NORMALIZE_EXTS:
        return (False, "unsupported_ext", 0.0, 0.0)
    try:
        y, sr = sf.read(str(src), dtype='float32')
    except Exception:
        return (False, "sf_read_failed", 0.0, 0.0)
    y_norm, pk_before, pk_after = peak_normalize_array(y, target_dbfs=target_dbfs)
    try:
        dst_ext = dst.suffix.lower()
        if dst_ext in ('.aif', '.aiff'):
            sf.write(str(dst), y_norm, sr, format='AIFF', subtype='PCM_24')
        elif dst_ext == '.wav':
            sf.write(str(dst), y_norm, sr, format='WAV', subtype='PCM_24')
        elif dst_ext == '.flac':
            sf.write(str(dst), y_norm, sr, format='FLAC', subtype='PCM_24')
        else:
            sf.write(str(dst), y_norm, sr)
        try:
            shutil.copystat(src, dst)
        except Exception:
            pass
        return (True, "ok", pk_before, pk_after)
    except Exception:
        return (False, "sf_write_failed", pk_before, pk_after)


def classify_by_heuristics(f: Features) -> Tuple[str, str, str]:
    dur = f.duration
    low = f.low_ratio
    mid = f.mid_ratio
    high = f.high_ratio
    cent = f.centroid
    flat = f.flatness
    zcr = f.zcr
    on = f.n_onsets
    perc = f.perc_ratio
    harm = f.harm_ratio

    if perc < 0.3 and on <= 1 and dur > 5.0 and flat > 0.3:
        return ("8_Field_Recordings", "Ambience", "low-perc long noisy ambience")
    if perc >= 0.4 and on <= 6 and dur <= 5.0:
        if low > 0.55 and cent < 1200 and on <= 2:
            return ("1_Kicks", "Electronic", "low-heavy percussive short")
        if (mid > 0.35 or high > 0.25) and flat > 0.2:
            if high > mid and dur < 1.0:
                return ("2_Snares_Claps", "Claps", "percussive high short noisy")
            else:
                return ("2_Snares_Claps", "Snares_Electronic", "percussive mid-high noisy")
        if high > 0.45 and cent > 4000 and zcr > 0.1:
            if on >= 3:
                return ("3_HiHats", "Shaker", "many fine onsets high freq")
            else:
                return ("3_HiHats", "Closed", "bright percussive hat-like")
        if low > 0.25 and mid > 0.25 and harm > 0.3 and on <= 3:
            return ("4_Percussion", "Toms", "low-mid tonal percussive")
        if high > 0.5 and dur > 1.5 and flat > 0.25:
            return ("4_Percussion", "Cymbals", "high energy long decay")
    if harm >= 0.5:
        if low > 0.5 and cent < 800:
            return ("5_Synth", "Bass", "harmonic low-dominant")
        if dur > 2.5 and on <= 2 and flat < 0.2:
            return ("5_Synth", "Pad", "long sustained harmonic")
        if dur <= 1.2 and on <= 2 and cent > 1500:
            return ("5_Synth", "Pluck", "short bright harmonic")
        if cent >= 1200 and on <= 3:
            return ("5_Synth", "Lead", "mid-high harmonic single")
        if dur <= 1.0 and on <= 2:
            return ("5_Synth", "Stabs", "short harmonic chord-like")
        if dur > 1.0 and on <= 4:
            return ("5_Synth", "Chords", "sustained harmonic chord")
    if perc >= 0.3 and dur >= 0.8 and on <= 4 and flat >= 0.25 and low < 0.4 and high >= 0.3:
        return ("6_OneShots", "Impacts", "broadband impact-like")
    if dur >= 1.0 and on <= 3 and high >= 0.25 and flat >= 0.2:
        if zcr < 0.05:
            return ("6_OneShots", "Textures", "noisy sustained texture")
        else:
            return ("6_OneShots", "Transitions", "noisy transitional")
    if dur > 4.0 and perc < 0.4:
        return ("8_Field_Recordings", "Ambience", "fallback ambience")
    if perc >= 0.4:
        return ("4_Percussion", "Foley", "fallback percussive")
    return ("6_OneShots", "Atonal", "fallback atonal")


class YamnetWrap:
    def __init__(self):
        if not TF_OK:
            raise RuntimeError("TensorFlow/Hub nicht verfügbar.")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path) as f:
            rdr = csv.DictReader(f)
            self.labels = [row['display_name'].lower() for row in rdr]

    def infer(self, path: Path) -> Optional[str]:
        try:
            y, sr = librosa.load(str(path), sr=16000, mono=True)
            y = y.astype(np.float32)
            scores, _, _ = self.model(y)
            ms = scores.numpy().mean(axis=0)
            idx = int(np.argmax(ms))
            return self.labels[idx]
        except Exception:
            return None


def map_yamnet_label(lbl: Optional[str]) -> Optional[Tuple[str, str, str]]:
    if not lbl:
        return None
    l = lbl.lower()
    if any(k in l for k in ["speech", "talk", "conversation", "narration"]):
        return ("7_Vocals", "Spoken", f"yamnet:{l}")
    if any(k in l for k in ["singing", "choir", "chant"]):
        return ("7_Vocals", "Sung", f"yamnet:{l}")
    if "bass drum" in l or "kick drum" in l:
        return ("1_Kicks", "Electronic", f"yamnet:{l}")
    if "snare" in l:
        return ("2_Snares_Claps", "Snares_Electronic", f"yamnet:{l}")
    if "hi-hat" in l or "cymbal" in l:
        return ("3_HiHats", "Closed", f"yamnet:{l}")
    if "tom-tom" in l:
        return ("4_Percussion", "Toms", f"yamnet:{l}")
    if any(k in l for k in ["rain", "water", "ocean", "stream", "river"]):
        return ("8_Field_Recordings", "Water", f"yamnet:{l}")
    if any(k in l for k in ["wind", "thunder", "storm"]):
        return ("8_Field_Recordings", "Nature", f"yamnet:{l}")
    if any(k in l for k in ["vehicle", "car", "train", "airplane", "traffic", "urban", "crowd"]):
        return ("8_Field_Recordings", "Urban", f"yamnet:{l}")
    if any(k in l for k in ["animal", "bird", "dog", "cat", "insect"]):
        return ("8_Field_Recordings", "Animals", f"yamnet:{l}")
    if "music" in l or "instrument" in l:
        return ("5_Synth", "FX", f"yamnet:{l}")
    return None


def decide_category(path: Path, use_ml: bool) -> Tuple[str, str, str]:
    m = match_keywords(path.name)
    if m:
        return m
    if use_ml and TF_OK:
        try:
            yw = YamnetWrap()
            lbl = yw.infer(path)
            mm = map_yamnet_label(lbl)
            if mm:
                return mm
        except Exception:
            pass
    feat = extract_features(path)
    if feat:
        return classify_by_heuristics(feat)
    return ("6_OneShots", "Atonal", "fallback")


def iter_progress(items: List[Path], desc: str) -> Iterable[Path]:
    if tqdm is None:
        for x in items:
            yield x
    else:
        for x in tqdm(items, unit='file', desc=desc):
            yield x


def run_sort(src: Path, dst: Path, use_ml: bool, copy_only: bool, log_path: Path, exts_csv: str) -> None:
    ensure_dir(dst)
    exts = tuple(e.strip().lower() for e in exts_csv.split(','))
    files = [p for p in src.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print('Keine Audiodateien gefunden.')
        sys.exit(1)
    log_rows: List[List[str]] = []
    header = [
        'source_path','dest_path','category','subcategory','reason','duration','sr','n_onsets','perc_ratio','harm_ratio','zcr','centroid','rolloff','flatness','low_ratio','mid_ratio','high_ratio',
        'normalized','peak_before','peak_after','target_dbfs'
    ]
    log_rows.append(header)

    for f in iter_progress(files, desc='Sorting'):
        is_export = any(parent.name.lower() in ("export", "exports") for parent in f.parents)
        if is_export:
            target_dir = dst / "exports"
            ensure_dir(target_dir)
            target = unique_target(target_dir / f.name)
            normalized = False
            norm_pk_before = 0.0
            norm_pk_after = 0.0
            norm_target_dbfs = -1.0
            ok, norm_reason, norm_pk_before, norm_pk_after = try_normalize_file_to_target(f, target, target_dbfs=norm_target_dbfs)
            if ok:
                normalized = True
            else:
                try:
                    shutil.copy2(f, target)
                except Exception as e:
                    print(f"Fehler bei {f}: {e}")
                    continue
            feat = extract_features(target)
            row = [
                str(f), str(target), "exports", "", "from_export_folder"
            ]
            if feat:
                row += [
                    f"{feat.duration:.3f}", str(feat.sr), str(feat.n_onsets),
                    f"{feat.perc_ratio:.3f}", f"{feat.harm_ratio:.3f}", f"{feat.zcr:.4f}",
                    f"{feat.centroid:.1f}", f"{feat.rolloff:.1f}", f"{feat.flatness:.3f}",
                    f"{feat.low_ratio:.3f}", f"{feat.mid_ratio:.3f}", f"{feat.high_ratio:.3f}",
                ]
            else:
                row += ["", "", "", "", "", "", "", "", "", "", "", ""]
            row += [
                "1" if normalized else "0",
                f"{norm_pk_before:.6f}" if normalized and norm_pk_before > 0 else "",
                f"{norm_pk_after:.6f}" if normalized and norm_pk_after > 0 else "",
                f"{norm_target_dbfs:.2f}"
            ]
            log_rows.append(row)
            continue

        cat, sub, reason = decide_category(f, use_ml=use_ml)
        cat_prefix = category_prefix(cat)
        target_dir = dst / cat / sub
        ensure_dir(target_dir)
        new_name = f"{cat_prefix}_{sub}_{f.stem}{f.suffix}"
        target = target_dir / new_name
        target = unique_target(target)
        normalized = False
        norm_pk_before = 0.0
        norm_pk_after = 0.0
        norm_target_dbfs = -1.0
        ok, norm_reason, norm_pk_before, norm_pk_after = try_normalize_file_to_target(f, target, target_dbfs=norm_target_dbfs)
        if ok:
            normalized = True
        else:
            try:
                shutil.copy2(f, target)
            except Exception as e:
                print(f"Fehler bei {f}: {e}")
                continue
        feat = extract_features(target)
        row = [
            str(f), str(target), cat, sub, reason
        ]
        if feat:
            row += [
                f"{feat.duration:.3f}", str(feat.sr), str(feat.n_onsets),
                f"{feat.perc_ratio:.3f}", f"{feat.harm_ratio:.3f}", f"{feat.zcr:.4f}",
                f"{feat.centroid:.1f}", f"{feat.rolloff:.1f}", f"{feat.flatness:.3f}",
                f"{feat.low_ratio:.3f}", f"{feat.mid_ratio:.3f}", f"{feat.high_ratio:.3f}",
            ]
        else:
            row += ["", "", "", "", "", "", "", "", "", "", "", ""]
        row += [
            "1" if normalized else "0",
            f"{norm_pk_before:.6f}" if normalized and norm_pk_before > 0 else "",
            f"{norm_pk_after:.6f}" if normalized and norm_pk_after > 0 else "",
            f"{norm_target_dbfs:.2f}"
        ]
        log_rows.append(row)

    with open(log_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerows(log_rows)
    print(f"\nLog gespeichert: {Path(log_path).resolve()}")


# -------------------------------
# Unified CLI
# -------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Audio Tools: cleanup and sort")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("cleanup", help="Trim/skip silence and dedupe without touching originals (use --dst)")
    p_clean.add_argument("input", type=Path, help="Input directory to scan")
    p_clean.add_argument("--dst", type=Path, help="Destination root where a working copy of INPUT will be created")
    p_clean.add_argument("--no-skip-silent", action="store_true", help="Do not skip pure digital silence")
    p_clean.add_argument("--no-trim", action="store_true", help="Do not trim leading/trailing silence")
    p_clean.add_argument("--silence-threshold-db", type=float, default=-50.0)
    p_clean.add_argument("--min-silence-ms", type=int, default=200)
    p_clean.add_argument("--dedupe", action="store_true", help="Search duplicates after processing")
    p_clean.add_argument("--dedupe-path", type=Path, help="Directory to dedupe (default: input)")
    p_clean.add_argument("--name-similarity", type=float, default=0.85)
    p_clean.add_argument("--duration-tolerance-ms", type=int, default=50)
    p_clean.add_argument("--delete-duplicates", action="store_true")
    p_clean.add_argument("--move-duplicates-to", type=Path)
    p_clean.add_argument("--probe-timeout-s", type=float, default=60.0)
    p_clean.add_argument("--ffmpeg-timeout-s", type=float, default=900.0)
    p_clean.add_argument("--hash-timeout-s", type=float, default=900.0)

    p_sort = sub.add_parser("sort", help="Classify and copy/normalize to destination")
    p_sort.add_argument("--src", required=True)
    p_sort.add_argument("--dst", required=True)
    p_sort.add_argument("--ml", action='store_true', help='Use YAMNet if available')
    p_sort.add_argument("--copy", action='store_true', help='Copy instead of move (not used; we always copy/normalize)')
    p_sort.add_argument("--log", default='audio_sort_log.csv')
    p_sort.add_argument("--exts", default='.wav,.aiff,.aif,.mp3,.flac,.m4a')

    p_all = sub.add_parser("all", help="Cleanup source, then sort into destination")
    p_all.add_argument("--src", required=True)
    p_all.add_argument("--dst", required=True)
    p_all.add_argument("--ml", action='store_true')
    p_all.add_argument("--log", default='audio_sort_log.csv')
    p_all.add_argument("--exts", default='.wav,.aiff,.aif,.mp3,.flac,.m4a')
    p_all.add_argument("--no-skip-silent", action="store_true")
    p_all.add_argument("--no-trim", action="store_true")
    p_all.add_argument("--silence-threshold-db", type=float, default=-50.0)
    p_all.add_argument("--min-silence-ms", type=int, default=200)
    p_all.add_argument("--dedupe", action="store_true")
    p_all.add_argument("--duration-tolerance-ms", type=int, default=50)
    p_all.add_argument("--delete-duplicates", action="store_true")

    args = parser.parse_args()

    if args.cmd == "cleanup":
        ensure_binaries_available()
        input_dir: Path = Path(args.input).resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Error: input directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
            return 2
        global PROBE_TIMEOUT_S, FFMPEG_TIMEOUT_S, HASH_TIMEOUT_S
        PROBE_TIMEOUT_S = float(args.probe_timeout_s)
        FFMPEG_TIMEOUT_S = float(args.ffmpeg_timeout_s)
        HASH_TIMEOUT_S = float(args.hash_timeout_s)
        skip_silent = not args.no_skip_silent
        trim = not args.no_trim

        # Decide processing root: prefer working copy inside --dst if provided
        work_root: Path
        if getattr(args, "dst", None):
            dst_root = Path(args.dst).expanduser().resolve()
            try:
                work_root = make_working_copy(input_dir, dst_root)
                print(f"Working copy created: {work_root}")
            except Exception as e:
                print(f"Error creating working copy in {dst_root}: {e}", file=sys.stderr)
                return 2
        else:
            # Back-compat: if --dst not provided, operate in-place (original behavior)
            work_root = input_dir

        default_move_dir: Optional[Path] = (
            (work_root / "tobedeleted") if args.delete_duplicates and args.move_duplicates_to is None else args.move_duplicates_to
        )
        trim_log_rows: List[List[str]] = [[
            "source_path","duration_before_s","duration_after_s","trimmed_ms","threshold_db","min_silence_ms","mean_volume_db",
        ]]
        process_in_place(
            input_dir=work_root,
            skip_silent=skip_silent,
            trim=trim,
            silence_threshold_db=args.silence_threshold_db,
            min_silence_ms=args.min_silence_ms,
            silent_move_to=default_move_dir,
            trim_log_rows=trim_log_rows,
        )
        if args.dedupe or args.delete_duplicates:
            target_dir = (args.dedupe_path or work_root).resolve()
            default_move_dir = (work_root / "tobedeleted") if args.delete_duplicates and args.move_duplicates_to is None else args.move_duplicates_to
            dedupe_similar(
                target_dir=target_dir,
                name_similarity_threshold=args.name_similarity,
                duration_tolerance_ms=args.duration_tolerance_ms,
                delete=args.delete_duplicates,
                move_to=default_move_dir,
            )
        log_path = work_root / "log.csv"
        try:
            with open(log_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerows(trim_log_rows)
            print(f"Trim log written: {log_path.resolve()}")
        except Exception as e:
            print(f"[error] failed to write trim log to {log_path}: {e}")
        return 0

    if args.cmd == "sort":
        src = Path(args.src).expanduser().resolve()
        dst = Path(args.dst).expanduser().resolve()
        run_sort(src=src, dst=dst, use_ml=bool(args.ml), copy_only=bool(args.copy), log_path=Path(args.log), exts_csv=str(args.exts))
        return 0

    if args.cmd == "all":
        ensure_binaries_available()
        src = Path(args.src).expanduser().resolve()
        dst = Path(args.dst).expanduser().resolve()
        skip_silent = not args.no_skip_silent
        trim = not args.no_trim
        # Create working copy of src inside dst before any processing
        try:
            work_src = make_working_copy(src, dst)
            print(f"Working copy created for 'all': {work_src}")
        except Exception as e:
            print(f"Error creating working copy in {dst}: {e}", file=sys.stderr)
            return 2
        trim_log_rows: List[List[str]] = [[
            "source_path","duration_before_s","duration_after_s","trimmed_ms","threshold_db","min_silence_ms","mean_volume_db",
        ]]
        process_in_place(
            input_dir=work_src,
            skip_silent=skip_silent,
            trim=trim,
            silence_threshold_db=args.silence_threshold_db,
            min_silence_ms=args.min_silence_ms,
            silent_move_to=(work_src / "tobedeleted") if args.delete_duplicates else None,
            trim_log_rows=trim_log_rows,
        )
        if args.dedupe or args.delete_duplicates:
            dedupe_similar(
                target_dir=work_src,
                name_similarity_threshold=0.85,
                duration_tolerance_ms=args.duration_tolerance_ms,
                delete=args.delete_duplicates,
                move_to=(work_src / "tobedeleted") if args.delete_duplicates else None,
            )
        try:
            with open((work_src / "log.csv"), "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerows(trim_log_rows)
        except Exception:
            pass
        run_sort(src=work_src, dst=dst, use_ml=bool(args.ml), copy_only=False, log_path=Path(args.log), exts_csv=str(args.exts))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


