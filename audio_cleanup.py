import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )


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
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def parse_duration_seconds(info: dict) -> Optional[float]:
    # Prefer container duration
    fmt = info.get("format") or {}
    dur = fmt.get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    # Fallback to max stream duration
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
    # Use volumedetect to estimate mean volume; -inf means pure digital silence
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
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        return None
    mean_db: Optional[float] = None
    for line in proc.stderr.splitlines():
        line = line.strip()
        if "mean_volume:" in line:
            # Example: mean_volume: -23.0 dB
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
    # Choose codec by extension to keep format sensible
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
    proc = run_cmd(cmd)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def pcm_content_hash(
    path: Path,
    sample_rate_hz: int = 16000,
    channels: int = 1,
) -> Optional[str]:
    # Decode to a canonical PCM stream and hash it
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
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        hasher = hashlib.sha256()
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        proc.wait()
        if proc.returncode != 0:
            return None
        return hasher.hexdigest()
    except Exception:
        return None


def find_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def normalize_name_for_similarity(name: str) -> str:
    base = name.lower().strip()
    # Remove trailing date-time suffix like "[2024-08-10 151650]" optionally preceded by space/underscore/dash
    # Accept optional 'T' between date and time as well
    base = re.sub(r"(?:[ _-]*)\[\d{4}-\d{2}-\d{2}[ t]?\d{6}\]$", "", base)
    base = base.strip(" _-")
    # Remove common punctuation and extra spaces
    keep = []
    for ch in base:
        if ch.isalnum() or ch in {" ", "-", "_"}:
            keep.append(ch)
    simplified = "".join(keep)
    while "  " in simplified:
        simplified = simplified.replace("  ", " ")
    return simplified.strip()


def similarity_ratio(a: str, b: str) -> float:
    # Lightweight similarity using SequenceMatcher
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


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
    for src in find_audio_files(input_dir):
        info = ffprobe_json(src)
        if info is None:
            print(f"[skip] ffprobe failed: {src}")
            continue

        if not has_audio_stream(info):
            print(f"[skip] no audio stream: {src}")
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
                print(f"[silent->move] {src} -> {dest}")
                try:
                    shutil.move(str(src), str(dest))
                except Exception as e:
                    print(f"[error] failed to move silent file: {src} -> {dest} ({e})")
            else:
                print(f"[skip] digital silence only: {src}")
            continue

        if trim:
            # Keep original extension for proper codec selection; write to a temp sibling
            tmp_dst = src.with_name(src.stem + ".tmp" + src.suffix)
            ok = trim_silence(
                src,
                tmp_dst,
                noise_threshold_db=silence_threshold_db,
                min_silence_ms=min_silence_ms,
            )
            if ok and tmp_dst.exists():
                # Probe trimmed duration before replacing
                info_after = ffprobe_json(tmp_dst)
                after_duration_s: Optional[float] = parse_duration_seconds(info_after) if info_after else None
                try:
                    os.replace(str(tmp_dst), str(src))
                    processed_paths.append(src)
                    # Compute and report trim details if possible
                    trimmed_ms: Optional[int] = None
                    if orig_duration_s is not None and after_duration_s is not None:
                        delta_ms = int(round((orig_duration_s - after_duration_s) * 1000.0))
                        trimmed_ms = max(0, delta_ms)
                        print(
                            f"[trim] {src} before={orig_duration_s:.3f}s after={after_duration_s:.3f}s trimmed={trimmed_ms}ms "
                            f"(threshold={silence_threshold_db}dB, min_silence={min_silence_ms}ms)"
                        )
                    else:
                        print(f"[ok] trimmed in-place: {src}")
                    # Append to CSV log if enabled
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
                except Exception as e:
                    print(f"[error] failed to replace original with trimmed file: {src} -> {e}")
                    try:
                        tmp_dst.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                print(f"[warn] trimming failed, left unchanged: {src}")
                try:
                    tmp_dst.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            # No trimming requested; leave file as-is
            processed_paths.append(src)
            print(f"[ok] checked (no trim): {src}")

    return processed_paths


def dedupe_similar(
    target_dir: Path,
    name_similarity_threshold: float,
    duration_tolerance_ms: int,
    delete: bool,
    move_to: Optional[Path] = None,
) -> List[Tuple[Path, Path]]:
    # Returns list of (kept, deleted) pairs
    files = list(find_audio_files(target_dir))
    if not files:
        return []

    # Probe all durations first
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

    # Group by quantized duration to reduce pair checks
    by_duration: Dict[int, List[AudioFileInfo]] = {}
    for afi in infos.values():
        q = quantize_ms(afi.duration_seconds)
        by_duration.setdefault(q, []).append(afi)

    duplicates: List[Tuple[Path, Path]] = []
    hash_cache: Dict[Path, Optional[str]] = {}

    for qdur, group in by_duration.items():
        if len(group) < 2:
            continue
        # Also consider near neighbors within tolerance
        candidates: List[AudioFileInfo] = []
        for delta in range(-duration_tolerance_ms, duration_tolerance_ms + 1, 1):
            candidates.extend(by_duration.get(qdur + delta, []))
        if len(candidates) < 2:
            continue

        # Pairwise compare; always verify audio content hash for duration-near candidates
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                if a.path == b.path:
                    continue
                # Do not gate on name similarity; duration grouping already restricts candidates.
                # Some true duplicates have very different names. We still compute hashes once per file.

                # If either file has been moved/deleted by a previous iteration, skip this pair
                if not a.path.exists() or not b.path.exists():
                    hash_cache.pop(a.path, None)
                    hash_cache.pop(b.path, None)
                    continue

                # Hash audio content in canonical PCM
                ha = hash_cache.get(a.path)
                if ha is None:
                    ha = pcm_content_hash(a.path)
                    hash_cache[a.path] = ha
                hb = hash_cache.get(b.path)
                if hb is None:
                    hb = pcm_content_hash(b.path)
                    hash_cache[b.path] = hb
                if ha is None or hb is None:
                    continue
                if ha != hb:
                    continue

                # If either file disappeared after hashing (e.g., moved), skip
                if not a.path.exists() or not b.path.exists():
                    hash_cache.pop(a.path, None)
                    hash_cache.pop(b.path, None)
                    continue

                # Identical content; decide which to keep
                keep: Path
                drop: Path
                # Keep older file (smaller mtime) to be conservative
                try:
                    stat_a = a.path.stat()
                    stat_b = b.path.stat()
                except FileNotFoundError:
                    # One of the files disappeared after hashing
                    hash_cache.pop(a.path, None)
                    hash_cache.pop(b.path, None)
                    continue
                if stat_a.st_mtime <= stat_b.st_mtime:
                    keep, drop = a.path, b.path
                else:
                    keep, drop = b.path, a.path

                if delete and drop.exists():
                    if move_to is not None:
                        move_to.mkdir(parents=True, exist_ok=True)
                        dest = move_to / drop.name
                        print(f"[dupe->move] {drop} -> {dest} (keep {keep})")
                        shutil.move(str(drop), str(dest))
                        hash_cache.pop(drop, None)
                    else:
                        print(f"[dupe->delete] {drop} (keep {keep})")
                        drop.unlink(missing_ok=True)
                        hash_cache.pop(drop, None)
                else:
                    print(f"[dupe] would remove {drop} (keep {keep})")

                duplicates.append((keep, drop))

    return duplicates


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audio cleanup: skip silent/no-audio files, trim leading/trailing silence, "
            "and de-duplicate very similar files by name and duration with PCM hash verification."
        )
    )
    parser.add_argument("input", type=Path, help="Input directory to scan")
    parser.add_argument(
        "--output",
        type=Path,
        help="[Ignored] Processing is done in-place; this option is deprecated",
    )
    parser.add_argument(
        "--no-skip-silent",
        action="store_true",
        help="Do not skip files that are pure digital silence",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim leading/trailing silence",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-50.0,
        help="Silence threshold in dBFS for trimming (default: -50)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=200,
        help="Minimum silence duration in milliseconds to trim (default: 200)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Search for duplicates in the INPUT directory after processing",
    )
    parser.add_argument(
        "--dedupe-path",
        type=Path,
        help="Directory to dedupe (default: input directory)",
    )
    parser.add_argument(
        "--name-similarity",
        type=float,
        default=0.85,
        help="Name similarity ratio threshold 0..1 (default: 0.85)",
    )
    parser.add_argument(
        "--duration-tolerance-ms",
        type=int,
        default=50,
        help="Duration tolerance in ms to consider files same length (default: 50)",
    )
    parser.add_argument(
        "--delete-duplicates",
        action="store_true",
        help="If set, delete/move detected duplicates (otherwise just report)",
    )
    parser.add_argument(
        "--move-duplicates-to",
        type=Path,
        help="If set with --delete-duplicates, move duplicates into this directory instead of deleting",
    )
    # Trim log is always written to input_dir/log.csv

    args = parser.parse_args(argv)

    ensure_binaries_available()

    input_dir: Path = args.input.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 2

    skip_silent = not args.no_skip_silent
    trim = not args.no_trim

    print(f"Input:  {input_dir}")
    print(f"Options: skip_silent={skip_silent}, trim={trim}, threshold={args.silence_threshold_db} dB, min_silence={args.min_silence_ms} ms")

    # Determine duplicates move directory upfront so we can also move silent files there
    # Always create the "tobedeleted" folder in the source (input) directory by default
    default_move_dir: Optional[Path] = (
        (input_dir / "tobedeleted") if args.delete_duplicates and args.move_duplicates_to is None else args.move_duplicates_to
    )

    # Prepare trim log (always generated in input_dir as log.csv)
    trim_log_rows: List[List[str]] = [[
        "source_path",
        "duration_before_s",
        "duration_after_s",
        "trimmed_ms",
        "threshold_db",
        "min_silence_ms",
        "mean_volume_db",
    ]]

    processed = process_in_place(
        input_dir=input_dir,
        skip_silent=skip_silent,
        trim=trim,
        silence_threshold_db=args.silence_threshold_db,
        min_silence_ms=args.min_silence_ms,
        silent_move_to=default_move_dir,
        trim_log_rows=trim_log_rows,
    )

    print(f"Processed {len(processed)} files.")

    if args.dedupe or args.delete_duplicates:
        target_dir = (args.dedupe_path or input_dir).resolve()
        print(
            f"De-duplicating in: {target_dir} (name_similarity>={args.name_similarity}, duration_tolerance={args.duration_tolerance_ms} ms)"
        )
        # By default, when deleting duplicates, move them into a "tobedeleted" folder in the source (input) directory
        default_move_dir = (input_dir / "tobedeleted") if args.delete_duplicates and args.move_duplicates_to is None else args.move_duplicates_to
        dups = dedupe_similar(
            target_dir=target_dir,
            name_similarity_threshold=args.name_similarity,
            duration_tolerance_ms=args.duration_tolerance_ms,
            delete=args.delete_duplicates,
            move_to=default_move_dir,
        )
        print(f"Duplicates found: {len(dups)}")

    # Write trim log to input_dir/log.csv
    log_path = input_dir / "log.csv"
    try:
        with open(log_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerows(trim_log_rows)
        print(f"Trim log written: {log_path.resolve()}")
    except Exception as e:
        print(f"[error] failed to write trim log to {log_path}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


