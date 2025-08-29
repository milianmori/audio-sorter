#!/usr/bin/env python3
"""
Audio Sorter PRO (CLI)
======================

Ziel: Deutlich weniger "Uncategorized" durch **Content-Heuristiken + (optional) ML**.

Neu in dieser Version:
- **Robuste Audio-Heuristiken** (librosa):
  • Onset-Zahl, Dauer, Harmonic/Percussive-Ratio (HPSS)
  • Spektral-Zentroid, -Rolloff, -Flatness, Zero-Crossing-Rate
  • Low-/Mid-/Highband-Energieanteile
- **Besseres Keyword-Matching**: Tokenisierung, Wortgrenzen, Regex-Prioritäten
- **Priorisierte Entscheidungslogik**: Keywords → ML (YAMNet) → Heuristiken → Fallback
- **Synth-Subtypen-Heuristik** (Bass/Pad/Lead/Pluck/Arp/Keys/Stabs/Chords/FX)
- **Drum-Subtypen-Heuristik** (808/909/Acoustic/Electronic/Distorted/Layered)
- **Vocals/Field-Recordings** via ML oder Rausch-/Tonalitätsmerkmale
- Konsistente **Naming Convention**: `Kategorie_Subkategorie_Originalname.ext`
- Ausführliches **Logging** (CSV) inkl. getroffenen Regeln/Features

Voraussetzungen (macOS empfohlen):
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  # Apple Silicon (ohne ML):
  pip install librosa soundfile numpy scipy tqdm pandas
  # Optional ML (YAMNet):
  pip install tensorflow-macos==2.16.2 tensorflow-metal tensorflow-hub
  # Intel/mac/Linux (stattdessen):
  # pip install tensorflow==2.16.1 tensorflow-hub

Beispiel:
  python audio_sorter_pro.py \
    --src "/Volumes/OldDrive/Audio" \
    --dst "/Volumes/NewDrive/Audio_Sortiert" \
    --ml --log out.csv

"""
from __future__ import annotations
import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import soundfile as sf
import os

# Audio analysis
import librosa

# Optional ML (YAMNet)
try:
    import tensorflow_hub as hub
    import tensorflow as tf
    TF_OK = True
except Exception:
    hub = None
    tf = None
    TF_OK = False

# -------------------------------
# Category schema (target folders)
# -------------------------------
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

# -------------------------------
# Keyword dictionaries (broad + DE/EN variants)
# -------------------------------
KW = {
    # Kicks
    ("1_Kicks", "808"): [r"\b808\b"],
    ("1_Kicks", "909"): [r"\b909\b"],
    ("1_Kicks", "Distorted"): [r"distort", r"crush", r"dirty", r"drive"],
    ("1_Kicks", "Layered"): [r"layer"],
    ("1_Kicks", "Acoustic"): [r"acoustic", r"live", r"room"],
    ("1_Kicks", "Electronic"): [r"\bkick\b", r"\bbd\b", r"bass\s?drum", r"drum\s?machine"],

    # Snares/Claps
    ("2_Snares_Claps", "Rimshot"): [r"\brim(\s?shot)?\b"],
    ("2_Snares_Claps", "Claps"): [r"\bclap(s)?\b", r"hand\s?clap"],
    ("2_Snares_Claps", "Rolls"): [r"\broll\b", r"buzz"],
    ("2_Snares_Claps", "Snares_Acoustic"): [r"snare.*(acoustic|live|room)", r"acoustic.*snare"],
    ("2_Snares_Claps", "Snares_Electronic"): [r"\bsnare\b", r"\bsd\b"],

    # HiHats
    ("3_HiHats", "Closed"): [r"(closed\s?hat|\bchh\b|\bhhc\b)"],
    ("3_HiHats", "Open"): [r"(open\s?hat|\bohh\b|\bhho\b)"],
    ("3_HiHats", "Shaker"): [r"\bshaker\b"],

    # Percussion
    ("4_Percussion", "Cymbals"): [r"cymbal|crash|ride|splash|china"],
    ("4_Percussion", "Toms"): [r"\btom(s)?\b"],
    ("4_Percussion", "Bells"): [r"bell|glock|tubular"],
    ("4_Percussion", "Wood"): [r"wood(\s?block)?|clave|claves"],
    ("4_Percussion", "Metal"): [r"metal|iron|anvil|steel|sheet"],
    ("4_Percussion", "Ethnic"): [r"bongo|conga|djembe|tabla|udu|cuica|taiko|dar(b|p)uka"],
    ("4_Percussion", "Foley"): [r"foley|coin|door|paper|keys|steps|whoosh"],

    # Synths
    ("5_Synth", "Bass"): [r"\bbass\b", r"\bsub\b"],
    ("5_Synth", "Lead"): [r"\blead\b"],
    ("5_Synth", "Pad"): [r"\bpad\b"],
    ("5_Synth", "Pluck"): [r"pluck"],
    ("5_Synth", "Keys"): [r"keys|piano|rhodes|clav"],
    ("5_Synth", "Arp"): [r"\barp\b|arpeggio"],
    ("5_Synth", "Stabs"): [r"stab(s)?\b"],
    ("5_Synth", "Chords"): [r"chord(s)?\b"],
    ("5_Synth", "FX"): [r"\bfx\b|sfx"],

    # OneShots
    ("6_OneShots", "Impacts"): [r"impact|boom|hit|slam|thud|explosion|exploder"],
    ("6_OneShots", "Risers"): [r"riser|rise|uplift"],
    ("6_OneShots", "Downers"): [r"downer|fall|drop"],
    ("6_OneShots", "Transitions"): [r"transition|whoosh|swoosh|swipe"],
    ("6_OneShots", "Textures"): [r"texture|drone|granular|noise"],
    ("6_OneShots", "Tonal"): [r"tonal|note|pitch"],
    ("6_OneShots", "Atonal"): [r"atonal"],

    # Vocals
    ("7_Vocals", "Spoken"): [r"spoken|speech|talk|words"],
    ("7_Vocals", "Sung"): [r"sung|sing(ing)?|choir|vocal\s?melody"],
    ("7_Vocals", "FX"): [r"vocal\s?fx|voxfx|\bvox\b"],
    ("7_Vocals", "Adlibs"): [r"ad[- ]?lib|uh+|oh+|hey+|yeah+"],
    ("7_Vocals", "Phrases"): [r"phrase|sentence|line"],
    ("7_Vocals", "OneShots"): [r"vocal.*(stab|one[- ]?shot)"],

    # Field Recordings
    ("8_Field_Recordings", "Nature"): [r"wind|storm|thunder|forest|birds?ong"],
    ("8_Field_Recordings", "Urban"): [r"street|city|urban|crowd|market"],
    ("8_Field_Recordings", "Transport"): [r"car|bus|train|plane|tram|subway|metro"],
    ("8_Field_Recordings", "Water"): [r"water|ocean|sea|river|stream|rain|rainfall|drip"],
    ("8_Field_Recordings", "Animals"): [r"bird|dog|cat|insect|cow|sheep|animal"],
    ("8_Field_Recordings", "Ambience"): [r"ambience|ambient|atmo|atmosphere|room\s?amb"],
    ("8_Field_Recordings", "RoomTone"): [r"room[- ]?tone|silence|bg\s?noise|background"],
}

# Priority order for keyword matches (most specific first)
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
        # HPSS
        harm, perc = librosa.effects.hpss(y)
        energy = np.sum(y**2) + 1e-12
        perc_energy = np.sum(perc**2)
        harm_energy = np.sum(harm**2)
        perc_ratio = float(perc_energy / energy)
        harm_ratio = float(harm_energy / energy)
        # Onsets
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        n_onsets = int(len(onsets))
        # Spectral
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        # Band energies (linear power)
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

# -------------------------------
# Normalization helpers (peak to target dBFS)
# -------------------------------

SUPPORTED_NORMALIZE_EXTS = {".wav", ".aiff", ".aif", ".flac"}


def dbfs_to_linear(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def peak_normalize_array(y: np.ndarray, target_dbfs: float = -1.0) -> Tuple[np.ndarray, float, float]:
    if y is None or y.size == 0:
        return y, 0.0, 0.0
    peak_before = float(np.max(np.abs(y)))
    if peak_before <= 0.0:
        return y, peak_before, peak_before
    target_peak = dbfs_to_linear(target_dbfs)
    scale = target_peak / peak_before
    y_out = y * scale
    # Safety clamp
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
        # For AIFF, many players have limited support for float subtypes (AIFC/float).
        # Force writing as PCM to ensure broad compatibility.
        dst_ext = dst.suffix.lower()
        if dst_ext in ('.aif', '.aiff'):
            sf.write(str(dst), y_norm, sr, format='AIFF', subtype='PCM_24')
        elif dst_ext == '.wav':
            # Write WAV as 24-bit PCM
            sf.write(str(dst), y_norm, sr, format='WAV', subtype='PCM_24')
        elif dst_ext == '.flac':
            # Ensure FLAC is encoded with 24-bit samples
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

# -------------------------------
# Heuristic decision logic
# -------------------------------

def classify_by_heuristics(f: Features) -> Tuple[str, str, str]:
    """Return (cat, sub, reason) best guess based on audio features."""
    # Quick helpers
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

    # 1) Field/ambience
    if perc < 0.3 and on <= 1 and dur > 5.0 and flat > 0.3:
        return ("8_Field_Recordings", "Ambience", "low-perc long noisy ambience")

    # 2) Drums (percussive, short to medium, onsets 1-6)
    if perc >= 0.4 and on <= 6 and dur <= 5.0:
        # Kick: strong low, low centroid
        if low > 0.55 and cent < 1200 and on <= 2:
            return ("1_Kicks", "Electronic", "low-heavy percussive short")
        # Snare/Clap: broadband/noisy mid-high, flatness höher
        if (mid > 0.35 or high > 0.25) and flat > 0.2:
            if high > mid and dur < 1.0:
                return ("2_Snares_Claps", "Claps", "percussive high short noisy")
            else:
                return ("2_Snares_Claps", "Snares_Electronic", "percussive mid-high noisy")
        # HiHats/Shaker: sehr hohe Helligkeit, hohe ZCR, ggf. mehrere feine Onsets
        if high > 0.45 and cent > 4000 and zcr > 0.1:
            if on >= 3:
                return ("3_HiHats", "Shaker", "many fine onsets high freq")
            else:
                return ("3_HiHats", "Closed", "bright percussive hat-like")
        # Toms
        if low > 0.25 and mid > 0.25 and harm > 0.3 and on <= 3:
            return ("4_Percussion", "Toms", "low-mid tonal percussive")
        # Cymbals
        if high > 0.5 and dur > 1.5 and flat > 0.25:
            return ("4_Percussion", "Cymbals", "high energy long decay")

    # 3) Synth material / musical one-shots
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

    # 4) OneShots FX etc.
    if perc >= 0.3 and dur >= 0.8 and on <= 4 and flat >= 0.25 and low < 0.4 and high >= 0.3:
        return ("6_OneShots", "Impacts", "broadband impact-like")
    if dur >= 1.0 and on <= 3 and high >= 0.25 and flat >= 0.2:
        if zcr < 0.05:
            return ("6_OneShots", "Textures", "noisy sustained texture")
        else:
            return ("6_OneShots", "Transitions", "noisy transitional")

    # 5) Fallbacks
    if dur > 4.0 and perc < 0.4:
        return ("8_Field_Recordings", "Ambience", "fallback ambience")
    if perc >= 0.4:
        return ("4_Percussion", "Foley", "fallback percussive")
    return ("6_OneShots", "Atonal", "fallback atonal")

# -------------------------------
# ML wrapper (YAMNet) → coarse mapping
# -------------------------------
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


def map_yamnet_label(lbl: Optional[str]) -> Optional[Tuple[str,str,str]]:
    if not lbl:
        return None
    l = lbl.lower()
    if any(k in l for k in ["speech","talk","conversation","narration"]):
        return ("7_Vocals","Spoken", f"yamnet:{l}")
    if any(k in l for k in ["singing","choir","chant"]):
        return ("7_Vocals","Sung", f"yamnet:{l}")
    if "bass drum" in l or "kick drum" in l:
        return ("1_Kicks","Electronic", f"yamnet:{l}")
    if "snare" in l:
        return ("2_Snares_Claps","Snares_Electronic", f"yamnet:{l}")
    if "hi-hat" in l or "cymbal" in l:
        return ("3_HiHats","Closed", f"yamnet:{l}")
    if "tom-tom" in l:
        return ("4_Percussion","Toms", f"yamnet:{l}")
    if any(k in l for k in ["rain","water","ocean","stream","river"]):
        return ("8_Field_Recordings","Water", f"yamnet:{l}")
    if any(k in l for k in ["wind","thunder","storm"]):
        return ("8_Field_Recordings","Nature", f"yamnet:{l}")
    if any(k in l for k in ["vehicle","car","train","airplane","traffic","urban","crowd"]):
        return ("8_Field_Recordings","Urban", f"yamnet:{l}")
    if any(k in l for k in ["animal","bird","dog","cat","insect"]):
        return ("8_Field_Recordings","Animals", f"yamnet:{l}")
    if "music" in l or "instrument" in l:
        return ("5_Synth","FX", f"yamnet:{l}")
    return None

# -------------------------------
# Keyword matching
# -------------------------------
REGEX_CACHE: Dict[Tuple[str,str], List[re.Pattern]] = {}
for key, pats in KW.items():
    REGEX_CACHE[key] = [re.compile(p, re.I) for p in pats]


def match_keywords(name: str) -> Optional[Tuple[str,str,str]]:
    tokens = tokenize(name)
    s = " ".join(tokens)
    for cat_sub in PRIORITY:
        pats = REGEX_CACHE.get(cat_sub, [])
        for rx in pats:
            if rx.search(s):
                return (cat_sub[0], cat_sub[1], f"kw:{rx.pattern}")
    return None

# -------------------------------
# IO helpers
# -------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def category_prefix(cat: str) -> str:
    return cat.split('_',1)[1] if '_' in cat else cat


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

# -------------------------------
# Main pipeline
# -------------------------------

def decide_category(path: Path, use_ml: bool) -> Tuple[str,str,str]:
    # 1) Keywords
    m = match_keywords(path.name)
    if m:
        return m
    # 2) ML (optional)
    if use_ml and TF_OK:
        try:
            yw = YamnetWrap()
            lbl = yw.infer(path)
            mm = map_yamnet_label(lbl)
            if mm:
                return mm
        except Exception:
            pass
    # 3) Heuristics
    feat = extract_features(path)
    if feat:
        return classify_by_heuristics(feat)
    # 4) Last resort
    return ("6_OneShots","Atonal","fallback")


def main():
    ap = argparse.ArgumentParser(description="Audio Sorter PRO — inhaltsbasierte Sortierung + Naming")
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--ml', action='store_true', help='YAMNet verwenden, falls verfügbar')
    ap.add_argument('--copy', action='store_true', help='Dateien kopieren statt verschieben')
    ap.add_argument('--log', default='audio_sort_log.csv')
    ap.add_argument('--exts', default='.wav,.aiff,.aif,.mp3,.flac,.m4a')
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    ensure_dir(dst)

    exts = tuple(e.strip().lower() for e in args.exts.split(','))
    files = [p for p in src.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print('Keine Audiodateien gefunden.')
        sys.exit(1)

    # Log setup
    log_rows: List[List[str]] = []
    header = [
        'source_path','dest_path','category','subcategory','reason','duration','sr','n_onsets','perc_ratio','harm_ratio','zcr','centroid','rolloff','flatness','low_ratio','mid_ratio','high_ratio',
        'normalized','peak_before','peak_after','target_dbfs'
    ]
    log_rows.append(header)

    for f in tqdm(files, unit='file', desc='Sorting'):
        # Sonderfall: Liegt die Datei in einem Ordner namens "export" oder "exports"?
        # Dann in Ziel unter "exports" ablegen, ohne Umbenennung.
        is_export = any(parent.name.lower() in ("export", "exports") for parent in f.parents)
        if is_export:
            target_dir = dst / "exports"
            ensure_dir(target_dir)
            target = unique_target(target_dir / f.name)
            # Normalize if possible; otherwise copy/move
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
            # Feature row for log (Kategorie: exports, ohne Subkategorie)
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
            # Normalization columns
            row += [
                "1" if normalized else "0",
                f"{norm_pk_before:.6f}" if normalized and norm_pk_before > 0 else "",
                f"{norm_pk_after:.6f}" if normalized and norm_pk_after > 0 else "",
                f"{norm_target_dbfs:.2f}"
            ]
            log_rows.append(row)
            continue
        cat, sub, reason = decide_category(f, use_ml=args.ml)
        cat_prefix = category_prefix(cat)
        target_dir = dst / cat / sub
        ensure_dir(target_dir)
        new_name = f"{cat_prefix}_{sub}_{f.stem}{f.suffix}"
        target = target_dir / new_name
        target = unique_target(target)
        # Normalize if possible; otherwise copy/move
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
        # Feature row for log
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
        # Normalization columns
        row += [
            "1" if normalized else "0",
            f"{norm_pk_before:.6f}" if normalized and norm_pk_before > 0 else "",
            f"{norm_pk_after:.6f}" if normalized and norm_pk_after > 0 else "",
            f"{norm_target_dbfs:.2f}"
        ]
        log_rows.append(row)

    # write log
    with open(args.log, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerows(log_rows)
    print(f"\nLog gespeichert: {Path(args.log).resolve()}")

if __name__ == '__main__':
    main()
