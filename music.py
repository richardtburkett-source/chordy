"""
Music analysis for Chordy: tempo, key, chords and melody notes from raw audio.

Everything here runs locally with librosa, so it works without any API key.
It's an estimate, not a transcription: good enough to talk about a song,
honest about when it isn't sure.
"""

from __future__ import annotations

import io
from collections import Counter

import librosa
import numpy as np
import soundfile as sf

SR = 22050
NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_audio(data: bytes) -> np.ndarray:
    """Decode WAV, FLAC, OGG or MP3 bytes to mono float32 at SR."""
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _chord_templates() -> tuple[list[str], np.ndarray]:
    names, temps = [], []
    for root in range(12):
        for quality, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            for i in intervals:
                t[(root + i) % 12] = 1
            names.append(NOTES[root] + quality)
            temps.append(t / np.linalg.norm(t))
    return names, np.array(temps)


CHORD_NAMES, CHORD_TEMPLATES = _chord_templates()


def _key_from_chroma(chroma_mean: np.ndarray) -> tuple[str, float]:
    best, best_r, scores = None, -2.0, []
    for root in range(12):
        for mode, prof in (("major", MAJOR), ("minor", MINOR)):
            r = float(np.corrcoef(chroma_mean, np.roll(prof, root))[0, 1])
            scores.append(r)
            if r > best_r:
                best, best_r = f"{NOTES[root]} {mode}", r
    scores.sort(reverse=True)
    margin = scores[0] - scores[1]
    confidence = "clear" if best_r > 0.7 and margin > 0.05 else "likely" if best_r > 0.5 else "unsure"
    return best or "unknown", confidence


def _melody(y: np.ndarray) -> list[dict]:
    """Note-by-note line of a hummed, whistled or sung melody."""
    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=2100, sr=SR, frame_length=2048, hop_length=512)
    times = librosa.times_like(f0, sr=SR, hop_length=512)
    notes: list[dict] = []
    cur, start = None, 0.0
    for t, f, v in zip(times, f0, voiced):
        midi = int(round(librosa.hz_to_midi(f))) if (v and f and not np.isnan(f)) else None
        if midi != cur:
            if cur is not None and t - start >= 0.12:
                notes.append({"note": librosa.midi_to_note(cur, unicode=False), "start": round(start, 2), "dur": round(t - start, 2)})
            cur, start = midi, t
    if cur is not None and times.size and times[-1] - start >= 0.12:
        notes.append({"note": librosa.midi_to_note(cur, unicode=False), "start": round(start, 2), "dur": round(times[-1] - start, 2)})
    # merge repeated notes separated by tiny gaps
    merged: list[dict] = []
    for n in notes:
        if merged and merged[-1]["note"] == n["note"] and n["start"] - (merged[-1]["start"] + merged[-1]["dur"]) < 0.02:
            merged[-1]["dur"] = round(n["start"] + n["dur"] - merged[-1]["start"], 2)
        else:
            merged.append(n)
    return merged[:64]


def analyze(y: np.ndarray, *, source: str = "microphone") -> dict:
    """Estimate what's in a clip. Returns plain values Chordy can talk about."""
    dur = len(y) / SR
    rms = librosa.feature.rms(y=y)[0]
    loud = float(np.percentile(rms, 90)) if rms.size else 0.0
    out: dict = {"source": source, "seconds": round(dur, 1), "level_db": round(20 * np.log10(loud + 1e-9))}
    # With the voice filters off (Music mic) a song across the room can be very quiet, so the gate is low.
    if dur < 2 or loud < 0.0004:
        out["heard"] = False
        out["summary"] = f"Almost nothing audible: too quiet or too short (level {out['level_db']} dB)."
        return out
    out["heard"] = True
    if loud < 0.01:
        out["level_note"] = "very quiet: the result may be unreliable; suggest turning the music up or moving closer to the mic"
    y = y / max(1e-6, float(np.max(np.abs(y))))

    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=512)
    frame_rms = librosa.feature.rms(y=harmonic, hop_length=512)[0][: chroma.shape[1]]
    active = frame_rms > 0.25 * np.max(frame_rms)
    act = chroma[:, active] if active.sum() > 8 else chroma

    # One clear pitch at a time (humming, whistling, a single line) or full music?
    strong = float(np.mean((act >= 0.6 * act.max(axis=0, keepdims=True)).sum(axis=0)))
    perc_share = float(np.sum(percussive**2) / (np.sum(y**2) + 1e-9))
    out["_pitch_classes_per_frame"] = round(strong, 2)
    is_melody = strong < 2.0 and perc_share < 0.15
    out["kind"] = "single melody (hummed, whistled or one instrument line)" if is_melody else "full music (chords and/or drums)"

    key, key_conf = _key_from_chroma(act.mean(axis=1))
    out["key"], out["key_confidence"] = key, key_conf

    onset_env = librosa.onset.onset_strength(y=y, sr=SR)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR)
    bpm = float(np.atleast_1d(tempo)[0])
    pulse = float(np.mean(onset_env[beats]) / (np.mean(onset_env) + 1e-9)) if len(beats) > 3 else 0.0
    out["bpm"] = round(bpm) if bpm > 0 else None
    out["tempo_confidence"] = "steady beat" if pulse > 1.6 else "rough estimate" if pulse > 1.2 else "no clear beat"

    if is_melody:
        notes = _melody(y)
        out["notes"] = notes
        out["note_names"] = " ".join(n["note"] for n in notes[:24])
        if notes:
            midis = [librosa.note_to_midi(n["note"]) for n in notes]
            out["range"] = f"{librosa.midi_to_note(min(midis), unicode=False)} to {librosa.midi_to_note(max(midis), unicode=False)}"
    else:
        sync = librosa.util.sync(chroma, beats, aggregate=np.median) if len(beats) > 3 else chroma[:, :: max(1, chroma.shape[1] // 32)]
        sync = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
        labels = [CHORD_NAMES[i] for i in np.argmax(CHORD_TEMPLATES @ sync, axis=0)]
        prog: list[str] = []
        for lab in labels:
            if not prog or prog[-1] != lab:
                prog.append(lab)
        # drop one-beat blips between identical neighbours
        cleaned = [c for i, c in enumerate(prog) if not (0 < i < len(prog) - 1 and prog[i - 1] == prog[i + 1] and labels.count(c) <= 1)]
        out["chords"] = cleaned[:16]
        out["common_chords"] = [c for c, _ in Counter(labels).most_common(4)]

    bits = [f"{out['kind']}", f"key {key} ({key_conf})"]
    if out.get("bpm"):
        bits.append(f"about {out['bpm']} BPM ({out['tempo_confidence']})")
    if out.get("chords"):
        bits.append("chords " + " ".join(out["chords"][:8]))
    if out.get("note_names"):
        bits.append("notes " + out["note_names"])
    out["summary"] = "; ".join(bits)
    return out


_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def pitch_class(note: str) -> int:
    return _SHARP.index(note) if note in _SHARP else _FLAT.index(note)


def check_chord(y: np.ndarray, wanted: str, notes: list[str]) -> dict:
    """Does a strummed clip contain the notes of the chord the person says they are playing?"""
    out: dict = {"wanted": wanted, "wanted_notes": notes}
    rms = librosa.feature.rms(y=y)[0]
    loud = float(np.percentile(rms, 90)) if rms.size else 0.0
    out["level_db"] = round(20 * np.log10(loud + 1e-9))
    if len(y) / SR < 1 or loud < 0.0004:
        out.update(heard=False, verdict="no sound", summary=f"I couldn't hear anything to check (level {out['level_db']} dB).")
        return out
    out["heard"] = True
    if loud < 0.01:
        out["level_note"] = "quiet: the check may be less reliable; ask them to move the instrument closer to the mic"
    y = y / max(1e-6, float(np.max(np.abs(y))))
    harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=512)
    frame_rms = librosa.feature.rms(y=harmonic, hop_length=512)[0][: chroma.shape[1]]
    active = frame_rms > 0.3 * np.max(frame_rms)
    prof = (chroma[:, active] if active.sum() > 4 else chroma).mean(axis=1)
    prof = prof / (prof.max() + 1e-9)

    want = {pitch_class(n) for n in notes}
    missing = [NOTES[i] for i in sorted(want) if prof[i] < 0.4]
    extra = [NOTES[i] for i in range(12) if i not in want and prof[i] > 0.6]
    best = CHORD_NAMES[int(np.argmax(CHORD_TEMPLATES @ (prof / (np.linalg.norm(prof) + 1e-9))))]
    out.update(missing_notes=missing, unexpected_notes=extra, closest_chord=best)
    if not missing and not extra:
        out.update(verdict="match", summary=f"All the notes of {wanted} are there and nothing else stands out.")
    elif not missing:
        out.update(verdict="mostly", summary=f"{wanted} is there, but {', '.join(extra)} also rings out (a stray string or a different voicing). Closest chord: {best}.")
    else:
        out.update(verdict="mismatch", summary=f"Missing {', '.join(missing)} from {wanted}" + (f" and I hear {', '.join(extra)} instead" if extra else "") + f". It sounds closest to {best}.")
    return out


def to_wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()
