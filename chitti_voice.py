"""
Chitti's voice.

  speech_text()   turns written text into text that sounds right when read aloud
  engines         Edge neural voices (free, needs internet), Piper (offline neural, English),
                  Windows built-in voice (offline, robotic) - tried in that order
  Player          plays audio and can be stopped instantly (for interruptions)
  Speaker         speaks sentence by sentence, preparing the next sentence while the current one plays

Settings (in .env): TTS_ENGINE, TTS_VOICE_EN, TTS_VOICE_TE, SPEECH_SPEED, SPEECH_PITCH_HZ, VOICE_VOLUME,
SENTENCE_GAP_MS, PIPER_MODEL, OUTPUT_DEVICE. See .env.example.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

import chitti_config as cfg

log = logging.getLogger("chitti.voice")

OUT_RATE = 48000          # every voice is converted to this before playing (all sound cards accept it)

# ================================================================ TEXT FOR SPEECH ====
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]

DEFAULT_PRONUNCIATIONS = {
    "YouTube": "You Tube", "GitHub": "Git Hub", "Wi-Fi": "Wi Fi", "WiFi": "Wi Fi", "VS Code": "V S Code",
    "ChatGPT": "Chat G P T", "OK": "okay", "PDF": "P D F", "USB": "U S B", "URL": "U R L", "API": "A P I",
    "CPU": "C P U", "GPU": "G P U", "SQL": "S Q L",
}


def load_pronunciations() -> dict:
    """Add your own in chitti_pronunciations.txt, one per line:   word = how it should sound"""
    table = dict(DEFAULT_PRONUNCIATIONS)
    try:
        for line in (cfg.APP_DIR / "chitti_pronunciations.txt").read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                word, _, spoken = line.partition("=")
                if word.strip() and spoken.strip():
                    table[word.strip()] = spoken.strip()
    except OSError:
        pass
    return table


def detect_lang(text: str, default: str = "en") -> str:
    """Telugu if the text contains Telugu letters, otherwise the default language."""
    telugu = sum(1 for ch in text if "\u0c00" <= ch <= "\u0c7f")
    return "te" if telugu >= 2 else default


def _unit(singular: str, plural: str):
    def repl(m):
        return f"{m.group(1)} {singular if m.group(1) == '1' else plural}"
    return repl


def _iso_date(m):
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{d} {_MONTHS[mo - 1]} {y}" if 1 <= mo <= 12 and 1 <= d <= 31 else m.group(0)


def _slash_date(m):
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    y = y + 2000 if y < 100 else y
    day, month = (b, a) if cfg.get("DATE_ORDER", "dmy").lower() == "mdy" else (a, b)
    return f"{day} {_MONTHS[month - 1]} {y}" if 1 <= month <= 12 and 1 <= day <= 31 else m.group(0)


_ABBREVIATIONS = [
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\betc\.", "et cetera"),
    (r"\bvs\.?(?=\s)", "versus"), (r"\bapprox\.", "approximately"), (r"\bDr\.", "Doctor"),
    (r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"), (r"\bProf\.", "Professor"),
    (r"\bNo\.\s?(?=\d)", "number "), (r"\ba\.m\.", "A M"), (r"\bp\.m\.", "P M"),
]


def speech_text(text: str, lang: str = "en", pronunciations: Optional[dict] = None) -> str:
    """Clean written text so a voice reads it naturally (no symbols read out, numbers and dates spoken)."""
    t = text
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                          # code blocks are not read aloud
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"https?://\S+|www\.\S+", " a link ", t)
    t = re.sub(r"\S+@\S+\.\S+", " an email address ", t)
    t = re.sub("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]", "", t)  # emoji
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)                    # headings
    t = re.sub(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+", "", t, flags=re.M)    # bullet / list markers
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = t.replace("*", "").replace("_", " ").replace("#", " ")
    t = re.sub(r"([^.!?\u2026:;,\n])[ \t]*\n+[ \t]*", r"\1. ", t)           # line break = sentence break
    t = re.sub(r"\s*\n+\s*", " ", t)

    if lang == "en":
        for pattern, spoken in _ABBREVIATIONS:
            t = re.sub(pattern, spoken, t)
        t = re.sub(r"(\d+(?:\.\d+)?)\s?(?:km/h|kmph)\b", _unit("kilometer per hour", "kilometers per hour"), t)
        for sym, sing, plur in (("km", "kilometer", "kilometers"), ("kg", "kilogram", "kilograms"),
                                ("cm", "centimeter", "centimeters"), ("mm", "millimeter", "millimeters"),
                                ("ml", "milliliter", "milliliters"), ("GB", "gigabyte", "gigabytes"),
                                ("MB", "megabyte", "megabytes")):
            t = re.sub(rf"(\d+(?:\.\d+)?)\s?{sym}\b", _unit(sing, plur), t)
        t = re.sub(r"(\d)\s?\u00b0\s?C\b", r"\1 degrees Celsius", t)
        t = re.sub(r"(\d)\s?\u00b0\s?F\b", r"\1 degrees Fahrenheit", t)
        t = re.sub(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b", lambda m: m.group(0).replace(",", ""), t)
        t = re.sub(r"(?:\u20b9|\bRs\.?)\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1).replace(",", "") + " rupees", t)
        t = re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1).replace(",", "") + " dollars", t)
        t = re.sub(r"\u20ac\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1).replace(",", "") + " euros", t)
        t = re.sub(r"\u00a3\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1).replace(",", "") + " pounds", t)
        t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _iso_date, t)
        t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", _slash_date, t)
        t = t.replace("%", " percent").replace("&", " and ")
        for word, spoken in (pronunciations if pronunciations is not None else load_pronunciations()).items():
            t = re.sub(rf"(?<![\w]){re.escape(word)}(?![\w])", spoken, t)

    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    if t and t[-1] not in ".!?\u2026\u0964\u0965,;:\"')":
        t += "."                                        # a final full stop gives the voice a natural ending
    return t


# =================================================================== AUDIO HELPERS ====
def to_out_rate(pcm: np.ndarray, rate: int, gain: float = 1.0, out_rate: int = OUT_RATE) -> np.ndarray:
    x = pcm.astype(np.float32)
    if rate != out_rate and len(x) > 1:
        n = int(round(len(x) * out_rate / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(x * gain, -32768, 32767).astype(np.int16)


def trim_silence(pcm: np.ndarray, rate: int, thresh: int = 250, keep_ms: int = 25) -> np.ndarray:
    """Remove the silence that voices put before and after a sentence (so sentences join smoothly)."""
    idx = np.flatnonzero(np.abs(pcm.astype(np.int32)) > thresh)
    if idx.size == 0:
        return pcm
    keep = int(rate * keep_ms / 1000)
    return pcm[max(0, idx[0] - keep): min(len(pcm), idx[-1] + keep)]


def fade_edges(pcm: np.ndarray, rate: int, ms: int = 6) -> np.ndarray:
    n = min(int(rate * ms / 1000), len(pcm) // 2)
    if n < 2:
        return pcm
    x = pcm.astype(np.float32)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x.astype(np.int16)


def read_wav(path: str) -> "Audio":
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise TTSError("unsupported WAV format")
    pcm = np.frombuffer(data, dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return Audio(pcm.copy(), rate)


# ======================================================================= VOICES ====
class TTSError(Exception):
    pass


@dataclass
class Audio:
    pcm: np.ndarray        # int16 mono
    rate: int


class _AsyncRunner:
    """One private event loop in a background thread, used to call async libraries from normal code."""

    def __init__(self):
        self.loop = asyncio.SelectorEventLoop()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except BaseException:
            fut.cancel()
            raise


class EdgeEngine:
    """Microsoft Edge neural voices through the `edge-tts` package. Free, natural, needs internet."""
    name = "edge"
    cost = "free (uses Microsoft's online service; unofficial, can change)"

    def __init__(self, voices: dict, speed: float = 1.0, pitch_hz: int = 0, timeout: float = 15):
        self.voices = voices
        self.rate = f"{round((speed - 1) * 100):+d}%"
        self.pitch = f"{pitch_hz:+d}Hz"
        self.timeout = timeout
        self._runner: Optional[_AsyncRunner] = None

    def available(self) -> bool:
        try:
            import edge_tts, miniaudio  # noqa: F401
            return True
        except ImportError:
            return False

    def supports(self, lang: str) -> bool:
        return lang in self.voices

    async def _collect(self, text: str, voice: str) -> bytes:
        import edge_tts
        comm = edge_tts.Communicate(text, voice, rate=self.rate, pitch=self.pitch,
                                    connect_timeout=8, receive_timeout=20)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        if not buf:
            raise TTSError("the voice service returned no audio")
        return bytes(buf)

    def synth(self, text: str, lang: str) -> Audio:
        import miniaudio
        if self._runner is None:
            self._runner = _AsyncRunner()
        try:
            data = self._runner.run(self._collect(text, self.voices[lang]), self.timeout)
            dec = miniaudio.decode(data, output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=1, sample_rate=24000)
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"{type(e).__name__}") from e
        return Audio(np.frombuffer(dec.samples, dtype=np.int16).copy(), 24000)


class PiperEngine:
    """Piper neural voice running on this PC. Offline and fast. English only here."""
    name = "piper"
    cost = "free, offline"

    def __init__(self, model_path: str, timeout: float = 25):
        self.model, self.timeout = model_path, timeout

    def available(self) -> bool:
        import importlib.util
        return bool(self.model) and os.path.isfile(self.model) and importlib.util.find_spec("piper") is not None

    def supports(self, lang: str) -> bool:
        return lang == "en"

    def synth(self, text: str, lang: str) -> Audio:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            flags = 0x08000000 if sys.platform == "win32" else 0
            r = subprocess.run([sys.executable, "-m", "piper", "-m", self.model, "-f", wav],
                               input=text.encode("utf-8"), capture_output=True, timeout=self.timeout,
                               creationflags=flags)
            if r.returncode != 0:
                raise TTSError("piper failed")
            return read_wav(wav)
        except subprocess.TimeoutExpired as e:
            raise TTSError("piper timed out") from e
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass


class WindowsEngine:
    """Windows' built-in voice. Always available offline, but sounds robotic. Used as the last fallback."""
    name = "windows"
    cost = "free, offline"

    def __init__(self, speed: float = 1.0, voice_hint: str = ""):
        self.rate = max(-10, min(10, round((speed - 1) * 10)))
        self.voice_hint = voice_hint
        self.proc = None
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()

    def available(self) -> bool:
        return sys.platform == "win32"

    def supports(self, lang: str) -> bool:
        return lang == "en"

    def _reader(self, proc):
        for line in proc.stdout:
            self.lines.put(line.strip())

    def _start(self):
        hint = self.voice_hint.replace("'", "")
        script = (
            "[Console]::InputEncoding=[Text.Encoding]::UTF8;"
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$v=$s.GetInstalledVoices()|Where-Object{$_.VoiceInfo.Gender -eq 'Female'}|Select-Object -First 1;"
            f"$h=$s.GetInstalledVoices()|Where-Object{{'{hint}' -ne '' -and $_.VoiceInfo.Name -like '*{hint}*'}}|Select-Object -First 1;"
            "if($h){$s.SelectVoice($h.VoiceInfo.Name)}elseif($v){$s.SelectVoice($v.VoiceInfo.Name)};"
            f"$s.Rate={self.rate};"
            "[Console]::Out.WriteLine('ready');"
            "while(($l=[Console]::In.ReadLine()) -ne $null){"
            "try{$i=$l.IndexOf('|');$p=$l.Substring(0,$i);$t=$l.Substring($i+1);"
            "$s.SetOutputToWaveFile($p);$s.Speak($t);$s.SetOutputToNull();[Console]::Out.WriteLine('done')}"
            "catch{[Console]::Out.WriteLine('error')}}"
        )
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NoLogo", "-Command", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, creationflags=0x08000000)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.lines.get(timeout=15)

    def synth(self, text: str, lang: str) -> Audio:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with self.lock:
                try:
                    if self.proc is None or self.proc.poll() is not None:
                        self._start()
                    while not self.lines.empty():
                        self.lines.get_nowait()
                    self.proc.stdin.write(f"{wav}|{' '.join(text.split())}\n")
                    self.proc.stdin.flush()
                    answer = self.lines.get(timeout=max(8.0, len(text) * 0.12 + 5))
                except Exception as e:
                    try:
                        if self.proc:
                            self.proc.kill()
                    except Exception:
                        pass
                    self.proc = None
                    raise TTSError(f"windows voice: {type(e).__name__}") from e
            if answer != "done":
                raise TTSError("windows voice failed")
            return read_wav(wav)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass


class TTSChain:
    """Tries the voices in order. A voice that keeps failing is skipped for a while."""

    def __init__(self, engines: list, fail_limit: int = 2, cooldown: float = 60):
        self.engines = engines
        self.fail_limit, self.cooldown = fail_limit, cooldown
        self._fails: dict = collections.Counter()
        self._down_until: dict = {}
        self.last_engine = ""

    def supports(self, lang: str) -> bool:
        return any(e.supports(lang) for e in self.engines)

    def synth(self, text: str, lang: str) -> Audio:
        errors = []
        skipped = []
        for attempt in (0, 1):
            candidates = [e for e in self.engines if e.supports(lang)] if attempt == 0 else skipped
            for e in candidates:
                if attempt == 0 and self._down_until.get(e.name, 0) > time.time():
                    skipped.append(e)
                    continue
                try:
                    audio = e.synth(text, lang)
                    self._fails[e.name] = 0
                    self.last_engine = e.name
                    return audio
                except Exception as ex:
                    self._fails[e.name] += 1
                    if self._fails[e.name] >= self.fail_limit:
                        self._down_until[e.name] = time.time() + self.cooldown
                        self._fails[e.name] = 0
                    errors.append(f"{e.name}: {type(ex).__name__}")
                    log.warning("voice '%s' failed (%s)", e.name, type(ex).__name__)
            if not skipped:
                break
        raise TTSError("no voice could speak (" + "; ".join(errors) + ")")


def build_chain() -> TTSChain:
    choice = cfg.get("TTS_ENGINE", "auto").lower()
    speed = cfg.get_float("SPEECH_SPEED", 1.0)
    edge = EdgeEngine({"en": cfg.get("TTS_VOICE_EN", "en-IN-NeerjaNeural"),
                       "te": cfg.get("TTS_VOICE_TE", "te-IN-ShrutiNeural")},
                      speed=speed, pitch_hz=cfg.get_int("SPEECH_PITCH_HZ", 0),
                      timeout=cfg.get_float("TTS_TIMEOUT", 15))
    piper = PiperEngine(cfg.get("PIPER_MODEL", ""))
    win = WindowsEngine(speed=speed, voice_hint=cfg.get("WINDOWS_VOICE", ""))
    by_name = {"edge": edge, "piper": piper, "windows": win}
    if choice == "auto" or choice not in by_name:
        order = [edge, piper, win]
    else:
        order = [by_name[choice]] + ([e for e in (edge, piper, win) if e is not by_name[choice]]
                                     if cfg.get_bool("TTS_FALLBACK", True) else [])
    return TTSChain([e for e in order if e.available()])


# ======================================================================= PLAYER ====
class Player:
    """
    Plays audio through the sound card and can be stopped instantly with a tiny fade (no click).
    It also keeps the loudness of what it plays, which the interruption detector uses to
    tell Chitti's own voice apart from yours.
    """

    def __init__(self, rate: int = OUT_RATE, device=None, blocksize: int = 1024, stream_factory=None):
        self.rate, self.device, self.block = rate, device, blocksize
        self._factory = stream_factory
        self._lock = threading.Lock()
        self._segs: collections.deque = collections.deque()        # [pcm, offset, item_id, started]
        self._stopping = False
        self.started_ids: set = set()
        self._hist: collections.deque = collections.deque(maxlen=400)   # (time, loudness)
        self.stream = None

    def start(self) -> None:
        if self._factory:
            self.stream = self._factory(self._callback)
        else:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="int16",
                                          blocksize=self.block, device=self.device, callback=self._callback)
        self.stream.start()

    def close(self) -> None:
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None

    def push(self, pcm: np.ndarray, item_id: int) -> None:
        with self._lock:
            self._segs.append([pcm, 0, item_id, False])
            if len(self.started_ids) > 500:
                self.started_ids = set(sorted(self.started_ids)[-200:])

    def clear(self) -> None:
        with self._lock:
            if self._segs:
                self._stopping = True

    def busy(self) -> bool:
        with self._lock:
            return bool(self._segs) or self._stopping

    def queued_items(self) -> int:
        with self._lock:
            return len(self._segs)

    def recent_output_rms(self, window: float = 0.4) -> float:
        now = time.monotonic()
        vals = [r for t, r in list(self._hist) if now - t <= window]
        return max(vals) if vals else 0.0

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.int16)
        with self._lock:
            if self._stopping:
                if self._segs:
                    arr, off = self._segs[0][0], self._segs[0][1]
                    n = min(frames, len(arr) - off)
                    if n > 0:
                        out[:n] = (arr[off:off + n].astype(np.float32)
                                   * np.linspace(1.0, 0.0, n, dtype=np.float32)).astype(np.int16)
                self._segs.clear()
                self._stopping = False
            else:
                pos = 0
                while pos < frames and self._segs:
                    seg = self._segs[0]
                    if not seg[3]:
                        seg[3] = True
                        self.started_ids.add(seg[2])
                    n = min(frames - pos, len(seg[0]) - seg[1])
                    out[pos:pos + n] = seg[0][seg[1]:seg[1] + n]
                    seg[1] += n
                    pos += n
                    if seg[1] >= len(seg[0]):
                        self._segs.popleft()
        self._hist.append((time.monotonic(), float(np.sqrt(np.mean(out.astype(np.float32) ** 2)))))
        outdata[:, 0] = out


# ====================================================================== SPEAKER ====
@dataclass
class _Item:
    id: int
    text: str
    lang: str
    gen: int


class Speaker:
    """
    Speaks sentence by sentence. While one sentence is playing the next one is already being prepared,
    so there is no gap. stop() silences everything at once (used when you interrupt).
    """

    def __init__(self, chain: TTSChain, player: Player, default_lang: str = "en", volume: float = 1.0,
                 gap_ms: int = 140, prefetch: int = 2):
        self.chain, self.player = chain, player
        self.default_lang, self.volume, self.gap_ms, self.prefetch = default_lang, volume, gap_ms, prefetch
        self.audio_ok = True
        self.last_error = ""
        self.last_synth_ms = 0
        self._lock = threading.Lock()
        self._gen = 0
        self._q: "queue.Queue[_Item]" = queue.Queue()
        self._pending: dict = collections.Counter()
        self._next_id = 0
        self._pron = load_pronunciations()
        self._cache: dict = {}                       # (text, language) -> audio prepared ahead of time

    def start(self) -> bool:
        try:
            self.player.start()
        except Exception as e:
            self.audio_ok = False
            self.last_error = f"audio output unavailable ({type(e).__name__})"
            log.warning(self.last_error)
        self._spawn_worker()
        return self.audio_ok

    def close(self) -> None:
        self.stop()
        self.player.close()

    def _spawn_worker(self) -> None:
        threading.Thread(target=self._worker, args=(self._gen, self._q), daemon=True).start()

    def say(self, text: str, lang: Optional[str] = None) -> int:
        """Queue one sentence. Returns its id (use started(id) to know whether it began playing)."""
        with self._lock:
            self._next_id += 1
            item = _Item(self._next_id, text, lang or detect_lang(text, self.default_lang), self._gen)
            self._pending[item.gen] += 1
            q = self._q
        q.put(item)
        return item.id

    def started(self, item_id: int) -> bool:
        return item_id in self.player.started_ids

    def busy(self) -> bool:
        with self._lock:
            waiting = self._pending[self._gen] > 0
        return waiting or self.player.busy()

    def stop(self) -> None:
        """Silence everything now: current sound, queued sentences, and sentences being prepared."""
        with self._lock:
            self._gen += 1
            old, self._q = self._q, queue.Queue()
        try:
            while True:
                old.get_nowait()
        except queue.Empty:
            pass
        self.player.clear()
        deadline = time.time() + 0.15                    # let the 10 ms fade-out finish
        while self.player.busy() and time.time() < deadline:
            time.sleep(0.003)
        self._spawn_worker()

    def wait_idle(self, tick=None, poll: float = 0.02) -> bool:
        """Wait until everything has been spoken. If tick() returns True (you spoke) stop and return False."""
        while self.busy():
            if tick is not None and tick():
                self.stop()
                return False
            time.sleep(poll)
        return True

    def recent_output_rms(self, window: float = 0.4) -> float:
        return self.player.recent_output_rms(window)

    def _worker(self, gen: int, q: "queue.Queue[_Item]") -> None:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if gen != self._gen:
                    return
                continue
            try:
                if item.gen == self._gen:
                    self._speak_item(item)
            except Exception as e:                       # never let one sentence kill the voice
                self.last_error = f"{type(e).__name__}"
                log.warning("speaking failed: %s", self.last_error)
            finally:
                with self._lock:
                    self._pending[item.gen] -= 1
            if gen != self._gen:
                return

    def _render(self, text: str, lang: str) -> np.ndarray:
        """Text -> audio that is ready to play. Raises TTSError if no voice can do it."""
        t0 = time.time()
        audio = self.chain.synth(text, lang)
        self.last_synth_ms = int((time.time() - t0) * 1000)
        pcm = trim_silence(audio.pcm, audio.rate)
        pcm = fade_edges(to_out_rate(pcm, audio.rate, self.volume), OUT_RATE)
        gap = 50 if text[-1] in ",;:" else self.gap_ms
        return np.concatenate([pcm, np.zeros(int(OUT_RATE * gap / 1000), dtype=np.int16)])

    def prewarm(self, phrases, lang: str = "en") -> None:
        """Prepare short phrases in the background, so they play with no delay (acknowledgements, fillers)."""
        def run():
            for phrase in phrases:
                if (phrase, lang) in self._cache or not self.audio_ok or not self.chain.supports(lang):
                    continue
                try:
                    text = speech_text(phrase, lang, self._pron)
                    if text:
                        self._cache[(phrase, lang)] = self._render(text, lang)
                except Exception as e:
                    log.debug("could not prepare a phrase ahead of time: %s", type(e).__name__)
        threading.Thread(target=run, daemon=True).start()

    def is_cached(self, text: str, lang: str = "en") -> bool:
        return (text, lang) in self._cache

    def _speak_item(self, item: _Item) -> None:
        if not self.audio_ok:
            return
        pcm = self._cache.get((item.text, item.lang))        # prepared ahead of time: no waiting at all
        if pcm is None:
            text = speech_text(item.text, item.lang, self._pron)
            if not text:
                return
        while self.player.queued_items() >= self.prefetch and item.gen == self._gen:
            time.sleep(0.02)
        if item.gen != self._gen:
            return
        if pcm is None:
            try:
                pcm = self._render(text, item.lang)
            except TTSError as e:
                self.last_error = str(e)
                log.warning("%s", e)
                return
            if item.gen != self._gen:
                return
        self.player.push(pcm, item.id)


def build_speaker(default_lang: str = "en") -> Speaker:
    device = cfg.get("OUTPUT_DEVICE", "")
    player = Player(device=int(device) if device.isdigit() else (device or None))
    return Speaker(build_chain(), player, default_lang=default_lang,
                   volume=cfg.get_float("VOICE_VOLUME", 1.0), gap_ms=cfg.get_int("SENTENCE_GAP_MS", 140))
