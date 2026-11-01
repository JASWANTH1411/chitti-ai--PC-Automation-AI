"""
CHITTI - a natural voice assistant for Windows.

  hears you      Whisper speech recognition (faster-whisper), with end-of-speech detection and confidence checks
  thinks         an AI model of your choice (Ollama on this PC, Claude, or any OpenAI-compatible API)
  speaks         a natural neural voice (Edge voices, Piper offline, or the Windows voice as a fallback)
  acts           opens/closes apps, volume, search, WhatsApp chats, and more (the original Chitti commands)

Files (keep them all in the same folder):
  chitti.py  chitti_brain.py  chitti_voice.py  chitti_stt.py  chitti_config.py  setup_ai.py  chitti_prompt.txt  .env

Install (one time):  pip install -r requirements.txt
Settings:            copy .env.example to .env and edit it (voices, AI model, speed, microphone, ...)

Run:
    python chitti.py              start the assistant
    python chitti.py --setup      choose and test the AI model (and optional fast speech recognition)
    python chitti.py --diag       check microphone, AI, and voice speed on this PC
    python chitti.py --tts-test   hear the voice (English and Telugu)
    python chitti.py --test       live meter + shows what she heard and what she WOULD do (nothing runs)
    python chitti.py --selftest   checks command understanding with many phrasings (no mic needed)
    python chitti.py --mics       list microphones and speakers
    python chitti.py --debug      show extra detail (the log file chitti.log never contains your words unless
                                  LOG_TRANSCRIPTS=true)

Say "chitti" and then talk. She keeps listening for a while after each answer, so you can just continue the
conversation. Interrupt her any time by speaking. Say "goodbye" to end the conversation, "quit chitti" to close her.
    chitti open chrome            chitti how are rainbows formed        chitti search for python tutorial
    chitti speak in telugu        chitti forget this conversation       chitti ravi  (opens Ravi's WhatsApp chat)
"""

from __future__ import annotations

import collections
import datetime
import difflib
import logging
import logging.handlers
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

import numpy as np

import chitti_brain
import chitti_config as cfg
import chitti_stt
import chitti_voice

# =============================================================== SETTINGS ====
# Everything here can be changed in the .env file (see .env.example), without editing this code.
WAKE_WORD = cfg.get("WAKE_WORD", "chitti").lower()
# Spellings / sound-alikes the speech engine may produce for the wake word (add more in WAKE_EXTRA)
WAKE_VARIANTS = ["chitti", "chitty", "chiti", "chittie", "cheeti", "cheety", "chity",
                 "kitty", "city", "cheetie", "sheety", "\u0c1a\u0c3f\u0c1f\u0c4d\u0c1f\u0c3f",
                 "\u0c1a\u0c3f\u0c1f\u0c4d\u0c1f\u0c40"] + cfg.get_list("WAKE_EXTRA")
if WAKE_WORD not in WAKE_VARIANTS:
    WAKE_VARIANTS.insert(0, WAKE_WORD)

# --- speech recognition ------------------------------------------------------------------
# "tiny.en" (fastest) < "base.en" < "small.en" < "medium.en" (most accurate, slowest).
# Telugu needs a multilingual model: "small" (or "base" / "medium").
WHISPER_MODEL = cfg.get("WHISPER_MODEL", "base.en")
# "en" or "te". You can also switch while talking: "speak in Telugu" / "speak in English".
LANGUAGE = cfg.get("STT_LANGUAGE", "en")
BEAM_SIZE = cfg.get_int("BEAM_SIZE", 1)            # 1 = fastest, 5 = most careful

# --- listening / end-of-speech detection ---------------------------------------------------
_mic = cfg.get("MIC_DEVICE", "")
MIC_DEVICE = int(_mic) if _mic.isdigit() else (_mic or None)
MIC_SENSITIVITY = max(0.2, cfg.get_float("MIC_SENSITIVITY", 1.0))   # higher = hears quieter speech
SILENCE_TIMEOUT = cfg.get_float("SILENCE_TIMEOUT", 0.55)           # seconds of quiet that end a sentence
MIN_SPEECH_SECONDS = cfg.get_float("MIN_SPEECH_SECONDS", 0.25)     # ignore bumps/clicks shorter than this
MAX_UTTERANCE_SECONDS = cfg.get_float("MAX_UTTERANCE_SECONDS", 20)
PREROLL_SECONDS = 0.4                                              # audio kept from just before you spoke
SPEECH_THRESHOLD = cfg.get_float("SPEECH_THRESHOLD", 0) or None    # None = automatic (adapts to the room)
NOISE_MARGIN = cfg.get_float("NOISE_MARGIN", 2.5)               # speech must be this much louder than the noise
MIN_SNR = cfg.get_float("MIN_SNR", 1.8)

# --- confidence and safety -----------------------------------------------------------------
MIN_CONFIDENCE = cfg.get_float("MIN_CONFIDENCE", 0.50)   # below this she asks you to repeat (0 to 1)
IGNORE_CONFIDENCE = 0.25   # inside a conversation, anything below this is treated as background noise
SAFE_CONFIDENCE = 0.70     # closing apps needs this, otherwise she asks first
NO_SPEECH_MAX = 0.60       # Whisper's own "this is not speech" probability
ACCEPT_SCORE = 0.85        # command match score needed to run without asking
ASK_SCORE = 0.60           # between ASK_SCORE and ACCEPT_SCORE she asks "Did you mean ...?"
PHRASE_ASK_SCORE = 0.70    # same idea for commands without a name
MARGIN = 0.08              # best command must beat the next one by this much, else she asks
DEDUP_SECONDS = cfg.get_float("DEDUP_SECONDS", 1.2)      # the same command inside this window runs once

# --- conversation and interruptions --------------------------------------------------------
# continuous: after you say "chitti" she keeps listening, no need to repeat it. wake: say it every time.
CONVERSATION_MODE = cfg.get("CONVERSATION_MODE", "continuous").lower()
CONVERSATION_IDLE_SECONDS = cfg.get_float("CONVERSATION_IDLE_SECONDS", 20)
# off | speaker (echo-aware, for laptop speakers) | headphones (most sensitive)
BARGE_IN = cfg.get("BARGE_IN", "speaker").lower()
BARGE_IN_MS = cfg.get_int("BARGE_IN_MS", 450)             # how long you must speak to interrupt her
BEEP = cfg.get_bool("BEEP", True)                         # short beep when she hears the wake word
POST_SPEAK_COOLDOWN = cfg.get_float("POST_SPEAK_COOLDOWN", 0.3)   # ignore the mic briefly after she speaks
LOG_TRANSCRIPTS = cfg.get_bool("LOG_TRANSCRIPTS", False)          # keep your words out of chitti.log
TEST_MODE = "--test" in sys.argv

# --- speed ---------------------------------------------------------------------------------
# Start recognising your sentence after SPEC_AFTER_SECONDS of quiet, before she is sure you have finished.
# If you stay quiet the result is already waiting; if you carry on, it is thrown away.
SPECULATIVE_STT = cfg.get_bool("SPECULATIVE_STT", True)
SPEC_AFTER_SECONDS = cfg.get_float("SPEC_AFTER_SECONDS", 0.25)
STT_VAD = cfg.get_bool("STT_VAD", True)                # False saves a little time but lets more noise through
# After a computer command she can: voice = say what she did, tone = a short beep (fastest), none = silent
COMMAND_FEEDBACK = cfg.get("COMMAND_FEEDBACK", "voice").lower()
FILLER_AFTER_SECONDS = cfg.get_float("FILLER_AFTER_SECONDS", 0.9)   # say "Hmm." if the AI is slower than this

# WhatsApp contacts. Just say the name: "chitti ravi" opens Ravi's chat. Also settable in .env:
#   CONTACTS=mom:919876543210,ravi:
#   "ravi": "919812345678"  -> opens the chat directly by phone number (most reliable;
#                              country code + number, no + or spaces)
#   "ravi": ""              -> no number: opens WhatsApp and searches for the name
CONTACTS = {
    # "mom": "919876543210",
    # "ravi": "",
}
for _pair in cfg.get_list("CONTACTS"):
    _name, _, _num = _pair.partition(":")
    if _name.strip():
        CONTACTS[_name.strip().lower()] = _num.strip()
WHATSAPP_WAIT = cfg.get_float("WHATSAPP_WAIT", 3)       # seconds to wait for WhatsApp to open before searching
WHATSAPP_PRESS_DOWN = cfg.get_bool("WHATSAPP_PRESS_DOWN", True)

SAMPLE_RATE = 16000
BLOCK = 1600                # 0.1 second chunks
LOG_FILE = str(cfg.APP_DIR / "chitti.log")

# ============================================================ DATA TABLES ====
HOME = os.path.expanduser("~")
FOLDERS = {
    "desktop": os.path.join(HOME, "Desktop"),
    "documents": os.path.join(HOME, "Documents"),
    "downloads": os.path.join(HOME, "Downloads"),
    "pictures": os.path.join(HOME, "Pictures"),
    "music": os.path.join(HOME, "Music"),
    "videos": os.path.join(HOME, "Videos"),
}

BUILTIN = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "command prompt": "cmd.exe",
    "task manager": "taskmgr.exe",
    "file explorer": "explorer.exe",
    "settings": "ms-settings:",
}

# Websites (open in Chrome, need internet)
SITES = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "google drive": "https://drive.google.com",
    "github": "https://github.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://x.com",
    "linkedin": "https://www.linkedin.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "chatgpt": "https://chatgpt.com",
}

# Store apps that have no Start Menu shortcut can be opened with a protocol
PROTOCOLS = {
    "whatsapp": "whatsapp:",
    "settings": "ms-settings:",
    "calendar": "outlookcal:",
    "mail": "mailto:",
    "store": "ms-windows-store:",
    "camera": "microsoft.windows.camera:",
}

# Program names used to close apps. Others are found automatically from the Start Menu.
PROC_MAP = {
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "firefox": "firefox.exe",
    "notepad": "notepad.exe",
    "calculator": "CalculatorApp.exe",
    "paint": "mspaint.exe",
    "task manager": "Taskmgr.exe",
    "vs code": "Code.exe",
    "whatsapp": "WhatsApp.exe",
    "word": "WINWORD.EXE",
    "excel": "EXCEL.EXE",
    "powerpoint": "POWERPNT.EXE",
    "spotify": "Spotify.exe",
    "vlc": "vlc.exe",
    "telegram": "Telegram.exe",
    "settings": "SystemSettings.exe",
}

# Chitti will never close these (it would break your PC or Chitti itself)
PROTECTED = {
    "python.exe", "pythonw.exe", "py.exe", "explorer.exe", "cmd.exe", "powershell.exe",
    "pwsh.exe", "windowsterminal.exe", "conhost.exe", "svchost.exe", "winlogon.exe", "csrss.exe",
}

# Common mis-hearings (Indian English and fast speech) -> what you meant. Add your own!
# Applied to the whole sentence before she tries to understand it.
ALIASES = {
    "what's up app": "whatsapp", "what's up": "whatsapp", "whats up": "whatsapp",
    "what's app": "whatsapp", "whats app": "whatsapp", "watts app": "whatsapp",
    "what sap": "whatsapp", "whatsapp web": "whatsapp", "vatsapp": "whatsapp", "whatsap": "whatsapp",
    "crome": "chrome", "krome": "chrome", "chrom": "chrome", "google chrome": "chrome",
    "visual studio code": "vs code", "v s code": "vs code", "vscode": "vs code",
    "you tube": "youtube", "u tube": "youtube", "utube": "youtube", "yutube": "youtube",
    "g mail": "gmail", "gee mail": "gmail",
    "note pad": "notepad", "notpad": "notepad",
    "screen shot": "screenshot", "screenshoot": "screenshot",
    "wolume": "volume", "volum": "volume", "walum": "volume", "valume": "volume",
    "calcultor": "calculator", "calculater": "calculator",
    "file explorer": "file explorer",
}

# First-word fixes (the verb is the most important word)
VERB_FIXES = {
    "opan": "open", "ohpen": "open", "oepn": "open", "opened": "open", "opens": "open",
    "opening": "open", "launch": "open", "launched": "open", "start": "open", "run": "open",
    "clothes": "close", "cloths": "close", "cloze": "close", "clause": "close",
    "closed": "close", "closing": "close", "closes": "close", "kill": "close",
    "serch": "search", "sarch": "search", "surch": "search", "sirch": "search", "searching": "search",
}

# Commands without an argument. Several ways to say each; matching is by similarity.
INTENT_PHRASES = {
    "volume_up": ["volume up", "increase volume", "increase the volume", "turn up the volume", "raise volume",
                  "louder", "make it louder", "turn volume up", "volume increase", "more volume"],
    "volume_down": ["volume down", "decrease volume", "decrease the volume", "lower volume", "lower the volume",
                    "turn down the volume", "quieter", "make it quieter", "turn volume down", "less volume",
                    "reduce volume"],
    "mute": ["mute", "unmute", "mute volume", "mute sound", "mute the sound"],
    "pause": ["pause", "pause music", "pause the music", "pause video", "stop music"],
    "play": ["play", "resume", "continue playing", "play music", "play pause", "resume music"],
    "next": ["next song", "next track", "skip", "skip song", "play next", "play next song", "next"],
    "previous": ["previous song", "previous track", "play previous", "play previous song", "go back song",
                 "last song", "previous"],
    "screenshot": ["screenshot", "take screenshot", "take a screenshot", "capture screen", "capture the screen",
                   "screen capture"],
    "time": ["time", "what is the time", "what time is it", "tell me the time", "current time", "whats the time"],
    "date": ["date", "what is the date", "what is today date", "what day is it", "today date",
             "tell me the date", "what is today", "today"],
    "lock": ["lock", "lock computer", "lock the computer", "lock screen", "lock my pc", "lock pc"],
    "shutdown": ["shut down", "shutdown", "shut down computer", "shut down the computer", "turn off computer",
                 "turn off the computer", "power off", "switch off pc"],
    "restart": ["restart", "reboot", "restart computer", "restart the computer", "restart pc"],
    "press_enter": ["press enter", "enter", "hit enter"],
    "end_session": ["goodbye", "bye", "stop listening", "go to sleep", "that is all", "thats all", "that will be all",
                    "stop", "thank you that is all"],
    "quit": ["exit", "quit", "quit chitti", "exit chitti", "turn off chitti"],
    "lang_telugu": ["speak telugu", "speak in telugu", "talk in telugu", "talk telugu", "switch to telugu",
                    "telugu mode", "reply in telugu"],
    "lang_english": ["speak english", "speak in english", "talk in english", "talk english", "switch to english",
                     "english mode", "reply in english"],
    "clear_memory": ["forget this conversation", "forget our conversation", "forget everything", "new conversation",
                     "clear memory", "clear the conversation"],
    "search_prompt": ["search", "google", "search google"],
    "chat_prompt": ["open chat", "whatsapp chat", "chat"],
    "type_prompt": ["type", "start typing", "dictate"],
}
# Words that mean the opposite. If the heard text has one and a template has the other, the match is penalised
# so "volume up" can never be mistaken for "volume down".
OPPOSITES = [("up", "down"), ("next", "previous"), ("open", "close"), ("start", "stop"),
             ("increase", "decrease"), ("louder", "quieter"), ("raise", "lower"), ("play", "pause"),
             ("mute", "unmute"), ("more", "less"), ("telugu", "english")]
# These need an extra-sure match before they run
DANGEROUS_INTENTS = {"shutdown", "restart", "quit", "end_session", "clear_memory"}
LABELS = {
    "volume_up": "volume up", "volume_down": "volume down", "mute": "mute", "pause": "pause",
    "play": "play", "next": "next song", "previous": "previous song", "screenshot": "take a screenshot",
    "time": "tell the time", "date": "tell the date", "lock": "lock the computer",
    "shutdown": "shut down", "restart": "restart", "press_enter": "press enter",
    "end_session": "end our conversation", "quit": "close Chitti", "clear_memory": "forget our conversation",
    "lang_telugu": "speak Telugu", "lang_english": "speak English",
    "search_prompt": "search Google", "chat_prompt": "open a WhatsApp chat", "type_prompt": "type something",
    "close_window": "close this window", "close_tab": "close this tab",
}
WINDOW_WORDS = {"window", "this window", "this", "it", "current window", "that"}
TAB_WORDS = {"tab", "this tab", "current tab", "that tab"}
PLAY_WORDS = {"music", "song", "video", "something", "it", "next", "previous", "next song", "previous song",
              "next track", "previous track", "pause", "resume", "the music", "the song"}
YES_WORDS = {"yes", "yeah", "yep", "yup", "yea", "sure", "correct", "right", "okay", "ok", "haan", "ha", "confirm"}
NO_WORDS = {"no", "nope", "nah", "cancel", "wrong", "dont", "don't", "never", "stop"}

# Whisper sometimes invents these from silence/noise
HALLUCINATIONS = {"you", "thank you", "thanks", "thanks for watching", "thank you for watching",
                  "okay", "oh", "uh", "um", "hmm", "mm", "bye bye"}

# ================================================================ LOGGING ====
log = logging.getLogger("chitti")
DEBUG = "--debug" in sys.argv or TEST_MODE


class _Redact(logging.Filter):
    """Never let an API key reach the screen or the log file."""

    def filter(self, record):
        try:
            record.msg = cfg.redact(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


def setup_logging(debug: bool = False) -> None:
    global DEBUG
    DEBUG = DEBUG or debug
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    console.addFilter(_Redact())
    log.addHandler(console)
    try:
        fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        fh.addFilter(_Redact())
        log.addHandler(fh)
    except OSError:
        pass


def sensitive(text: str) -> str:
    """Your words are not written to the log unless you ask for it (LOG_TRANSCRIPTS=true or --debug)."""
    return text if (LOG_TRANSCRIPTS or DEBUG) else f"<{len(text)} chars>"


def echo(kind: str, text: str) -> None:
    """Show what was said on the screen only. This is never written to the log file."""
    print(f"  {kind}: {text}")


class State:
    """What Chitti is doing right now."""
    language = LANGUAGE            # "en" or "te": language she listens for and speaks
    last_interrupted = False       # the user just interrupted her
    end_session = False            # the user asked to end the conversation
    in_conversation = False        # a conversation is open (only then may audio go to the cloud recogniser)


state = State()
speaker = None                     # chitti_voice.Speaker (created in init_engine)
brain = None                       # chitti_brain.Brain   (created in init_engine)
audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=300)     # 30 s of audio at most, never grows forever
_carry: list = []                  # audio of the moment you interrupted her (becomes the start of your sentence)


# ============================================================= VOICE OUTPUT ====
def drain_audio() -> None:
    while True:
        try:
            audio_q.get_nowait()
        except queue.Empty:
            return


class BargeIn:
    """
    Notices that you started talking while Chitti is speaking.

    Her own voice reaches the microphone too, so the loudness of what she plays is known exactly and
    is taken into account: she only stops when you are clearly louder than her echo for BARGE_IN_MS.
    (With headphones there is no echo, so the "headphones" mode is more sensitive.)
    """

    def __init__(self) -> None:
        self.coupling = 0.35      # how loud her voice is at the mic compared with what she plays (learned)
        self.loud_run = 0
        self.quiet = 0
        self.warm = 6
        self.pre: collections.deque = collections.deque(maxlen=12)

    def reset(self) -> None:
        self.loud_run = self.quiet = 0
        self.warm = 6                  # the next thing she says: first listen to her own echo, then watch for you
        self.pre.clear()

    def _user_voice(self, rms: float) -> bool:
        start_th, _ = thresholds()
        ref = speaker.recent_output_rms(0.35) if speaker is not None else 0.0
        if ref < 150:                                   # she is (nearly) silent: normal sensitivity
            self.quiet += 1
            return rms > max(start_th * 1.3, NOISE * 3.0)
        if self.quiet >= 3:                             # a new sentence just started: listen to her echo first
            self.warm = 6
        self.quiet = 0
        if BARGE_IN == "headphones":
            return rms > max(start_th * 1.3, NOISE * 3.0)
        if self.warm > 0:
            self.warm -= 1
            self.coupling = min(3.0, max(0.02, 0.8 * self.coupling + 0.2 * (rms / ref)))
            return False
        loud = rms > max(start_th * 1.5, 2.2 * self.coupling * ref + NOISE * 2.0)
        if not loud:                                    # keep learning how loud her echo is
            self.coupling = min(3.0, max(0.02, 0.95 * self.coupling + 0.05 * (rms / ref)))
        return loud

    def tick(self) -> bool:
        """Look at the microphone. True means: the user is speaking over her, stop talking."""
        global _carry
        need = max(2, int(round(BARGE_IN_MS / 1000 / (BLOCK / SAMPLE_RATE))))
        while True:
            try:
                data = audio_q.get_nowait()
            except queue.Empty:
                return False
            if BARGE_IN == "off":
                continue                                # her own voice is never treated as yours
            self.pre.append(data)
            self.loud_run = self.loud_run + 1 if self._user_voice(chunk_rms(data)) else 0
            if self.loud_run >= need:
                _carry = list(self.pre)[-(need + 3):]
                log.info("interrupted by the user")
                self.reset()
                return True


barge = BargeIn()


def after_speaking() -> None:
    """Short pause, then forget what the microphone heard while she was talking."""
    time.sleep(POST_SPEAK_COOLDOWN)
    drain_audio()
    barge.reset()


def speak(text: str, interruptible: bool = True) -> bool:
    """Say a short sentence and wait for it. Returns False if you interrupted her by speaking."""
    echo("Chitti", text)
    if speaker is None or not speaker.audio_ok:
        return True
    speaker.say(text)
    finished = speaker.wait_idle(barge.tick if interruptible else None)
    if finished:
        after_speaking()
    else:
        state.last_interrupted = True
    return finished


def beep(low: bool = False) -> None:
    if not BEEP:
        return
    try:
        import winsound
        winsound.Beep(600 if low else 1000, 70)
    except Exception:
        pass


def confirm_action(text: str) -> None:
    """Tell the user what was done, AFTER doing it. voice = say it, tone = a short beep (fastest), none = silent."""
    if COMMAND_FEEDBACK == "voice":
        speak(text)
    else:
        echo("Chitti", text)
        if COMMAND_FEEDBACK == "tone":
            beep()


# ============================================== LISTENING / END-OF-SPEECH ====
NOISE = 100.0       # background loudness, learned at startup and kept up to date


def chunk_rms(data: bytes) -> float:
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0


def thresholds():
    """(start_speech, keep_speaking) loudness levels. Two levels stop the sentence flickering on/off."""
    if SPEECH_THRESHOLD:
        return float(SPEECH_THRESHOLD), float(SPEECH_THRESHOLD) * 0.6
    start = max(200.0, NOISE * NOISE_MARGIN) / MIC_SENSITIVITY
    end = min(max(110.0, NOISE * 1.6), start * 0.7)
    return start, end


def audio_callback(indata, frames, t, status):
    data = bytes(indata)
    if TEST_MODE:
        rms = chunk_rms(data)
        start, _ = thresholds()
        bars = "#" * min(40, int(rms / 100))
        print(f"\rmic: {bars:<40} {int(rms):5d}  {'SPEECH' if rms > start else '      '}", end="", flush=True)
    try:
        audio_q.put_nowait(data)
    except queue.Full:                          # processing is behind: drop the oldest audio
        try:
            audio_q.get_nowait()
            audio_q.put_nowait(data)
        except (queue.Empty, queue.Full):
            pass


@dataclass
class Utterance:
    frames: list
    seconds: float
    speech_rms: float
    spec: object = None            # speech recognition that was started early (see Spec)


class Spec:
    """
    Speech recognition started a moment BEFORE she is sure you have finished talking.
    If you stay quiet, the answer is already there when she decides you are done (the recognition time hides
    inside the waiting time). If you carry on talking, the early result is thrown away.
    """

    def __init__(self, frames: list, seconds: float, speech_chunks: int):
        self.speech_chunks = speech_chunks
        self.result = None
        self.done = threading.Event()
        threading.Thread(target=self._run, args=(frames, seconds), daemon=True).start()

    def _run(self, frames, seconds):
        try:
            self.result = _stt(frames, seconds)
        except Exception:
            log.debug("early speech recognition failed")
        finally:
            self.done.set()


def listen_utterance(timeout=None):
    """
    Wait for one spoken sentence and return it when you stop talking.
    Returns an Utterance, or None if nothing was said before the timeout.
    """
    global NOISE, _carry
    block_s = BLOCK / SAMPLE_RATE
    silence_limit = max(1, int(round(SILENCE_TIMEOUT / block_s)))
    spec_after = max(1, int(round(SPEC_AFTER_SECONDS / block_s)))
    max_chunks = int(MAX_UTTERANCE_SECONDS / block_s)
    min_chunks = max(1, int(round(MIN_SPEECH_SECONDS / block_s)))
    pre = collections.deque(maxlen=int(round(PREROLL_SECONDS / block_s)) + 2)

    frames: list = []
    spec = None
    speaking, loud_run, silence, speech_chunks, speech_sum = False, 0, 0, 0, 0.0
    began = time.time()
    if _carry:                                          # you interrupted her: your sentence has already begun
        frames, _carry = list(_carry), []
        speaking = True
        end_th = thresholds()[1]
        levels = [chunk_rms(c) for c in frames]
        speech_chunks = max(2, sum(1 for r in levels if r >= end_th))
        speech_sum = sum(r for r in levels if r >= end_th) or float(NOISE * MIN_SNR * 2)

    while True:
        if not speaking and timeout and time.time() - began > timeout:
            return None
        try:
            data = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        rms = chunk_rms(data)
        start_th, end_th = thresholds()

        if not speaking:
            pre.append(data)
            if rms >= start_th:
                loud_run += 1
            else:
                loud_run = 0
                NOISE = min(3000.0, max(20.0, 0.98 * NOISE + 0.02 * rms))   # slowly follow the room
            if loud_run >= 2:                                                # 0.2 s of real speech
                speaking, frames = True, list(pre)
                speech_chunks, speech_sum, silence = loud_run, rms * loud_run, 0
            continue

        frames.append(data)
        if rms >= end_th:
            silence = 0
            speech_chunks += 1
            speech_sum += rms
            spec = None                                 # you are still talking: forget the early recognition
        else:
            silence += 1
            if (SPECULATIVE_STT and spec is None and silence == spec_after and spec_after < silence_limit
                    and speech_chunks >= min_chunks and speech_sum / max(1, speech_chunks) >= NOISE * MIN_SNR):
                keep = frames[:len(frames) - max(0, silence - 2)]
                spec = Spec(keep, len(keep) * block_s, speech_chunks)       # start recognising now

        if silence >= silence_limit or len(frames) >= max_chunks:
            trim = max(0, silence - 2)                  # do not send long trailing silence to Whisper
            if trim:
                frames = frames[:len(frames) - trim]
            mean_rms = speech_sum / max(1, speech_chunks)
            if speech_chunks < min_chunks:
                log.debug("ignored: too short (%d chunks)", speech_chunks)
            elif mean_rms < NOISE * MIN_SNR:
                log.debug("ignored: too quiet/unclear (speech %.0f vs noise %.0f)", mean_rms, NOISE)
            else:
                early = spec if (spec is not None and spec.speech_chunks == speech_chunks) else None
                return Utterance(frames, len(frames) * block_s, mean_rms, early)
            frames, speaking, loud_run, silence, speech_chunks, speech_sum, spec = [], False, 0, 0, 0, 0.0, None
            pre.clear()


@dataclass
class Heard:
    raw: str
    clean: str
    conf: float
    no_speech: float
    stt_ms: int
    seconds: float
    engine: str = "local"


def normalise(text: str) -> str:
    """Lower case, no punctuation, single spaces. Keeps letters of every language (Telugu vowel signs too)."""
    text = text.lower().replace("\u2019", "'")
    kept = "".join(ch if (ch.isalnum() or ch == "'" or unicodedata.category(ch).startswith("M")) else " "
                   for ch in text)
    return re.sub(r"\s+", " ", kept).strip()


stt_lock = threading.Lock()        # one recognition at a time on the local model
cloud_stt = None                   # chitti_stt.CloudSTT when STT_ENGINE=cloud (created in init_engine)


def _prepare_audio(frames: list):
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return None
    audio -= audio.mean()
    peak = float(np.max(np.abs(audio)))
    if 0 < peak < 0.5:                                  # quiet mic: bring it up (max 10x)
        audio *= min(0.8 / peak, 10.0)
    return audio


def _usable(raw: str, clean: str, conf: float, no_speech: float, avg_lp: float) -> bool:
    """Throw away noise, filler words Whisper invents from silence, and 'not speech' guesses."""
    if not clean or clean in HALLUCINATIONS:
        log.debug("ignored filler/hallucination: %s", sensitive(raw))
        return False
    words = clean.split()
    if len(words) > 5 and len(set(words)) <= 2:
        log.debug("ignored repeated-word noise: %s", sensitive(raw))
        return False
    if no_speech > NO_SPEECH_MAX and avg_lp < -1.0:
        log.debug("ignored: probably not speech (no_speech=%.2f conf=%.2f): %s", no_speech, conf, sensitive(raw))
        return False
    return True


def _stt_local(audio, seconds: float):
    english = state.language == "en"
    kwargs = dict(
        language=state.language, beam_size=BEAM_SIZE, best_of=1, temperature=0.0,
        initial_prompt=PROMPT if english else None, condition_on_previous_text=False,
        vad_filter=STT_VAD, vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
        without_timestamps=True,
    )
    t0 = time.time()
    with stt_lock:
        try:
            segments, _ = whisper.transcribe(audio, hotwords=HOTWORDS if english else None, **kwargs)
        except TypeError:                               # older faster-whisper without hotwords
            segments, _ = whisper.transcribe(audio, **kwargs)
        segments = list(segments)
    stt_ms = int((time.time() - t0) * 1000)
    if not segments:
        log.debug("no speech found in audio (%d ms)", stt_ms)
        return None

    raw = " ".join(s.text.strip() for s in segments).strip()
    clean = normalise(raw)
    weights = [max(1, len(s.text.split())) for s in segments]
    avg_lp = sum(s.avg_logprob * w for s, w in zip(segments, weights)) / sum(weights)
    conf = math.exp(avg_lp)
    no_speech = max(s.no_speech_prob for s in segments)
    if not _usable(raw, clean, conf, no_speech, avg_lp):
        return None
    return Heard(raw, clean, conf, no_speech, stt_ms, seconds, "local")


def _stt_cloud(audio, seconds: float):
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()
    t0 = time.time()
    r = cloud_stt.transcribe(pcm, state.language, PROMPT if state.language == "en" else "", SAMPLE_RATE)
    stt_ms = int((time.time() - t0) * 1000)
    raw = r["text"]
    clean = normalise(raw)
    conf, no_speech = r["conf"], r["no_speech"]
    if not _usable(raw, clean, conf, no_speech, math.log(max(conf, 1e-6))):
        return None
    return Heard(raw, clean, conf, no_speech, stt_ms, seconds, "cloud")


def _stt(frames: list, seconds: float):
    """Speech to text. Cloud (fast, only while a conversation is open) with the local Whisper as the fallback."""
    audio = _prepare_audio(frames)
    if audio is None:
        return None
    if cloud_stt is not None and state.in_conversation:
        try:
            return _stt_cloud(audio, seconds)
        except chitti_stt.STTError as e:
            log.warning("cloud speech recognition failed (%s): using the local one", e)
    return _stt_local(audio, seconds)


def transcribe(u: Utterance):
    """Speech-to-text with a confidence score. Returns Heard or None (noise / nothing)."""
    if u.spec is not None:                              # recognition already running since you went quiet
        u.spec.done.wait(timeout=60)
        h = u.spec.result
    else:
        h = _stt(u.frames, u.seconds)
    if h is not None:
        echo("You", h.raw)
        log.info("heard %s  conf=%.2f no_speech=%.2f stt=%dms audio=%.1fs engine=%s%s", sensitive(h.raw), h.conf,
                 h.no_speech, h.stt_ms, h.seconds, h.engine, " (early)" if u.spec is not None else "")
    return h


def listen_command(timeout=7):
    """Listen for one sentence (no wake word needed) and transcribe it."""
    u = listen_utterance(timeout)
    return transcribe(u) if u else None


def split_wake(text: str):
    """If the wake word is near the start of text, return what comes after it ('' if nothing)."""
    words = text.split()
    for i in range(min(3, len(words))):
        if words[i] in WAKE_VARIANTS:
            return " ".join(words[i + 1:])
    if words and difflib.get_close_matches(words[0], WAKE_VARIANTS, n=1, cutoff=0.8):
        return " ".join(words[1:])
    if len(words) > 1 and difflib.get_close_matches(words[0] + words[1], WAKE_VARIANTS, n=1, cutoff=0.85):
        return " ".join(words[2:])
    return None


def strip_wake(text: str) -> str:
    rest = split_wake(text)
    return text if rest is None else rest


# ============================================== UNDERSTANDING THE COMMAND ====
def squash(s: str) -> str:
    """'What's Up' -> 'whatsup' so spaces and apostrophes don't matter."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clean_name(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s.lower())).strip()


def apply_aliases(t: str) -> str:
    for bad in sorted(ALIASES, key=len, reverse=True):
        t = re.sub(r"(?<![\w'])" + re.escape(bad) + r"(?![\w'])", ALIASES[bad], t)
    return t


def normalize_command(text: str) -> str:
    """Capitals, punctuation, spacing, fillers and common mis-hearings -> a plain command."""
    t = apply_aliases(text.lower().replace("\u2019", "'").replace("`", "'"))
    t = re.sub(r"\b(what|it|that|there|who|how|where)'s\b", r"\1 is", t)
    t = t.replace("today's", "today")
    t = re.sub(r"[^\w' ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    def fix_verb(s: str) -> str:
        w = s.split()
        if w:
            w[0] = VERB_FIXES.get(w[0], w[0])
        return " ".join(w)

    t = fix_verb(t)
    prev = None
    while prev != t:                                    # strip polite padding at the start
        prev = t
        t = re.sub(r"^(?:please|hey|ok|okay|so|just|kindly|can you|could you|would you|will you|"
                   r"i want to|i want you to|i would like to|i need to|let us|lets|let's)\s+", "", t)
    t = re.sub(r"\s+(?:please|for me|right now|now|thank you|thanks)$", "", t)
    t = fix_verb(t)
    t = re.sub(r"^go to\s+", "open ", t)
    t = re.sub(r"^(open|close)\s+up\s+", r"\1 ", t)
    t = re.sub(r"^(open|close)\s+(?:the|my|a|an)\s+", r"\1 ", t)
    t = re.sub(r"^open\s+(?:folder|file)\s+", "open ", t)
    m = re.fullmatch(r"(open|close)\s+(.+?)\s+(?:app|application|browser|program|software|website|site|page|folder)", t)
    if m:
        t = f"{m.group(1)} {m.group(2)}"
    return t


@dataclass
class Intent:
    name: str
    arg: str = ""
    score: float = 0.0
    status: str = "ok"          # ok | ask | ambiguous | notfound | unknown
    options: list = field(default_factory=list)
    target: tuple = None        # (kind, path/url/exe/phone) once a name is resolved

    def label(self) -> str:
        if self.name == "open_app":
            return f"open {self.arg}"
        if self.name == "close_app":
            return f"close {self.arg}"
        if self.name == "chat":
            return f"open chat with {self.arg}"
        if self.name == "search":
            return f"search for {self.arg}"
        if self.name == "play_youtube":
            return f"play {self.arg} on YouTube"
        if self.name == "type":
            return f"type {self.arg}"
        return LABELS.get(self.name, self.name)


# ---- names: apps, folders, files, sites, contacts ----
PRIORITY = {"builtin": 0, "proc": 0, "contact": 0, "folder": 1, "app": 2, "protocol": 3, "site": 4, "file": 5}
OPEN_CANDS: list = []
CLOSE_CANDS: list = []
CONTACT_CANDS: list = []
APPS: dict = {}
FILES: dict = {}


def _junk(name: str) -> bool:
    return any(w in name for w in ("uninstall", "readme", "release notes", "license", "help", "documentation"))


def build_candidates() -> None:
    """Pre-compute everything she can open/close so matching is instant."""
    global OPEN_CANDS, CLOSE_CANDS, CONTACT_CANDS
    open_c = [("builtin", k, v) for k, v in BUILTIN.items()]
    open_c += [("folder", k, v) for k, v in FOLDERS.items()]
    open_c += [("app", k, v) for k, v in APPS.items() if not _junk(k)]
    open_c += [("protocol", k, v) for k, v in PROTOCOLS.items()]
    open_c += [("site", k, v) for k, v in SITES.items()]
    open_c += [("file", k, v) for k, v in FILES.items()]
    close_c = [("proc", k, v) for k, v in PROC_MAP.items()] + [("app", k, v) for k, v in APPS.items() if not _junk(k)]

    def with_alias_forms(items: list) -> list:
        """Each name is matchable as-is AND in its alias form ('visual studio code' is also 'vs code')."""
        out = []
        for kind, key, tgt in items:
            for sk in {squash(key), squash(apply_aliases(key.lower()))}:
                if sk:
                    out.append((kind, key, tgt, sk))
        return out

    OPEN_CANDS = with_alias_forms(open_c)
    CLOSE_CANDS = with_alias_forms(close_c)
    CONTACT_CANDS = [("contact", k, v, squash(k)) for k, v in CONTACTS.items() if squash(k)]


def name_score(sq: str, sk: str) -> float:
    if sq == sk:
        return 1.0
    short, long_ = (sq, sk) if len(sq) <= len(sk) else (sk, sq)
    if len(short) >= 4 and short in long_:
        return 0.88 + 0.10 * len(short) / len(long_)
    return difflib.SequenceMatcher(None, sq, sk).ratio()


def rank_candidates(name: str, cands: list, limit: int = 3) -> list:
    """Best matches as (score, kind, key, target), one per distinct name."""
    sq = squash(name)
    if not sq:
        return []
    best: dict = {}
    for kind, key, target, sk in cands:
        s = name_score(sq, sk)
        if kind == "file":
            s -= 0.07                                    # files are matched more carefully than apps
        cur = best.get(key)
        if cur is None or s > cur[0] + 1e-9 or (abs(s - cur[0]) < 1e-9 and PRIORITY[kind] < PRIORITY[cur[1]]):
            best[key] = (s, kind, key, target)
    return sorted(best.values(), key=lambda r: (-r[0], PRIORITY[r[1]]))[:limit]


def decide(ranked: list) -> str:
    """ok / ask / ambiguous / none for a ranked list of name matches."""
    if not ranked:
        return "none"
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    close_second = bool(second) and second[0] >= ASK_SCORE and top[0] - second[0] < 0.04 and top[0] < 0.999
    if top[0] >= 0.9:
        return "ambiguous" if close_second else "ok"
    if top[0] >= ASK_SCORE:
        return "ambiguous" if close_second else "ask"
    return "none"


def _resolve(name: str, arg: str, cands: list, wrap) -> Intent:
    ranked = rank_candidates(arg, cands)
    verdict = decide(ranked)
    if verdict == "none":
        return Intent(name, arg, ranked[0][0] if ranked else 0.0, "notfound")
    top = wrap(ranked[0])
    if verdict == "ok":
        return top
    top.status = verdict
    top.options = [wrap(r) for r in (ranked[:1] if verdict == "ask" else ranked[:2])]
    return top


def resolve_open(arg: str) -> Intent:
    return _resolve("open_app", arg, OPEN_CANDS, lambda r: Intent("open_app", r[2], r[0], "ok", target=(r[1], r[3])))


def resolve_close(arg: str) -> Intent:
    return _resolve("close_app", arg, CLOSE_CANDS, lambda r: Intent("close_app", r[2], r[0], "ok", target=(r[1], r[3])))


def resolve_contact(arg: str):
    ranked = rank_candidates(arg, CONTACT_CANDS)
    return ranked[0] if ranked else None


# ---- commands without a name: similarity to example phrases ----
def phrase_score(t: str, p: str) -> float:
    if t == p:
        return 1.0
    base = difflib.SequenceMatcher(None, t, p).ratio()
    ta, tb = set(t.split()), set(p.split())
    jac = len(ta & tb) / len(ta | tb)
    score = max(base, 0.95 * jac)
    for a, b in OPPOSITES:
        if (a in ta and b in tb and a not in tb and b not in ta) or (b in ta and a in tb and b not in tb and a not in ta):
            score *= 0.4                                  # "up" vs "down" can never be the same command
    return score


def match_templates(t: str) -> list:
    scores = {name: max(phrase_score(t, p) for p in phrases) for name, phrases in INTENT_PHRASES.items()}
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def parse_intent(text: str) -> Intent:
    t = normalize_command(text)
    if not t:
        return Intent("none", status="unknown")

    # 1. commands that carry a word or name
    m = re.fullmatch(r"(?:search for|search google for|search|google|look up) (.+)", t)
    if m:
        return Intent("search", m.group(1), 1.0)
    m = re.fullmatch(r"type (.+)", t)
    if m:
        return Intent("type", m.group(1), 1.0)
    m = re.fullmatch(r"play (.+)", t)
    if m and m.group(1) not in PLAY_WORDS:
        return Intent("play_youtube", m.group(1), 0.9)
    m = re.fullmatch(r"(?:open |whatsapp )?chat (?:with |of |for )?(.+)", t)
    if m:
        r = resolve_contact(m.group(1))
        if r and r[0] >= 0.85:
            return Intent("chat", r[2], r[0], "ok", target=("contact", r[3]))
        if r and r[0] >= 0.7:
            opt = Intent("chat", r[2], r[0], "ok", target=("contact", r[3]))
            return Intent("chat", r[2], r[0], "ask", options=[opt], target=opt.target)
        return Intent("chat", m.group(1), 0.8, "ok", target=("search", ""))   # not in CONTACTS: search by name
    m = re.fullmatch(r"(open|close) (.+)", t)
    if m:
        verb, arg = m.groups()
        if verb == "close":
            if arg in WINDOW_WORDS:
                return Intent("close_window", score=1.0)
            if arg in TAB_WORDS:
                return Intent("close_tab", score=1.0)
            return resolve_close(arg)
        if arg in ("chat", "whatsapp chat"):
            return Intent("chat_prompt", score=1.0)
        return resolve_open(arg)

    # 2. volume in natural words ("make it louder", "turn the volume down")
    if (len(t.split()) <= 6 and not re.match(r"^(how|what|why|when|where|which|who|is|are|can i|should|do|does)\b", t)
            and re.search(r"\b(volume|sound|louder|quieter|softer)\b", t) and "mute" not in t):
        up = re.search(r"\b(up|increase|raise|higher|louder|more|boost)\b", t)
        down = re.search(r"\b(down|decrease|lower|reduce|quieter|softer|less)\b", t)
        if up and not down:
            return Intent("volume_up", score=0.9)
        if down and not up:
            return Intent("volume_down", score=0.9)

    # 3. everything else: closest example phrase, but only if clearly the best
    if len(t.split()) > 8:                                  # a long sentence is a conversation, not a command
        return Intent("none", status="unknown")
    ranked = match_templates(t)
    (name1, s1), (name2, s2) = ranked[0], ranked[1]

    # a bare contact name ("ravi") opens that chat, unless it is clearly a normal command
    if CONTACT_CANDS and len(t.split()) <= 3:
        r = resolve_contact(t)
        if r and r[0] >= 0.85 and r[0] > s1 + 0.05:
            return Intent("chat", r[2], r[0], "ok", target=("contact", r[3]))
        if r and r[0] >= 0.7 and r[0] > s1:
            opt = Intent("chat", r[2], r[0], "ok", target=("contact", r[3]))
            return Intent("chat", r[2], r[0], "ask", options=[opt], target=opt.target)

    need = 0.9 if name1 in DANGEROUS_INTENTS else ACCEPT_SCORE
    if s1 >= need and (s1 - s2 >= MARGIN or s1 >= 0.99):
        return Intent(name1, "", s1)
    if s1 >= PHRASE_ASK_SCORE:
        if len(t.split()) >= 4:                             # loosely like a command but long: leave it to the AI
            return Intent("none", score=s1, status="unknown")
        opts = [Intent(name1, "", s1)]
        if s2 >= PHRASE_ASK_SCORE and s1 - s2 < MARGIN * 2:
            opts.append(Intent(name2, "", s2))
        first = opts[0]
        first.status = "ambiguous" if len(opts) > 1 else "ask"
        first.options = opts
        return first
    return Intent("none", score=s1, status="unknown")


# ================================================================ ACTIONS ====
_pag = None


def pag():
    """pyautogui is imported only when a keyboard/screen action is needed."""
    global _pag
    if _pag is None:
        import pyautogui
        _pag = pyautogui
    return _pag


def open_url(url: str) -> None:
    try:
        subprocess.Popen(["cmd", "/c", "start", "", "chrome", url], creationflags=0x08000000)
    except Exception:
        import webbrowser
        webbrowser.open(url)


def web_search(query: str) -> None:
    open_url("https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
    confirm_action(f"Searching Google for {query}")


def windows_search(name: str) -> None:
    speak(f"Searching Windows for {name}")
    p = pag()
    p.press("win")
    time.sleep(0.6)
    p.write(name, interval=0.05)
    time.sleep(0.8)
    p.press("enter")


def open_whatsapp_chat(name: str, phone: str = "") -> None:
    if phone:
        os.startfile(f"whatsapp://send?phone={phone}")
        confirm_action(f"Opening chat with {name}")
        return
    began = time.time()
    try:
        os.startfile("whatsapp:")
    except OSError:
        os.startfile(APPS.get("whatsapp", "whatsapp:"))
    confirm_action(f"Opening WhatsApp and looking for {name}")      # said while WhatsApp is starting up
    time.sleep(max(0.0, WHATSAPP_WAIT - (time.time() - began)))
    p = pag()
    p.hotkey("ctrl", "f")                       # focus WhatsApp's search box
    time.sleep(0.6)
    p.write(name, interval=0.05)
    time.sleep(1.5)
    if WHATSAPP_PRESS_DOWN:
        p.press("down")
        time.sleep(0.2)
    p.press("enter")


def lnk_target_exe(path: str) -> str:
    """Find which .exe a Start Menu shortcut launches."""
    safe = path.replace("'", "''")
    ps = f"(New-Object -ComObject WScript.Shell).CreateShortcut('{safe}').TargetPath"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=8, creationflags=0x08000000).stdout.strip()
        return os.path.basename(out) if out.lower().endswith(".exe") else ""
    except Exception:
        return ""


def _kill(*criteria) -> bool:
    """Ask the app to close politely first; if Windows refuses, force it."""
    for extra in ([], ["/F"]):
        r = subprocess.run(["taskkill", *criteria, *extra], capture_output=True, creationflags=0x08000000)
        if r.returncode == 0:
            return True
    return False


def do_open(intent: Intent) -> str:
    kind, target = intent.target
    if kind == "builtin":                       # do it first ...
        subprocess.Popen(target, shell=True)
    elif kind == "site":
        open_url(target)
    else:                                       # app, folder, file, protocol
        os.startfile(target)
    confirm_action(f"Opening {intent.arg}")     # ... then say it
    return f"opened {kind} '{intent.arg}'"


def do_close(intent: Intent) -> str:
    kind, target = intent.target
    exe = target if kind == "proc" else lnk_target_exe(target)
    if exe and exe.lower() in PROTECTED:
        speak("I will not close that, it is a system program")
        return f"refused to close protected program {exe}"
    ok = _kill("/IM", exe) if exe else False
    if not ok:
        ok = _kill("/FI", f"WINDOWTITLE eq *{intent.arg}*")      # fall back to the window title
    if ok:
        confirm_action(f"Closed {intent.arg}")
    else:
        speak(f"I could not close {intent.arg}. Maybe it is not open")
    return f"closed {intent.arg}" if ok else f"could not close {intent.arg}"


# ================================================================ DIALOGUE ====
def ask_yes_no(question: str):
    """True / False, or None if she got no clear answer."""
    speak(question)
    h = listen_command(6)
    if not h:
        return None
    words = set(h.clean.split())
    if words & YES_WORDS and not words & NO_WORDS:
        return True
    if words & NO_WORDS and not words & YES_WORDS:
        return False
    return None


def ask_free(prompt: str, timeout: int = 8) -> str:
    """Ask a question and return what you said, with original capitalisation."""
    speak(prompt)
    h = listen_command(timeout)
    return h.raw.strip(" .?!") if h else ""


_recent: dict = {}


def is_duplicate(key: str) -> bool:
    now = time.time()
    for k in [k for k, ts in _recent.items() if now - ts > DEDUP_SECONDS]:
        del _recent[k]
    if key in _recent:
        return True
    _recent[key] = now
    return False


def confirm_intent(intent: Intent):
    """Return the Intent to run, or None if she is not sure and the user did not confirm."""
    if intent.status == "ok":
        return intent
    if intent.status in ("ask", "ambiguous"):
        for opt in intent.options[:2]:
            ans = ask_yes_no(f"Did you mean {opt.label()}?")
            if ans:
                return opt
            if ans is None:
                break
        speak("Okay, cancelled")
    return None


def execute(intent: Intent, heard: Heard):
    """Run an Intent. Returns (keep_running, result_text)."""
    n, a = intent.name, intent.arg
    p = pag

    if n == "end_session":
        state.end_session = True
        speak(random.choice(["Okay, talk to you later.", "Alright. Just say my name when you need me.",
                             "Sure, I'll be here."]))
        return True, "conversation ended"
    if n == "quit":
        speak("Goodbye.")
        return False, "quit"
    if n == "clear_memory":
        if brain is not None:
            brain.conv.clear()
        speak("Okay, I've forgotten our conversation.")
        return True, "memory cleared"
    if n in ("lang_telugu", "lang_english"):
        want = "te" if n == "lang_telugu" else "en"
        if want == "te" and WHISPER_MODEL.endswith(".en"):
            speak("To understand Telugu I need the multilingual speech model. "
                  "Please set WHISPER_MODEL to small in the settings file.")
            return True, "telugu needs a multilingual speech model"
        state.language = want
        if speaker is not None:
            speaker.default_lang = want
        if want == "te":
            if speaker is not None and not speaker.chain.supports("te"):
                speak("Okay. I'll listen in Telugu, but my Telugu voice needs the internet.")
            else:
                speak("\u0c38\u0c30\u0c47, \u0c07\u0c2a\u0c4d\u0c2a\u0c1f\u0c3f \u0c28\u0c41\u0c02\u0c1a\u0c3f "
                      "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41\u0c32\u0c4b \u0c2e\u0c3e\u0c1f\u0c4d\u0c32\u0c3e\u0c21\u0c26\u0c3e\u0c02.")
        else:
            speak("Okay, English it is.")
        return True, f"language set to {want}"
    if n == "open_app":
        return True, do_open(intent)
    if n == "close_app":
        if heard.conf < SAFE_CONFIDENCE and ask_yes_no(f"Close {a}?") is not True:
            speak("Okay, I will leave it open")
            return True, "close cancelled (low confidence)"
        return True, do_close(intent)

    if n in ("search", "search_prompt"):
        query = a or ask_free("What should I search for?")
        if not query:
            return True, "search: nothing heard"
        web_search(query)
        return True, f"searched '{query}'"
    if n == "play_youtube":
        speak(f"Playing {a} on YouTube")
        open_url("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(a))
        return True, f"youtube '{a}'"

    if n in ("chat", "chat_prompt"):
        name, phone = a, ""
        if n == "chat_prompt":
            name = ask_free("Whose chat should I open?")
            if not name:
                return True, "chat: nothing heard"
            r = resolve_contact(name)
            if r and r[0] >= 0.85:
                name, phone = r[2], r[3]
        elif intent.target and intent.target[0] == "contact":
            phone = intent.target[1]
        open_whatsapp_chat(name, phone)
        return True, f"whatsapp chat '{name}'"

    if n in ("type", "type_prompt"):
        text = a
        if n == "type":                           # keep the capital letters you spoke
            m = re.search(r"\btype\b[\s,:]*(.*)$", heard.raw, flags=re.I)
            text = (m.group(1) if m and m.group(1).strip() else a).strip()
        else:
            text = ask_free("What should I type?")
        if text:
            p().write(text, interval=0.02)
        return True, f"typed {len(text)} characters"

    if n == "press_enter":
        p().press("enter")
    elif n == "close_window":
        p().hotkey("alt", "f4")
    elif n == "close_tab":
        p().hotkey("ctrl", "w")
    elif n == "volume_up":
        for _ in range(5):
            p().press("volumeup")
    elif n == "volume_down":
        for _ in range(5):
            p().press("volumedown")
    elif n == "mute":
        p().press("volumemute")
    elif n in ("pause", "play"):
        p().press("playpause")
    elif n == "next":
        p().press("nexttrack")
    elif n == "previous":
        p().press("prevtrack")
    elif n == "screenshot":
        folder = FOLDERS["pictures"]
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"shot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
        p().screenshot(path)
        speak("Screenshot saved in Pictures")
    elif n == "time":
        speak(datetime.datetime.now().strftime("It is %I:%M %p"))
    elif n == "date":
        speak(datetime.datetime.now().strftime("Today is %A, %d %B %Y"))
    elif n == "lock":
        subprocess.Popen("rundll32.exe user32.dll,LockWorkStation")
    elif n in ("shutdown", "restart"):
        if ask_yes_no(f"Say yes to confirm {LABELS[n]}") is True:
            subprocess.Popen(f"shutdown /{'s' if n == 'shutdown' else 'r'} /t 5", shell=True)
        else:
            speak("Cancelled")
            return True, f"{n} cancelled"
    else:
        speak("Sorry, I can't do that yet")
        return True, f"no action for {n}"
    return True, f"done {n}"


def run_command(cmd_text: str, heard: Heard, retry: bool = True) -> bool:
    """Understand and run one command. Returns False when the assistant should quit."""
    # 1. confidence gate: never act on speech she was not sure about
    if heard.conf < MIN_CONFIDENCE:
        log.info("LOW CONFIDENCE %.2f (<%.2f) for %s", heard.conf, MIN_CONFIDENCE, sensitive(cmd_text))
        if not retry:
            speak("I am still not sure what you said. Please try again later.")
            return True
        speak("Sorry, I did not catch that. Please say it again.")
        h2 = listen_command(7)
        return run_command(strip_wake(h2.clean), h2, retry=False) if h2 else True

    # 2. understand
    intent = parse_intent(cmd_text)
    shown_arg = intent.arg if intent.name in ("open_app", "close_app") else sensitive(intent.arg)
    log.info("intent=%s arg=%s score=%.2f status=%s conf=%.2f", intent.name, shown_arg, intent.score,
             intent.status, heard.conf)
    if DEBUG:
        log.debug("normalized: %s  top matches: %s", normalize_command(cmd_text),
                  [(n, round(s, 2)) for n, s in match_templates(normalize_command(cmd_text))[:3]])

    if intent.status == "unknown":                          # not a computer command: have a conversation
        return chat_reply(heard)
    if intent.status == "notfound":
        if intent.name == "open_app" and ask_yes_no(f"I could not find {intent.arg}. Shall I search Windows for it?"):
            windows_search(intent.arg)
        else:
            speak(f"I could not find {intent.arg} on this computer")
        return True

    # 3. ask when not sure
    chosen = confirm_intent(intent)
    if chosen is None:
        log.info("result: cancelled / not confirmed")
        return True

    # 4. one spoken command runs once
    dup_key = f"{chosen.name}|{chosen.arg}"
    if is_duplicate(dup_key):
        log.info("result: ignored duplicate %s", chosen.label())
        return True

    # 5. run it
    try:
        keep, result = execute(chosen, heard)
    except Exception:
        log.exception("action failed")
        speak("Sorry, something went wrong with that.")
        return True
    log.info("result: %s", result)
    if "cancel" in result:
        _recent.pop(dup_key, None)                              # you said no: asking again is allowed at once
    if brain is not None and keep:
        brain.note_action(strip_wake_raw(heard.raw), result)   # so follow-up questions make sense
    return keep


# ============================================================ CONVERSATION ====
# Words that only mean "be quiet" when you say them while she is talking
INTERRUPT_WORDS = {"stop", "wait", "hold on", "enough", "okay stop", "ok stop", "quiet", "silence", "be quiet",
                   "shh", "one second", "just a second", "sorry", "never mind", "okay", "ok"}
ACKS_EN = ["Yes?", "I'm listening.", "Go ahead.", "Mm-hm?", "I'm here."]
ACKS_TE = ["\u0c1a\u0c46\u0c2a\u0c4d\u0c2a\u0c02\u0c21\u0c3f.", "\u0c35\u0c3f\u0c02\u0c1f\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c28\u0c41."]
_last_ack = [""]
# Said when the AI is slow to start, like a person saying "hmm". Prepared in advance so they play instantly.
FILLERS = ["Hmm.", "Let me think.", "Okay.", "One moment."]
_last_filler = [""]
PREWARM_PHRASES = (ACKS_EN + FILLERS + [
    "Sure.", "Alright.", "Okay, talk to you later.", "Alright. Just say my name when you need me.",
    "Sure, I'll be here.", "Goodbye.", "Okay, English it is.", "Sorry, I did not understand that.",
    "Sorry, I did not catch that. Please say it again.", "Okay, cancelled", "Cancelled", "Okay, I've forgotten our conversation.",
])


def pick_filler() -> str:
    """A prepared 'hmm' (never the same twice in a row), or '' if none is ready yet."""
    ready = [f for f in FILLERS if speaker is not None and speaker.is_cached(f, "en") and f != _last_filler[0]]
    if not ready:
        return ""
    _last_filler[0] = random.choice(ready)
    return _last_filler[0]


def pick_ack() -> str:
    """A short 'I'm listening', never the same one twice in a row."""
    pool = ACKS_TE if state.language == "te" else ACKS_EN
    choices = [a for a in pool if a != _last_ack[0]] or pool
    _last_ack[0] = random.choice(choices)
    return _last_ack[0]


_WAKE_RAW = None


def strip_wake_raw(raw: str) -> str:
    """Remove the wake word from the start of what you said, keeping capitals and punctuation for the AI."""
    global _WAKE_RAW
    if _WAKE_RAW is None:
        variants = sorted((re.escape(v) for v in WAKE_VARIANTS), key=len, reverse=True)
        _WAKE_RAW = re.compile(r"^\W*(?:(?:hey|ok|okay|hi|hello)\W+)?(?:" + "|".join(variants) + r")[\W_]*", re.I)
    out = _WAKE_RAW.sub("", raw, count=1).strip()
    return out or raw.strip()


def chat_reply(heard: Heard) -> bool:
    """
    Answer an ordinary sentence with the AI. Each sentence is spoken as soon as it is ready, while the
    rest is still being written. Speaking over her stops the answer at once.
    """
    user_text = strip_wake_raw(heard.raw)
    if brain is None or not brain.enabled:
        speak("I can only run computer commands right now. To chat with me, set up an AI model in the settings file.")
        return True

    cancel = chitti_brain.CancelToken()
    gate = threading.Lock()                 # makes "check cancelled" and "queue a sentence" one step
    sentences: list = []
    ids: list = []
    errors: list = []
    first: list = []
    done = threading.Event()
    began = time.time()

    def produce() -> None:
        try:
            for s in brain.reply_stream(user_text, cancel, state.language):
                with gate:
                    if cancel.is_set():
                        break
                    if not first:
                        first.append(time.time() - began)
                    sentences.append(s)
                    ids.append(speaker.say(s))
                echo("Chitti", s)
        except chitti_brain.LLMError as e:
            errors.append(e)
        except Exception as e:
            errors.append(chitti_brain.LLMError("unknown", type(e).__name__))
        finally:
            done.set()

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()

    interrupted = False
    filler_said = False
    while True:
        if barge.tick():                                    # you started talking
            interrupted = True
            break
        if done.is_set() and not speaker.busy():            # everything has been written and spoken
            break
        if (not filler_said and not sentences and state.language == "en"
                and time.time() - began > FILLER_AFTER_SECONDS):
            filler_said = True                              # the AI is slow: say "hmm" like a person would
            f = pick_filler()
            if f:
                speaker.say(f)
        time.sleep(0.02)
    if interrupted:
        with gate:
            cancel.set()
            speaker.stop()
        state.last_interrupted = True
    worker.join(timeout=1.0)

    if speaker.audio_ok:
        spoken = " ".join(s for s, i in zip(sentences, ids) if speaker.started(i))
    else:
        spoken = " ".join(sentences)
    if interrupted:
        spoken = (spoken + " [the user interrupted here]").strip()
    if sentences or not errors:
        brain.commit(user_text, spoken)                     # remember only what was really said
    log.info("chat: provider=%s first_sentence=%s sentences=%d interrupted=%s error=%s",
             brain.last_provider or "-", f"{first[0]:.2f}s" if first else "-", len(sentences), interrupted,
             errors[0].kind if errors else "-")

    if errors and not sentences:
        speak(errors[0].spoken())
    elif not interrupted:
        after_speaking()
    return True


def route(cmd_text: str, heard: Heard) -> bool:
    """One thing the user said: a computer command, or a normal conversation sentence."""
    if state.last_interrupted:
        state.last_interrupted = False
        if normalize_command(cmd_text) in INTERRUPT_WORDS:      # "stop" / "wait" while she was talking
            log.info("stop word after interruption: just listening")
            return True
    return run_command(cmd_text, heard)


def handle_request(cmd_text: str, heard: Heard) -> bool:
    if not cmd_text.strip():                      # only the wake word was said
        speak(pick_ack())
        h2 = listen_command(7)
        if not h2:
            return True
        cmd_text, heard = strip_wake(h2.clean), h2
    return route(cmd_text, heard)


# ============================================================ START-UP ====
def build_app_index() -> dict:
    roots = [
        os.path.join(os.environ.get("ProgramData", ""), r"Microsoft\Windows\Start Menu\Programs"),
        os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
    ]
    apps = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".lnk", ".url")):
                    apps[os.path.splitext(f)[0].lower()] = os.path.join(dirpath, f)
    return apps


def build_file_index(max_depth: int = 3) -> dict:
    found = {}
    for base in FOLDERS.values():
        if not os.path.isdir(base):
            continue
        base_depth = base.count(os.sep)
        for dirpath, dirs, files in os.walk(base):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
                continue
            for name in dirs + files:
                found.setdefault(os.path.splitext(name)[0].lower(), os.path.join(dirpath, name))
    return found


whisper = None
PROMPT = ""
HOTWORDS = None


def init_engine() -> None:
    """Load Whisper, scan the PC, prepare the vocabulary hints."""
    global whisper, APPS, FILES, PROMPT, HOTWORDS, speaker, brain, cloud_stt
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("Missing package. Run:  pip install faster-whisper sounddevice pyautogui numpy")

    log.info("Loading Whisper model '%s' (first run downloads it, needs internet once)...", WHISPER_MODEL)
    threads = max(2, min(8, (os.cpu_count() or 4) // 2))        # logical cores / 2 ~ real cores: fastest
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=threads)

    log.info("Scanning your PC...")
    APPS = build_app_index()
    FILES = build_file_index()
    build_candidates()
    log.info("Indexed %d apps and %d files/folders.", len(APPS), len(FILES))

    names = [clean_name(k) for k in list(BUILTIN) + list(APPS) if clean_name(k) and not _junk(k)]
    names = [n for n in dict.fromkeys(names) if len(n.split()) <= 3][:25]
    PROMPT = "Chitti. Chitti, open Chrome. Chitti, close WhatsApp. Chitti, volume up. " + " ".join(f"{c}." for c in CONTACTS)
    HOTWORDS = ", ".join(["Chitti", "Chrome", "WhatsApp", "YouTube"] + list(CONTACTS)[:6])

    if state.language != "en" and WHISPER_MODEL.endswith(".en"):
        log.warning("'%s' only understands English; starting in English. Use WHISPER_MODEL=small for Telugu.",
                    WHISPER_MODEL)
        state.language = "en"
    list(whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language=state.language)[0])   # warm-up

    # voice
    speaker = chitti_voice.build_speaker(default_lang=state.language)
    speaker.start()
    log.info("Voice engines (tried in this order): %s%s", ", ".join(e.name for e in speaker.chain.engines) or "NONE",
             "" if speaker.audio_ok else "   [AUDIO OUTPUT NOT AVAILABLE: " + speaker.last_error + "]")

    # AI brain
    usable = []
    for p in chitti_brain.build_providers():
        ok, why = p.check()
        log.info("AI model '%s' (%s): %s", p.name, getattr(p, "cost", ""), why)
        if ok:
            usable.append(p)
    brain = chitti_brain.Brain(usable, max_tokens=cfg.get_int("LLM_MAX_TOKENS", 300),
                               retries=cfg.get_int("LLM_RETRIES", 1))
    if not brain.enabled:
        log.info("No AI model is available, so Chitti will only run computer commands. See .env.example.")

    # fast cloud speech recognition (optional)
    cloud_stt = chitti_stt.build_cloud_stt()
    if cfg.get("STT_ENGINE", "local").lower() == "cloud":
        if cloud_stt is None:
            log.warning("STT_ENGINE=cloud but STT_API_KEY / STT_MODEL are not set: using local Whisper only")
        else:
            log.info("Speech recognition: local Whisper while waiting for her name; cloud '%s' during a conversation "
                     "(your audio is sent to that service only then)", cloud_stt.model)

    # get ready: instant phrases, open connections
    speaker.prewarm(PREWARM_PHRASES, "en")
    if state.language == "te":
        speaker.prewarm(ACKS_TE, "te")
    brain.warm()
    if cloud_stt is not None:
        cloud_stt.warm()


def calibrate_noise() -> None:
    """Measure the quiet room so she knows what counts as speech."""
    global NOISE
    log.info("Measuring background noise, stay quiet for 2 seconds...")
    levels, t0 = [], time.time()
    while len(levels) < 20 and time.time() - t0 < 6:
        try:
            levels.append(chunk_rms(audio_q.get(timeout=1)))
        except queue.Empty:
            pass
    NOISE = max(20.0, float(np.median(levels))) if levels else 100.0
    start, end = thresholds()
    log.info("Background level %.0f -> speech starts above %.0f", NOISE, start)


class Session:
    """A conversation stays open for a while after the wake word, so you do not have to repeat it."""

    def __init__(self) -> None:
        self.active = False
        self.until = 0.0

    def start(self) -> None:
        self.active = True
        self.touch()

    def touch(self) -> None:
        self.until = time.time() + CONVERSATION_IDLE_SECONDS

    def remaining(self) -> float:
        return max(0.5, self.until - time.time())

    def end(self) -> None:
        self.active = False


def assistant_loop() -> None:
    if TEST_MODE:
        log.info("TEST MODE: speak and watch what she heard and what she WOULD do. Ctrl+C to stop.")
        while True:
            u = listen_utterance()
            h = transcribe(u) if u else None
            if not h:
                continue
            rest = split_wake(h.clean)
            if rest is None:
                print(f"\n  (no wake word '{WAKE_WORD}' heard, so she would ignore this)")
                continue
            i = parse_intent(rest)
            low = "  LOW CONFIDENCE -> would ask you to repeat" if h.conf < MIN_CONFIDENCE else ""
            what = "chat with the AI" if i.status == "unknown" else f"intent={i.name} arg={i.arg!r}"
            print(f"\n  command={rest!r} {what} score={i.score:.2f} status={i.status} conf={h.conf:.2f}{low}")

    speak(f"I'm ready. Just say {WAKE_WORD} when you want me.")
    session = Session()
    while True:
        u = listen_utterance(session.remaining() if session.active else None)
        if u is None:                                       # quiet for the whole conversation window
            if session.active:
                session.end()
                state.in_conversation = False
                beep(low=True)
                log.info("conversation ended (quiet for %d s)", CONVERSATION_IDLE_SECONDS)
            continue
        h = transcribe(u)
        if not h:
            continue

        if session.active:
            if h.conf < IGNORE_CONFIDENCE:
                log.debug("ignored unclear speech inside the conversation (conf %.2f)", h.conf)
                continue
            rest = strip_wake(h.clean)
        else:
            rest = split_wake(h.clean)
            if rest is None:
                log.debug("no wake word, ignoring")
                continue
            beep()
            if brain is not None:
                brain.warm()                                # open the connection to the AI while you are still talking
            if cloud_stt is not None:
                cloud_stt.warm()
            if CONVERSATION_MODE == "continuous":
                session.start()

        state.in_conversation = True
        try:
            keep = handle_request(rest, h)
        except Exception:
            log.exception("error while handling a request")
            keep = True
            speak("Sorry, something went wrong.")
        if not keep:
            state.in_conversation = False
            return
        if state.end_session:
            state.end_session = False
            session.end()
        elif session.active:
            session.touch()                                 # the quiet timer starts after she finishes talking
        state.in_conversation = session.active


def run_forever() -> None:
    import sounddevice as sd
    calibrated = False
    while True:
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype="int16", channels=1,
                                   device=MIC_DEVICE, callback=audio_callback):
                if not calibrated:
                    calibrate_noise()
                    calibrated = True
                assistant_loop()
                return
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("Microphone problem, restarting listening in 2 seconds")
            time.sleep(2)


# ================================================================= TOOLS ====
def tts_test() -> None:
    """--tts-test: hear the voice, and see how fast each sentence is prepared."""
    print(f"\nVoices, in the order they are tried: {[e.name for e in speaker.chain.engines] or 'NONE'}")
    if not speaker.audio_ok:
        print("Audio output is not available:", speaker.last_error)
        return
    now = datetime.datetime.now()
    samples = [
        ("en", f"Hi, I'm Chitti. It's a pleasure to meet you. Today is {now:%A}, the {now.day} of {now:%B}, "
               f"and it's {now:%I:%M %p} right now. I can open apps, search the web, or just have a chat."),
        ("te", "\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02, \u0c28\u0c47\u0c28\u0c41 \u0c1a\u0c3f\u0c1f\u0c4d\u0c1f\u0c3f. "
               "\u0c2e\u0c40\u0c15\u0c41 \u0c0e\u0c32\u0c3e \u0c38\u0c39\u0c3e\u0c2f\u0c02 \u0c1a\u0c47\u0c2f\u0c17\u0c32\u0c28\u0c41?"),
    ]
    for lang, text in samples:
        if not speaker.chain.supports(lang):
            print(f"\n[{lang}] no voice available for this language (Telugu needs the Edge voice and internet)")
            continue
        sentences = chitti_brain.SentenceChunker()
        parts = sentences.feed(text + " ") + sentences.flush()
        print(f"\n[{lang}] speaking {len(parts)} sentence(s)...")
        t0 = time.time()
        first_ms = None
        ids = [speaker.say(p, lang) for p in parts]
        while speaker.busy():
            if first_ms is None and speaker.started(ids[0]):
                first_ms = int((time.time() - t0) * 1000)
            time.sleep(0.01)
        print(f"    first sound after {first_ms} ms, voice used: {speaker.chain.last_engine}"
              f"{'   <-- NOT the neural voice, check internet/edge-tts' if speaker.chain.last_engine == 'windows' else ''}")
        if speaker.last_error:
            print("    last problem:", speaker.last_error)


def diagnostics() -> None:
    """--diag: measure every stage of a conversation turn on this PC."""
    print("\n=== Chitti diagnostics ===")
    print(f"Speech recognition : Whisper '{WHISPER_MODEL}', language {state.language}, beam {BEAM_SIZE}"
          + (f"   + cloud '{cloud_stt.model}' during conversations" if cloud_stt else ""))
    print(f"Speed settings     : silence {SILENCE_TIMEOUT}s, early recognition {SPECULATIVE_STT}, "
          f"command feedback '{COMMAND_FEEDBACK}', first AI chunk {cfg.get_int('FIRST_CHUNK_CHARS', 45)} chars")
    print(f"AI model           : {brain.describe()}")
    print(f"Voice engines      : {[e.name for e in speaker.chain.engines] or 'NONE'}   audio output ok: {speaker.audio_ok}")
    print(f"Interruptions      : {BARGE_IN}   conversation: {CONVERSATION_MODE} ({CONVERSATION_IDLE_SECONDS:.0f} s)")

    ai_ms = tts_ms = stt_ms = None
    if brain.enabled:
        times = []
        for _ in range(2):                            # the second question shows the benefit of the open connection
            cancel = chitti_brain.CancelToken()
            t0 = time.time()
            try:
                for sentence in brain.reply_stream("In one short sentence, say hello and tell me your name.", cancel):
                    times.append(int((time.time() - t0) * 1000))
                    cancel.set()
                    break
            except chitti_brain.LLMError as e:
                print(f"\nAI test failed     : {e.kind} - {e}")
                break
        if times:
            ai_ms = times[-1]
            print(f"\nAI first sentence  : {times[0]} ms the first time"
                  + (f", {times[1]} ms the next time (connection already open)" if len(times) > 1 else "")
                  + f"   [{brain.last_provider or 'streaming'}]")
    else:
        print("\nAI test skipped (no AI model is set up)")

    for e in speaker.chain.engines:
        if not e.supports("en"):
            continue
        t0 = time.time()
        try:
            e.synth("Hello, this is a speed test of my voice.", "en")
            ms = int((time.time() - t0) * 1000)
            print(f"Voice '{e.name}' makes a sentence in {ms} ms")
            if e is speaker.chain.engines[0]:
                tts_ms = ms
        except Exception as ex:
            print(f"Voice '{e.name}' failed: {type(ex).__name__}")

    import sounddevice as sd
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype="int16", channels=1,
                               device=MIC_DEVICE, callback=audio_callback):
            calibrate_noise()
            print("\nSay a sentence now, for example 'chitti what's the time' ...")
            u = listen_utterance(timeout=12)
            if u is None:
                print("Nothing heard. Check the microphone (python chitti.py --mics) or MIC_SENSITIVITY.")
            else:
                h = transcribe(u)
                if h:
                    stt_ms = h.stt_ms
                    print(f"Heard: {h.raw!r}  confidence {h.conf:.2f}  recognised in {h.stt_ms} ms")
    except Exception as ex:
        print(f"\nMicrophone test failed: {type(ex).__name__}. Check Windows microphone permission and MIC_DEVICE.")

    parts = [("waiting for end of speech", int(SILENCE_TIMEOUT * 1000)), ("speech recognition", stt_ms),
             ("AI first sentence", ai_ms), ("voice first sentence", tts_ms)]
    known = [v for _, v in parts if v is not None]
    print("\nEstimated time from the end of your sentence to the first sound of her answer:")
    for label, v in parts:
        print(f"   {label:28} {v if v is not None else '?':>6} ms")
    if known:
        print(f"   {'TOTAL':28} {sum(known):>6} ms   (under 2500 ms feels natural)")
    print("\nSlow? Smaller WHISPER_MODEL, smaller/faster AI model, or a lower SILENCE_TIMEOUT. See .env.example.")


# =============================================================== SELF-TEST ====
def selftest() -> int:
    """Checks understanding of many phrasings without a microphone. Returns 0 if everything passes."""
    global APPS, FILES, CONTACTS
    APPS = {"google chrome": "x", "microsoft edge": "x", "whatsapp": "x", "visual studio code": "x",
            "microsoft word": "x", "microsoft excel": "x", "spotify": "x", "vlc media player": "x",
            "telegram desktop": "x", "uninstall google chrome": "x"}
    FILES = {"resume": "x", "budget 2026": "x", "ravi photo": "x"}
    CONTACTS = {"ravi": "919800000000", "mom": ""}
    build_candidates()

    def summary(i: Intent) -> str:
        if i.status in ("ask", "ambiguous"):
            return f"{i.status}:" + "|".join(o.label() for o in i.options)
        if i.status == "notfound":
            return f"notfound:{i.arg}"
        if i.status == "unknown":
            return "unknown"
        return f"{i.name}:{i.arg}" if i.arg else i.name

    cases = [
        # --- same request, different wording ---
        ("open youtube", "open_app:youtube"), ("launch youtube", "open_app:youtube"),
        ("Can you open YouTube?", "open_app:youtube"), ("  OPEN   YouTube!!  ", "open_app:youtube"),
        ("could you please open the youtube website", "open_app:youtube"), ("go to youtube", "open_app:youtube"),
        ("start chrome", "open_app:google chrome"), ("open chrome", "open_app:google chrome"),
        ("open the chrome browser", "open_app:google chrome"), ("I want to open google chrome", "open_app:google chrome"),
        # --- accent / mis-hearing fixes ---
        ("opan crome", "open_app:google chrome"), ("open krome", "open_app:google chrome"),
        ("open what's up", "open_app:whatsapp"), ("open whats app", "open_app:whatsapp"),
        ("open vatsapp", "open_app:whatsapp"), ("open note pad", "open_app:notepad"),
        ("open calculater", "open_app:calculator"), ("open vs code", "open_app:visual studio code"),
        ("open downloads folder", "open_app:downloads"), ("open folder documents", "open_app:documents"),
        ("open edge", "open_app:microsoft edge"), ("open word", "open_app:microsoft word"),
        ("open resume", "open_app:resume"),
        # --- must ask, never guess ---
        ("open microsoft", "ambiguous"), ("open spotfy", "open_app:spotify"), ("open whatsup", "ask"),
        ("open xyzzy", "notfound"),
        # --- close ---
        ("close chrome", "close_app:chrome"), ("clothes chrome", "close_app:chrome"),
        ("close the tab", "close_tab"), ("close this window", "close_window"), ("close it", "close_window"),
        ("please close whatsapp", "close_app:whatsapp"),
        # --- opposites must never swap ---
        ("volume up", "volume_up"), ("volume down", "volume_down"), ("turn up the volume", "volume_up"),
        ("turn the volume down", "volume_down"), ("increase the volume please", "volume_up"),
        ("lower the volume", "volume_down"), ("make it louder", "volume_up"), ("make it quieter", "volume_down"),
        ("wolume up", "volume_up"), ("volume", "ask"),
        ("next song", "next"), ("previous song", "previous"), ("play next song", "next"), ("skip", "next"),
        ("pause", "pause"), ("pause the music", "pause"), ("resume", "play"), ("mute", "mute"),
        # --- similar sounding words must not run the wrong command ---
        ("meet", "safe"), ("stock", "safe"), ("dime", "safe"), ("stop", "end_session"), ("lock the computer", "lock"),
        ("shut down the computer", "shutdown"), ("shutdown", "shutdown"), ("reboot", "restart"),
        ("restart", "restart"),
        # --- time, date, screen ---
        ("what's the time", "time"), ("what time is it", "time"), ("tell me the time", "time"),
        ("what is today's date", "date"), ("what day is it", "date"),
        ("take a screenshot", "screenshot"), ("screen shot", "screenshot"), ("capture the screen", "screenshot"),
        # --- search, type, play ---
        ("search for python tutorial", "search:python tutorial"), ("google cheap flights", "search:cheap flights"),
        ("look up the weather", "search:the weather"), ("search", "search_prompt"),
        ("play despacito", "play_youtube:despacito"), ("type hello world", "type:hello world"), ("type", "type_prompt"),
        # --- WhatsApp ---
        ("open chat with ravi", "chat:ravi"), ("chat with mom", "chat:mom"), ("ravi", "chat:ravi"),
        ("open chat", "chat_prompt"), ("rovi", "ask"),
        # --- conversation control ---
        ("goodbye", "end_session"), ("that is all", "end_session"), ("stop listening", "end_session"),
        ("quit chitti", "quit"), ("exit", "quit"), ("speak in telugu", "lang_telugu"),
        ("talk in english", "lang_english"), ("forget this conversation", "clear_memory"),
        # --- sentences that are conversation, not commands: they go to the AI ---
        ("what is the time in tokyo", "unknown"), ("how do i turn the volume up on my phone", "unknown"),
        ("what is the capital of france", "unknown"), ("can you explain how rainbows form", "unknown"),
        ("tell me about the history of hyderabad", "unknown"),
        # --- not commands ---
        ("tell me a joke", "unknown"), ("blah blah blah", "unknown"), ("", "unknown"),
    ]
    failed = 0
    print(f"{'heard':44} {'expected':32} got")
    for phrase, expected in cases:
        got = summary(parse_intent(phrase))
        if expected == "safe":              # must NOT run anything: asking or ignoring are both fine
            ok = got.startswith(("ask", "ambiguous", "unknown", "notfound"))
        elif expected == "ask":             # must ask the user (one or two options)
            ok = got.startswith(("ask", "ambiguous"))
        elif expected in ("ambiguous", "notfound", "unknown"):
            ok = got.startswith(expected)
        else:
            ok = got == expected
        if not ok:
            failed += 1
        print(f"{'PASS' if ok else 'FAIL'} {phrase!r:40} {expected:32} {got}")

    wake_cases = [("chitti open chrome", "open chrome"), ("kitty volume up", "volume up"), ("chitty", ""),
                  ("a chitti open notepad", "open notepad"), ("hit the road", None), ("open chrome", None),
                  ("what is the time", None), ("chittu open chrome", "open chrome")]
    print("\nwake word")
    for phrase, expected in wake_cases:
        got = split_wake(phrase)
        ok = got == expected
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {phrase!r:30} expected={expected!r} got={got!r}")

    print("\nduplicate guard")
    _recent.clear()
    first, second = is_duplicate("volume_up|"), is_duplicate("volume_up|")
    ok = (first, second) == (False, True)
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} same command twice in {DEDUP_SECONDS}s -> run, ignore")

    total = len(cases) + len(wake_cases) + 1
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


# ==================================================================== MAIN ====
def main() -> None:
    setup_logging("--debug" in sys.argv)
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if "--mics" in sys.argv:
        import sounddevice as sd
        print(sd.query_devices())
        print("\nPut the number of your microphone in MIC_DEVICE (and speaker in OUTPUT_DEVICE) in the .env file")
        return
    if "--setup" in sys.argv:
        import setup_ai
        raise SystemExit(setup_ai.main())
    init_engine()
    try:
        if "--tts-test" in sys.argv:
            tts_test()
        elif "--diag" in sys.argv:
            diagnostics()
        else:
            run_forever()
    finally:
        if speaker is not None:
            speaker.close()


if __name__ == "__main__":
    main()
