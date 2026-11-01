"""
Optional cloud speech recognition, for PCs that are too slow to run Whisper quickly.

It talks to any service that offers the OpenAI-style  /audio/transcriptions  endpoint, for example Groq
(hosted Whisper, free tier with limits) or OpenAI. Set STT_ENGINE=cloud and STT_BASE_URL, STT_API_KEY, STT_MODEL
in .env (see .env.example).

PRIVACY: Chitti only sends audio here AFTER you have said her name and a conversation is open. While she is idle,
waiting for her name, everything stays on your PC (local Whisper). Nothing is saved to disk.
If this service fails, Chitti falls back to the local Whisper automatically.
"""

from __future__ import annotations

import io
import json
import math
import time
import uuid
import wave

import chitti_config as cfg
from chitti_brain import CancelToken, LLMError, _Conn


class STTError(Exception):
    pass


def wav_bytes(pcm: bytes, rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono audio in a WAV header (held in memory, never written to disk)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def multipart(fields: dict, filename: str, file_bytes: bytes, content_type: str) -> tuple:
    """Build a multipart/form-data body. Returns (body, content_type_header)."""
    boundary = "----chitti" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                  f'Content-Type: {content_type}\r\n\r\n').encode("utf-8") + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class CloudSTT:
    name = "cloud"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 10, retries: int = 1):
        self.base, self.key, self.model = base_url.rstrip("/"), api_key, model
        self.retries = retries
        self._c = _Conn(self.base, timeout)

    @property
    def enabled(self) -> bool:
        return bool(self.base and self.key and self.model)

    def warm(self) -> None:
        self._c.warm()

    def transcribe(self, pcm: bytes, language: str = "en", prompt: str = "", rate: int = 16000) -> dict:
        """Returns {"text", "conf", "no_speech"}. Raises STTError (the caller then uses the local Whisper)."""
        fields = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt[:600]
        body, ctype = multipart(fields, "speech.wav", wav_bytes(pcm, rate), "audio/wav")
        headers = {"Content-Type": ctype, "Authorization": f"Bearer {self.key}"}

        last = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._c.request_raw("/audio/transcriptions", body, headers, CancelToken())
                try:
                    data = resp.read()
                finally:
                    self._c.release(resp)
                break
            except LLMError as e:
                last = e
                if not e.retryable:
                    raise STTError(f"{e.kind}: {cfg.redact(str(e))}") from e
                time.sleep(0.15)
            except Exception as e:
                raise STTError(type(e).__name__) from e
        else:
            raise STTError(f"{last.kind}: {cfg.redact(str(last))}" if last else "failed")

        try:
            obj = json.loads(data.decode("utf-8", "replace"))
        except ValueError as e:
            raise STTError("the service sent something unreadable") from e
        text = (obj.get("text") or "").strip()
        segs = obj.get("segments") or []
        if segs:
            weights = [max(1, len(str(s.get("text", "")).split())) for s in segs]
            avg_lp = sum(float(s.get("avg_logprob", -0.3)) * w for s, w in zip(segs, weights)) / sum(weights)
            conf = math.exp(avg_lp)
            no_speech = max(float(s.get("no_speech_prob", 0.0)) for s in segs)
        else:
            conf, no_speech = (0.8 if text else 0.0), (0.0 if text else 1.0)
        return {"text": text, "conf": conf, "no_speech": no_speech}


def build_cloud_stt():
    """The cloud recogniser from the settings, or None when it is not switched on."""
    if cfg.get("STT_ENGINE", "local").lower() != "cloud":
        return None
    stt = CloudSTT(cfg.get("STT_BASE_URL", "https://api.groq.com/openai/v1"),
                   cfg.get("STT_API_KEY", cfg.get("OPENAI_API_KEY", "")),
                   cfg.get("STT_MODEL", "whisper-large-v3-turbo"),
                   timeout=cfg.get_float("STT_TIMEOUT", 10))
    return stt if stt.enabled else None
