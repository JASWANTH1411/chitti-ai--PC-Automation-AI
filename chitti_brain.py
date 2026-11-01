"""
Chitti's brain: personality, conversation memory, streaming AI providers, sentence chunking.

No third-party packages are needed: the AI servers are called with Python's built-in urllib.

Providers (choose with LLM_PROVIDER in .env):
  ollama     free, runs on this PC (install Ollama, then `ollama pull llama3.2:3b`). Works offline.
  anthropic  Claude API. Paid, billed by usage. Needs ANTHROPIC_API_KEY.
  openai     Any OpenAI-compatible API (OpenAI, Groq, Gemini's compatible endpoint, OpenRouter, ...).
             Needs OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL. Some of these have free tiers with limits.
  auto       Uses the first one that is configured: anthropic, then openai, then ollama.
"""

from __future__ import annotations

import datetime
import http.client
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Optional

import chitti_config as cfg

log = logging.getLogger("chitti.brain")

LANG_NAMES = {"en": "English", "te": "Telugu"}

# ---------------------------------------------------------------- personality ----
DEFAULT_PERSONA = """You are Chitti, a friendly, intelligent, natural-sounding personal voice assistant. Speak like a thoughtful human conversational partner, not a robotic narrator. Listen carefully, understand context, answer directly, use natural language, and keep spoken responses concise unless the user asks for detail. Use a warm and calm tone. Never claim to have performed an action unless it actually succeeded.

Your words are read aloud, so write for the ear:
- Use short, clear sentences and contractions like I'm, you're, that's.
- No markdown, no bullet points, no headings, no emojis, no web addresses. If you must list things, say them in a natural sentence.
- Say numbers, dates, and units the way a person would say them.
- For a simple question, answer in one or two sentences. Go into detail only when asked.
- Do not start every reply with the same phrase, and do not keep asking how you can help.
- If a request is unclear, ask one short, specific question instead of guessing.
- If you are not sure about a fact, say so plainly. Never invent facts, names, or numbers.
- You cannot see the screen or the user's files. Opening apps, volume, search and similar computer actions are handled separately before you, so never say you did something on the computer unless the note in the conversation says it was done.
- Reply in the language the user is speaking. If they speak Telugu, reply in Telugu script."""


def load_persona() -> str:
    """Personality text lives in chitti_prompt.txt (edit it freely). Falls back to the built-in text."""
    path = cfg.APP_DIR / "chitti_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            return text
    except OSError:
        pass
    return DEFAULT_PERSONA


def build_system_prompt(lang: str = "en", now: Optional[datetime.datetime] = None) -> str:
    now = now or datetime.datetime.now()
    return (
        f"{load_persona()}\n\n"
        f"Current date and time: {now:%A, %d %B %Y, %I:%M %p}.\n"
        f"The user is currently speaking {LANG_NAMES.get(lang, lang)}."
    )


def wants_detail(text: str) -> bool:
    return bool(re.search(r"\b(detail|detailed|explain|elaborate|in depth|step by step|tell me more|"
                          r"long answer|full story|everything about)\b", text.lower()))


# ------------------------------------------------------------ conversation memory ----
class Conversation:
    """Recent turns only (so cost and delay stay small). Kept in memory, never written to disk."""

    def __init__(self, max_turns: int = 8, max_chars: int = 6000, idle_reset_s: float = 900):
        self.max_turns, self.max_chars, self.idle_reset_s = max_turns, max_chars, idle_reset_s
        self.msgs: list = []
        self.last = time.time()

    def add(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if content:
            self.msgs.append({"role": role, "content": content})
            self.last = time.time()

    def add_exchange(self, user: str, assistant: str) -> None:
        self.add("user", user)
        self.add("assistant", assistant)

    def maybe_reset_idle(self) -> bool:
        if self.msgs and time.time() - self.last > self.idle_reset_s:
            self.clear()
            return True
        return False

    def clear(self) -> None:
        self.msgs.clear()

    def messages(self) -> list:
        msgs = self.msgs[-self.max_turns * 2:]
        while len(msgs) > 2 and sum(len(m["content"]) for m in msgs) > self.max_chars:
            msgs = msgs[2:] if msgs[0]["role"] == "user" else msgs[1:]
        return _clean_roles(msgs)


def _clean_roles(msgs: list) -> list:
    """APIs want the first message from the user and roles alternating: merge neighbours."""
    out: list = []
    for m in msgs:
        if not out and m["role"] != "user":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


# ------------------------------------------------------------- sentence chunking ----
_BOUNDARY = re.compile(r"([.!?…।॥]+[\"')\]”’]*)(\s+)(?=\S)")
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "no",
          "inc", "ltd", "co", "mt", "ft", "dept", "est", "fig", "al"}


class SentenceChunker:
    """
    Turns a stream of text pieces into whole sentences, so speech can start with the first
    sentence while the rest is still being written. Never splits "Dr. Rao", "3.5", or "e.g.".
    A very long first sentence is split once at a comma so speech starts sooner.
    """

    def __init__(self, first_soft: int = 70, soft_max: int = 170):
        self.buf = ""
        self.emitted = 0
        self.first_soft = first_soft
        self.soft_max = soft_max

    def feed(self, text: str) -> list:
        self.buf += text
        out = []
        while True:
            s = self._next()
            if s is None:
                return out
            out.append(s)

    def flush(self) -> list:
        rest = self.buf.strip()
        self.buf = ""
        if rest:
            self.emitted += 1
            return [rest]
        return []

    def _take(self, n: int) -> str:
        sentence = self.buf[:n].strip()
        self.buf = self.buf[n:].lstrip()
        self.emitted += 1
        return sentence

    @staticmethod
    def _false_boundary(sent: str, rest: str) -> bool:
        if len(sent.strip()) < 4:
            return True                                   # "1." or "OK." alone: join with the next one
        if re.search(r"[.!?…।॥]{2,}[\"')\]”’]*$", sent):
            return False                                  # "..." is a real pause
        if sent.rstrip("\"')]”’").endswith("."):
            m = re.search(r"([^\s.]+)\.[\"')\]”’]*$", sent)
            tok = m.group(1).lower().strip("(\"'") if m else ""
            if tok in ABBREV or (len(tok) == 1 and tok.isalpha()):
                return True                               # Dr.  Mr.  J.  e.g.
            if rest[:1].isascii() and rest[:1].islower():
                return True                               # sentence continues in lower case
        return False

    def _next(self) -> Optional[str]:
        self.buf = self.buf.lstrip()
        buf = self.buf
        if not buf:
            return None

        cut = None
        for m in _BOUNDARY.finditer(buf):
            if not self._false_boundary(buf[:m.end(1)], buf[m.end():]):
                cut = m.end()
                break
        nl = buf.find("\n")
        if nl >= 12 and (cut is None or nl < cut):
            cut = nl
        if cut is not None:
            return self._take(cut)

        limit = self.first_soft if self.emitted == 0 else self.soft_max
        if len(buf) >= limit:
            idx = max(buf.rfind(", "), buf.rfind("; "), buf.rfind(": "), buf.rfind(" - "), buf.rfind(" — "))
            if idx >= min(30, limit // 2):
                return self._take(idx + 1)
            if len(buf) >= limit * 2:                     # extreme: no punctuation at all
                idx = buf.rfind(" ")
                if idx >= 30:
                    return self._take(idx)
        return None


# ------------------------------------------------------------------ AI providers ----
class LLMError(Exception):
    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.kind, self.retryable = kind, retryable

    def spoken(self) -> str:
        """A short, friendly sentence for the user (no technical detail, no secrets)."""
        return {
            "auth": "My AI key isn't being accepted. Please check the settings file.",
            "not_found": "The AI model I'm set to use isn't available. Please check the settings file.",
            "rate": "I'm being rate limited right now. Please try again in a moment.",
            "timeout": "My thinking is taking too long. Please try again.",
            "network": "I can't reach my AI right now. Please check the internet or the AI settings.",
            "none": "I don't have an AI model set up yet, so I can only run commands. See the settings file.",
        }.get(self.kind, "Sorry, I had trouble thinking about that. Please try again.")


class CancelToken:
    """Stops a running AI answer immediately, even while waiting for the network."""

    def __init__(self):
        self._ev = threading.Event()
        self._closers: list = []
        self._lock = threading.Lock()

    def set(self) -> None:
        self._ev.set()
        with self._lock:
            closers, self._closers = self._closers, []
        for fn in closers:
            try:
                fn()
            except Exception:
                pass

    def is_set(self) -> bool:
        return self._ev.is_set()

    def add_closer(self, fn) -> None:
        with self._lock:
            if not self._ev.is_set():
                self._closers.append(fn)
                return
        try:
            fn()
        except Exception:
            pass


def _http_error(code: int, body: str) -> LLMError:
    snippet = cfg.redact(re.sub(r"\s+", " ", body))[:160]
    if code in (401, 403):
        return LLMError("auth", f"HTTP {code}: {snippet}", False)
    if code == 404:
        return LLMError("not_found", f"HTTP 404: {snippet}", False)
    if code == 429:
        return LLMError("rate", f"HTTP 429: {snippet}", True)
    if code >= 500:
        return LLMError("server", f"HTTP {code}: {snippet}", True)
    return LLMError("bad_request", f"HTTP {code}: {snippet}", False)


USER_AGENT = "Chitti/2.0 (personal voice assistant)"      # some services block Python's default one


class _Conn:
    """
    One reusable connection to one server.

    Opening a secure connection takes 100-400 ms. Keeping it open (and opening it before the question
    arrives, see warm()) removes that wait from every answer. If the server closed the idle connection,
    the request is quietly repeated on a fresh one. Behind a proxy it falls back to urllib.
    """

    def __init__(self, base_url: str, timeout: float):
        u = urllib.parse.urlsplit(base_url)
        self.https = u.scheme == "https"
        self.host = u.hostname or ""
        self.port = u.port or (443 if self.https else 80)
        self.prefix = u.path.rstrip("/")
        self.scheme = u.scheme
        self.timeout = timeout
        self._conn = None
        self._lock = threading.Lock()
        self.opened = 0                                   # how many connections were opened (for the tests)
        try:
            proxies = urllib.request.getproxies()
            self.use_proxy = bool(proxies.get(u.scheme)) and not urllib.request.proxy_bypass(self.host)
        except Exception:
            self.use_proxy = False

    def _new(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        self.opened += 1
        return cls(self.host, self.port, timeout=self.timeout)

    def warm(self) -> None:
        """Open the connection now, so the next question does not wait for it. Never raises."""
        if self.use_proxy:
            return
        with self._lock:
            if self._conn is not None:
                return
            try:
                conn = self._new()
                conn.connect()
                self._conn = conn
            except Exception:
                pass

    def request(self, path: str, payload: dict, headers: dict, cancel: "CancelToken"):
        return self.request_raw(path, json.dumps(payload).encode("utf-8"),
                                {"Content-Type": "application/json", **headers}, cancel)

    def request_raw(self, path: str, data: bytes, headers: dict, cancel: "CancelToken"):
        """POST and return the open response (status 200). Call release(resp) when finished with it."""
        if self.use_proxy:
            return self._via_urllib(path, data, headers, cancel)
        hdrs = {"User-Agent": USER_AGENT, "Content-Length": str(len(data)), **headers}
        resp = conn = None
        for attempt in (0, 1):
            with self._lock:
                conn, self._conn = self._conn, None
            reused = conn is not None
            if conn is None:
                conn = self._new()
            try:
                conn.request("POST", self.prefix + path, body=data, headers=hdrs)
                resp = conn.getresponse()
                break
            except (socket.timeout, TimeoutError):
                conn.close()
                raise LLMError("timeout", "connection timed out", True)
            except (http.client.HTTPException, OSError) as e:
                conn.close()
                if reused and attempt == 0:
                    continue                              # the server had closed the idle connection: use a fresh one
                raise LLMError("network", f"cannot connect: {type(e).__name__}", True)
        if resp.status != 200:
            try:
                body = resp.read(600).decode("utf-8", "replace")
            except Exception:
                body = ""
            finally:
                conn.close()
            raise _http_error(resp.status, body)
        resp._chitti_conn = conn
        cancel.add_closer(lambda: self._abort(conn, resp))
        return resp

    @staticmethod
    def _abort(conn, resp=None) -> None:
        """
        Cut the connection now. Shutting the socket down wakes a read that is waiting for the next word.
        (For one-shot responses Python hands the socket from the connection to the response, so try both.)
        """
        socks = []
        if getattr(conn, "sock", None) is not None:
            socks.append(conn.sock)
        try:
            socks.append(resp.fp.raw._sock)
        except Exception:
            pass
        for sock in socks:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass

    def release(self, resp, drain: bool = False) -> None:
        """
        Finished with a response: keep the connection for the next request if it is clean.
        drain=True first reads the last few bytes of a finished answer (after "[DONE]" there is still a blank line).
        """
        conn = getattr(resp, "_chitti_conn", None)
        if conn is None:
            try:
                resp.close()
            except Exception:
                pass
            return
        if drain:
            try:
                if getattr(conn, "sock", None) is not None:
                    conn.sock.settimeout(1.0)
                resp.read()
            except Exception:
                pass
        try:
            reusable = resp.isclosed() and not resp.will_close
        except Exception:
            reusable = False
        if reusable:
            with self._lock:
                if self._conn is None:
                    self._conn = conn
                    return
        try:
            conn.close()
        except Exception:
            pass

    def _via_urllib(self, path: str, data: bytes, headers: dict, cancel: "CancelToken"):
        url = f"{self.scheme}://{self.host}:{self.port}{self.prefix}{path}"
        req = urllib.request.Request(url, data=data, method="POST", headers={"User-Agent": USER_AGENT, **headers})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            try:
                body = e.read(600).decode("utf-8", "replace")
            except Exception:
                body = ""
            finally:
                e.close()
            raise _http_error(e.code, body)
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise LLMError("timeout", "connection timed out", True)
            raise LLMError("network", f"cannot connect: {type(e.reason).__name__}", True)
        except (socket.timeout, TimeoutError):
            raise LLMError("timeout", "connection timed out", True)
        except OSError as e:
            raise LLMError("network", f"connection failed: {type(e).__name__}", True)
        cancel.add_closer(lambda: (resp.close(),))
        return resp


def _lines(resp, cancel: CancelToken, pool: _Conn) -> Iterator[str]:
    broken = False
    try:
        for raw in resp:
            if cancel.is_set():
                return
            yield raw.decode("utf-8", "replace").strip()
    except (socket.timeout, TimeoutError):
        broken = True
        if not cancel.is_set():
            raise LLMError("timeout", "no data from the AI", True)
    except Exception as e:                     # the connection was closed under us (cancel) or was lost
        broken = True
        if not cancel.is_set():
            raise LLMError("network", f"connection lost: {type(e).__name__}", True)
    finally:
        pool.release(resp, drain=not broken and not cancel.is_set())


class OllamaProvider:
    name = "ollama"
    cost = "free, runs on this PC"

    def __init__(self, host: str, model: str, timeout: float = 45, keep_alive: str = "30m"):
        self.host, self.model, self.timeout, self.keep_alive = host.rstrip("/"), model, timeout, keep_alive
        self._c = _Conn(self.host, timeout)

    def warm(self) -> None:
        self._c.warm()

    def check(self) -> tuple:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                names = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
        except Exception:
            return False, "Ollama is not running (start it, or choose another LLM_PROVIDER)"
        if not any(n == self.model or n.split(":")[0] == self.model or n == self.model + ":latest" for n in names):
            return False, f"model '{self.model}' is not downloaded (run: ollama pull {self.model})"
        return True, "ready"

    def stream(self, system: str, messages: list, max_tokens: int, cancel: CancelToken) -> Iterator[str]:
        payload = {"model": self.model, "stream": True, "keep_alive": self.keep_alive,
                   "messages": [{"role": "system", "content": system}] + messages,
                   "options": {"num_predict": max_tokens, "temperature": 0.7}}
        resp = self._c.request("/api/chat", payload, {}, cancel)
        for line in _lines(resp, cancel, self._c):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("error"):
                raise LLMError("server", str(obj["error"])[:160], False)
            piece = (obj.get("message") or {}).get("content")
            if piece:
                yield piece
            if obj.get("done"):
                return


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30,
                 tokens_param: str = "max_tokens", free: bool = False):
        self.base, self.key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout
        self.tokens_param = tokens_param
        self.cost = "free tier (limits apply)" if free else "paid API (billed by usage)"
        self._c = _Conn(self.base, timeout)

    def warm(self) -> None:
        self._c.warm()

    def check(self) -> tuple:
        return (bool(self.key and self.model), "ready" if self.key and self.model else "key or model missing")

    def stream(self, system: str, messages: list, max_tokens: int, cancel: CancelToken) -> Iterator[str]:
        payload = {"model": self.model, "stream": True, "temperature": 0.7, self.tokens_param: max_tokens,
                   "messages": [{"role": "system", "content": system}] + messages}
        resp = self._c.request("/chat/completions", payload, {"Authorization": f"Bearer {self.key}"}, cancel)
        for line in _lines(resp, cancel, self._c):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("error"):
                err = obj["error"]
                raise LLMError("server", cfg.redact(str(err.get("message", err) if isinstance(err, dict) else err))[:160], False)
            choices = obj.get("choices") or []
            if choices:
                piece = (choices[0].get("delta") or {}).get("content")
                if isinstance(piece, str) and piece:
                    yield piece


class AnthropicProvider:
    name = "anthropic"
    cost = "paid API (billed by usage)"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.anthropic.com", timeout: float = 30):
        self.key, self.model, self.base, self.timeout = api_key, model, base_url.rstrip("/"), timeout
        self._c = _Conn(self.base, timeout)

    def warm(self) -> None:
        self._c.warm()

    def check(self) -> tuple:
        return (bool(self.key), "ready" if self.key else "ANTHROPIC_API_KEY missing")

    def stream(self, system: str, messages: list, max_tokens: int, cancel: CancelToken) -> Iterator[str]:
        payload = {"model": self.model, "max_tokens": max_tokens, "system": system,
                   "messages": messages, "stream": True}
        resp = self._c.request("/v1/messages", payload,
                               {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, cancel)
        for line in _lines(resp, cancel, self._c):
            if not line.startswith("data:"):
                continue
            try:
                obj = json.loads(line[5:].strip())
            except ValueError:
                continue
            kind = obj.get("type")
            if kind == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]
            elif kind == "error":
                err = obj.get("error") or {}
                raise LLMError("server", cfg.redact(str(err.get("message", "error")))[:160], False)
            elif kind == "message_stop":
                return


def make_provider(kind: str):
    """Build one provider from the settings, or None if it is not configured."""
    kind = kind.lower()
    if kind == "ollama":
        return OllamaProvider(cfg.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                              cfg.get("OLLAMA_MODEL", "llama3.2:3b"),
                              timeout=cfg.get_float("LLM_TIMEOUT", 45))
    if kind == "anthropic":
        key = cfg.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        return AnthropicProvider(key, cfg.get("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                                 cfg.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                                 timeout=cfg.get_float("LLM_TIMEOUT", 30))
    if kind == "openai":
        key, model = cfg.get("OPENAI_API_KEY"), cfg.get("OPENAI_MODEL")
        if not key or not model:
            return None
        return OpenAICompatProvider(cfg.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), key, model,
                                    timeout=cfg.get_float("LLM_TIMEOUT", 30),
                                    tokens_param=cfg.get("OPENAI_TOKENS_PARAM", "max_tokens"),
                                    free=cfg.get_bool("OPENAI_FREE_TIER", False))
    return None


def build_providers() -> list:
    """
    Which AI models to use, in order. A later one is only tried if an earlier one fails.
      LLM_PROVIDER = auto | ollama | anthropic | openai | none
      LLM_FALLBACK = comma list of extra providers to try after the main one (e.g. ollama)
    In auto mode: the first of anthropic / openai that has a key, with Ollama as the free local backup.
    """
    choice = cfg.get("LLM_PROVIDER", "auto").lower()
    if choice == "none":
        return []
    kinds = []
    if choice == "auto":
        for kind in ("anthropic", "openai"):
            if make_provider(kind):
                kinds.append(kind)
                break
        kinds.append("ollama")
    else:
        kinds.append(choice)
    kinds += [k.lower() for k in cfg.get_list("LLM_FALLBACK")]

    providers, seen = [], set()
    for kind in kinds:
        if kind in seen:
            continue
        seen.add(kind)
        p = make_provider(kind)
        if p:
            providers.append(p)
    return providers


class Brain:
    """Streams an answer as whole sentences, with retries, fallback providers and cancellation."""

    def __init__(self, providers: list, conversation: Optional[Conversation] = None,
                 max_tokens: int = 300, retries: int = 1):
        self.providers = providers
        self.conv = conversation or Conversation(
            max_turns=cfg.get_int("HISTORY_TURNS", 8),
            max_chars=cfg.get_int("HISTORY_CHARS", 6000),
            idle_reset_s=cfg.get_float("HISTORY_IDLE_MINUTES", 15) * 60)
        self.max_tokens, self.retries = max_tokens, retries
        self.last_provider = ""
        self.last_error: Optional[LLMError] = None

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    def describe(self) -> str:
        if not self.providers:
            return "no AI model configured (commands only)"
        return " -> ".join(f"{p.name} [{getattr(p, 'cost', '')}]" for p in self.providers)

    def reply_stream(self, user_text: str, cancel: CancelToken, lang: str = "en") -> Iterator[str]:
        """Yield the answer one sentence at a time. Raises LLMError only if nothing could be said."""
        self.conv.maybe_reset_idle()
        msgs = _clean_roles(self.conv.messages() + [{"role": "user", "content": user_text}])
        system = build_system_prompt(lang)
        max_tokens = self.max_tokens * (3 if wants_detail(user_text) else 1)
        chunker = SentenceChunker(first_soft=cfg.get_int("FIRST_CHUNK_CHARS", 45))
        got_any = False
        last: Optional[LLMError] = None
        self.last_error = None

        for prov in self.providers:
            for attempt in range(self.retries + 1):
                if cancel.is_set():
                    return
                try:
                    for piece in prov.stream(system, msgs, max_tokens, cancel):
                        got_any = True
                        for sentence in chunker.feed(piece):
                            if cancel.is_set():
                                return
                            yield sentence
                    if cancel.is_set():                           # cancelled: never speak a half sentence
                        return
                    for sentence in chunker.flush():
                        yield sentence
                    self.last_provider = prov.name
                    return
                except LLMError as e:
                    last = e
                    self.last_error = e
                    log.warning("AI provider %s failed (%s): %s", prov.name, e.kind, e)
                    if got_any:                                   # failed in the middle: keep what was said
                        for sentence in chunker.flush():
                            yield sentence
                        return
                    if not e.retryable:
                        break                                     # try the next provider
                    time.sleep(0.4 * (attempt + 1))
        raise last or LLMError("none", "no AI provider available", False)

    def warm(self) -> None:
        """Open the connection to the AI now (in the background) so the next answer starts sooner."""
        for p in self.providers:
            fn = getattr(p, "warm", None)
            if fn:
                threading.Thread(target=fn, daemon=True).start()

    def commit(self, user_text: str, assistant_text: str) -> None:
        """Remember what was actually said (the interrupted part is not remembered as spoken)."""
        self.conv.add_exchange(user_text, assistant_text or "(no answer was spoken)")

    def note_action(self, user_text: str, result: str) -> None:
        """Remember a computer action so follow-up questions make sense."""
        self.conv.add_exchange(user_text, f"[Computer action result: {result}]")
