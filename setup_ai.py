"""
Chitti AI setup wizard.

    python chitti.py --setup        (or:  python setup_ai.py)

It asks which AI you want, saves your choice in the .env file (creating it if it does not exist), and then
TESTS the connection and shows how fast the AI answers. Your key is typed hidden and is never printed or logged.
You can run it again at any time to change your mind.
"""

from __future__ import annotations

import getpass
import os
import shutil
import sys
import time

import chitti_brain as B
import chitti_config as cfg

GROQ_URL = "https://api.groq.com/openai/v1"
MENU = """
Which AI should Chitti use to answer your questions?

  1) Claude (Anthropic)     Paid by usage. Smart and very quick. Claude Haiku 5.5 costs about
                            $0.10 per million words in and $0.50 per million out: roughly 15 cents
                            for 1,000 spoken questions. Key from console.anthropic.com
  2) Groq                   Very fast. Has a FREE tier with limits (check console.groq.com).
                            Can ALSO make speech recognition much faster on a slow PC.
  3) Another service        Any "OpenAI-compatible" service (OpenAI, Google Gemini, OpenRouter ...)
  4) Ollama on this PC      Free and offline, but slow on an older PC
  5) No AI                  Chitti only runs computer commands (open apps, volume, search ...)
"""


def ensure_env_file(say=print) -> None:
    """Create .env (from the template) if it does not exist yet."""
    if cfg.ENV_FILE.exists():
        return
    for name in (".env.example", "env_example.txt"):
        template = cfg.APP_DIR / name
        if template.exists():
            shutil.copyfile(template, cfg.ENV_FILE)
            say(f"Created {cfg.ENV_FILE.name} from {name}.")
            return
    cfg.ENV_FILE.write_text("# Chitti settings\n", encoding="utf-8")
    say(f"Created a new {cfg.ENV_FILE.name}.")


def save_settings(values: dict) -> None:
    """Set KEY=value lines in .env (changing the line if it exists, adding it if not). Other lines are kept."""
    ensure_env_file(lambda *_: None)
    lines = cfg.ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
    remaining = dict(values)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].replace("export ", "").strip()
            if key in remaining:
                lines[i] = f"{key}={remaining.pop(key)}"
    if remaining:
        lines += ["", "# added by the setup wizard"] + [f"{k}={v}" for k, v in remaining.items()]
    cfg.ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ.update({k: str(v) for k, v in values.items()})


def test_ai(say=print) -> bool:
    """Ask the configured AI one question, twice, and report the speed."""
    providers = [p for p in B.build_providers() if p.check()[0]]
    if not providers:
        say("No AI is ready to test.")
        return False
    brain = B.Brain(providers, retries=0)
    say(f"\nTesting '{providers[0].name}' ({getattr(providers[0], 'cost', '')}) ...")
    times = []
    for attempt in range(2):
        cancel = B.CancelToken()
        began = time.time()
        try:
            for sentence in brain.reply_stream("In one short sentence, say hello and tell me your name.", cancel):
                times.append(int((time.time() - began) * 1000))
                if len(times) == 1:
                    say(f"  She says: {sentence}")
                cancel.set()
                break
        except B.LLMError as e:
            say(f"  That did not work: {e.spoken()}")
            say(f"  (technical detail: {e.kind})")
            return False
        if len(times) < attempt + 1:
            say("  The AI answered with nothing. Check the model name.")
            return False
    say(f"  First sentence after {times[0]} ms, and {times[1]} ms the next time (connection already open).")
    if times[1] <= 700:
        say("  That is quick: she will feel natural.")
    elif times[1] <= 1500:
        say("  That is fine. A faster service or model would feel snappier.")
    else:
        say("  That is slow for a conversation. Try Groq or Claude Haiku, or a smaller model.")
    return True


def test_stt(say=print) -> bool:
    """Check that the cloud speech recognition service accepts a request."""
    import chitti_stt
    stt = chitti_stt.build_cloud_stt()
    if stt is None:
        return False
    import numpy as np
    noise = (np.random.default_rng(1).standard_normal(16000) * 300).astype(np.int16).tobytes()
    began = time.time()
    try:
        stt.transcribe(noise, "en", "")
    except chitti_stt.STTError as e:
        say(f"  The speech recognition service did not accept the request: {e}")
        say("  Check the key and the model name (STT_MODEL). Chitti will fall back to the local Whisper.")
        return False
    say(f"  Speech recognition service reachable ({int((time.time() - began) * 1000)} ms for 1 second of audio).")
    return True


def main(ask=input, ask_secret=getpass.getpass, say=print) -> int:
    say("=== Chitti AI setup ===")
    ensure_env_file(say)
    say(MENU)
    choice = ask("Type 1, 2, 3, 4 or 5 and press Enter: ").strip()
    values: dict = {}

    def secret(prompt: str) -> str:
        value = ask_secret(prompt).strip().strip("'\"")
        if not value:
            say("No key entered. Nothing was changed.")
            raise SystemExit(1)
        say(f"  (key saved as {cfg.mask(value)}; it is never shown again)")
        return value

    def named(prompt: str, default: str = "") -> str:
        value = ask(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        return value or default

    if choice == "1":
        say("\nGet a key at console.anthropic.com (API keys). It starts with 'sk-ant-'. Add a little credit first.")
        values = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": secret("Paste your Claude key (hidden): "),
                  "ANTHROPIC_MODEL": named("Model", "claude-haiku-5-5")}
    elif choice == "2":
        say("\nGet a free key at console.groq.com (API Keys).")
        key = secret("Paste your Groq key (hidden): ")
        say("Model names change. Copy a current chat model name from console.groq.com/docs/models")
        values = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": key, "OPENAI_BASE_URL": GROQ_URL,
                  "OPENAI_MODEL": named("Model", "llama-3.1-8b-instant"), "OPENAI_FREE_TIER": "true"}
        say("\nGroq can also do the speech recognition (hosted Whisper), which is MUCH faster than a slow PC.")
        say("Privacy: audio is sent to Groq only AFTER you say 'chitti' and a conversation is open.")
        say("While she waits for her name, everything stays on your PC.")
        if ask("Use Groq for fast speech recognition too? (y/n) [y]: ").strip().lower() in ("", "y", "yes"):
            values.update({"STT_ENGINE": "cloud", "STT_BASE_URL": GROQ_URL, "STT_API_KEY": key,
                           "STT_MODEL": named("Speech model", "whisper-large-v3-turbo")})
    elif choice == "3":
        say("\nYou need the service's address, a key, and a model name from its website.")
        values = {"LLM_PROVIDER": "openai", "OPENAI_BASE_URL": named("Address (ends with /v1)", "https://api.openai.com/v1"),
                  "OPENAI_API_KEY": secret("Paste the key (hidden): "), "OPENAI_MODEL": named("Model name"),
                  "OPENAI_FREE_TIER": "true" if ask("Is it a free tier? (y/n) [n]: ").strip().lower() in ("y", "yes") else "false"}
        if not values["OPENAI_MODEL"]:
            say("A model name is required. Nothing was changed.")
            return 1
    elif choice == "4":
        model = named("Model to use (try llama3.2:1b if llama3.2:3b is too slow)", "llama3.2:3b")
        values = {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": model}
        ok, why = B.OllamaProvider(cfg.get("OLLAMA_HOST", "http://127.0.0.1:11434"), model).check()
        if not ok:
            say(f"\nOllama is not ready yet: {why}")
            say("  1) Install Ollama from ollama.com   2) In a new window run:  ollama pull " + model)
            say("  Your choice is saved; run this setup again to test it.")
    elif choice == "5":
        values = {"LLM_PROVIDER": "none"}
    else:
        say("That was not 1, 2, 3, 4 or 5. Nothing was changed.")
        return 1

    save_settings(values)
    say(f"\nSaved in {cfg.ENV_FILE}  (keep that file private: never share it).")
    if values.get("LLM_PROVIDER") == "none":
        say("Chitti will only run computer commands.")
        return 0

    ok = test_ai(say)
    if values.get("STT_ENGINE") == "cloud":
        ok = test_stt(say) and ok
    if ok:
        say("\nAll set. Start Chitti with:  python chitti.py")
    else:
        say("\nFix the problem above (wrong key or model name is the usual cause) and run this setup again.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
