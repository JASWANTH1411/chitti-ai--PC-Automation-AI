# chitti-ai--PC-Automation-AI
CHITTI AI is a Python-based personal desktop assistant featuring AI-powered conversations, voice interaction, speech recognition, and desktop automation. It supports multiple AI providers, including Gemini, Groq, Claude, OpenAI-compatible services, and Ollama, making everyday computer interactions smarter and easier.
# 🤖 CHITTI AI – Your Personal Desktop AI Assistant

Chitti AI is a Python-powered personal desktop assistant designed to make computer interaction smarter, easier, and more natural.

It combines voice interaction, artificial intelligence, and desktop automation to create a personal AI companion for everyday computer tasks.

## ✨ Features

- 🎙️ Voice-based interaction
- 🧠 AI-powered conversations
- 🗣️ Natural text-to-speech responses
- 💻 Desktop automation
- 🔊 System volume controls
- 🔍 Web search capabilities
- ⚡ Fast speech recognition using Faster-Whisper
- 🔌 Multiple AI provider support
- ⚙️ Interactive setup wizard
- 📴 Local AI support through Ollama

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application |
| Faster-Whisper | Speech recognition |
| SoundDevice | Audio input |
| Edge-TTS | Voice responses |
| PyAutoGUI | Desktop automation |
| Miniaudio | Audio processing |
| AI APIs | Intelligent responses |
| Ollama | Local AI models |

## 🧠 Supported AI Providers

Chitti supports five AI configuration options:

1. **Claude (Anthropic)** – Cloud-based AI responses.
2. **Groq** – Fast AI responses with a free tier.
3. **OpenAI-Compatible Services** – OpenAI, Gemini, OpenRouter and other compatible providers.
4. **Ollama** – Run local AI models on your computer.
5. **No AI** – Use supported desktop automation commands without an AI model.

## 📂 Project Structure

```text
CHITTI-AI/
│
├── chitti.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
└── assets/
```

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/chitti-ai.git
```

### Step 2: Open the Project

```bash
cd chitti-ai
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Chitti

Run the interactive setup wizard:

```bash
python chitti.py --setup
```

## ⚙️ AI Setup Wizard

During setup, Chitti displays:

```text
=== Chitti AI setup ===

Which AI should Chitti use to answer your questions?

1) Claude (Anthropic)
2) Groq
3) Another service
4) Ollama on this PC
5) No AI

Type 1, 2, 3, 4 or 5 and press Enter:
```

Choose your preferred AI provider and follow the instructions to configure it.

### Gemini Configuration Example

For Google's OpenAI-compatible API, use:

**API Endpoint:**
```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

**Model:**

```text
gemini-3.8-flash
```

Use a valid API key from Google AI Studio and a model currently available to your API key.

## ▶️ Run Chitti

After completing the configuration:

```bash
python chitti.py
```

Start interacting with your personal desktop assistant.

## 🔐 Security

- Never upload API keys to GitHub.
- Keep `.env` private.
- Add `.env` to `.gitignore`.
- Never publish passwords or private credentials.
- Use environment variables for sensitive configuration.

## 🎯 Future Improvements

- More natural human-like conversations.
- Improved voice response speed.
- Better AI provider switching.
- Long-term conversational memory.
- Advanced desktop automation.
- Improved error handling and connection recovery.
- More intelligent task execution.

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome!

Feel free to report bugs, suggest features, or contribute improvements.

## 👨‍💻 Developer

**Project:** CHITTI AI  
**Developer:** Jaswanth  
**Programming Language:** Python  
**Project Type:** Personal AI Desktop Assistant

---

⭐ If you like Chitti AI, consider giving this repository a star!

**CHITTI AI – Your Desktop. Your Assistant. Your Chitti.**
