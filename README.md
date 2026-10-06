# 🚀 AI Engineering

Welcome to the **AI Engineering** repository! This repository documents practical implementations, architectures, and experiments across modern Generative AI, LLM frameworks, agentic workflows, and production systems.

---

## 📁 Repository Structure

```
AI_Engineering/
├── langchain/
│   ├── langchain_oneshot/                         # Complete LangChain hands-on course
│   │   ├── Chapter-1/                             # Basics, Prompts, Structured Outputs
│   │   ├── Chapter-2/                             # Chains, LCEL Runnables, Tools, Agents
│   │   ├── Chapter-3/                             # Memory & Context Management
│   │   ├── main.py                                # Entry point script
│   │   ├── pyproject.toml                         # Project dependencies (uv)
│   │   └── README.md                              # Detailed module guide
│   └── langchain_oneshot_techSimPlus_learnings/   # Visual architecture diagrams & session notes
│       ├── images/
│       └── README.md
├── .gitignore                                     # Comprehensive security and cache ignore rules
└── README.md                                      # Repository overview
```


---

## ⚡ Quickstart

### 1. LangChain Module
To get started with the LangChain course:
```bash
cd langchain/langchain_oneshot
```

Copy the environment template and add your Gemini API key:
```bash
# On Linux/macOS:
cp .env.example .env

# On Windows PowerShell:
Copy-Item .env.example .env
```

Install dependencies using [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

Run test script or open any notebook in VS Code / Jupyter:
```bash
uv run python main.py
```

---

## 🔐 Security Notice
* Never commit `.env` or sensitive API keys.
* `.env` files are ignored by git rules across all subdirectories. Use `.env.example` templates for configuration guidelines.
