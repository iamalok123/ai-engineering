# 🦜️🔗 LangChain One-Shot: Complete Hands-On Guide

A comprehensive, hands-on repository covering LangChain fundamentals, Runnables (LCEL), tool calling, agents, and advanced memory management with Google Gemini models.

---

## 📚 Curriculum & Notebooks

### 🔹 Chapter 1: Foundations & Structured Outputs
* **`Chapter-1/1_Basics_and_Messages.ipynb`**: Introduction to LangChain, Chat Models, and Message types (`HumanMessage`, `AIMessage`, `SystemMessage`).
* **`Chapter-1/2_prompts.ipynb`**: `ChatPromptTemplate`, message placeholders, and prompt composition.
* **`Chapter-1/3_structured_outputs.ipynb`**: Structured extraction with Pydantic schemas using `.with_structured_output()`.

### 🔹 Chapter 2: Chains, Tools & Agents
* **`Chapter-2/1_first_chain.ipynb`**: Composing chains with LangChain Expression Language (LCEL).
* **`Chapter-2/2_runnables.ipynb`**: Deep dive into Runnables (`RunnableLambda`, `RunnableParallel`, `RunnablePassthrough`).
* **`Chapter-2/3_tools.ipynb`**: Creating and binding custom tools using `@tool` decorator.
* **`Chapter-2/4_agents.ipynb`**: Building tool-calling agents and multi-step reasoning agents.

### 🔹 Chapter 3: Memory & Context Management
* **`Chapter-3/1_memory_basics.ipynb`**: Fundamentals of conversation history and chat message memory.
* **`Chapter-3/2_buffer_window_contextManagement.ipynb`**: Sliding window buffers and token-based trimming.
* **`Chapter-3/3_Summery_Memory.ipynb`**: Dynamic summarization memory to conserve context windows.
* **`Chapter-3/4_vector_semantic_memory.ipynb`**: Semantic memory search using vector similarity.
* **`Chapter-3/5_Production_Memory.ipynb`**: Production-ready memory architectures.

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.12+
* [uv](https://docs.astral.sh/uv/) (recommended package manager) or standard `pip`

### 2. Environment Setup
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   # On Windows PowerShell:
   Copy-Item .env.example .env
   ```
2. Open `.env` and configure your API key:
   ```env
   GEMINI_API_KEY="your_google_gemini_api_key"
   ```
   > 💡 You can obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 3. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or activate the virtual environment manually:
```powershell
.venv\Scripts\activate
```

### 4. Running the Code & Notebooks
Run the entry point script:
```bash
uv run python main.py
```
Or launch Jupyter / VS Code:
* Select `.venv` as your Python kernel in VS Code / Jupyter.
* Run any notebook in `Chapter-1/`, `Chapter-2/`, or `Chapter-3/`.
