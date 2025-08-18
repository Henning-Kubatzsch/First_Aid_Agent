# MR First Aid Training – Local RAG Prototype

🚑 **Goal:**  
This project explores how to build a **local, offline Retrieval-Augmented Generation (RAG) system** for First Aid training.  
The long-term vision is an **MR multi-user training environment (Quest 3)** with an **agentic pedagogical assistant** that guides users through **ERC (European Resuscitation Council) guidelines**.

This repo is **Phase 1**:  
✅ Local RAG pipeline on **Mac M1 Pro**  
✅ ERC guidelines as knowledge base  
✅ Local LLM only (no external API calls)  
✅ Conversational loop with **progression tracking** of resuscitation steps  

---

## 🚀 Roadmap (Phases)

1. **Local RAG baseline (Mac M1 Pro)** ← *this repo*  
   - Local LLM + embeddings  
   - ERC guideline ingestion & retrieval  
   - Conversational loop with state tracking  

2. **Graph-based progression (LangGraph)**  
   - Model progression of resuscitation steps as a state machine  
   - Natural language conversation updates progression state  

3. **Agentic RAG experimentation**  
   - Tune ERC text (chunking, formatting)  
   - Test different small, local LLMs  
   - Evaluate clarity & correctness of outputs  

4. **Standalone deployment on Quest 3**  
   - Offline pipeline  
   - Voice input / output  

5. **DummyStation integration**  
   - Stream vitals (BP, compression depth, O₂, ventilation)  
   - Agent reacts dynamically  

---

## 🛠️ Phase 1 Setup

### Requirements
- macOS (Apple Silicon, M1 Pro or similar)  
- Python 3.10+  
- [Ollama](https://ollama.ai) installed locally  
- Optional: `uv` or `conda` for environment management  

### Installation
```bash
# Clone the repo
git clone https://github.com/yourname/mr-first-aid-rag.git
cd mr-first-aid-rag

# Setup environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

### Models

Pull a small local LLM (fast & lightweight):

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

Optional embeddings model (runs locally via sentence-transformers or nomic):

* `nomic-embed-text`
* `bge-small-en`
* `gte-small`

---

## 📚 Data

* ERC Guidelines (PDF/text provided locally by user)
* Ingested into vector store (`/data/indexes/erc_v1`)

Scripts:

```bash
python scripts/build_index.py
```

---

## 💬 Run the CLI Prototype

Start the chat:

```bash
python src/app/main.py
```

Example session:

```
> What do I do next?
Agent: Please call an ambulance immediately. [ERC §Basic Life Support]
State updated: {"called_ambulance": true}
```

Commands:

* `/state` → show current resuscitation progress
* `/reset` → clear state
* `/undo` → revert last action
* `/trace` → debug retrieved context

---

## 📊 Progression Tracking

The system keeps a minimal state:

```json
{
  "checked_responsiveness": false,
  "called_ambulance": false,
  "started_cpr": false,
  "aed_requested": false,
  "airway_checked": false
}
```

* State is updated automatically based on conversation
* In Phase 1, only **conversation → state** mapping is implemented
* In later phases, **dummy monitoring values** will also update state

---

## 🧪 Evaluation (Phase 1)

* Small scenario set (`/tests/scenarios.jsonl`) with typical training questions
* Logs retrieved chunks, answers, and state updates
* Manual pass/fail evaluation of clarity & ERC correctness

---

## 📂 Repo Structure

```
/src/app/
  main.py         # CLI chat entry
  rag.py          # Retriever + RAG chain
  state.py        # State schema & reducer
  prompts.py      # Prompt templates
  models.py       # LLM & embeddings adapters
  config.py       # Settings

/scripts/
  build_index.py  # ERC ingestion & vector store builder

/data/
  erc/            # ERC guideline source text
  indexes/        # FAISS/Chroma indexes

/tests/
  test_state.py   # Unit tests for state handling
  scenarios.jsonl # Evaluation scenarios
```

---

## ⚠️ Disclaimer

This project is **for training and research purposes only**.
It does **not replace certified medical training** or real emergency protocols.
Always follow official ERC guidelines and seek certified instruction.

---

## 📅 Next Steps

* [ ] Finish ingestion pipeline for ERC guidelines
* [ ] Implement minimal RAG chain with local LLM
* [ ] Add progression state updates from conversation
* [ ] Build evaluation set for Phase 1
* [ ] Prepare migration to LangGraph for Phase 2

---

```

