# AI Agentic Project Management — Phase 1

## Setup

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import openai; print('ok', openai.__version__)"
```

Make sure your `.env` file exists at the project root:

```text
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

---

## Running Agents

All runners are in `phase_1/runners/`. Run each command from the **project root**. Results are saved automatically to the `results/` folder.

### 1. DirectPromptAgent

Sends prompts directly to GPT-3.5-turbo using the model's own pre-trained knowledge.

```bash
venv/bin/python -m phase_1.runners.run_direct_prompt_agent
```

Output saved to: `results/direct_prompt_agent.txt`

---

### 2. AugmentedPromptAgent

Injects a persona via a system prompt and clears prior context before each call.

```bash
venv/bin/python -m phase_1.runners.run_augmented_prompt_agent
```

Output saved to: `results/augmented_prompt_agent.txt`

---

### 3. KnowledgeAugmentedPromptAgent

Grounds the LLM to answer strictly from injected knowledge, ignoring its own training data.

```bash
venv/bin/python -m phase_1.runners.run_knowledge_augmented_prompt_agent
```

Output saved to: `results/knowledge_augmented_prompt_agent.txt`

---

### 4. RAGKnowledgePromptAgent

Chunks a knowledge corpus, embeds it, and retrieves the most relevant chunk to answer each prompt.

```bash
venv/bin/python -m phase_1.runners.run_rag_agent
```

Output saved to: `results/rag_agent.txt`

> Note: This runner makes embedding API calls and creates temporary CSV files (`chunks-*.csv`, `embeddings-*.csv`) in the working directory.

---

### 5. EvaluationAgent

Runs a worker agent in a feedback loop, evaluating and correcting its responses up to a maximum number of iterations.

```bash
venv/bin/python -m phase_1.runners.run_evaluation_agent
```

Output saved to: `results/evaluation_agent.txt`

---

## Running Tests

Unit and integration tests are in `phase_1/workflow_agents/`.

```bash
# Run all tests
venv/bin/python -m unittest discover -s phase_1/workflow_agents -p "*_test.py" -v

# Run a specific test file
venv/bin/python -m unittest phase_1.workflow_agents.direct_prompt_agent_test -v
venv/bin/python -m unittest phase_1.workflow_agents.augmented_prompt_agent_test -v
venv/bin/python -m unittest phase_1.workflow_agents.knowledge_augmented_prompt_agent -v
```

---

## Project Structure

```text
.
├── phase_1/
│   ├── runners/                              # Agent runner scripts
│   │   ├── run_direct_prompt_agent.py
│   │   ├── run_augmented_prompt_agent.py
│   │   ├── run_knowledge_augmented_prompt_agent.py
│   │   ├── run_rag_agent.py
│   │   └── run_evaluation_agent.py
│   └── workflow_agents/                      # Agent classes and unit tests
│       ├── base_agents.py
│       ├── direct_prompt_agent_test.py
│       ├── augmented_prompt_agent_test.py
│       └── knowledge_augmented_prompt_agent.py
├── results/                                  # Saved agent outputs (auto-created on first run)
├── requirements.txt
└── .env
```
