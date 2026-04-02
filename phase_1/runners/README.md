# Agent Runners

Each script runs a specific agent, prints the full interaction to the terminal, and saves all output to `results/<agent_name>.txt`.

Run all commands from the **project root**.

---

## Scripts

### run_direct_prompt_agent.py

Sends prompts directly to GPT-3.5-turbo with no system prompt or injected knowledge.
The model answers entirely from its own pre-trained knowledge.

```bash
venv/bin/python -m phase_1.runners.run_direct_prompt_agent
```

Output: `results/direct_prompt_agent.txt`

---

### run_augmented_prompt_agent.py

Injects a persona via a system prompt before each call.
The model is told to forget prior context and respond as the defined persona.

```bash
venv/bin/python -m phase_1.runners.run_augmented_prompt_agent
```

Output: `results/augmented_prompt_agent.txt`

---

### run_knowledge_augmented_prompt_agent.py

Injects a custom knowledge string into the system prompt.
The model is instructed to answer **only** from that knowledge, not its own training data.

```bash
venv/bin/python -m phase_1.runners.run_knowledge_augmented_prompt_agent
```

Output: `results/knowledge_augmented_prompt_agent.txt`

---

### run_rag_agent.py

Chunks a knowledge corpus, generates embeddings, and retrieves the most relevant chunk
via cosine similarity to answer each prompt (Retrieval-Augmented Generation).

```bash
venv/bin/python -m phase_1.runners.run_rag_agent
```

Output: `results/rag_agent.txt`

> Note: Makes embedding API calls and writes temporary CSV files (`chunks-*.csv`, `embeddings-*.csv`) to the working directory.

---

### run_evaluation_agent.py

Runs a worker agent in an evaluation loop. After each response, an evaluator agent
judges it against defined criteria and generates correction instructions if needed.
The loop repeats until the response passes or `max_interactions` is reached.

```bash
venv/bin/python -m phase_1.runners.run_evaluation_agent
```

Output: `results/evaluation_agent.txt`

### run_routing_agent.py

Uses cosine similarity between prompt embeddings and agent description embeddings to
automatically route each user prompt to the most relevant agent (Texas, Europe, or Math).

```bash
venv/bin/python -m phase_1.runners.run_routing_agent
```

Output: `results/routing_agent.txt`

---

## How Results Are Saved

All runners use the shared `_tee.py` helper, which mirrors every line printed to the
terminal into the corresponding result file simultaneously — so the full interaction
log is always captured, not just the final answer.
