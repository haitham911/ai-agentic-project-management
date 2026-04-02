# AI-Powered Agentic Workflow for Project Management

An agentic AI system that builds a full project development plan from a product specification — automatically generating user stories, product features, and engineering tasks using a pipeline of specialized LLM agents.

---

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

Create a `.env` file at the project root:

```text
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

---

## Project Structure

```text
.
├── phase_1/                                    # Agent toolkit — individual agent classes and demos
│   ├── workflow_agents/
│   │   ├── base_agents.py                      # All agent class definitions
│   │   ├── direct_prompt_agent_test.py         # Test: DirectPromptAgent
│   │   ├── augmented_prompt_agent_test.py      # Test: AugmentedPromptAgent
│   │   ├── knowledge_augmented_prompt_agent.py # Test: KnowledgeAugmentedPromptAgent
│   │   ├── rag_knowledge_prompt_agent.py       # Test: RAGKnowledgePromptAgent
│   │   ├── action_planning_agent_test.py       # Test: ActionPlanningAgent
│   │   └── README.md
│   └── runners/                                # Scripts that run each agent and save output
│       ├── _tee.py                             # Shared stdout-to-file helper
│       ├── run_direct_prompt_agent.py
│       ├── run_augmented_prompt_agent.py
│       ├── run_knowledge_augmented_prompt_agent.py
│       ├── run_rag_agent.py
│       ├── run_evaluation_agent.py
│       ├── run_routing_agent.py
│       ├── run_action_planning_agent.py
│       └── README.md
│
├── phase_2/                                    # Agentic workflow — full project management pipeline
│   ├── agentic_workflow.py                     # Main workflow orchestration script
│   ├── Product-Spec-Email-Router.txt           # Input product specification
│   └── workflow_agents/
│       └── base_agents.py                      # Agent classes (env loading removed — caller's responsibility)
│
├── results/                                    # All saved agent outputs (auto-created on first run)
│   ├── direct_prompt_agent.txt                 # Phase 1 — DirectPromptAgent output
│   ├── augmented_prompt_agent.txt              # Phase 1 — AugmentedPromptAgent output
│   ├── knowledge_augmented_prompt_agent.txt    # Phase 1 — KnowledgeAugmentedPromptAgent output
│   ├── rag_agent.txt                           # Phase 1 — RAGKnowledgePromptAgent output
│   ├── evaluation_agent.txt                    # Phase 1 — EvaluationAgent output
│   ├── routing_agent.txt                       # Phase 1 — RoutingAgent output
│   ├── action_planning_agent.txt               # Phase 1 — ActionPlanningAgent output
│   └── agentic_workflow.txt                    # Phase 2 — Full workflow output (user stories + features + tasks)
│
├── requirements.txt
└── .env
```

---

## Phase 1 — Agent Toolkit

Six agent classes are implemented in `phase_1/workflow_agents/base_agents.py`:

| Agent | Method | Description |
|-------|--------|-------------|
| `DirectPromptAgent` | `respond()` | Sends prompt directly to GPT-3.5-turbo, no system prompt |
| `AugmentedPromptAgent` | `respond()` | Injects a persona via system prompt |
| `KnowledgeAugmentedPromptAgent` | `respond()` | Answers strictly from injected knowledge |
| `RAGKnowledgePromptAgent` | `find_prompt_in_knowledge()` | Retrieves relevant chunks via embeddings |
| `EvaluationAgent` | `evaluate()` | Iterative feedback loop with worker agent |
| `RoutingAgent` | `route()` | Routes prompts to best agent via cosine similarity |
| `ActionPlanningAgent` | `extract_steps_from_prompt()` | Extracts ordered action steps from a prompt |

### Running Phase 1 Agents

Run all commands from the **project root**. Output is saved to `results/`.

```bash
venv/bin/python -m phase_1.runners.run_direct_prompt_agent
venv/bin/python -m phase_1.runners.run_augmented_prompt_agent
venv/bin/python -m phase_1.runners.run_knowledge_augmented_prompt_agent
venv/bin/python -m phase_1.runners.run_rag_agent
venv/bin/python -m phase_1.runners.run_evaluation_agent
venv/bin/python -m phase_1.runners.run_routing_agent
venv/bin/python -m phase_1.runners.run_action_planning_agent
```

### Phase 1 Results

All output files are in `results/`:

| File | Contents |
|------|----------|
| `results/direct_prompt_agent.txt` | Responses to general knowledge questions |
| `results/augmented_prompt_agent.txt` | Responses from a travel guide persona |
| `results/knowledge_augmented_prompt_agent.txt` | Responses grounded in injected knowledge |
| `results/rag_agent.txt` | Responses retrieved from a knowledge corpus |
| `results/evaluation_agent.txt` | Full evaluation loop log with iterations |
| `results/routing_agent.txt` | Routing decisions and agent responses |
| `results/action_planning_agent.txt` | Extracted action steps for cooking tasks |

### Running Phase 1 Tests

```bash
# Run all unit and integration tests
venv/bin/python -m unittest discover -s phase_1/workflow_agents -p "*_test.py" -v

# Run a specific test
venv/bin/python -m unittest phase_1.workflow_agents.direct_prompt_agent_test -v
venv/bin/python -m unittest phase_1.workflow_agents.augmented_prompt_agent_test -v
venv/bin/python -m unittest phase_1.workflow_agents.knowledge_augmented_prompt_agent -v
venv/bin/python -m unittest phase_1.workflow_agents.action_planning_agent_test -v
```

---

## Phase 2 — Agentic Workflow

The full project management pipeline in `phase_2/agentic_workflow.py` orchestrates multiple agents to generate a complete development plan from a product specification.

**Pipeline:**
```
ActionPlanningAgent → extracts workflow steps
       ↓
RoutingAgent → routes each step to the best role:
   ├── ProductManager   → writes user stories   → EvaluationAgent validates
   ├── ProgramManager   → defines features      → EvaluationAgent validates
   └── DevelopmentEngineer → creates tasks      → EvaluationAgent validates
       ↓
Final output: User Stories + Product Features + Engineering Tasks
```

### Running Phase 2 Workflow

```bash
cd phase_2
../venv/bin/python agentic_workflow.py
```

### Phase 2 Results

| File | Contents |
|------|----------|
| `results/agentic_workflow.txt` | Complete workflow log: all 7 steps, routing decisions, evaluation loops, and the final structured development plan including user stories, product features, and engineering tasks (T-001 → T-006) for the Email Router product |
