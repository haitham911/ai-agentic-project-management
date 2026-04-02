# agentic_workflow.py

# TODO: 1 - Import agents from local workflow_agents
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

# TODO: 3 - Load the product spec
spec_path = os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt")
with open(spec_path, "r", encoding="utf-8") as f:
    product_spec = f.read()

# ── Action Planning Agent ──────────────────────────────────────────────────────
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

# TODO: 4 - Instantiate action_planning_agent
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    knowledge=knowledge_action_planning,
)

# ── Product Manager ────────────────────────────────────────────────────────────
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Append the product spec
    + product_spec
)

# TODO: 6 - Instantiate product_manager_knowledge_agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
)

# TODO: 7 - Product Manager Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager = (
    "The answer should be user stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=3,
)

# ── Program Manager ────────────────────────────────────────────────────────────
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
)

# TODO: 8 - Program Manager Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=3,
)

# ── Development Engineer ───────────────────────────────────────────────────────
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
)

# TODO: 9 - Development Engineer Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=3,
)

# ── Support Functions (TODO: 11) ───────────────────────────────────────────────

def product_manager_support_function(query):
    result = product_manager_evaluation_agent.evaluate(query)
    return result["response"]


def program_manager_support_function(query):
    result = program_manager_evaluation_agent.evaluate(query)
    return result["response"]


def development_engineer_support_function(query):
    result = development_engineer_evaluation_agent.evaluate(query)
    return result["response"]


# TODO: 10 - Instantiate routing_agent
routes = [
    {
        "name": "ProductManager",
        "description": "Defines user stories for a product based on a product specification. Handles persona, action, and desired outcome for each story.",
        "func": product_manager_support_function,
    },
    {
        "name": "ProgramManager",
        "description": "Defines product features by grouping and organizing related user stories into cohesive functional areas.",
        "func": program_manager_support_function,
    },
    {
        "name": "DevelopmentEngineer",
        "description": "Defines development tasks and engineering work required to implement user stories and build the product.",
        "func": development_engineer_support_function,
    },
]

routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    base_url=base_url,
    agents=routes,
)

# ── Run the Workflow (TODO: 12) ────────────────────────────────────────────────

import sys

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "agentic_workflow.txt")


class _Tee:
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


with open(results_path, "w", encoding="utf-8") as _f:
    sys.stdout = _Tee(_f)
    try:
        print("\n*** Workflow execution started ***\n")
        workflow_prompt = "What would the development tasks for this product be?"
        print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

        print("\nDefining workflow steps from the workflow prompt")
        workflow_steps = action_planning_agent.respond(workflow_prompt)
        print(f"Workflow steps extracted: {len(workflow_steps)}")
        for i, step in enumerate(workflow_steps, 1):
            print(f"  Step {i}: {step}")

        completed_steps = []
        for i, step in enumerate(workflow_steps, 1):
            print(f"\n--- Executing Step {i}: {step} ---")
            result = routing_agent.route(step)
            completed_steps.append(result)
            print(f"Result:\n{result}")

        print("\n\n*** Workflow Final Output ***")
        print(completed_steps[-1])
    finally:
        sys.stdout = sys.stdout._stdout

print(f"\nResults saved to results/agentic_workflow.txt")
