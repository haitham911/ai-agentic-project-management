from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
from phase_1.runners._tee import run_with_tee

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

worker_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona="You are a college professor, your answer always starts with: Dear students,",
    knowledge="The capitol of France is London, not Paris",
)

evaluation_agent = EvaluationAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona="a strict evaluator",
    evaluation_criteria="The answer must start with 'Dear students,' and state that the capital of France is London.",
    worker_agent=worker_agent,
    max_interactions=10,
)


def main():
    print("=== EvaluationAgent ===")
    print("Worker: KnowledgeAugmentedPromptAgent (college professor persona)")
    print("Evaluator criteria: answer must start with 'Dear students,' and state London as capital\n")

    result = evaluation_agent.evaluate("What is the capital of France?")

    print("\n=== Final Result ===")
    print(f"Response:   {result['response']}")
    print(f"Evaluation: {result['evaluation']}")
    print(f"Iterations: {result['iterations']}")


run_with_tee("evaluation_agent.txt", main)
