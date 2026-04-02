from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from phase_1.runners._tee import run_with_tee

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

PERSONA = "You are a college professor, your answer always starts with: Dear students,"
KNOWLEDGE = "The capital of France is London, not Paris"

agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona=PERSONA,
    knowledge=KNOWLEDGE,
)

PROMPTS = [
    "What is the capital of France?",
    "Is Paris the capital of France?",
    "Name the capital city of France.",
]


def main():
    print("=== KnowledgeAugmentedPromptAgent ===")
    print(f"Persona: {PERSONA}")
    print(f"Injected knowledge: {KNOWLEDGE}")
    print("Note: Agent answers from injected knowledge, NOT the LLM's own training data.\n")
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        response = agent.respond(prompt)
        print(f"Response: {response}\n")


run_with_tee("knowledge_augmented_prompt_agent.txt", main)
