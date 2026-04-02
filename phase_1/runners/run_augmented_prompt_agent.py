from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import AugmentedPromptAgent
from phase_1.runners._tee import run_with_tee

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

agent = AugmentedPromptAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona="a knowledgeable travel guide who speaks in an enthusiastic tone",
)

PROMPTS = [
    "What are the top 3 must-see attractions in Rome?",
    "What should I do on my first trip to Tokyo?",
    "Why should I visit Barcelona?",
]


def main():
    print("=== AugmentedPromptAgent ===")
    print("Persona: enthusiastic travel guide")
    print("Knowledge source: GPT-3.5-turbo pre-trained knowledge + persona system prompt\n")
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        response = agent.respond(prompt)
        print(f"Response: {response}\n")


run_with_tee("augmented_prompt_agent.txt", main)
