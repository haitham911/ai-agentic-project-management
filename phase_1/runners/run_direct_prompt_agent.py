from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import DirectPromptAgent
from phase_1.runners._tee import run_with_tee

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

agent = DirectPromptAgent(openai_api_key=api_key, base_url=base_url)

PROMPTS = [
    "What is the Capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
]


def main():
    print("=== DirectPromptAgent ===")
    print("Knowledge source: GPT-3.5-turbo pre-trained general knowledge\n")
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        response = agent.respond(prompt)
        print(f"Response: {response}\n")


run_with_tee("direct_prompt_agent.txt", main)
