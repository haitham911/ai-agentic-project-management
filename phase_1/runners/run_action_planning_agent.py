from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import ActionPlanningAgent
from phase_1.runners._tee import run_with_tee

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

agent = ActionPlanningAgent(
    openai_api_key=api_key,
    base_url=base_url,
    knowledge="General cooking knowledge and kitchen procedures.",
)

PROMPTS = [
    "One morning I wanted to have scrambled eggs",
    "I need to bake a chocolate cake",
    "I want to make a cup of coffee",
]


def main():
    print("=== ActionPlanningAgent ===\n")
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        steps = agent.respond(prompt)
        print("Action steps:")
        for step in steps:
            print(f"  {step}")
        print()


run_with_tee("action_planning_agent.txt", main)
