from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
from phase_1.runners._tee import run_with_tee

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

# Instantiate Texas agent
texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona="a Texas historian",
    knowledge=(
        "Texas is the second-largest U.S. state by area and population. "
        "It was an independent republic from 1836 to 1845 before joining the United States. "
        "Rome, Texas is a small unincorporated community in Webb County near Laredo. "
        "Texas is known for its ranching, oil industry, and diverse culture."
    ),
)

# Instantiate Europe agent
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona="a European historian",
    knowledge=(
        "Europe is a continent with a rich history spanning thousands of years. "
        "Rome, Italy is the capital city of Italy and was the center of the Roman Empire. "
        "It is known for landmarks such as the Colosseum, the Vatican, and the Trevi Fountain. "
        "The Roman Empire at its peak controlled much of Europe, North Africa, and the Middle East."
    ),
)

# Instantiate Math agent
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=api_key,
    base_url=base_url,
    persona="a math tutor",
    knowledge=(
        "Mathematics covers arithmetic, algebra, geometry, and calculus. "
        "To calculate total time: multiply the number of items by the time per item. "
        "For example, if one story takes 2 days and there are 20 stories, total = 2 x 20 = 40 days."
    ),
)

# Define agent lambdas
texas_fn  = lambda prompt: texas_agent.respond(prompt)
europe_fn = lambda prompt: europe_agent.respond(prompt)
math_fn   = lambda prompt: math_agent.respond(prompt)

# Assign agents to the router
router = RoutingAgent(
    openai_api_key=api_key,
    base_url=base_url,
    agents=[
        {
            "name": "TexasAgent",
            "description": "Answers questions about Texas history, geography, culture, and Texas-specific locations.",
            "func": texas_fn,
        },
        {
            "name": "EuropeAgent",
            "description": "Answers questions about European history, geography, culture, and European cities and landmarks.",
            "func": europe_fn,
        },
        {
            "name": "MathAgent",
            "description": "Solves math problems, arithmetic, time calculations, and numerical reasoning.",
            "func": math_fn,
        },
    ],
)

PROMPTS = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
]


def main():
    print("=== RoutingAgent ===\n")
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        response = router.route(prompt)
        print(f"Response: {response}\n")
        print("-" * 60)


run_with_tee("routing_agent.txt", main)
