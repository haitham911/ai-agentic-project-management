import unittest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import ActionPlanningAgent

load_dotenv()


class TestActionPlanningAgent(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = ActionPlanningAgent(
            openai_api_key=api_key,
            base_url=base_url,
            knowledge="General cooking knowledge and kitchen procedures.",
        )

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_respond_returns_list(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="1. Crack eggs\n2. Whisk eggs\n3. Heat pan\n4. Cook eggs"))]
        )
        # Re-instantiate so the patched OpenAI is used
        agent = ActionPlanningAgent(
            openai_api_key="test-key",
            base_url="https://openai.vocareum.com/v1",
            knowledge="General cooking knowledge.",
        )
        result = agent.respond("One morning I wanted to have scrambled eggs")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_empty_lines_removed(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="1. Crack eggs\n\n2. Cook\n\n"))]
        )
        agent = ActionPlanningAgent(openai_api_key="test-key", base_url="https://openai.vocareum.com/v1")
        result = agent.respond("Make eggs")
        self.assertEqual(result, ["1. Crack eggs", "2. Cook"])

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_knowledge_injected_in_system_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="1. Step one"))]
        )
        agent = ActionPlanningAgent(
            openai_api_key="test-key",
            base_url="https://openai.vocareum.com/v1",
            knowledge="Cooking basics",
        )
        agent.respond("Make eggs")
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        self.assertIn("Cooking basics", system_msg["content"])
        self.assertIn("Action Planning Agent", system_msg["content"])

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_user_prompt_passed_to_api(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="1. Step one"))]
        )
        agent = ActionPlanningAgent(openai_api_key="test-key", base_url="https://openai.vocareum.com/v1")
        agent.respond("One morning I wanted to have scrambled eggs")
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertEqual(user_msg["content"], "One morning I wanted to have scrambled eggs")


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestActionPlanningAgentIntegration(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = ActionPlanningAgent(
            openai_api_key=api_key,
            base_url=base_url,
            knowledge="General cooking knowledge and kitchen procedures.",
        )

    def test_scrambled_eggs_returns_action_steps(self):
        result = self.agent.respond("One morning I wanted to have scrambled eggs")
        print("\nExtracted action steps:")
        for step in result:
            print(step)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
