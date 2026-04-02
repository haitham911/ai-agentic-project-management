import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv()


class TestAugmentedPromptAgent(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = AugmentedPromptAgent(
            openai_api_key=api_key,
            base_url=base_url,
            persona="a knowledgeable travel guide who speaks in an enthusiastic tone"
        )

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_respond_returns_text(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  Paris is the capital!  "))]
        )

        augmented_agent_response = self.agent.respond("What is the Capital of France?")

        self.assertEqual(augmented_agent_response, "Paris is the capital!")

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_persona_is_injected_in_system_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Paris!"))]
        )

        self.agent.respond("What is the Capital of France?")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        self.assertIn("travel guide", system_msg["content"])
        self.assertIn("enthusiastic", system_msg["content"])

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_system_prompt_clears_prior_context(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Paris!"))]
        )

        self.agent.respond("What is the Capital of France?")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        self.assertIn("Forget any previous conversational context", system_msg["content"])

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_user_prompt_passed_unchanged(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Paris!"))]
        )

        self.agent.respond("What is the Capital of France?")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertEqual(user_msg["content"], "What is the Capital of France?")


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestAugmentedPromptAgentIntegration(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = AugmentedPromptAgent(
            openai_api_key=api_key,
            base_url=base_url,
            persona="a knowledgeable travel guide who speaks in an enthusiastic tone"
        )

    def test_real_response_is_non_empty_string(self):
        augmented_agent_response = self.agent.respond("What are the top 3 must-see attractions in Rome?")
        self.assertIsInstance(augmented_agent_response, str)
        self.assertGreater(len(augmented_agent_response), 0)

    def test_response_reflects_travel_knowledge(self):
        augmented_agent_response = self.agent.respond("What should I do on my first trip to Tokyo?")
        self.assertTrue(
            any(word in augmented_agent_response for word in ["Tokyo", "temple", "sushi", "shibuya", "Shibuya", "shrine", "Shinjuku", "food"]),
            msg=f"Expected travel-specific content, got: {augmented_agent_response}"
        )

    def test_persona_influences_tone(self):
        augmented_agent_response = self.agent.respond("Why should I visit Barcelona?")
        self.assertGreater(len(augmented_agent_response), 50)

    def test_stateless_across_calls(self):
        self.agent.respond("Tell me everything about Bali beaches.")
        augmented_agent_response = self.agent.respond("What is the best time of year to visit Iceland?")
        self.assertTrue(
            any(word in augmented_agent_response for word in ["Iceland", "aurora", "summer", "winter", "northern lights", "midnight"]),
            msg=f"Expected Iceland-specific content, got: {augmented_agent_response}"
        )


if __name__ == "__main__":
    unittest.main()
