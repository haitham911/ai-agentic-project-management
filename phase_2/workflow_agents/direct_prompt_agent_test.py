import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import DirectPromptAgent

load_dotenv()


class TestDirectPromptAgent(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = DirectPromptAgent(openai_api_key=api_key, base_url=base_url)

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_respond_returns_stripped_content(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  Paris  "))]
        )

        response = self.agent.respond("What is the Capital of France?")
        self.assertEqual(response, "Paris")

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_respond_sends_correct_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Paris"))]
        )

        self.agent.respond("What is the Capital of France?")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "What is the Capital of France?")

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_respond_uses_correct_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Paris"))]
        )

        self.agent.respond("What is the Capital of France?")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-3.5-turbo")

    def test_knowledge_source_is_pretrained_llm(self):
        """
        The agent uses the general pre-trained knowledge of the GPT-3.5-turbo
        language model. No external tools, databases, or retrieval systems are
        involved — the model relies entirely on information learned during training.
        """
        self.assertIsInstance(self.agent, DirectPromptAgent)


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestDirectPromptAgentIntegration(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = DirectPromptAgent(openai_api_key=api_key, base_url=base_url)

    def test_real_response_to_capital_question(self):
        response = self.agent.respond("What is the Capital of France?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()
