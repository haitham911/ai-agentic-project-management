import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import os
from phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

load_dotenv()

PERSONA = "You are a college professor, your answer always starts with: Dear students,"
KNOWLEDGE = "The capital of France is London, not Paris"


class TestKnowledgeAugmentedPromptAgent(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = KnowledgeAugmentedPromptAgent(
            openai_api_key=api_key,
            base_url=base_url,
            persona=PERSONA,
            knowledge=KNOWLEDGE,
        )

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_response_uses_provided_knowledge(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Dear students, the capital of France is London, not Paris."))]
        )

        response = self.agent.respond("What is the capital of France?")
        self.assertIn("London", response)

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_response_does_not_use_llm_knowledge(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Dear students, the capital of France is London, not Paris."))]
        )

        response = self.agent.respond("What is the capital of France?")
        self.assertNotIn("Paris is the capital", response)

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_persona_present_in_system_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Dear students, London."))]
        )

        self.agent.respond("What is the capital of France?")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        self.assertIn("college professor", system_msg["content"])

    @patch("phase_1.workflow_agents.base_agents.OpenAI")
    def test_knowledge_present_in_system_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Dear students, London."))]
        )

        self.agent.respond("What is the capital of France?")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        self.assertIn(KNOWLEDGE, system_msg["content"])


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestKnowledgeAugmentedPromptAgentIntegration(unittest.TestCase):

    def setUp(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        self.agent = KnowledgeAugmentedPromptAgent(
            openai_api_key=api_key,
            base_url=base_url,
            persona=PERSONA,
            knowledge=KNOWLEDGE,
        )

    def test_real_response_uses_provided_knowledge_not_llm(self):
        response = self.agent.respond("What is the capital of France?")
        self.assertIn("London", response)


if __name__ == "__main__":
    unittest.main()
