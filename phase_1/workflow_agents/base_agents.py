from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime

load_dotenv()




class DirectPromptAgent:
    """An agent that directly prompts a pre-trained language model (like GPT-3.5-turbo) without using any external tools, databases, or retrieval systems. The agent relies entirely on the general knowledge and reasoning capabilities of the language model to generate responses based on the input prompt."""

    def __init__(self, openai_api_key, base_url):
        self.openaiKey = openai_api_key
        self.base_url = base_url

    def respond(self, prompt):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.openaiKey
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

class AugmentedPromptAgent:
    """An agent that augments prompts with a persona-based system message before calling the LLM.
    Unlike DirectPromptAgent, it injects a system prompt that sets a defined persona and clears
    prior context, then passes the (optionally augmented) user prompt to GPT-3.5-turbo."""

    def __init__(self, openai_api_key, base_url, persona: str = "a helpful assistant"):
        self.openaiKey = openai_api_key
        self.base_url = base_url
        self.persona = persona

    def respond(self, prompt):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.openaiKey
        )
        system_prompt = (
            f"You are {self.persona}. "
            "Forget any previous conversational context. "
            "Respond only based on the current message."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

class KnowledgeAugmentedPromptAgent:
    """An agent that grounds the LLM strictly in provided knowledge rather than its own training data.
    A persona and an explicit knowledge base are injected into the system prompt, and the model is
    instructed to answer only from that knowledge."""

    def __init__(self, openai_api_key, base_url, persona: str = "a helpful assistant", knowledge: str = ""):
        self.openaiKey = openai_api_key
        self.base_url = base_url
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, prompt):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.openaiKey
        )
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
            "Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # break the loop if we have reached the end of the text
            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content
    
class EvaluationAgent:

    def __init__(self, openai_api_key, base_url, persona, evaluation_criteria, worker_agent, max_interactions):
        # TODO: 1 - Declare class attributes
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):  # TODO: 2 - loop up to max_interactions
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)  # TODO: 3 - worker response
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"  # TODO: 4 - evaluation criteria
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[  # TODO: 5 - evaluation message structure
                    {"role": "system", "content": f"You are {self.persona}."},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[  # TODO: 6 - correction instruction message structure
                        {"role": "system", "content": f"You are {self.persona}."},
                        {"role": "user", "content": instruction_prompt},
                    ],
                    temperature=0,
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"Your previous response was:\n{response_from_worker}\n\n"
                    f"That response was rejected. Apply ONLY these corrections and rewrite your full answer:\n{instructions}\n\n"
                    f"Output only the corrected answer. Do not repeat these instructions."
                )
        return {  # TODO: 7 - return final response, evaluation, and iteration count
            "response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
        }

class RoutingAgent:
    def __init__(self, openai_api_key, base_url, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.agents = agents  # TODO: 1 - attribute to hold the agents

    def get_embedding(self, text):
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        # TODO: 2 - calculate embedding using text-embedding-3-large
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):  # TODO: 3 - routing method
        input_emb = self.get_embedding(user_input)  # TODO: 4 - embed the user input
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])  # TODO: 5 - embed agent description
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # TODO: 6 - select best agent by highest similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """An agent that uses provided knowledge to dynamically extract and list
    the steps required to execute a task described in a user's prompt."""

    def __init__(self, openai_api_key: str, base_url: str, knowledge: str = ""):
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.knowledge = knowledge
        self.client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)

    def respond(self, prompt: str) -> list[str]:
        system_prompt = (
            "You are an Action Planning Agent. "
            "Your job is to extract a clear, ordered list of action steps required to complete the task described by the user. "
            "Use only the following knowledge to inform your steps: "
            f"{self.knowledge} "
            "Return the steps as a numbered list, one step per line. "
            "Do not include any introduction, summary, or extra commentary."
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content

        actions = [
            line.strip()
            for line in response_text.splitlines()
            if line.strip()
        ]
        return actions