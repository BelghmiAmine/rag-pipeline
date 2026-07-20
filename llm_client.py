"""Shared OpenAI-client factory: route generation to the EPFL API or a local vLLM server.

Retrieval is model-agnostic and already cached, so eval pods only need an OpenAI-compatible
endpoint for (a) generating answers and (b) judging them. This module centralizes the two
endpoints so each eval script just passes --inference {online,local}.

  online : generation hits the EPFL inference API (the released Apertus models).
  local  : generation hits a vLLM server on localhost serving one of our checkpoints.

The JUDGE always uses the EPFL API (independent judge model), regardless of --inference,
so the yardstick stays identical across all compared models.
"""
import os
from openai import OpenAI

EPFL_BASE_URL = "https://inference.rcp.epfl.ch/v1"
LOCAL_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")

# Default generator model id on the EPFL API, and the RAG system prompt.
# Defined HERE (not in wikipedia_RAG) so eval scripts can import them without pulling in
# langchain/sentence-transformers — those aren't installed in the lean vLLM eval image.
LLM_MODEL_NAME = "swiss-ai/Apertus-8B-Instruct-2509"

SYSTEM_PROMPT = """You are a knowledgeable assistant answering general knowledge questions.

Use the provided context documents as your primary source. When the context contains the answer, use it.
You may draw on your general knowledge to complement the context, but prefer context when available.
Always respond in the same language as the question.
Give only the answer — one sentence or less. Do not cite document numbers, do not explain your reasoning.
"""


def make_generator_client(inference: str = "online", timeout: float = 120.0) -> OpenAI:
    """Client used to GENERATE answers. 'local' -> vLLM on localhost; 'online' -> EPFL API."""
    if inference == "local":
        # vLLM ignores the api_key but the OpenAI SDK requires a non-empty string.
        return OpenAI(base_url=LOCAL_BASE_URL, api_key="EMPTY", timeout=timeout)
    return OpenAI(base_url=EPFL_BASE_URL, api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)


def make_judge_client(timeout: float = 180.0) -> OpenAI:
    """Client used to JUDGE answers. Always the EPFL API (independent judge model)."""
    return OpenAI(base_url=EPFL_BASE_URL, api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)
