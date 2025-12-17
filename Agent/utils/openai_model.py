"""
OpenAI model setup for the MineRL agent.
"""

import os
import sys
import json

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_openai_config():
    """Get OpenAI configuration from environment variables."""
    return {
        "model_name": os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
    }


def ensure_openai_api_key():
    """
    Ensure the OpenAI API key is available.

    Checks config.json first, then environment variables.
    Sets the OPENAI_API_KEY environment variable if found.

    Returns:
        str: The OpenAI API key.

    Raises:
        ValueError: If no API key is found.
    """
    # Try config.json first
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        openai_key = config.get('OPENAI_API_KEY')
    except Exception:
        openai_key = None

    # Fall back to environment variable
    if not openai_key:
        openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found (required when USE_OPENAI_LLM=true)")

    os.environ["OPENAI_API_KEY"] = openai_key
    return openai_key


def setup_openai_agent(tools: list | None = None, config: dict | None = None):
    """
    Setup OpenAI model-based agent with optional tool binding.

    Args:
        tools: Optional list of tools to bind to the model.
        config: Optional configuration dictionary. If None, reads from environment variables.
               Expected keys: model_name, temperature

    Returns:
        ChatOpenAI: The configured chat model, with tools bound if provided.

    Raises:
        ImportError: If langchain-openai is not installed.
        ValueError: If OPENAI_API_KEY is not found.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "OpenAI mode requires langchain-openai. "
            "Install with: pip install langchain-openai"
        )

    # Ensure API key is available
    ensure_openai_api_key()

    # Use provided config or get from environment
    if config is None:
        config = get_openai_config()

    model_name = config["model_name"]
    temperature = config["temperature"]

    print(f"\nSetting up OpenAI agent with model: {model_name}")
    sys.stdout.flush()

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature
    )

    # Bind tools if provided
    if tools:
        llm = llm.bind_tools(tools)

    print("OpenAI agent ready!")
    sys.stdout.flush()
    return llm
