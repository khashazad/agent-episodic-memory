"""
Utility modules for the MineRL agent.
"""

from .constants import OPENAI_SYSTEM_PROMPT, LOCAL_SYSTEM_PROMPT
from .local_model import setup_local_agent
from .openai_model import setup_openai_agent

__all__ = [
    "OPENAI_SYSTEM_PROMPT",
    "LOCAL_SYSTEM_PROMPT",
    "setup_local_agent",
    "setup_openai_agent",
]
