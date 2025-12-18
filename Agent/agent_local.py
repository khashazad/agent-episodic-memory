"""
Local HuggingFace Model Agent for MineRL

This agent supports both local HuggingFace models and OpenAI's API.
Configuration is read from environment variables (see .env.example).

Usage:
    python Agent/agent_local.py

Or via sbatch:
    sbatch run_local_agent.sbatch

Environment Variables:
    USE_OPENAI_LLM: Set to "true" to use OpenAI instead of local model
    OPENAI_API_KEY: Required if USE_OPENAI_LLM is true
"""

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import subprocess
import time
import requests
import base64
from PIL import Image
import io
import atexit
import json
import os
import re
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Check which LLM to use
USE_OPENAI_LLM = os.environ.get("USE_OPENAI_LLM", "false").lower() == "true"

# Import model setup utilities
from utils.constants import (  # noqa: E402
    OPENAI_SYSTEM_PROMPT,
    LOCAL_SYSTEM_PROMPT,
    MINERL_ACTION_TEMPLATE,
    PITCH_MIN,
    PITCH_MAX,
)

if USE_OPENAI_LLM:
    from utils.openai_model import setup_openai_agent, ensure_openai_api_key
    # Ensure API key is set up early
    ensure_openai_api_key()
else:
    from utils.local_model import setup_local_agent

# Import RAG system (optional)
try:
    from RAG_server import create_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# =============================================================================
# Configuration from environment variables
# =============================================================================

# OpenAI LLM configuration
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))

# Local LLM configuration
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
LOCAL_MODEL_DEVICE = os.environ.get("LOCAL_MODEL_DEVICE", "auto")  # 'auto', 'cuda', 'cpu', 'mps'
LOCAL_MODEL_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS", "256"))
LOCAL_MODEL_TEMPERATURE = float(os.environ.get("LOCAL_MODEL_TEMPERATURE", "0.2"))
LOCAL_MODEL_DTYPE = os.environ.get("LOCAL_MODEL_DTYPE", "float16")  # 'float16', 'float32', 'bfloat16'

# Agent configuration
USE_RAG = os.environ.get("USE_RAG", "false").lower() == "true"
RAG_CONFIG = os.environ.get("RAG_CONFIG", "v1")  # v1, v2, or v3
RENDER = os.environ.get("RENDER", "false").lower() == "true"
TEST_RUNS = int(os.environ.get("TEST_RUNS", "5"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "500"))
USE_MAX_FRAMES = os.environ.get("USE_MAX_FRAMES", "true").lower() == "true"
EMBEDDING_METHOD = os.environ.get("EMBEDDING_METHOD", "local_llm")

# Remote server configuration
USE_REMOTE_SERVER = os.environ.get("USE_REMOTE_SERVER", "false").lower() == "true"
SERVER_URL = os.environ.get("MINERL_SERVER_URL", "http://127.0.0.1:5001")
ENV_ID = None  # Used to identify environment session on remote server

print("Configuration:")
print(f"  Use OpenAI LLM: {USE_OPENAI_LLM}")
if USE_OPENAI_LLM:
    print(f"  OpenAI Model: {OPENAI_MODEL_NAME}")
    print(f"  OpenAI Temperature: {OPENAI_TEMPERATURE}")
else:
    print(f"  Local Model: {LOCAL_MODEL_NAME}")
    print(f"  Device: {LOCAL_MODEL_DEVICE}")
    print(f"  Max New Tokens: {LOCAL_MODEL_MAX_NEW_TOKENS}")
    print(f"  Temperature: {LOCAL_MODEL_TEMPERATURE}")
print(f"  Use RAG: {USE_RAG}")
print(f"  RAG Config: {RAG_CONFIG}")
print(f"  Render: {RENDER}")
print(f"  Test Runs: {TEST_RUNS}")
print(f"  Max Frames: {MAX_FRAMES}")
print(f"  Use Remote Server: {USE_REMOTE_SERVER}")
print(f"  Server URL: {SERVER_URL}")

# =============================================================================
# State tracking
# =============================================================================

ACTION_HISTORY = deque(maxlen=5)
FRAME_HISTORY = deque(maxlen=16)

CAMERA_STATE = {"pitch": 0.0, "yaw": 0.0}

# For rendering the frames
_RENDER = {
    "fig": None,
    "ax": None,
    "im": None,
}

# Setting the RAG system
rag = None
if USE_RAG:
    if RAG_AVAILABLE:
        rag = create_rag(RAG_CONFIG)
    else:
        print("Warning: RAG requested but RAG_server not available")

# Load example images for training examples
try:
    with open('example_images.json', 'r') as f:
        example_images = json.load(f)
    EXAMPLE_IMAGES_LOADED = True
except FileNotFoundError:
    # Try alternate path (when running from project root)
    try:
        with open('Agent/example_images.json', 'r') as f:
            example_images = json.load(f)
        EXAMPLE_IMAGES_LOADED = True
    except FileNotFoundError:
        print("Warning: example_images.json not found, using text-only examples")
        example_images = {}
        EXAMPLE_IMAGES_LOADED = False

# Create base64 data URLs for example images
example_image_1 = "data:image/png;base64," + example_images.get("example_1", "") if EXAMPLE_IMAGES_LOADED else None
example_image_2 = "data:image/png;base64," + example_images.get("example_2", "") if EXAMPLE_IMAGES_LOADED else None
example_image_3 = "data:image/png;base64," + example_images.get("example_3", "") if EXAMPLE_IMAGES_LOADED else None
example_image_4 = "data:image/png;base64," + example_images.get("example_4", "") if EXAMPLE_IMAGES_LOADED else None

# Example actions and explanations
example_action_1 = {"forward": 1}
example_action_2 = {"attack": 1}
example_action_3 = {"camera": [0, -30]}
example_action_4 = {"camera": [30, 0]}

explanation_1 = "The agent saw the tree (object in white) in the distance and started moving towards it."
explanation_2 = "The agent is now at the tree. It continuously attacks the tree until the block is gone. Once the block is broken the reward is given."
explanation_3 = "The agent can only see dirt. There is nothing useful here and it should not be attacked. The agent should look around to find trees."
explanation_4 = "The agent just destroyed wood. To collect more the viewpoint needs to be changed. The agent can look up or down."

# =============================================================================
# System prompt for the agent
# =============================================================================

def build_openai_system_prompt():
    """Build the system prompt for OpenAI models with detailed behavioral examples including images."""
    base_prompt = OPENAI_SYSTEM_PROMPT

    # Add training examples with images (similar to agent.py)
    if EXAMPLE_IMAGES_LOADED:
        training_examples = f"""

Training examples:

EXAMPLE 1:
Input image: [{example_image_1}]
Taken action: {example_action_1}
The reason for the action: {explanation_1}

EXAMPLE 2:
Input image: [{example_image_2}]
Taken action: {example_action_2}
The reason for the action: {explanation_2}

EXAMPLE 3:
Input image: [{example_image_3}]
Taken action: {example_action_3}
The reason for the action: {explanation_3}

EXAMPLE 4:
Input image: [{example_image_4}]
Taken action: {example_action_4}
The reason for the action: {explanation_4}

Key visual cues for trees:
- Trees have brown/tan vertical trunks
- Oak logs appear as brown/beige blocks
- Trees are taller than the ground and stand out against the sky
- When close enough to attack, the trunk should be in the CENTER of your view
"""
    else:
        # Fallback to text-only examples if images not available
        training_examples = """

Detailed behavioral examples:

EXAMPLE 1 - Tree visible in distance:
- What you see: A tree trunk (brown/white vertical structure) visible ahead
- Best action: {"forward": 1} - Move towards the tree
- Why: The agent saw the tree in the distance and started moving towards it.

EXAMPLE 2 - Close to tree trunk:
- What you see: Tree trunk fills center of view, very close
- Best action: {"attack": 1} - Attack to break the wood block
- Why: The agent is now at the tree. Attack continuously until the block breaks and reward is given.

EXAMPLE 3 - No trees visible (dirt/grass only):
- What you see: Only ground (dirt, grass) with no trees in view
- Best action: {"camera": [0, -30]} or {"camera": [0, 30]} - Look around
- Why: The agent can only see dirt. There is nothing useful here. Look around to find trees.

EXAMPLE 4 - Just collected wood, need to find more:
- What you see: Tree partially destroyed or wood block just collected
- Best action: {"camera": [15, 0]} or {"camera": [-15, 0]} - Adjust view up/down
- Why: The agent just destroyed wood. To collect more, adjust the viewpoint to see remaining tree or find new trees.

Key visual cues for trees:
- Trees have brown/tan vertical trunks
- Oak logs appear as brown/beige blocks
- Trees are taller than the ground and stand out against the sky
- When close enough to attack, the trunk should be in the CENTER of your view
"""
    return base_prompt + training_examples

# Select the appropriate system prompt based on LLM type
if USE_OPENAI_LLM:
    system_msg = SystemMessage(content=build_openai_system_prompt())
else:
    system_msg = SystemMessage(content=LOCAL_SYSTEM_PROMPT)

container_started = False


# =============================================================================
# Utility functions
# =============================================================================

def log_result(wood):
    """Save the results of a run."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wood_collected": wood
    }

    path = f'Agent/Results/{RAG_CONFIG}.jsonl'

    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def process_image(obs_b64):
    """Convert base64 to PIL Image."""
    obs_bytes = base64.b64decode(obs_b64)
    obs = Image.open(io.BytesIO(obs_bytes))
    return obs


def init_renderer(first_obs_b64: str):
    """Create the matplotlib window ONCE."""
    frame = process_image(first_obs_b64)

    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_axis_off()

    im = ax.imshow(frame, interpolation="nearest")
    fig.tight_layout(pad=0)

    _RENDER["fig"] = fig
    _RENDER["ax"] = ax
    _RENDER["im"] = im

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()


def show_obs(obs_b64: str):
    """Update the existing window instead of clearing/recreating."""
    if _RENDER["fig"] is None:
        init_renderer(obs_b64)
        return

    frame = process_image(obs_b64)
    _RENDER["im"].set_data(frame)

    fig = _RENDER["fig"]
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def exit_cleanup():
    """Clean up resources on exit."""
    global container_started, ENV_ID

    if USE_REMOTE_SERVER:
        if ENV_ID:
            try:
                requests.post(
                    f"{SERVER_URL}/destroy",
                    json={"env_id": ENV_ID},
                    timeout=5
                )
                print(f"Destroyed remote environment session: {ENV_ID}")
            except Exception as e:
                print(f"Failed to destroy remote environment: {e}")
    else:
        if container_started:
            try:
                subprocess.run(
                    ["docker", "compose", "stop", "minerl-env"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                print("Stopping docker-compose service minerl-env")
            except Exception as e:
                print(f"Failed to stop docker-compose service: {e}")


atexit.register(exit_cleanup)


# =============================================================================
# Environment management
# =============================================================================

def create_env():
    """Start the environment (local Docker or remote server)."""
    global container_started, ENV_ID

    if USE_REMOTE_SERVER:
        print(f"Connecting to remote MineRL server at {SERVER_URL}...")

        try:
            resp = requests.post(
                f"{SERVER_URL}/create",
                json={"env_type": "MineRLTreechop-v0"},
            )


        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout connecting to MineRL server at {SERVER_URL}. Is the server running?")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Could not connect to MineRL server at {SERVER_URL}. Is the server running?\nError: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f'Failed to create remote environment: {resp.status_code} - {resp.text}')


        data = resp.json()
        ENV_ID = data['env_id']

        time.sleep(60) # wait for the environment to start

        print(f"Created remote environment session: {ENV_ID}")

        return data['obs']
    else:
        subprocess.run(
            ["docker", "compose", "up", "-d", "minerl-env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        container_started = True
        print("Started environment server using docker-compose (service: minerl-env). Waiting for setup.")
        time.sleep(20)

        print('Setting up environment...')
        print('LONG WAIT')
        resp = requests.get(url=f'{SERVER_URL}/start')

        if resp.status_code != 200:
            raise RuntimeError(f'Status code error: {resp.status_code}')

        return resp.json()['obs']


def reset_env():
    """Reset the environment for different runs."""
    if USE_REMOTE_SERVER:
        resp = requests.post(
            f"{SERVER_URL}/reset",
            json={"env_id": ENV_ID}
        )
    else:
        resp = requests.post(url=f'{SERVER_URL}/reset')

    if resp.status_code != 200:
        raise RuntimeError(f'Status code error: {resp.status_code}')

    return resp.json()['obs']


def submit_action(action_dict: dict):
    """Submit actions to the server."""
    action_body = MINERL_ACTION_TEMPLATE.copy()

    global CAMERA_STATE

    for action, value in action_dict.items():
        if action == "camera":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("Camera action requires value=[pitch_delta, yaw_delta]")

            pitch_delta, yaw_delta = float(value[0]), float(value[1])

            new_pitch = CAMERA_STATE["pitch"] + pitch_delta
            new_pitch = max(min(new_pitch, PITCH_MAX), PITCH_MIN)
            CAMERA_STATE["pitch"] = new_pitch
            CAMERA_STATE["yaw"] += yaw_delta

            actual_pitch_delta = new_pitch - (CAMERA_STATE["pitch"] - pitch_delta)
            action_body["camera"] = [actual_pitch_delta, yaw_delta]
        else:
            if action not in action_body:
                print(f"Warning: Ignoring unknown action name: {action}")
                continue
            action_body[action] = int(value)

    body = {"actions": action_body}
    if USE_REMOTE_SERVER:
        body["env_id"] = ENV_ID

    resp = requests.post(
        url=f"{SERVER_URL}/action",
        json=body,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Status code error: {resp.status_code}")

    data = resp.json()
    obs = data["obs"]
    reward = data["reward"]
    done = data["done"]

    return obs, reward, done, action_body


# =============================================================================
# Tool definition for LLM
# =============================================================================

@tool
def step_env(action: dict) -> dict:
    """Step the MineRL environment by executing a combined action.

    Args:
        action: Dictionary specifying controls to activate.
            Valid keys: "forward", "back", "left", "right", "jump", "attack", "camera"
            Movement/attack: 0 or 1
            Camera: [pitch_delta, yaw_delta]

    Returns:
        Dictionary with obs, reward, and done status.
    """
    obs, reward, done, action_body = submit_action(action)

    print(f"Action: {action}")

    ACTION_HISTORY.append({
        "action": action,
        "action_body": action_body,
        "reward": reward,
    })

    return {
        "obs": obs,
        "reward": reward,
        "done": done,
    }


# =============================================================================
# Action parsing
# =============================================================================

def parse_action_from_text(text: str) -> dict | None:
    """
    Attempt to parse an action dict from model output text.
    Handles various formats that local models might output.
    """
    # Try to find JSON-like action dict in the text
    # Pattern 1: {"forward": 1, "camera": [0, 15]}
    json_pattern = r'\{[^{}]*(?:"forward"|"back"|"left"|"right"|"jump"|"attack"|"camera")[^{}]*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            action = json.loads(match)
            # Validate it's a valid action dict
            valid_keys = {"forward", "back", "left", "right", "jump", "attack", "camera"}
            if any(k in valid_keys for k in action.keys()):
                return action
        except json.JSONDecodeError:
            continue

    # Pattern 2: action={"forward": 1}
    action_pattern = r'action\s*=\s*(\{[^{}]+\})'
    match = re.search(action_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 3: step_env(action={"forward": 1})
    tool_call_pattern = r'step_env\s*\(\s*action\s*=\s*(\{[^{}]+\})\s*\)'
    match = re.search(tool_call_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Message formatting
# =============================================================================

def format_prev_actions():
    """Format the previous actions list."""
    if ACTION_HISTORY:
        history_lines = [
            f"{i+1}. action={entry['action']}, reward={entry['reward']}"
            for i, entry in enumerate(list(ACTION_HISTORY))
        ]
        context = "Last actions taken:\n" + "\n".join(history_lines)
    else:
        context = "No previous actions yet."

    return context


def format_user_msg(context: str, memory_str: str | None = None, obs: str | None = None):
    """
    Format user message.

    For OpenAI models: includes vision (image) and uses tool calling.
    For local models: text-only, expects JSON action response.

    Args:
        context: Extra context string
        memory_str: Episodic memory string
        obs: Base64 encoded observation image (used for OpenAI vision)
    """
    prev_actions = format_prev_actions()
    goal = 'Goal: Collect as much wood as possible.'

    memory_block = memory_str or "No relevant episodic memory was retrieved for this state."

    camera_info = (
        f"Estimated camera pitch: {CAMERA_STATE['pitch']:.1f} degrees "
        "(0 = horizon; positive = down, negative = up).\n"
        "Try to keep pitch between about -20 and +20 degrees unless you are "
        "deliberately looking up or down briefly.\n"
    )

    if USE_OPENAI_LLM and obs is not None:
        # OpenAI mode: include image and use tool calling
        user_msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You see the current Minecraft frame below.\n\n"
                        f"Extra context:\n{context}\n\n"
                        f"Camera state:\n{camera_info}\n"
                        f"Previous actions:\n{prev_actions}\n\n"
                        f"Episodic memory (similar past situation):\n{memory_block}\n\n"
                        f"Goal: {goal}\n\n"
                        "Remember: You only get reward when you collect WOOD LOGS from trees. "
                        "If there is no reward, you did not hit the wood. "
                        "Decide the next action and call the `step_env` tool."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{obs}"
                    },
                },
            ]
        )
    else:
        # Local model mode: text-only, expects JSON response
        user_msg = HumanMessage(
            content=(
                "You are in a Minecraft world. Based on your memory and previous actions, "
                "decide the next action to collect wood.\n\n"
                f"Extra context:\n{context}\n\n"
                f"Camera state:\n{camera_info}\n"
                f"Previous actions:\n{prev_actions}\n\n"
                f"Episodic memory (similar past situation):\n{memory_block}\n\n"
                f"Goal: {goal}\n\n"
                "Respond with ONLY a JSON action dict like: {\"forward\": 1} or {\"camera\": [0, 15]}\n"
            )
        )

    return user_msg


# =============================================================================
# Agent episode runner
# =============================================================================

def run_agent_episode(agent, obs):
    """Run a single agent episode step."""
    reward = 0
    done = False

    # Query episodic memory if available
    memory_str = "No episodic memory yet (not enough frames)."
    if USE_RAG and rag and len(FRAME_HISTORY) == 16:
        memory_str = rag.get_action(FRAME_HISTORY)

    context = ""
    user_msg = format_user_msg(context, memory_str=memory_str, obs=obs)

    # Invoke the model with retry logic for rate limiting
    ai_msg = None
    max_retries = 5
    retry_delay = 6  # Start with 6 seconds (based on error message suggestion)

    for attempt in range(max_retries):
        try:
            ai_msg = agent.invoke([system_msg, user_msg])
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in str(e):
                print(f"Hit rate limit. Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"Error invoking model: {e}")
                raise

    if ai_msg is None:
        print("Warning: Failed to get response from model after retries. Skipping action (no-op).")
        FRAME_HISTORY.append(obs)
        return obs, 0, False

    action_executed = False

    # Try standard tool_calls first (works with OpenAI and some HF models)
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
            if tool_name == "step_env":
                raw_args = tool_call.get("args") if isinstance(tool_call, dict) else tool_call.args

                # Handle case where raw_args is not a dict
                if not isinstance(raw_args, dict):
                    print(f"Warning: Tool call args is not a dict: {raw_args}. Skipping action (no-op).")
                    FRAME_HISTORY.append(obs)
                    return obs, 0, False
                if "action" not in raw_args:
                    raw_args = {"action": raw_args}

                # Validate that action is a dictionary
                action_value = raw_args.get("action")
                if not isinstance(action_value, dict):
                    print(f"Warning: action is not a dict: {action_value}. Skipping action (no-op).")
                    FRAME_HISTORY.append(obs)
                    return obs, 0, False

                try:
                    result = step_env.invoke(raw_args)
                    obs = result["obs"]
                    reward = result["reward"]
                    done = result["done"]
                    action_executed = True
                except Exception as e:
                    print(f"Warning: Tool invocation failed: {e}. Skipping action (no-op).")
                    FRAME_HISTORY.append(obs)
                    return obs, 0, False

    # Fallback for local models: parse action from text output
    if not action_executed and not USE_OPENAI_LLM:
        text_content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
        parsed_action = parse_action_from_text(text_content)

        if parsed_action:
            print(f"Parsed action from text: {parsed_action}")
            result = step_env.invoke({"action": parsed_action})
            obs = result["obs"]
            reward = result["reward"]
            done = result["done"]
            action_executed = True
        else:
            print(f"Warning: Could not parse action from model output: {text_content[:300]}...")
            print("Skipping action (no-op).")
            FRAME_HISTORY.append(obs)
            return obs, 0, False
    elif not action_executed and USE_OPENAI_LLM:
        # OpenAI mode but no tool call - skip action
        print("Warning: OpenAI model did not make a tool call. Skipping action (no-op).")
        FRAME_HISTORY.append(obs)
        return obs, 0, False

    FRAME_HISTORY.append(obs)

    return obs, reward, done


# =============================================================================
# Main agent loop
# =============================================================================

def run_agent():
    """Main agent execution loop."""
    print("\n" + "="*50)
    print("Starting agent...")
    print("="*50 + "\n")

    # Setup the agent based on configuration
    if USE_OPENAI_LLM:
        print("Setting up OpenAI LLM agent...")
        agent = setup_openai_agent(tools=[step_env])
    else:
        print("Setting up local LLM agent...")
        agent = setup_local_agent()

    # Start the env
    print("\nConnecting to MineRL environment...")
    print(f"  Server URL: {SERVER_URL}")
    print(f"  Remote mode: {USE_REMOTE_SERVER}")
    obs = create_env()

    print("Environment connected successfully!")

    # Start rendering
    if RENDER:
        init_renderer(obs)

    done = False
    reward = 0
    wood_count = 0
    cur_frames = 0

    for run_idx in range(TEST_RUNS):
        print(f"\n{'='*50}")
        print(f"Starting run {run_idx + 1}/{TEST_RUNS}")
        print(f"{'='*50}\n")

        done = False

        while not done:
            obs, reward, done = run_agent_episode(agent, obs)

            if reward != 0:
                wood_count += 1
                print(f'Wood count: {wood_count}')

            # Render this frame
            if RENDER:
                show_obs(obs)

            cur_frames += 1
            print(f'Frame {cur_frames}')

            # Check frame limit
            if USE_MAX_FRAMES and cur_frames >= MAX_FRAMES:
                cur_frames = 0
                break

        # Save the results of that run
        log_result(wood_count)
        print(f"\nRun {run_idx + 1} complete. Wood collected: {wood_count}")

        # Reset the env and tracking
        wood_count = 0
        cur_frames = 0
        ACTION_HISTORY.clear()
        FRAME_HISTORY.clear()

        if run_idx < TEST_RUNS - 1:  # Don't reset after last run
            obs = reset_env()

    print("\nAll runs complete!")


if __name__ == "__main__":
    run_agent()
