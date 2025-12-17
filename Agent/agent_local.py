"""
Local HuggingFace Model Agent for MineRL

This agent uses a local HuggingFace model instead of OpenAI's API.
Configuration is read from environment variables (see .env.example).

Usage:
    python Agent/agent_local.py

Or via sbatch:
    sbatch run_local_agent.sbatch
"""

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import subprocess
import sys
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

# Enable HuggingFace progress bars and ensure output is unbuffered
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Import HuggingFace dependencies
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "This agent requires langchain-huggingface and transformers. "
        "Install with: pip install langchain-huggingface transformers torch"
    ) from e

# Import RAG system (optional)
try:
    from RAG_server import RAG, create_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# =============================================================================
# Configuration from environment variables
# =============================================================================

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

print(f"Configuration:")
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
PITCH_MIN, PITCH_MAX = -60.0, 60.0

MINERL_ACTION_TEMPLATE = {
    "attack": 0,
    "back": 0,
    "camera": [0.0, 0.0],   # pitch, yaw
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
}

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

# =============================================================================
# System prompt for the agent
# =============================================================================

system_msg = SystemMessage(
    content=(
        "You are a Minecraft-playing agent in a 3D world.\n"
        "\n"
        "Available controls:\n"
        "- Movement: forward, back, left, right, jump\n"
        "- Camera: camera (pitch, yaw)\n"
        "- Breaking blocks: attack\n"
        "\n"
        "Your goal is to collect as much wood as possible. To do this, you must:\n"
        "1) Use camera to look around until you can see a tree.\n"
        "2) Use forward/left/right/back/jump to walk up to the tree.\n"
        "3) When close enough and looking at the trunk, use attack.\n"
        "\n"
        "Very important behavior guidelines:\n"
        "- Do NOT only use forward and attack. Use camera to turn and explore.\n"
        "- Use left/right/back/jump to navigate around obstacles or adjust position.\n"
        "- Prefer using camera at the start of an episode or when you see no tree.\n"
        "- If you seem stuck, adjust camera and position.\n"
        "- When attack is triggered, attack will be performed enough times to break the block automatically.\n"
        "- If you attack and no reward is given, you were not close enough to the tree.\n"
        "- The tree block must also be in the centre of the frame. You might have to adjust the camera so this is true.\n"
        "- If you attacked as the last option and the log did not break you know something is not right.\n"
        "- You should never need to attack twice in a row.\n"
        "- Once wood is destroyed, it still needs to be picked up by walking over the fallen wood.\n"
        "- MAKE SURE YOU ARE IN FRONT OF A TREE BEFORE YOU ATTACK.\n"
        "\n"
        "Response format:\n"
        "You MUST respond with a JSON action dictionary on a single line.\n"
        "Valid keys: \"forward\", \"back\", \"left\", \"right\", \"jump\", \"attack\", \"camera\"\n"
        "Movement/attack values: 0 or 1\n"
        "Camera value: [pitch_delta, yaw_delta]\n"
        "\n"
        "Examples of valid responses:\n"
        '{"forward": 1}\n'
        '{"attack": 1}\n'
        '{"camera": [0, 15]}\n'
        '{"forward": 1, "camera": [0, 10]}\n'
        "\n"
        "Respond ONLY with the JSON action, no explanation needed.\n"
    )
)

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

    path = f'Agent/Results/{EMBEDDING_METHOD}.jsonl'

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
                raise ValueError(f"Unknown action name: {action}")
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
# Local LLM setup
# =============================================================================

def setup_local_agent():
    """Setup local Hugging Face model-based agent."""
    print(f"\nLoading local model: {LOCAL_MODEL_NAME}")
    print(f"Configured device: {LOCAL_MODEL_DEVICE}")
    sys.stdout.flush()

    # Determine the device
    if LOCAL_MODEL_DEVICE == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = LOCAL_MODEL_DEVICE

    print(f"Using device: {device}")

    # Determine dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(LOCAL_MODEL_DTYPE, torch.float16)
    if device == 'cpu':
        torch_dtype = torch.float32  # CPU typically needs float32

    print(f"Using dtype: {torch_dtype}")
    sys.stdout.flush()

    # Load tokenizer with progress indication
    print("\n[1/5] Loading tokenizer...")
    sys.stdout.flush()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_NAME,
            trust_remote_code=True,
            resume_download=True,
        )
        print("✓ Tokenizer loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.stdout.flush()
        raise

    # Set device_map for automatic device placement
    device_map = "auto" if device in ['cuda', 'mps'] else None

    # Check if this is a vision-language model by inspecting the config
    print("\n[2/5] Loading model configuration...")
    sys.stdout.flush()
    try:
        config = AutoConfig.from_pretrained(
            LOCAL_MODEL_NAME,
            trust_remote_code=True,
            resume_download=True,
        )
        config_class_name = config.__class__.__name__
        print(f"✓ Config loaded: {config_class_name}")
        sys.stdout.flush()

        # Vision-language models typically have "VL" in their config name
        # or are not supported by AutoModelForCausalLM
        is_vision_language = (
            'VL' in config_class_name or
            'Vision' in config_class_name or
            'Multimodal' in config_class_name
        )

        print("\n[3/5] Loading model weights (this may take several minutes)...")
        print("      If this is the first time, the model will be downloaded from HuggingFace.")
        print("      Large models can take 5-15 minutes to download depending on network speed.")
        print("      Progress bars should appear below if downloading...")
        sys.stdout.flush()

        if is_vision_language:
            print(f"      Detected vision-language model ({config_class_name}), using AutoModel")
            sys.stdout.flush()
            model = AutoModel.from_pretrained(
                LOCAL_MODEL_NAME,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_NAME,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
        print("✓ Model weights loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        # Fallback: try AutoModelForCausalLM first, then AutoModel if it fails
        print(f"✗ Config check failed ({e}), trying fallback loading methods...")
        print("\n[3/5] Attempting to load with AutoModelForCausalLM...")
        sys.stdout.flush()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_NAME,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded with AutoModelForCausalLM")
            sys.stdout.flush()
        except ValueError as ve:
            print(f"✗ AutoModelForCausalLM failed: {ve}")
            print("   Trying AutoModel...")
            sys.stdout.flush()
            model = AutoModel.from_pretrained(
                LOCAL_MODEL_NAME,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded with AutoModel")
            sys.stdout.flush()
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}")
            sys.stdout.flush()
            raise

    # Move to device if not using device_map
    if device_map is None:
        print(f"\n[4/5] Moving model to {device}...")
        sys.stdout.flush()
        model = model.to(device)
        print(f"✓ Model moved to {device}")
        sys.stdout.flush()

    # Create the text generation pipeline
    print("\n[5/5] Creating text generation pipeline...")
    sys.stdout.flush()
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=LOCAL_MODEL_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=LOCAL_MODEL_TEMPERATURE,
            top_p=0.9,
            return_full_text=False,
        )
        print("✓ Pipeline created successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        sys.stdout.flush()
        raise

    # Wrap in LangChain's HuggingFacePipeline
    print("\nWrapping in LangChain interface...")
    sys.stdout.flush()
    try:
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        chat_model = ChatHuggingFace(llm=hf_llm)
        print("✓ LangChain wrapper created successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to create LangChain wrapper: {e}")
        sys.stdout.flush()
        raise

    print(f"\n{'='*50}")
    print(f"Local model {LOCAL_MODEL_NAME} loaded successfully!")
    print(f"{'='*50}\n")
    sys.stdout.flush()
    return chat_model


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


def format_user_msg(context: str, memory_str: str | None = None):
    """
    Format user message for local models (text-only, no vision).
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
    user_msg = format_user_msg(context, memory_str=memory_str)

    # Invoke the model
    ai_msg = agent.invoke([system_msg, user_msg])

    action_executed = False

    # Try standard tool_calls first (may work with some HF models)
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call.get("name") == "step_env":
                raw_args = tool_call["args"]

                if "action" not in raw_args:
                    raw_args = {"action": raw_args}

                result = step_env.invoke(raw_args)
                obs = result["obs"]
                reward = result["reward"]
                done = result["done"]
                action_executed = True

    # Fallback: parse action from text output
    if not action_executed:
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
            # Default action: look around to explore
            print("Using default action: camera turn")
            result = step_env.invoke({"action": {"camera": [0, 15]}})
            obs = result["obs"]
            reward = result["reward"]
            done = result["done"]

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

    # Setup the local agent first (so we know model loading works)
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
