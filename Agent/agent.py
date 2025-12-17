from langchain_openai import ChatOpenAI
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
from collections import deque
import matplotlib.pyplot as plt
# import keyboard
from datetime import datetime, timezone
from dotenv import load_dotenv
from RAG_server import RAG
from tqdm import tqdm 

# Load environment variables from .env file
load_dotenv()

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    openai_key = config.get('OPENAI_API_KEY')
except Exception:
    openai_key = None

if not openai_key:
    openai_key = os.environ.get("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = openai_key

with open('test_setup.json', 'r') as f:
    test_setup = json.load(f)

with open('example_images.json', 'r') as f:
    example_images = json.load(f)

# Remote server configuration (from .env)
USE_REMOTE_SERVER = os.environ.get("USE_REMOTE_SERVER", "false").lower() == "true"
SERVER_URL = os.environ.get("MINERL_SERVER_URL", "http://127.0.0.1:5001")
ENV_ID = None  # Used to identify environment session on remote server

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

# Valid keyboard keys and action mapping
VALID_KEYS = set(['w', 'a', 's', 'd', 'space', 'i', 'j', 'k', 'l', 'shift', 'esc'])
ACTION_MAP = {
    'w': ("forward", 1),
    'a': ("left", 1),
    's': ("back", 1),
    'd': ("right", 1),
    'space': ("jump", 1),
    'i': ("camera", (-15, 0)),
    'k': ("camera", (15, 0)),
    'j': ("camera", (0, -15)),
    'l': ("camera", (0, 15)),
    'shift': ("attack", 1)
}

# For rendering the frames
_RENDER = {
    "fig": None,
    "ax": None,
    "im": None,
}

# Setting the rag system
if test_setup['use_rag']:
    rag = RAG()

example_image_1 = "data:image/png;base64," + example_images["example_1"]
example_image_2 = "data:image/png;base64," + example_images["example_2"]
example_image_3 = "data:image/png;base64," + example_images["example_3"]
example_image_4 = "data:image/png;base64," + example_images["example_4"]

example_action_1 = {"forward": 1}
example_action_2 = {"attack": 1}
example_action_3 = {"camera", (0, -30)}
example_action_4 = {"camera", (30, 0)}

explanation_1 = "The agent saw the tree (object in white) in the distance and started moving towards it."
explanation_2 = "The agent is now at the tree. It continuously attacks the tree until the block is gone. Once the block is broken the reward is given."
explanation_3 = "The agent can only see dirt. There is nothing useful here and it should not be attacked. The agent should look around to find trees."
explanation_4 = "The agent just destroyed wood. To collect more the viewpoint needs to be changed. The agent can look up or down."

system_msg = SystemMessage(
    content=(
        f"You are a Minecraft-playing agent in a 3D world.\n"
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
        "- When attack is triggered, attack will be performed enough times to break the block automatically."
        "- If you attack and no reward is given, you were not close enough to the tree."
        "- The tree block must also be in the centre of the frame. You might have to adjust the camera so this is true."
        "- If you attacked as the last option and the log did not break you know something is not right."
        "- You should never need to attack twice in a row."
        "- Once wood is destroyed, it still needs to be picked up by walking over the fallen wood."
        "- Make sure to jump if you are stuck!!!"
        "\n"
        "Tool usage:\n"
        "- On each turn, you MUST respond ONLY by calling the `step_env` tool.\n"
        "- The tool takes a single argument: a dict named `action`.\n"
        "  * Keys: \"forward\", \"back\", \"left\", \"right\", \"jump\", \"attack\", \"camera\".\n"
        "  * Movement/attack: 0 or 1.\n"
        "  * Camera: [pitch_delta, yaw_delta].\n"
        "\n"
        "Training examples:\n"
        "\n"
        "EXAMPLE 1:\n"
        f"Input image: [{example_image_1}]\n"
        f"Taken action: {example_action_1}\n"
        f"The reason for the action: {explanation_1}\n"
        "\n"
        "EXAMPLE 2:\n"
        f"Input image: [{example_image_2}]\n"
        f"Taken action: {example_action_2}\n"
        f"The reason for the action: {explanation_2}\n"
        "\n"
        "EXAMPLE 3:\n"
        f"Input image: [{example_image_3}]\n"
        f"Taken action: {example_action_3}\n"
        f"The reason for the action: {explanation_3}\n"
        "\n"
        "EXAMPLE 4:\n"
        f"Input image: [{example_image_4}]\n"
        f"Taken action: {example_action_4}\n"
        f"The reason for the action: {explanation_4}\n"
    )
)

container_started = False

# Saves the progress
def log_result(wood):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wood_collected": wood
    }

    path = 'Agent/Results/' + test_setup['embedding_method'] + '.jsonl'

    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

# Convert to image from base64
def process_image(obs_b64):
    obs_bytes = base64.b64decode(obs_b64)
    obs = Image.open(io.BytesIO(obs_bytes))

    return obs

def init_renderer(first_obs_b64: str):
    """Create the matplotlib window ONCE."""
    frame = process_image(first_obs_b64)

    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)  # bigger window
    ax.set_axis_off()

    im = ax.imshow(frame, interpolation="nearest")  # crisp pixels
    fig.tight_layout(pad=0)

    _RENDER["fig"] = fig
    _RENDER["ax"] = ax
    _RENDER["im"] = im

    # show without blocking
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
    global container_started, ENV_ID

    if USE_REMOTE_SERVER:
        # Clean up remote environment session
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
        # Clean up local Docker container
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

# Starts the environment (local Docker or remote server)
def create_env():
    global container_started, ENV_ID

    if USE_REMOTE_SERVER:
        # Remote mode: call the multi-env server's /create endpoint
        print(f"Connecting to remote MineRL server at {SERVER_URL}...")

        resp = requests.post(
            f"{SERVER_URL}/create",
            json={"env_type": "MineRLTreechop-v0"}
        )

        if resp.status_code != 200:
            raise RuntimeError(f'Failed to create remote environment: {resp.status_code} - {resp.text}')

        data = resp.json()
        ENV_ID = data['env_id']
        print(f"Created remote environment session: {ENV_ID}")

        time.sleep(20) # wait for the environment to start

        return data['obs']
    else:
        # Local mode: start Docker container
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

# Resets the environment for different runs
def reset_env():
    if USE_REMOTE_SERVER:
        # Remote mode: include env_id in request
        resp = requests.post(
            f"{SERVER_URL}/reset",
            json={"env_id": ENV_ID}
        )
    else:
        # Local mode: simple reset
        resp = requests.post(url=f'{SERVER_URL}/reset')

    if resp.status_code != 200:
        raise RuntimeError(f'Status code error: {resp.status_code}')

    return resp.json()['obs']

# Submits the actions to the server
def submit_action(
    action_dict: dict,
):
    # Build the full MineRL action template
    action_body = MINERL_ACTION_TEMPLATE.copy()

    global CAMERA_STATE

    for action, value in action_dict.items():
        if action == "camera":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("Camera action requires value=[pitch_delta, yaw_delta]")

            pitch_delta, yaw_delta = float(value[0]), float(value[1])

            # Update our estimate
            new_pitch = CAMERA_STATE["pitch"] + pitch_delta
            new_pitch = max(min(new_pitch, PITCH_MAX), PITCH_MIN)
            CAMERA_STATE["pitch"] = new_pitch
            CAMERA_STATE["yaw"] += yaw_delta

            # Use the (possibly clamped) delta actually sent to env
            # Here we recompute delta so we never exceed bounds
            actual_pitch_delta = new_pitch - (CAMERA_STATE["pitch"] - pitch_delta)
            action_body["camera"] = [actual_pitch_delta, yaw_delta]
        else:
            if action not in action_body:
                raise ValueError(f"Unknown action name: {action}")
            action_body[action] = int(value)

    # Build request body (include env_id for remote server)
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

# The tool that the LLM can use
@tool
def step_env(
    action: dict,     # SINGLE DICT ONLY
) -> dict:
    """Step the MineRL environment by executing a *combined action* in a single environment step.

    This tool takes **one argument only**: a dictionary called `action`.

    The `action` dictionary specifies which controls to activate during this step.
    Each key is the name of a control, and each value specifies how that control
    should be applied.

    Valid action keys:
        - "forward", "back", "left", "right", "jump", "attack"
            → Use 1 to press the control, or 0 to leave it inactive.
        - "camera"
            → Use a two-element list: [pitch_delta, yaw_delta]
            * pitch_delta < 0 looks up;  > 0 looks down
            * yaw_delta   < 0 looks left; > 0 looks right

    Examples:
        step_env(action={"forward": 1})
        step_env(action={"jump": 1, "attack": 1})
        step_env(action={"camera": [0.0, 15.0]})
        step_env(action={"forward": 1, "camera": [0.0, 15.0]})

    All specified controls are applied simultaneously within a single MineRL step.
    Do not include multiple entries for the same key—use only one unified dict.
    You do not have to include actions in the dictonary that do not want to perform.
    """

    obs, reward, done, action_body = submit_action(action)

    print(action)

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

# This will init the agent and return the tool caller
def setup_agent():
    # Setting the LLM
    llm = ChatOpenAI(
        model="gpt-5-nano-2025-08-07",
        temperature=0.2
    )
    llm_with_tools = llm.bind_tools([step_env])

    return llm_with_tools

# Format the previous actions list
def format_rev_actions():
    if ACTION_HISTORY:
        history_lines = [
            f"{i+1}. action={entry['action']}, reward={entry['reward']}"
            for i, entry in enumerate(list(ACTION_HISTORY))
        ]
        context = "Last actions taken:\n" + "\n".join(history_lines)
    else:
        context = "No previous actions yet."

    return context

def format_user_msg(obs, context, memory_str: str | None = None):
    prev_actions = format_rev_actions()
    goal = 'Goal: Collect as much wood as possible.'

    memory_block = memory_str or "No relevant episodic memory was retrieved for this state."

    camera_info = (
        f"Estimated camera pitch: {CAMERA_STATE['pitch']:.1f} degrees "
        "(0 ≈ looking at the horizon; positive = down, negative = up).\n"
        "Try to keep pitch between about -20 and +20 degrees unless you are "
        "deliberately looking up or down briefly.\n"
    )

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

    return user_msg

# Runs the agent loop calling the tools
def run_agent_episode(agent, obs):
    reward = 0
    done = False

    # Querying the memory
    memory_str = "No episodic memory yet (not enough frames)."
    if test_setup['use_rag'] and len(FRAME_HISTORY) == 16:
        memory_str = rag.get_action(FRAME_HISTORY)

    # Write the query
    context = ""

    user_msg = format_user_msg(obs, context, memory_str=memory_str)
    ai_msg = agent.invoke([system_msg, user_msg])

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call["name"] == "step_env":
                raw_args = tool_call["args"]

                if "action" not in raw_args:
                    raw_args = {"action": raw_args}

                request_complete = False
                
                while not request_complete:
                    try:
                        result = step_env.invoke(raw_args)
                    except Exception as e:
                        print('Hit rate limit. Retrying.')
                        time.sleep(3)
                    else:
                        request_complete = True

                obs = result["obs"]
                reward = result["reward"]
                done = result["done"]

    FRAME_HISTORY.append(obs)

    return obs, reward, done

# Runs the human controlled episode
def run_human_episode():
    # keyboard_input = keyboard.read_key()
    keyboard_input = input("Enter a key: ")

    if keyboard_input in VALID_KEYS:
        if keyboard_input == 'esc':
            return

        action, value = ACTION_MAP[keyboard_input]

        print('Action: {action}, value: {value}')

        obs, reward, done, action_body = submit_action({action: value})

        time.sleep(0.01)

        return obs, reward, done

    return original_obs, 0, False

def run_agent():
    max_frames = test_setup["max_frames"]

    # Start the env
    obs = create_env()

    # Start rendering
    if test_setup['render']:
        init_renderer(obs)

    if test_setup['agent_mode']:
        agent = setup_agent()

    done = False
    reward = 0
    wood_count = 0
    cur_frames = 0

    for _ in range(test_setup['test_runs']):
        # Start progress
        if test_setup['use_max_frames']:
            pbar = tqdm(total=max_frames)

        while not done:
            # What mode to run in
            if test_setup['agent_mode']:
                obs, reward, done = run_agent_episode(agent, obs)
            else:
                obs, reward, done = run_human_episode(obs)

            if reward != 0:
                wood_count += 1
                print(f'Wood count: {wood_count}')

            # Render this frame
            if test_setup['render']:
                show_obs(obs)

            if test_setup['use_max_frames']:
                cur_frames += 1
                
                tqdm.write(f"Step {cur_frames} done")
                pbar.update(1)

                # the model isn't doing anything
                if cur_frames >= max_frames:
                    cur_frames = 0
                    pbar.close()
                    break

        # Saving the results of that run
        log_result(wood_count)

        # Resetting the env and tracking
        wood_count = 0
        ACTION_HISTORY.clear()
        FRAME_HISTORY.clear()
        obs = reset_env()


run_agent()
