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
from typing import Literal
import json
import os
from collections import deque
import matplotlib.pyplot as plt
import keyboard

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    openai_key = config.get('openai_token')
except Exception:
    openai_key = None

if not openai_key:
    openai_key = os.environ.get("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = openai_key

with open('test_setup.json', 'r') as f:
    test_setup = json.load(f)

ACTION_HISTORY = deque(maxlen=5)

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

system_msg = SystemMessage(
    content=(
        "You are a Minecraft playing agent in a 3D world.\n"
        "- You can move with: forward, back, left, right, jump.\n"
        "- You can look around with: camera (pitch, yaw).\n"
        "- You can break blocks (like tree trunks) with: attack.\n\n"
        "Your goal is to collect as much wood as possible. To do this, you must:\n"
        "1) Use camera to look around until you can see a tree.\n"
        "2) Use forward/left/right/back/jump to walk up to the tree.\n"
        "3) When close enough and looking at the trunk, use attack repeatedly.\n\n"
        "Very important:\n"
        "- Do NOT only use forward and attack. Use camera to turn and explore.\n"
        "- Use left/right/back/jump to navigate around obstacles or adjust position.\n"
        "- Prefer using camera at the start of an episode or when you see no tree.\n"
        "On each turn, you MUST respond ONLY by calling the `step_env` tool."
    )
)

container_id = None

# Convert to image from base64
def process_image(obs_b64):
    obs_bytes = base64.b64decode(obs_b64)
    obs = Image.open(io.BytesIO(obs_bytes))

    return obs

# Renders the observation
def show_obs(obs_b64):
    obs = process_image(obs_b64)
    plt.clf()
    plt.imshow(obs)
    plt.axis("off")
    plt.pause(0.001)  # tiny pause to update the window

def exit_cleanup():
    global container_id
    if container_id:
        try:
            # Stop just our container
            subprocess.run(
                ["docker", "stop", container_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            print(f"Stopping docker container {container_id}")
        except Exception as e:
            print(f"Failed to stop container {container_id}: {e}")

atexit.register(exit_cleanup)

# Starts the docker container and gets first observation
def create_env():
    global container_id

    # Start the docker container in detached mode and capture its ID
    proc = subprocess.run(
        ["docker", "run", "--rm", "-d", "-p", "5001:5000", "strangeman44/minerl-env-server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    container_id = proc.stdout.strip()
    print(f"Started environment server (container id: {container_id}). Waiting for setup.")
    time.sleep(20)

    print('Setting up environment...')
    print('LONG WAIT')
    resp = requests.get(url='http://127.0.0.1:5001/start')

    if resp.status_code != 200:
        raise RuntimeError(f'Status code error: {resp.status_code}')

    #obs = process_image(resp.json()['obs'])

    return resp.json()['obs']

# Actually making the action request
def submit_action(action: Literal["attack", "back", "forward", "left", "right", "jump", "camera"],
            value=1):
    
    # Copy default template
    action_body = MINERL_ACTION_TEMPLATE.copy()

    # Handle special case for camera movement
    if action == "camera":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Camera action requires value=[pitch_delta, yaw_delta]")
        action_body["camera"] = value
    else:
        # Normal discrete MineRL action
        if action not in action_body:
            raise ValueError(f"Unknown action name: {action}")
        action_body[action] = value

    body = {"actions": action_body}

    # Send request to env server
    resp = requests.post(
        url="http://127.0.0.1:5001/action",
        json=body
    )

    if resp.status_code != 200:
        raise RuntimeError(f'Status code error: {resp.status_code}')

    # Extracting the information
    resp = resp.json()
    obs = resp['obs'] #process_image(resp['obs'])
    reward = resp['reward']
    done = resp['done']

    return obs, reward, done, action_body

# The tool that the LLM can use
@tool
def step_env(action: Literal["attack", "back", "forward", "left", "right", "jump", "camera"],
            value=1) -> tuple[str, int, bool]:
    """Step the MineRL environment by taking a single action.

    Actions and when to use them:
    - "forward": walk straight ahead. Use this to move toward a tree you are facing.
    - "back": step backward. Use this to back away from walls or trees.
    - "left": strafe left (without turning). Use this to sidestep or circle a tree.
    - "right": strafe right (without turning). Use this to sidestep or circle a tree.
    - "jump": jump up. Use this to climb small blocks or jump while moving forward.
    - "camera": rotate your view. value must be [pitch_delta, yaw_delta].
        * pitch_delta < 0 looks up, > 0 looks down.
        * yaw_delta < 0 turns left, > 0 turns right.
        Use camera to look around for trees.
    - "attack": swing your tool to break blocks. Use this when close to the tree trunk
      and looking directly at it.

    Strategy for collecting wood:
    1) If you do NOT see a tree, call camera with a moderate yaw change like [0.0, 15.0]
       or [0.0, -15.0] to turn and search.
    2) When a tree is in view but far away, move with forward/left/right to approach it.
    3) If you get stuck, use back or jump to reposition.
    4) When the tree trunk is close, call attack several times in a row.

    - value: for discrete actions, 0 or 1; for "camera", [pitch_delta, yaw_delta].
    The tool sends the full MineRL action dict and returns the new observation JSON.
    """

    obs, reward, done, action_body = submit_action(action, value)

    print(action_body)

    ACTION_HISTORY.append({
        "action": action,
        "value": value,
        "action_body": action_body,
        "reward": reward
    })

    return obs, reward, done

# This will init the agent and return the tool caller
def setup_agent():
    # Setting the LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1)
    llm_with_tools = llm.bind_tools([step_env])

    return llm_with_tools

# Format the previous actions list
def format_rev_actions():
    if ACTION_HISTORY:
        history_lines = [
            f"{i+1}. action={entry['action']}, value={entry['value']}, reward={entry['reward']}"
            for i, entry in enumerate(list(ACTION_HISTORY))
        ]
        context = "Last actions taken:\n" + "\n".join(history_lines)
    else:
        context = "No previous actions yet."

    return context

def format_user_msg(obs, context):
    prev_actions = format_rev_actions()
    goal = 'Collect as much wood as possible. To do this go up to a tree and chop it.'

    user_msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You see the current Minecraft frame below.\n\n"
                        f"Extra context:\n{context}\n\n"
                        f"Previous actions\n{prev_actions}\n\n"
                        f"Goal: {goal}\n\n"
                        "Remember: You only get reward when you collect WOOD LOGS from trees. "
                        "If there is no reward, you might still be hitting wood. "
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
def run_agent_episode():
    # Start the env
    obs = create_env()
    
    # Start rendering
    if test_setup['render']:
        show_obs(obs)
        time.sleep(5)

    agent = setup_agent()
    done = False
    reward = 0

    while not done:
        # Write the query
        context = ""

        user_msg = format_user_msg(obs, context)
        ai_msg = agent.invoke([system_msg, user_msg])

        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                if tool_call["name"] == "step_env":
                    args = tool_call["args"]
                    obs, reward, done = step_env.invoke(args)

        if reward != 0:
            print(f'Got wood! {reward}')

        # Render this frame
        if test_setup['render']:
            show_obs(obs)

# Runs the human controlled episode
def run_human_episode():
    # Start the env
    obs = create_env()
    
    # Start rendering
    show_obs(obs)
    time.sleep(5)

    done = False
    reward = 0

    while not done:
        keyboard_input = keyboard.read_key()

        if keyboard_input in VALID_KEYS:
            if keyboard_input == 'esc':
                return
            
            action, value = ACTION_MAP[keyboard_input]

            obs, reward, done, action_body = submit_action(action, value)

            if reward != 0:
                print(f'Got wood! {reward}')

            show_obs(obs)
        
        time.sleep(0.01)

# What mode to run in
if test_setup['agent_mode']:
    run_agent_episode()
else:
    run_human_episode()
