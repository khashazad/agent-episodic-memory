import gym
import minerl
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
env = None

def encode_frame(frame: np.ndarray) -> str:
    """Convert MineRL pov (H, W, 3) ndarray -> base64 PNG string."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def obs_to_b64(obs):
    # pov image
    frame = obs["pov"]                # ndarray (H, W, 3)
    frame_b64 = encode_frame(frame)

    return frame_b64

@app.route('/start', methods=["GET"])
def create_env():
    global env
    data = request.get_json(silent=True)

    if data is None or data.get('env') is None:
        env = gym.make('MineRLTreechop-v0')
    else:
        env = gym.make(data.get('env'))

    obs = env.reset()

    return jsonify({'obs': obs_to_b64(obs)})

@app.route('/action', methods=["POST"])
def do_action():
    global env
    data = request.get_json()

    # Take an action
    # data['actions'] will be a dict of the actions and their values
    action = env.action_space.noop()
    action.update(data['actions'])

    obs, reward, done, _ = env.step(action)
    
    return jsonify({'obs': obs_to_b64(obs), 'reward': reward, 'done': done})

# This resets the environments
@app.route('/reset', methods=["POST"])
def reset_env():
    global env
    obs = env.reset()

    return jsonify({'obs': obs_to_b64(obs)})

# Start the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)