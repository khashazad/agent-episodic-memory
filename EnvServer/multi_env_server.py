"""
Multi-environment MineRL server that supports multiple concurrent environments.
Each environment is identified by a unique session ID (env_id).
Includes automatic cleanup of inactive environments.
"""

import gym
import minerl  # This registers MineRL environments with gym
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import io
import base64
import uuid
import threading
import time
import os
from datetime import datetime

app = Flask(__name__)

# Configuration (can be set via environment variables)
INACTIVITY_TIMEOUT = int(os.environ.get("INACTIVITY_TIMEOUT", "300"))  # 5 minutes default
CLEANUP_INTERVAL = int(os.environ.get("CLEANUP_INTERVAL", "60"))  # 60 seconds default

# Thread-safe storage for environments
environments = {}  # {env_id: {"env": gym_env, "last_activity": timestamp}}
env_lock = threading.Lock()


def encode_frame(frame: np.ndarray) -> str:
    """Convert MineRL pov (H, W, 3) ndarray -> base64 PNG string."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def obs_to_b64(obs):
    """Convert observation to base64 encoded image."""
    frame = obs["pov"]  # ndarray (H, W, 3)
    return encode_frame(frame)


def update_activity(env_id: str):
    """Update the last activity timestamp for an environment."""
    with env_lock:
        if env_id in environments:
            environments[env_id]["last_activity"] = time.time()


def cleanup_inactive_environments():
    """Background thread that removes inactive environments."""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        current_time = time.time()
        to_remove = []

        with env_lock:
            for env_id, env_data in environments.items():
                if current_time - env_data["last_activity"] > INACTIVITY_TIMEOUT:
                    to_remove.append(env_id)

            for env_id in to_remove:
                try:
                    environments[env_id]["env"].close()
                except Exception as e:
                    print(f"Error closing environment {env_id}: {e}")
                del environments[env_id]
                print(f"Cleaned up inactive environment: {env_id}")


# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_inactive_environments, daemon=True)
cleanup_thread.start()


@app.route('/create', methods=["POST"])
def create_env():
    """Create a new MineRL environment and return its unique ID."""
    data = request.get_json(silent=True) or {}
    env_type = data.get('env_type', 'MineRLTreechop-v0')

    try:
        env = gym.make(env_type)
        obs = env.reset()

        env_id = str(uuid.uuid4())

        with env_lock:
            environments[env_id] = {
                "env": env,
                "last_activity": time.time(),
                "env_type": env_type,
                "created_at": datetime.utcnow().isoformat()
            }

        print(f"Created environment {env_id} (type: {env_type})")

        return jsonify({
            'env_id': env_id,
            'obs': obs_to_b64(obs)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/action', methods=["POST"])
def do_action():
    """Execute an action in the specified environment."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    env_id = data.get('env_id')
    if not env_id:
        return jsonify({'error': 'env_id is required'}), 400

    with env_lock:
        if env_id not in environments:
            return jsonify({'error': f'Environment {env_id} not found'}), 404
        env = environments[env_id]["env"]

    update_activity(env_id)

    try:
        # Base action from client
        base_action = env.action_space.noop()
        base_action.update(data.get('actions', {}))

        # First step
        obs, reward, done, info = env.step(base_action)
        total_reward = reward
        last_obs = obs
        last_done = done

        # If attack pressed, repeat a few more attack-only steps
        if base_action.get('attack') == 1:
            attack_action = env.action_space.noop()
            attack_action['attack'] = 1

            sustained_attack_count = data.get('sustained_attack_count', 15)

            for _ in range(sustained_attack_count):  # repeat count
                if last_done:
                    break

                obs, reward, done, info = env.step(attack_action)
                total_reward += reward
                last_obs = obs
                last_done = done

        return jsonify({
            'obs': obs_to_b64(last_obs),
            'reward': float(total_reward),
            'done': bool(last_done),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=["POST"])
def reset_env():
    """Reset the specified environment."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    env_id = data.get('env_id')
    if not env_id:
        return jsonify({'error': 'env_id is required'}), 400

    with env_lock:
        if env_id not in environments:
            return jsonify({'error': f'Environment {env_id} not found'}), 404
        env = environments[env_id]["env"]

    update_activity(env_id)

    try:
        obs = env.reset()
        return jsonify({'obs': obs_to_b64(obs)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/destroy', methods=["POST"])
def destroy_env():
    """Explicitly destroy an environment and free its resources."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    env_id = data.get('env_id')
    if not env_id:
        return jsonify({'error': 'env_id is required'}), 400

    with env_lock:
        if env_id not in environments:
            return jsonify({'error': f'Environment {env_id} not found'}), 404

        try:
            environments[env_id]["env"].close()
        except Exception as e:
            print(f"Error closing environment {env_id}: {e}")

        del environments[env_id]

    print(f"Destroyed environment: {env_id}")
    return jsonify({'success': True})


@app.route('/health', methods=["GET"])
def health_check():
    """Return server health status and number of active environments."""
    with env_lock:
        active_envs = len(environments)
        env_details = [
            {
                "env_id": env_id,
                "env_type": data.get("env_type", "unknown"),
                "created_at": data.get("created_at", "unknown"),
                "idle_seconds": int(time.time() - data["last_activity"])
            }
            for env_id, data in environments.items()
        ]

    return jsonify({
        'status': 'healthy',
        'active_envs': active_envs,
        'environments': env_details,
        'inactivity_timeout': INACTIVITY_TIMEOUT
    })


# Start the server
if __name__ == "__main__":
    print(f"Starting Multi-Environment MineRL Server")
    print(f"Inactivity timeout: {INACTIVITY_TIMEOUT} seconds")
    print(f"Cleanup interval: {CLEANUP_INTERVAL} seconds")

    # Use gunicorn if available (production), otherwise fall back to Flask dev server
    try:
        from gunicorn.app.base import BaseApplication

        class GunicornApp(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            'bind': '0.0.0.0:5000',
            'workers': 1,
            'threads': 4,
            'timeout': 180,  # 3 minutes for environment creation
            'worker_class': 'gthread',
        }
        print("Starting with Gunicorn (timeout: 180s)")
        GunicornApp(app, options).run()
    except ImportError:
        print("Gunicorn not available, using Flask dev server (may timeout on long requests)")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
