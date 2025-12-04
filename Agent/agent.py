# from langchain_openai import ChatOpenAI
# from langchain.tools import tool
# from langchain.prompts import ChatPromptTemplate
# from langchain.agents import create_tool_calling_agent, AgentExecutor
import subprocess
import time
import requests
import base64
from PIL import Image
import io
import atexit

container_id = None

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

def create_env():
    global container_id

    # Start the docker container in detached mode and capture its ID
    proc = subprocess.run(
        ["docker", "run", "--rm", "-d", "-p", "5000:5000", "strangeman44/minerl-env-server"],
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
    resp = requests.get(url='http://127.0.0.1:5000/start')

    if resp.status_code != 200:
        raise RuntimeError(f'Status code error: {resp.status_code}')

    print('Received an image')

    obs_b64 = resp.json()['obs']
    obs_bytes = base64.b64decode(obs_b64)
    obs_small = Image.open(io.BytesIO(obs_bytes))
    obs = obs_small.resize((256, 256), Image.NEAREST)
    obs.show()

create_env()

# # Setting the LLM
# llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# # -----------------------------
# # 2) Tool (environment step)
# # -----------------------------
# @tool
# def step_env(action: str) -> str:
#     """Take one action in the environment and return the observation."""
#     # placeholder
#     return f"new_observation_after_{action}"

# tools = [step_env]

# # -----------------------------
# # 3) Prompt Template
# # -----------------------------
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You decide the next action. Then call the tool."),
#     ("human",
#      "Current observation:\n{observation}\n\n"
#      "Extra context:\n{context}\n\n"
#      "Goal: {goal}")
# ])

# # -----------------------------
# # 4) Agent
# # -----------------------------
# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=prompt
# )

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True
# )

# # -----------------------------
# # 5) Loop
# # -----------------------------
# def run_episode():
#     observation = "initial_obs"

#     for _ in range(5):
#         # You would query your DB here:
#         context = "retrieved_context"

#         result = agent_executor.invoke({
#             "observation": observation,
#             "context": context,
#             "goal": "Do task X"
#         })

#         # The tool result appears in intermediate_steps
#         last_tool_output = result["intermediate_steps"][-1][1]
#         observation = last_tool_output  # update observation

#         print("New observation:", observation)

# run_episode()