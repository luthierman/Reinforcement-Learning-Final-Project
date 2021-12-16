import gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

import torch.cuda

# helper function for setting weight initialization
def initialize_weights_dqn(agent, init=torch.nn.init.xavier_uniform_):
  for i in agent.policy.q_net.q_net:
    if type(i)==torch.nn.modules.linear.Linear:
      init(i.weight)
  agent.policy.q_net_target.q_net.load_state_dict(agent.policy.q_net.q_net.state_dict())
  return agent

# main function to run
def run_dqn(config_dqn, project_name,initialize=False):
  run = wandb.init(
    project=project_name,
    config=config_dqn,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
  )
  def make_env_dqn():
    env = gym.make(config_dqn["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env
  env_dqn = DummyVecEnv([make_env_dqn])
  env_dqn = VecVideoRecorder(env_dqn, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

  agent_dqn = DQN(policy=config_dqn["policy_type"],
                    env=env_dqn,
                    batch_size=config_dqn["batch_size"],
                    learning_rate=config_dqn["learning_rate"],
                    policy_kwargs=config_dqn["policy_kwargs"],
                    seed=config_dqn["seed"],
                    tensorboard_log=f"runs/{run.id}")
  if initialize:
    agent_dqn = initialize_weights_dqn(agent_dqn)
  agent_dqn.learn(
      total_timesteps=config_dqn["total_timesteps"],
      callback=WandbCallback(
          gradient_save_freq=100,
          model_save_path=f"models/{run.id}",
          verbose=2,
      ),
  )
  run.finish()

# edit here !
config_dqn1 = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
    "batch_size":32,
    "policy_kwargs": dict(activation_fn=torch.nn.ReLU, net_arch=[128, 128]),
    "learning_rate":.0001,
    "seed": None
}
run_dqn(config_dqn1, "test_rl")
