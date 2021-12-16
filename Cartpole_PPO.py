import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import torch.cuda

def initialize_weights_ppo(agent, init=torch.nn.init.xavier_uniform_):
    init(agent.policy.action_net.weight)
    init(agent.policy.value_net.weight)

    for i in agent.policy.mlp_extractor.policy_net:
        if type(i) == torch.nn.modules.linear.Linear:
            init(i.weight)

    for i in agent.policy.mlp_extractor.value_net:
        if type(i) == torch.nn.modules.linear.Linear:
            init(i.weight)
    return agent


def run_ppo(config_ppo, initialize=False):
    run = wandb.init(
        project="test_rl",
        config=config_ppo,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    def make_env_ppo():
        env = gym.make(config_ppo["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env

    env_ppo = DummyVecEnv([make_env_ppo])
    env_ppo = VecVideoRecorder(env_ppo, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0,
                               video_length=200)

    agent_ppo = PPO(policy=config_ppo["policy_type"],
                    env=env_ppo,
                    batch_size=config_ppo["batch_size"],
                    learning_rate=config_ppo["learning_rate"],
                    policy_kwargs=config_ppo["policy_kwargs"],
                    seed=config_ppo["seed"],
                    tensorboard_log=f"runs/{run.id}")

    agent_ppo.learn(
        total_timesteps=config_ppo["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()
config_ppo = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
    "batch_size":32,
    "policy_kwargs": dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "learning_rate":.0001,
    "seed":None
}

run_ppo(config_ppo)