import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import torch.cuda

from stable_baselines3 import A2C


def run_a2c(config_a2c):
    run = wandb.init(
        project="test_rl",  # change as needed
        config=config_a2c,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    def make_env_a2c():
        env = gym.make(config_a2c["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env

    env_a2c = DummyVecEnv([make_env_a2c] * config_a2c["n_env"])
    env_a2c = VecVideoRecorder(env_a2c, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0,
                               video_length=200)
    agent_a2c = A2C(policy=config_a2c["policy_type"],
                    env=env_a2c,
                    policy_kwargs=config_a2c["policy_kwargs"],
                    seed=config_a2c["seed"],
                    tensorboard_log=f"runs/{run.id}")

    agent_a2c.learn(
        total_timesteps=config_a2c["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


config_a2c = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
    "n_env": 1,
    "policy_kwargs": dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "seed":None
}
run_a2c(config_a2c)