import retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 10000,
    "env_name": "SonicTheHedgehog-Genesis",
    "state": "GreenHillZone.Act1"
}
run = wandb.init(
    project="test_rl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

def make_env():
    env = retro.make(config["env_name"], state=config["state"])
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 10000 == 0, video_length=200)


modelname = "sonic_ac2"
model = A2C("CnnPolicy", env, verbose=1)
model.set_env(env)
model = A2C(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()