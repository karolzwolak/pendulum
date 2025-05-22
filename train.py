from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from cartpole import CartPoleSimulation
from model import Env

if __name__ == "__main__":
    sim = CartPoleSimulation()
    env = Env(sim)

    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    model.save("models/cartpole")
