from stable_baselines3.common.env_checker import check_env
from cartpole import CartPoleSimulation
from env import Env
from model import create_model

if __name__ == "__main__":
    sim = CartPoleSimulation()
    env = Env(sim)

    check_env(env, warn=True)

    model = create_model(env)
    model.learn(total_timesteps=100_000)

    model.save("models/cartpole")
