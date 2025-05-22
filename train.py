from stable_baselines3.common.env_checker import check_env
from cartpole import CartPoleSimulation
from env import Env
from model import create_model


def main():
    sim = CartPoleSimulation()
    env = Env(sim)

    check_env(env, warn=True)

    model = create_model(env)
    try:
        model.learn(total_timesteps=10_000_000)
    except KeyboardInterrupt:
        if input("Save model? (y/n): ").lower() == "y":
            model.save("models/cartpole")
        return
    model.save("models/cartpole")


if __name__ == "__main__":
    main()
