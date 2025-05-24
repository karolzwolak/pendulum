from stable_baselines3.common.env_checker import check_env
from cartpole import CartPoleSimulation
from env import Env
from model import create_model, make_parallel_envs


def main():
    # Use parallel environments
    env = make_parallel_envs(num_envs=4)

    # Check one of the environments
    check_env(Env(CartPoleSimulation()), warn=True)

    model = create_model(env)
    try:
        model.learn(total_timesteps=10_000_000, progress_bar=True)
    except KeyboardInterrupt:
        if input("Save model? (y/n): ").lower() == "y":
            model.save("models/cartpole")
        return
    model.save("models/cartpole")


if __name__ == "__main__":
    main()
