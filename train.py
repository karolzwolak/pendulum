from model import make_parallel_normalized_envs, create_model, save_model


def main():
    # Use parallel environments
    env = make_parallel_normalized_envs()

    model = create_model(env)

    try:
        model.learn(total_timesteps=1_000_000, progress_bar=True)
    except KeyboardInterrupt:
        if input("Save model? (y/n): ").lower() == "y":
            save_model(model, env)
        return
    save_model(model, env)


if __name__ == "__main__":
    main()
