from model import (
    create_model,
    make_parallel_normalized_envs,
    save_model,
    CurriculumLearningCallback,
)


def main():
    # Use parallel environments
    env = make_parallel_normalized_envs()

    model = create_model(env)

    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=CurriculumLearningCallback(),
        )
    except KeyboardInterrupt:
        if input("Save model? (y/n): ").lower() == "y":
            save_model(model, env)
        return
    save_model(model, env)


if __name__ == "__main__":
    main()
