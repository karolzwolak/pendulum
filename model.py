from stable_baselines3 import PPO


def create_model(env):
    return PPO("MlpPolicy", env, verbose=1)


def load_model(model_path, env):
    return PPO.load(model_path, env)
