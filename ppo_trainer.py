# ppo_trainer.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_rl_agent(env, timesteps=5000, save_path="ppo_sparse_model"):
    """
    使用 MlpPolicy 训练 RL 智能体（适用于小图像输入）
    """
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy="MlpPolicy",           # ✅ 使用 MLP 处理展平后的图像
        env=vec_env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        batch_size=64,
        device="auto"  # 自动使用 cpu/cuda
    )

    print(f"🚀 开始训练 RL 智能体，共 {timesteps} 步...")
    model.learn(total_timesteps=timesteps, tb_log_name="ppo_run")
    model.save(save_path)
    print(f"💾 模型已保存至: {save_path}.zip")

    return model
