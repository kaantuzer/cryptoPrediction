# core/rl_utils.py
# --------------------------------------------------------------------------- #
#  RL eğitim / test yardımcıları                                              #
# --------------------------------------------------------------------------- #
import random
from pathlib import Path
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core.data_utils import fetch_klines_and_compute_indicators
from core.spot_env   import SpotTradingEnv

# -------------------------- sabitler --------------------------------------- #
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# -------------------------- eğitim callback’i ------------------------------ #
class EpisodeCountCallback(BaseCallback):
    def __init__(self, max_episodes: int = 70, verbose: int = 1):
        super().__init__(verbose); self.max_episodes = max_episodes
    def _on_step(self) -> bool:
        if self.training_env.get_attr("reset_count")[0] >= self.max_episodes:
            if self.verbose:
                print(f"⏹️  Episode limiti ({self.max_episodes}) doldu – eğitim duruyor.")
            return False
        return True

# -------------------------- model eğitimi ---------------------------------- #
def train_model(symbol: str, interval: str, limit: int = 1000):
    df = fetch_klines_and_compute_indicators(symbol, interval, limit)
    train_df = df.iloc[: int(len(df) * 0.85)].copy()

    env     = SpotTradingEnv(train_df, fee=0.001, gradient_window=5, gradient_threshold=0.1)
    vec_env = VecNormalize(
        DummyVecEnv([lambda: env]),
        norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    model = PPO(
        "MlpPolicy", vec_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1, n_steps=4096, batch_size=256,
        gamma=0.98, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.25,
        learning_rate=1.5e-4, seed=SEED
    )

    out_dir = Path("artifacts") / symbol / interval
    out_dir.mkdir(parents=True, exist_ok=True)

    model.learn(
        total_timesteps=2_000_000,
        callback=[
            CheckpointCallback(save_freq=500_000, save_path=out_dir, name_prefix="ppo_trading"),
            EpisodeCountCallback(70, 1)
        ]
    )
    model.save(out_dir / "model.zip")
    vec_env.save(out_dir / "vec_stats.pkl")
    print("✅  Model & istatistikler kaydedildi →", out_dir)

# -------------------------- test (backtest) -------------------------------- #
class NoResetDummyVecEnv(DummyVecEnv):
    """terminated+truncated ➜ done, otomatik reset yok."""
    def step_wait(self):
        packed = []
        for env, act in zip(self.envs, self.actions):
            o, r, term, trunc, info = env.step(act)
            packed.append((o, r, term or trunc, info))
        obs, rews, dones, infos = zip(*packed)
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

def run_backtest(symbol: str, interval: str, limit: int = 1000):
    """
    • rows   → tablo
    • prices / buys / sells → Chart.js grafiği
    """
    df      = fetch_klines_and_compute_indicators(symbol, interval, limit)
    test_df = df.iloc[int(len(df) * 0.85):].copy()

    raw_env = SpotTradingEnv(test_df)
    vec_env = NoResetDummyVecEnv([lambda: raw_env])

    stats_p = Path("artifacts") / symbol / interval / "vec_stats.pkl"
    vec_env = VecNormalize.load(stats_p, vec_env)
    vec_env.training, vec_env.norm_reward = False, False

    model = PPO.load(Path("artifacts") / symbol / interval / "model.zip", env=vec_env)

    out = vec_env.reset()
    obs = out[0] if isinstance(out, (tuple, list)) else out

    rows, prices, buys, sells = [], [], [], []

    for step in range(len(test_df)):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(act)

        env = vec_env.envs[0]
        idx = max(env.current_step - 1, 0)
        price = float(test_df.iloc[idx]["Close"])

        rows.append({
            "step":      step,
            "action":    int(env.last_action),
            "price":     round(price, 2),
            "balance":   round(env.balance, 2),
            "btc_held":  round(env.btc_held, 6),
            "total_usd": round(env.total_portfolio_value, 2)
        })

        prices.append(price)
        buys .append(price if env.last_action == 1 else None)
        sells.append(price if env.last_action == 2 else None)

        if done:
            break

    final_val = rows[-1]["total_usd"]
    # … döngü bitti
    return {
        "rows": rows,
        "prices": prices,  # <- her zaman liste
        "buys": buys or [None] * len(prices),
        "sells": sells or [None] * len(prices),
        "finalPortfolio": round(final_val, 2),
        "netProfit": round(final_val - 10_000, 2)
    }

