import numpy as np
import gymnasium as gym
from gymnasium import spaces

MIN_BUY_AMOUNT = 5         # Alım için minimum emir büyüklüğü (5 USDT)
MIN_SELL_BTC  = 0.0001     # Satış için minimum trade miktarı
EPSILON = 0

class SpotTradingEnv(gym.Env):
    """
    Spot trading ortamı.
    Gerekli df sütunları: ["Close", "RSI", "MACD", "MACD_Signal",
                           "Bollinger_High", "Bollinger_Low", "ATR"]
    Aksiyon: 0 BEKLE, 1 AL, 2 SAT
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, fee=0.001, gradient_window=5, gradient_threshold=0.05):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.fee = fee
        self.gradient_window = gradient_window
        self.gradient_threshold = gradient_threshold
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.total_portfolio_value = self.initial_balance
        self.old_balance = None
        self.old_btc_held = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset_count = 0
        self.last_action = None
        self.max_steps = len(self.df)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.total_portfolio_value = self.initial_balance
        self.old_balance = None
        self.old_btc_held = None
        self.last_action = None
        self.reset_count += 1
        print(f"✅ Reset! Episode: {self.reset_count}, current_step=0.")
        return self._next_observation(), {}

    def step(self, action):
        """
        Aksiyon uygulanır ve (obs, reward, terminated, truncated, info) döner.
        """
        # -------------- ZORUNLU KURAL: 0. adımda BUY -----------------
        if self.current_step == 0 and action != 1:
            action = 1
        # -------------------------------------------------------------

        if self.current_step >= len(self.df):
            obs = self._next_observation(clamp=True)
            return obs, 0.0, True, False, {}

        if np.random.rand() < EPSILON:
            action = self.action_space.sample()

        prev_value = self._get_portfolio_value()
        reward = self._apply_action(action)
        current_value = self._get_portfolio_value()
        self.total_portfolio_value = current_value
        reward += (current_value - prev_value)

        if self.last_action == 1 and action == 1:
            reward -= 10
        self.last_action = action

        terminated = truncated = False
        if self.total_portfolio_value < self.initial_balance * 0.85:
            terminated = True
            reward -= 50
        if self.current_step >= len(self.df) - 1:
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        self.current_step += 1
        obs = self._next_observation(clamp=self.current_step >= len(self.df))
        self.old_balance, self.old_btc_held = self.balance, self.btc_held
        return obs, reward, terminated, truncated, {}

    def _apply_action(self, action):
        row_idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[row_idx]
        current_price = row["Close"]
        rsi = row["RSI"]
        boll_high = row.get("Bollinger_High", 0.0)
        boll_low = row.get("Bollinger_Low", 0.0)
        price_gradient = self._calculate_price_gradient()
        reward = 0.0

        if action == 1:  # BUY
            buy_ratio = self._get_dynamic_buy_ratio(current_price)
            if self.balance > MIN_BUY_AMOUNT:
                buy_amount = self.balance * buy_ratio
                if buy_amount >= MIN_BUY_AMOUNT:
                    fee_cost = buy_amount * self.fee
                    buy_net = buy_amount - fee_cost
                    self.btc_held += buy_net / current_price
                    self.balance -= buy_amount
            reward += 70 if price_gradient < -self.gradient_threshold else -10
            reward += 15 if rsi < 35 else -10 if rsi > 65 else 0
            reward += 70 if current_price <= boll_low else -30

        elif action == 2:  # SELL
            sell_ratio = self._get_dynamic_sell_ratio(current_price)
            if self.btc_held > MIN_SELL_BTC:
                sell_amount = self.btc_held * sell_ratio
                if sell_amount >= MIN_SELL_BTC:
                    fee_cost = (sell_amount * current_price) * self.fee
                    sell_net = (sell_amount * current_price) - fee_cost
                    self.balance += sell_net
                    self.btc_held -= sell_amount
            reward += 70 if price_gradient > self.gradient_threshold else -10
            reward += 15 if rsi > 65 else -10 if rsi < 35 else 0
            reward += 70 if current_price >= boll_high else -30

        else:  # WAIT
            reward -= 5

        current_val = self.balance + (self.btc_held * current_price)
        if current_val > 0:
            pos_ratio = (self.btc_held * current_price) / current_val
            if pos_ratio > 0.8:
                reward -= 50
        return reward

    def _calculate_price_gradient(self):
        idx = min(self.current_step, len(self.df) - 1)
        if idx < self.gradient_window:
            return 0.0
        prices = self.df["Close"].iloc[idx - self.gradient_window: idx].values
        slope = np.polyfit(np.arange(len(prices)), prices, 1)[0]
        return float(slope)

    def _get_portfolio_value(self):
        idx = min(self.current_step, len(self.df) - 1)
        price = self.df["Close"].iloc[idx]
        return self.balance + (self.btc_held * price)

    def _next_observation(self, clamp=False):
        idx = min(self.current_step, len(self.df) - 1) if clamp else self.current_step
        row = self.df.iloc[idx]
        current_price = row["Close"]
        gradient = 0.0
        if idx >= self.gradient_window:
            prices = self.df["Close"].iloc[idx - self.gradient_window: idx]
            gradient = np.polyfit(np.arange(len(prices)), prices, 1)[0]
        obs = np.array([
            current_price, row["RSI"], row["MACD"],
            row.get("Bollinger_High", 0.0), row.get("Bollinger_Low", 0.0),
            self.balance, self.btc_held, gradient
        ], dtype=np.float32)
        return np.nan_to_num(obs)

    def _get_dynamic_buy_ratio(self, current_price):
        value = self.balance + self.btc_held * current_price
        if value <= 0:
            return 0.0
        pos_ratio = (self.btc_held * current_price) / value
        return 0.40 if pos_ratio < 0.5 else 0.20

    def _get_dynamic_sell_ratio(self, current_price):
        value = self.balance + self.btc_held * current_price
        if value <= 0:
            return 0.0
        pos_ratio = (self.btc_held * current_price) / value
        return 0.40 if pos_ratio > 0.5 else 0.20

    def render(self):
        print(f"Step={self.current_step}, Balance={self.balance:.2f}, "
              f"BTC_Held={self.btc_held:.6f}, TotalValue={self.total_portfolio_value:.2f}")
