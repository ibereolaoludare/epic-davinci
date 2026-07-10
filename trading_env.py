import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class DerivTradingEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    This simulates directional prediction rewards instead of broker PnL
    to prevent policy collapse.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=30):
        super(DerivTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.current_step = self.window_size
        
        # Actions: 0 = Hold/Flat, 1 = Buy (expect price up), 2 = Sell (expect price down)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: rolling window of all features
        self.num_features = df.shape[1] 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.num_features), 
            dtype=np.float32
        )
        
        # Simulated accounting for metric tracking
        self.balance = 1000.0
        self.net_worth = self.balance
        self.position = 0      # 0: flat, 1: long, -1: short
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = 1000.0
        self.net_worth = self.balance
        self.position = 0
        self.entry_price = 0.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close'] if self.current_step < len(self.df) - 1 else current_price
        
        reward = 0.0
        price_change = next_price - current_price
        
        # Redefined Directional Reward Function:
        # We reward pure prediction accuracy rather than simulated portfolio balance
        if action == 1: # Expecting price to go UP
            if price_change > 0:
                reward = 1.0  # Correct prediction
            else:
                reward = -1.0 # Incorrect prediction
                
        elif action == 2: # Expecting price to go DOWN
            if price_change < 0:
                reward = 1.0  # Correct prediction
            else:
                reward = -1.0 # Incorrect prediction
                
        elif action == 0: # Hold flat / Refuse to predict
            reward = -0.05 # Tiny boredom penalty to discourage laziness
            
        # Update step
        self.current_step += 1
        
        # Simulating account tracking for log/info metrics
        commission_fee = 0.10
        if action == 1:
            self.position = 1
            self.net_worth += (price_change * 10) - commission_fee
        elif action == 2:
            self.position = -1
            self.net_worth += (-price_change * 10) - commission_fee
        else:
            self.position = 0

        # Check if done
        done = self.current_step >= len(self.df) - 2
        
        info = {'net_worth': self.net_worth}
        truncated = False
        
        return self._get_observation(), float(reward), done, truncated, info

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Position: {self.position}')
