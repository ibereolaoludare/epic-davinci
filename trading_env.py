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
        
        # Actions: 0 = Flat/Close, 1 = Buy, 2 = Sell
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
        self.position = 0      # 1: long, -1: short
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
        
        # Hardcoded SL/TP thresholds in points
        TP_POINTS = 50.0
        SL_POINTS = 20.0
        
        # 1. Check for automatic SL/TP if in a trade
        trade_closed_automatically = False
        if self.position == 1:
            unrealized = current_price - self.entry_price
            if unrealized >= TP_POINTS:
                reward = 1.0
                self.position = 0
                trade_closed_automatically = True
            elif unrealized <= -SL_POINTS:
                reward = -1.0
                self.position = 0
                trade_closed_automatically = True
        elif self.position == -1:
            unrealized = self.entry_price - current_price
            if unrealized >= TP_POINTS:
                reward = 1.0
                self.position = 0
                trade_closed_automatically = True
            elif unrealized <= -SL_POINTS:
                reward = -1.0
                self.position = 0
                trade_closed_automatically = True
                
        # 2. Process AI Action (if trade didn't close automatically)
        if not trade_closed_automatically:
            if self.position == 0:
                if action == 1: # Go Long
                    self.position = 1
                    self.entry_price = current_price
                elif action == 2: # Go Short
                    self.position = -1
                    self.entry_price = current_price
                elif action == 0: # Stay Flat
                    reward = 0.0 # Removed boredom penalty to stop thrashing
            elif self.position == 1:
                if action == 0: # Close manually
                    unrealized = current_price - self.entry_price
                    reward = 1.0 if unrealized > 0 else -1.0
                    self.position = 0
                elif action == 2: # Reverse to Short
                    unrealized = current_price - self.entry_price
                    reward = 1.0 if unrealized > 0 else -1.0
                    self.position = -1
                    self.entry_price = current_price
                elif action == 1: # Hold Long
                    reward = 0.0
            elif self.position == -1:
                if action == 0: # Close manually
                    unrealized = self.entry_price - current_price
                    reward = 1.0 if unrealized > 0 else -1.0
                    self.position = 0
                elif action == 1: # Reverse to Long
                    unrealized = self.entry_price - current_price
                    reward = 1.0 if unrealized > 0 else -1.0
                    self.position = 1
                    self.entry_price = current_price
                elif action == 2: # Hold Short
                    reward = 0.0
                    
        # Update step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        info = {'net_worth': self.net_worth}
        truncated = False
        
        return self._get_observation(), float(reward), done, truncated, info

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Position: {self.position}')
