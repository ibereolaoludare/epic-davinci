import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from timezone_utils import get_session

class USDJPYTradingEnv(gym.Env):
    """
    Custom Environment for USDJPY H4.
    Simulates dynamic position management with spread and swap costs.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=96):
        super(USDJPYTradingEnv, self).__init__()
        
        self.df = df
        # Force time to UTC
        if 'Time' in self.df.columns:
            self.df['Time'] = pd.to_datetime(self.df['Time'], utc=True)
            
        self.feature_cols = [col for col in df.columns if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50']]
        self.window_size = window_size
        self.current_step = self.window_size
        
        # Actions: 0 = Short, 1 = Flat, 2 = Long
        self.action_space = spaces.Discrete(3)
        
        self.num_features = len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.num_features), 
            dtype=np.float32
        )
        
        self.position = 0
        self.entry_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0
        self.entry_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df[self.feature_cols].iloc[self.current_step - self.window_size: self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        current_time = self.df.iloc[self.current_step]['Time']
        
        next_price = self.df.iloc[self.current_step + 1]['Close'] if self.current_step < len(self.df) - 1 else current_price
        
        step_pnl = next_price - current_price
        target_position = action - 1
        
        reward = 0.0
        
        # Calculate Variable Spread & Swap Cost
        session = get_session(current_time)
        spread_pips = 0.5 if session in ['London', 'NY', 'London/NY Overlap'] else 1.5
        spread_cost = spread_pips * 0.01
        
        # 0.2 pips per day swap cost (0.2 * 0.01). On H4, one step = 4 hours = 1/6 of a day.
        swap_cost_per_step = (0.2 / 6.0) * 0.01
        
        if target_position != self.position:
            # Pay spread on entry
            reward -= spread_cost
            self.entry_step = self.current_step
            
        if target_position != 0:
            # Pay swap for holding over this step
            reward -= swap_cost_per_step
            
        if target_position == 1:
            reward += step_pnl
        elif target_position == -1:
            reward -= step_pnl
            
        current_atr = self.df.iloc[self.current_step]['ATR']
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = 1.0 # Fallback for USDJPY
            
        reward = float(reward / current_atr)
        reward = max(-5.0, min(5.0, reward))
        
        self.position = target_position
        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}

    def render(self):
        pass
