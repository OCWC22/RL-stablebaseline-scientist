import numpy as np

class DummyCuriosityModule:
    """Dummy implementation of a curiosity module (e.g., Random Network Distillation).
    
    This is a placeholder that mimics the interface of a curiosity module
    without implementing the actual neural network logic.
    """
    
    def __init__(self, observation_space, action_space, beta_init=0.2):
        """Initialize the dummy curiosity module.
        
        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
            beta_init: Initial curiosity weight
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.beta = beta_init
        print(f"Initialized DummyCuriosityModule with beta={beta_init}")
    
    def intrinsic_reward(self, obs, action, next_obs):
        """Calculate the intrinsic reward for a transition.
        
        Args:
            obs: The current observation
            action: The action taken
            next_obs: The next observation
            
        Returns:
            reward: A dummy intrinsic reward
        """
        # In a real implementation, this would calculate curiosity based on prediction error
        # For the dummy version, just return a small random value
        reward = np.random.exponential(0.1)  # Small positive value with exponential falloff
        
        print(f"Curiosity intrinsic_reward: {reward:.4f}")
        return reward
    
    def update(self, buffer):
        """Update the curiosity module using collected experience.
        
        Args:
            buffer: The rollout buffer containing transitions
            
        Returns:
            loss: The dummy loss value
        """
        # In a real implementation, this would train the curiosity networks
        # For the dummy version, just print a message
        loss = np.random.random() * 0.5
        
        print(f"Curiosity module updated, loss={loss:.4f}")
        return loss
    
    def adapt_beta(self, trend):
        """Adapt the curiosity weight based on the reward trend.
        
        Args:
            trend: The slope of the moving average of external rewards
            
        Returns:
            beta: The updated curiosity weight
        """
        # Implement the logic from pseudocode section 2.7
        beta_adapt_rate = 0.01
        low_threshold = -0.01
        high_threshold = 0.01
        
        if trend < low_threshold:
            self.beta = min(1.0, self.beta * (1 + beta_adapt_rate))
        elif trend > high_threshold:
            self.beta = max(0.0, self.beta * (1 - beta_adapt_rate))
        
        print(f"Curiosity beta adapted to {self.beta:.4f} based on trend {trend:.4f}")
        return self.beta
