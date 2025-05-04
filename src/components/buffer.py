import numpy as np

class DummyRolloutBuffer:
    """Dummy implementation of a rollout buffer that stores both real and imagined transitions.
    
    This is a placeholder that mimics the interface of a rollout buffer
    without implementing the complex logic for advantage computation, etc.
    """
    
    def __init__(self, buffer_size=2048):
        """Initialize the dummy rollout buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
        """
        self.buffer_size = buffer_size
        self.reset()
        print(f"Initialized DummyRolloutBuffer with size={buffer_size}")
    
    def reset(self):
        """Reset the buffer, clearing all stored transitions."""
        # Simple lists to store transitions
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logps = []
        self.ext_rewards = []
        self.intr_rewards = []
        self.dones = []
        self.is_real = []
        # Computed values
        self.advantages = []
        self.returns = []
        print("RolloutBuffer reset")
    
    def add(self, obs, action, reward, value, logp, ext_reward, intr_reward, done, is_real):
        """Add a transition to the buffer.
        
        Args:
            obs: The observation
            action: The action taken
            reward: The combined reward (ext + intr)
            value: The value estimate
            logp: The log probability of the action
            ext_reward: The extrinsic reward
            intr_reward: The intrinsic reward
            done: Whether the episode terminated
            is_real: Whether this is a real or imagined transition
        """
        # Add the transition to the buffer
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logps.append(logp)
        self.ext_rewards.append(ext_reward)
        self.intr_rewards.append(intr_reward)
        self.dones.append(done)
        self.is_real.append(is_real)
        
        # Print a message every 100 additions to avoid spam
        if len(self.obs) % 100 == 0:
            print(f"Added transition #{len(self.obs)} to buffer, is_real={is_real}")
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute returns and advantages for the stored transitions.
        
        Args:
            last_value: The value estimate for the last observation
            gamma: The discount factor
            gae_lambda: The GAE lambda parameter
        """
        # In a real implementation, this would compute proper returns and advantages
        # For the dummy version, just set random values
        n = len(self.obs)
        self.advantages = np.random.normal(0, 1, n)
        self.returns = np.random.normal(0, 1, n)
        
        print(f"Computed dummy returns and advantages for {n} transitions")
    
    def sample_recent_state(self):
        """Sample a recent state from the buffer for imagination.
        
        Returns:
            A state from the buffer
        """
        if not self.obs:
            raise ValueError("Buffer is empty, cannot sample state")
        
        # Sample from the most recent 25% of states
        start_idx = max(0, int(0.75 * len(self.obs)))
        idx = np.random.randint(start_idx, len(self.obs))
        
        print(f"Sampled state from buffer at index {idx}")
        return self.obs[idx]
    
    def iterate(self, batch_size=64):
        """Iterate through the buffer in batches.
        
        Args:
            batch_size: The size of each batch
            
        Yields:
            batch: A dummy batch object with the required attributes
        """
        n = len(self.obs)
        indices = np.random.permutation(n)
        
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_indices = indices[start_idx:end_idx]
            
            # Create a simple namespace as a batch
            class Batch:
                pass
            
            batch = Batch()
            batch.obs = [self.obs[i] for i in batch_indices]
            batch.actions = [self.actions[i] for i in batch_indices]
            batch.old_logp = [self.logps[i] for i in batch_indices]
            batch.adv = [self.advantages[i] for i in batch_indices]
            batch.ret = [self.returns[i] for i in batch_indices]
            
            print(f"Yielding batch of size {len(batch_indices)}")
            yield batch
    
    def real_ext_rewards(self):
        """Get the external rewards from real transitions.
        
        Returns:
            A list of external rewards from real transitions
        """
        return [r for r, real in zip(self.ext_rewards, self.is_real) if real]
    
    def real_and_imagined(self):
        """Get all transitions (both real and imagined).
        
        Returns:
            A simple object containing all transitions
        """
        # Create a simple namespace as a container
        class TransitionBatch:
            pass
        
        batch = TransitionBatch()
        batch.obs = self.obs
        batch.actions = self.actions
        batch.next_obs = self.obs[1:] + [self.obs[-1]]  # Dummy next_obs
        batch.rewards = self.rewards
        batch.dones = self.dones
        batch.is_real = self.is_real
        
        print(f"Returning all {len(self.obs)} transitions")
        return batch
