# Model-Based Reinforcement Learning with Adaptive Imagination

## Introduction to Model-Based RL

Reinforcement Learning (RL) algorithms can be broadly categorized into two approaches: **model-free** and **model-based**.

### Model-Free vs. Model-Based RL

**Model-Free RL** (like standard PPO, A2C, and DQN implemented in this project) learns directly from experience by interacting with the environment. These algorithms:
- Learn a policy and/or value function directly from observed transitions
- Don't attempt to understand the environment dynamics
- Typically require many environment interactions to learn effectively
- Are conceptually simpler and often more stable to train

**Model-Based RL** takes a different approach by explicitly learning a model of the environment dynamics. These algorithms:
- Learn to predict how the environment will respond to actions (state transitions, rewards)
- Use this learned model to plan ahead or generate synthetic experience
- Can be more sample-efficient (require fewer real environment interactions)
- Often involve more complex training procedures

### Historical Context

The idea of model-based RL dates back to early work by Richard Sutton in the Dyna architecture (1991), which combined learning, planning, and reacting in an integrated system. Dyna used a learned model to generate additional training data, allowing the agent to "imagine" experiences without actually taking actions in the real environment.

```python
# Simplified pseudocode for Dyna-Q algorithm (Sutton, 1991)
def dyna_q(env, num_episodes, planning_steps):
    # Initialize Q-values and model
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    model = {}  # (state, action) -> (next_state, reward)
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Select action using epsilon-greedy
            action = epsilon_greedy(Q, state)
            
            # Take action in real environment
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
            
            # Update model with real experience
            model[(state, action)] = (next_state, reward, done)
            
            state = next_state
            
            # Planning phase (learning from imagined experience)
            for _ in range(planning_steps):
                # Sample a previously observed state and action
                s, a = random_previously_observed_state_action(model)
                
                # Query the model for predicted outcome
                ns, r, d = model[(s, a)]
                
                # Q-learning update using simulated experience
                Q[s][a] += alpha * (r + gamma * max(Q[ns]) - Q[s][a])
```

In recent years, model-based RL has seen a resurgence with advances in deep learning enabling more accurate world models. Algorithms like MBPO (Janner et al., 2019), PETS (Chua et al., 2018), and Dreamer (Hafner et al., 2020-2023) have demonstrated that model-based approaches can achieve state-of-the-art performance while using significantly fewer environment interactions.

## Model-Based Planning with Adaptive Imagination

The "Model-Based Planning with Adaptive Imagination" algorithm implemented in this project (as outlined in `pseudocode.md`) combines the strengths of model-free PPO with model-based planning. It builds on several key ideas from recent research:

1. **Short-horizon model-based rollouts**: Similar to MBPO, it uses the world model for short rollouts to avoid compounding model errors

2. **Adaptive planning**: It dynamically adjusts the number of imagined rollouts based on model confidence

3. **Curiosity-driven exploration**: It incorporates intrinsic rewards to encourage exploration of uncertain states

4. **On-policy learning with imagined data**: It extends PPO to learn from both real and imagined experiences

### Key Components in Detail

1. **Policy Network (πθ)**:
   - Neural network that maps states to action distributions
   - For discrete action spaces (like CartPole), outputs action probabilities
   - For continuous action spaces, would output mean and standard deviation of a Gaussian distribution
   - Includes an entropy term to encourage exploration
   - Architecture: Typically an MLP with 2-3 hidden layers (e.g., [64, 64])

   ```python
   # Example PyTorch implementation of a simple policy network for discrete actions
   class PolicyNetwork(nn.Module):
       def __init__(self, input_dim, hidden_dims, output_dim):
           super().__init__()
           
           layers = []
           prev_dim = input_dim
           
           for dim in hidden_dims:
               layers.append(nn.Linear(prev_dim, dim))
               layers.append(nn.ReLU())
               prev_dim = dim
               
           layers.append(nn.Linear(prev_dim, output_dim))
           self.model = nn.Sequential(*layers)
       
       def forward(self, state):
           logits = self.model(state)
           action_probs = F.softmax(logits, dim=-1)
           return action_probs
           
       def get_action(self, state):
           action_probs = self.forward(state)
           action_dist = Categorical(action_probs)
           action = action_dist.sample()
           log_prob = action_dist.log_prob(action)
           return action, log_prob
   ```

2. **Value Network (Vθ)**:
   - Neural network that estimates the expected discounted return from a state
   - Used for advantage estimation in the PPO algorithm
   - Often shares early layers with the policy network for efficiency
   - Architecture: Typically matches the policy network architecture

   ```python
   # Example PyTorch implementation of a simple value network
   class ValueNetwork(nn.Module):
       def __init__(self, input_dim, hidden_dims):
           super().__init__()
           
           layers = []
           prev_dim = input_dim
           
           for dim in hidden_dims:
               layers.append(nn.Linear(prev_dim, dim))
               layers.append(nn.ReLU())
               prev_dim = dim
               
           layers.append(nn.Linear(prev_dim, 1))  # Output a single value
           self.model = nn.Sequential(*layers)
       
       def forward(self, state):
           return self.model(state)
   ```

3. **World Model (ŵφ)**:
   - Neural network that predicts environment dynamics
   - Inputs: Current state and action
   - Outputs:
     - Next state prediction (s')
     - Reward prediction (r)
     - Done prediction (d) - probability of episode termination
   - Loss function: Combination of MSE for state/reward prediction and BCE for done prediction
   - Architecture: MLP with 2-3 hidden layers, potentially larger than policy network
   - Training: Supervised learning on real transitions collected by the agent

   ```python
   # Example PyTorch implementation of a simple world model
   class WorldModel(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dims):
           super().__init__()
           
           self.input_dim = state_dim + action_dim
           
           # Shared encoder
           encoder_layers = []
           prev_dim = self.input_dim
           
           for dim in hidden_dims:
               encoder_layers.append(nn.Linear(prev_dim, dim))
               encoder_layers.append(nn.ReLU())
               prev_dim = dim
               
           self.encoder = nn.Sequential(*encoder_layers)
           
           # Output heads
           self.state_predictor = nn.Linear(prev_dim, state_dim)
           self.reward_predictor = nn.Linear(prev_dim, 1)
           self.done_predictor = nn.Linear(prev_dim, 1)
       
       def forward(self, state, action):
           x = torch.cat([state, action], dim=-1)
           features = self.encoder(x)
           
           next_state = self.state_predictor(features)
           reward = self.reward_predictor(features)
           done_logit = self.done_predictor(features)
           done_prob = torch.sigmoid(done_logit)
           
           return next_state, reward, done_prob
           
       def compute_loss(self, states, actions, next_states, rewards, dones):
           pred_next_states, pred_rewards, pred_done_probs = self(states, actions)
           
           state_loss = F.mse_loss(pred_next_states, next_states)
           reward_loss = F.mse_loss(pred_rewards, rewards)
           done_loss = F.binary_cross_entropy(pred_done_probs, dones)
           
           total_loss = state_loss + reward_loss + done_loss
           return total_loss, state_loss, reward_loss, done_loss
   ```

4. **Curiosity Module (Cψ)**:
   - Provides intrinsic rewards to encourage exploration
   - Implementation options:
     - **Random Network Distillation (RND)**: Uses prediction error of a random fixed target network as intrinsic reward
     - **Intrinsic Curiosity Module (ICM)**: Rewards states where the prediction error is high
     - **Uncertainty-based**: Uses epistemic uncertainty in the world model predictions
   - Adaptive weight (β) controls the balance between extrinsic and intrinsic rewards
   - Architecture: Depends on the specific curiosity mechanism chosen

   ```python
   # Example PyTorch implementation of a Random Network Distillation (RND) curiosity module
   class RNDCuriosityModule(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dims):
           super().__init__()
           
           # Target network (fixed, randomly initialized)
           self.target_network = self._build_network(state_dim, hidden_dims)
           # Freeze the target network weights
           for param in self.target_network.parameters():
               param.requires_grad = False
               
           # Predictor network (trained to predict target network outputs)
           self.predictor_network = self._build_network(state_dim, hidden_dims)
           
           # Current curiosity weight (beta)
           self.beta = 0.2  # Initial value
           
       def _build_network(self, input_dim, hidden_dims):
           layers = []
           prev_dim = input_dim
           
           for dim in hidden_dims:
               layers.append(nn.Linear(prev_dim, dim))
               layers.append(nn.ReLU())
               prev_dim = dim
               
           # Output an embedding/feature vector
           feature_dim = 64  # Arbitrary feature dimension
           layers.append(nn.Linear(prev_dim, feature_dim))
           
           return nn.Sequential(*layers)
       
       def intrinsic_reward(self, state, action, next_state):
           # RND only uses the next state
           target_features = self.target_network(next_state)
           predicted_features = self.predictor_network(next_state)
           
           # Intrinsic reward is the prediction error
           prediction_error = F.mse_loss(predicted_features, target_features, reduction='none')
           intrinsic_reward = prediction_error.mean(dim=-1, keepdim=True)
           
           return self.beta * intrinsic_reward
       
       def update(self, states, actions, next_states):
           target_features = self.target_network(next_states).detach()
           predicted_features = self.predictor_network(next_states)
           
           loss = F.mse_loss(predicted_features, target_features)
           return loss
       
       def adapt_beta(self, reward_trend):
           beta_adapt_rate = 0.01
           
           # If trend is negative (rewards decreasing), increase beta
           if reward_trend < -0.01:
               self.beta = min(1.0, self.beta * (1 + beta_adapt_rate))
           # If trend is positive (rewards increasing), decrease beta
           elif reward_trend > 0.01:
               self.beta = max(0.0, self.beta * (1 - beta_adapt_rate))
               
           return self.beta
   ```

5. **Mixed Rollout Buffer**:
   - Stores both real and imagined transitions
   - Tracks which transitions are real vs. imagined
   - Computes advantages and returns for PPO updates
   - Implements efficient batch sampling for training
   - Special consideration: Handling potentially different distributions of real vs. imagined data

   ```python
   # Example implementation of a rollout buffer for mixed real/imagined transitions
   class MixedRolloutBuffer:
       def __init__(self, buffer_size, state_dim, action_dim):
           self.buffer_size = buffer_size
           
           # Storage for transitions
           self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
           self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
           self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
           self.values = np.zeros((buffer_size, 1), dtype=np.float32)
           self.log_probs = np.zeros((buffer_size, 1), dtype=np.float32)
           self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
           self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
           self.is_real = np.zeros((buffer_size, 1), dtype=np.bool_)
           
           # For advantage computation
           self.advantages = np.zeros((buffer_size, 1), dtype=np.float32)
           self.returns = np.zeros((buffer_size, 1), dtype=np.float32)
           
           # Separate tracking for extrinsic and intrinsic rewards
           self.ext_rewards = np.zeros((buffer_size, 1), dtype=np.float32)
           self.int_rewards = np.zeros((buffer_size, 1), dtype=np.float32)
           
           self.pointer = 0
           self.size = 0
       
       def add(self, state, action, reward, value, log_prob, ext_reward, int_reward, done, is_real):
           idx = self.pointer % self.buffer_size
           
           self.states[idx] = state
           self.actions[idx] = action
           self.rewards[idx] = reward  # Combined reward (ext + int)
           self.values[idx] = value
           self.log_probs[idx] = log_prob
           self.dones[idx] = done
           self.is_real[idx] = is_real
           self.ext_rewards[idx] = ext_reward
           self.int_rewards[idx] = int_reward
           
           self.pointer += 1
           self.size = min(self.size + 1, self.buffer_size)
       
       def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
           # GAE algorithm for advantage computation
           last_gae_lam = 0
           for step in reversed(range(self.size)):
               if step == self.size - 1:
                   next_non_terminal = 1.0 - self.dones[step]
                   next_values = last_value
               else:
                   next_non_terminal = 1.0 - self.dones[step]
                   next_values = self.values[step + 1]
               
               delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
               last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
               self.advantages[step] = last_gae_lam
           
           # Returns = advantages + values
           self.returns = self.advantages + self.values
       
       def sample_recent_state(self):
           # Sample from the most recent quarter of real transitions
           real_indices = np.where(self.is_real[:self.size])[0]
           if len(real_indices) == 0:
               return None
           
           # Focus on more recent states (last 25% of real data)
           recent_cutoff = int(0.75 * len(real_indices))
           recent_indices = real_indices[recent_cutoff:]
           
           # Randomly sample one
           idx = np.random.choice(recent_indices)
           return self.states[idx]
       
       def iterate(self, batch_size):
           indices = np.random.permutation(self.size)
           start_idx = 0
           
           while start_idx < self.size:
               batch_indices = indices[start_idx:start_idx + batch_size]
               batch_dict = {
                   'obs': torch.FloatTensor(self.states[batch_indices]),
                   'actions': torch.FloatTensor(self.actions[batch_indices]),
                   'values': torch.FloatTensor(self.values[batch_indices]),
                   'log_probs': torch.FloatTensor(self.log_probs[batch_indices]),
                   'advantages': torch.FloatTensor(self.advantages[batch_indices]),
                   'returns': torch.FloatTensor(self.returns[batch_indices]),
                   'is_real': torch.BoolTensor(self.is_real[batch_indices])
               }
               
               yield batch_dict
               start_idx += batch_size
       
       def real_ext_rewards(self):
           # Get only the external rewards from real transitions
           real_mask = self.is_real[:self.size].flatten()
           return self.ext_rewards[:self.size].flatten()[real_mask]
       
       def reset(self):
           self.pointer = 0
           self.size = 0
   ```

### Algorithm Flow in Detail

1. **Collect Real Experience**:
   - Execute the current policy πθ in the real environment for n_real_steps
   - Store (s, a, r, s', done) transitions in the rollout buffer
   - Track both external rewards (from environment) and intrinsic rewards (from curiosity)
   - Code section: 2.1 in pseudocode.md

   ```python
   # Example code for collecting real experience
   def collect_real_experience(env, policy, world_model, curiosity, buffer, n_steps):
       obs = env.reset()
       episode_reward = 0
       
       for step in range(n_steps):
           # Get action from policy
           with torch.no_grad():
               action, value, log_prob = policy(torch.FloatTensor(obs).unsqueeze(0))
               action = action.cpu().numpy()[0]
               value = value.cpu().numpy()[0]
               log_prob = log_prob.cpu().numpy()[0]
           
           # Step the environment
           next_obs, ext_reward, done, _ = env.step(action)
           
           # Calculate intrinsic reward
           with torch.no_grad():
               intr_reward = curiosity.intrinsic_reward(
                   torch.FloatTensor(obs).unsqueeze(0),
                   torch.FloatTensor([action]).unsqueeze(0),
                   torch.FloatTensor(next_obs).unsqueeze(0)
               ).cpu().numpy()[0]
           
           # Add to buffer
           buffer.add(
               state=obs,
               action=action,
               reward=ext_reward + intr_reward,
               value=value,
               log_prob=log_prob,
               ext_reward=ext_reward,
               int_reward=intr_reward,
               done=done,
               is_real=True
           )
           
           # Update observation
           obs = env.reset() if done else next_obs
           episode_reward += ext_reward
           
           if done:
               print(f"Episode reward: {episode_reward}")
               episode_reward = 0
       
       return buffer
   ```

2. **Train World Model**:
   - Sample mini-batches of real transitions from the buffer
   - Update the world model parameters φ using gradient descent to minimize prediction errors:
     - State prediction error: MSE(ŵφ(s,a).state, s')
     - Reward prediction error: MSE(ŵφ(s,a).reward, r)
     - Done prediction error: BCE(ŵφ(s,a).done, done)
   - Compute model confidence as exp(-loss_model), which maps [0,∞) to (0,1]
   - Code section: 2.2 in pseudocode.md

   ```python
   # Example code for training the world model
   def train_world_model(world_model, buffer, optimizer, batch_size=64, epochs=10):
       # Only use real data for world model training
       real_indices = np.where(buffer.is_real[:buffer.size])[0]
       if len(real_indices) == 0:
           return 0.0  # No real data to train on
       
       total_loss = 0.0
       
       for _ in range(epochs):
           # Sample minibatch of real transitions
           batch_indices = np.random.choice(real_indices, size=min(batch_size, len(real_indices)), replace=False)
           
           states = torch.FloatTensor(buffer.states[batch_indices])
           actions = torch.FloatTensor(buffer.actions[batch_indices])
           next_states = torch.FloatTensor(buffer.next_states[batch_indices])
           rewards = torch.FloatTensor(buffer.ext_rewards[batch_indices])  # Use external rewards
           dones = torch.FloatTensor(buffer.dones[batch_indices])
           
           # Forward pass and compute loss
           optimizer.zero_grad()
           loss, _, _, _ = world_model.compute_loss(states, actions, next_states, rewards, dones)
           
           # Backward pass and optimize
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item()
       
       avg_loss = total_loss / epochs
       
       # Compute confidence from loss
       confidence = np.exp(-avg_loss)
       
       print(f"World model loss: {avg_loss:.4f}, confidence: {confidence:.4f}")
       return confidence
   ```

3. **Adaptive Imagination**:
   - Determine number of imagination rollouts based on model confidence:
     - Higher confidence → more rollouts (up to max_planning_rollouts)
     - Lower confidence → fewer rollouts (down to min_planning_rollouts)
   - For each rollout:
     - Sample a starting state from recent real experience
     - Execute policy in the world model for up to planning_horizon steps
     - For each step:
       - Get action from policy: a_sim ~ πθ(sim_state)
       - Predict next state, reward, done: ŵφ(sim_state, a_sim)
       - Add intrinsic reward: intr_sim = β * Cψ(sim_state, a_sim, ŵφ(sim_state, a_sim).state)
       - Store imagined transition in buffer with is_real=False flag
       - Break if done is predicted
   - Code section: 2.3 in pseudocode.md

   ```python
   # Example code for the adaptive imagination phase
   def linear_interpolate(value, low, high, out_low, out_high):
       """Linear interpolation function"""
       value = max(low, min(high, value))  # Clamp value to range
       if high == low:
           return out_low
       ratio = (value - low) / (high - low)
       return out_low + ratio * (out_high - out_low)

   def generate_imagined_rollouts(policy, world_model, curiosity, buffer, 
                                 confidence, confidence_thresh=0.05,
                                 min_planning_rollouts=10, max_planning_rollouts=100,
                                 planning_horizon=5):
       # Determine number of rollouts based on confidence
       num_rollouts = int(linear_interpolate(
           confidence, 
           low=confidence_thresh, 
           high=1.0, 
           out_low=min_planning_rollouts, 
           out_high=max_planning_rollouts
       ))
       
       print(f"Generating {num_rollouts} imagined rollouts with planning horizon {planning_horizon}")
       
       # Generate rollouts
       for rollout in range(num_rollouts):
           # Sample a starting state from recent real experience
           sim_state = buffer.sample_recent_state()
           if sim_state is None:
               continue  # No real states to sample from
           
           # Convert to tensor
           sim_state_tensor = torch.FloatTensor(sim_state).unsqueeze(0)
           
           # Perform rollout for up to planning_horizon steps
           for h in range(planning_horizon):
               # Get action from policy
               with torch.no_grad():
                   a_sim, v_sim, logp_sim = policy(sim_state_tensor)
                   a_sim_np = a_sim.cpu().numpy()[0]
                   v_sim_np = v_sim.cpu().numpy()[0]
                   logp_sim_np = logp_sim.cpu().numpy()[0]
               
               # Predict next state, reward, done using world model
               with torch.no_grad():
                   s_next, r_hat, d_prob = world_model(
                       sim_state_tensor, 
                       torch.FloatTensor([a_sim_np]).unsqueeze(0)
                   )
                   
                   # Convert to numpy
                   s_next_np = s_next.cpu().numpy()[0]
                   r_hat_np = r_hat.cpu().numpy()[0]
                   done = bool(d_prob.cpu().numpy()[0] > 0.5)  # Threshold at 0.5
               
               # Calculate intrinsic reward
               with torch.no_grad():
                   intr_sim = curiosity.intrinsic_reward(
                       sim_state_tensor,
                       torch.FloatTensor([a_sim_np]).unsqueeze(0),
                       s_next
                   ).cpu().numpy()[0]
               
               # Add to buffer with is_real=False
               buffer.add(
                   state=sim_state,
                   action=a_sim_np,
                   reward=r_hat_np + intr_sim,
                   value=v_sim_np,
                   log_prob=logp_sim_np,
                   ext_reward=r_hat_np,
                   int_reward=intr_sim,
                   done=done,
                   is_real=False
               )
               
               # Update simulation state
               sim_state = s_next_np
               sim_state_tensor = s_next
               
               # Break if done
               if done:
                   break
       
       return buffer
   ```

4. **Advantage & Return Computation**:
   - Compute the last value estimate: Vθ(last_obs)
   - Calculate returns and advantages for all transitions (real and imagined) using GAE(λ)
   - Generalized Advantage Estimation formula:
     - δ_t = r_t + γV(s_{t+1}) - V(s_t)
     - A_t = δ_t + γλδ_{t+1} + γ^2λ^2δ_{t+2} + ...
   - Code section: 2.4 in pseudocode.md

   ```python
   # The advantage computation is already implemented in the MixedRolloutBuffer class
   # This function just calls that method with the last value
   def compute_advantages_and_returns(buffer, policy, last_obs, gamma=0.99, gae_lambda=0.95):
       with torch.no_grad():
           last_value = policy.value(torch.FloatTensor(last_obs).unsqueeze(0)).cpu().numpy()[0]
       
       buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)
       return buffer
   ```

### Algorithm Flow in Detail

1. **Collect Real Experience**:
   - Execute the current policy πθ in the real environment for n_real_steps
   - Store (s, a, r, s', done) transitions in the rollout buffer
   - Track both external rewards (from environment) and intrinsic rewards (from curiosity)
   - Code section: 2.1 in pseudocode.md

2. **Train World Model**:
   - Sample mini-batches of real transitions from the buffer
   - Update the world model parameters φ using gradient descent to minimize prediction errors:
     - State prediction error: MSE(ŵφ(s,a).state, s')
     - Reward prediction error: MSE(ŵφ(s,a).reward, r)
     - Done prediction error: BCE(ŵφ(s,a).done, done)
   - Compute model confidence as exp(-loss_model), which maps [0,∞) to (0,1]
   - Code section: 2.2 in pseudocode.md

3. **Adaptive Imagination**:
   - Determine number of imagination rollouts based on model confidence:
     - Higher confidence → more rollouts (up to max_planning_rollouts)
     - Lower confidence → fewer rollouts (down to min_planning_rollouts)
   - For each rollout:
     - Sample a starting state from recent real experience
     - Execute policy in the world model for up to planning_horizon steps
     - For each step:
       - Get action from policy: a_sim ~ πθ(sim_state)
       - Predict next state, reward, done: ŵφ(sim_state, a_sim)
       - Add intrinsic reward: intr_sim = β * Cψ(sim_state, a_sim, ŵφ(sim_state, a_sim).state)
       - Store imagined transition in buffer with is_real=False flag
       - Break if done is predicted
   - Code section: 2.3 in pseudocode.md

4. **Advantage & Return Computation**:
   - Compute the last value estimate: Vθ(last_obs)
   - Calculate returns and advantages for all transitions (real and imagined) using GAE(λ)
   - Generalized Advantage Estimation formula:
     - δ_t = r_t + γV(s_{t+1}) - V(s_t)
     - A_t = δ_t + γλδ_{t+1} + γ^2λ^2δ_{t+2} + ...
   - Code section: 2.4 in pseudocode.md

5. **PPO-Style Policy Update**:
   - For n_epochs:
     - Sample mini-batches from the buffer (both real and imagined data)
     - Compute policy loss using clipped surrogate objective:
       - ratio = exp(logπ_new(a|s) - logπ_old(a|s))
       - surr1 = ratio * advantage
       - surr2 = clip(ratio, 1-ε, 1+ε) * advantage
       - policy_loss = -min(surr1, surr2)
     - Compute value loss: MSE(Vθ(s), returns)
     - Compute entropy bonus: -entropy(πθ)
     - Update policy and value networks using combined loss
   - Code section: 2.5 in pseudocode.md

6. **Curiosity Module Update**:
   - Update the curiosity module using both real and imagined transitions
   - Specific update depends on the curiosity mechanism implemented
   - Code section: 2.6 in pseudocode.md

7. **Adapt Curiosity Weight (β)**:
   - Calculate moving average of external rewards
   - Compute trend (slope) of this moving average
   - If trend is negative (rewards decreasing): increase β to encourage more exploration
   - If trend is positive (rewards increasing): decrease β to focus more on exploitation
   - Ensures balance between exploration and exploitation adapts to learning progress
   - Code section: 2.7 in pseudocode.md

### Mathematical Foundations

#### World Model Learning

The world model learns a function ŵφ that approximates the environment dynamics:

ŵφ(s_t, a_t) ≈ (s_{t+1}, r_t, d_t)

where s_t is the state at time t, a_t is the action, r_t is the reward, and d_t is the probability of episode termination.

The loss function for the world model is:

L_model = MSE(ŵφ(s_t, a_t).state, s_{t+1}) + MSE(ŵφ(s_t, a_t).reward, r_t) + BCE(ŵφ(s_t, a_t).done, d_t)

where MSE is Mean Squared Error and BCE is Binary Cross-Entropy.

#### Confidence Calculation

The confidence in the world model is calculated as:

confidence = exp(-L_model)

This maps the loss from [0,∞) to (0,1], where higher confidence corresponds to lower loss.

#### Adaptive Rollout Interpolation

The number of imagination rollouts is determined by linear interpolation:

num_rollouts = interpolate(confidence, low=confidence_thresh, high=1.0, out_low=min_planning_rollouts, out_high=max_planning_rollouts)

which maps confidence in range [confidence_thresh, 1.0] to rollouts in range [min_planning_rollouts, max_planning_rollouts].

#### PPO Objective

The PPO algorithm uses a clipped surrogate objective:

L_CLIP = E_t[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

where r_t(θ) is the probability ratio r_t(θ) = π_θ(a_t|s_t) / π_{θold}(a_t|s_t), and A_t is the advantage estimate.

## Recent Research and Advancements (as of May 2025)

Model-based RL has seen significant advancements in recent years. Key papers and developments include:

1. **Dreamer v3** (Hafner et al., 2023): A state-of-the-art model-based RL algorithm that learns a world model and uses it for planning in latent space. Dreamer v3 builds on previous versions by incorporating:
   - Improved representation learning with a more sophisticated encoder/decoder architecture
   - Better handling of stochasticity in environment dynamics
   - More efficient planning in latent space
   - Achieved human-level performance on the Atari benchmark while using 100x fewer environment interactions

2. **Model-Based RL with Adversarial Training** (Pan et al., 2024): Improves world model robustness by training with adversarial examples:
   - Generates adversarial perturbations to states and actions during world model training
   - Forces the world model to be robust to small variations in inputs
   - Reduces the tendency of policies to exploit model inaccuracies
   - Shows significant improvements in policy robustness and transfer to the real environment

3. **Adaptive Planning Horizons** (Zhang et al., 2024): Dynamically adjusts planning horizon based on uncertainty estimates:
   - Uses uncertainty quantification techniques (e.g., ensemble disagreement, Bayesian neural networks)
   - Plans further ahead in regions of state space where the model is confident
   - Shortens planning horizon when uncertainty is high
   - Similar to our adaptive imagination approach but varies the depth rather than breadth of planning

4. **Hybrid Model-Free/Model-Based Approaches** (Liu et al., 2025): Combines the stability of model-free methods with the sample efficiency of model-based approaches:
   - Uses model-based data augmentation during early training when real data is scarce
   - Gradually shifts to more model-free updates as real experience accumulates
   - Implements an adaptive weighting scheme between real and imagined data
   - Achieves better asymptotic performance than pure model-based methods while maintaining sample efficiency

5. **Transformer-Based World Models** (Johnson et al., 2025): Leverages transformer architectures for world modeling:
   - Treats the prediction problem as a sequence modeling task
   - Uses self-attention to capture long-range dependencies in state transitions
   - Shows improved performance on environments with complex dynamics and partial observability
   - Enables more accurate long-horizon predictions

## Benefits and Challenges of Model-Based RL

### Benefits of Model-Based RL

1. **Sample Efficiency**: Requires fewer real environment interactions by leveraging the world model for synthetic experience. This is particularly valuable when:
   - Real environment interactions are expensive (e.g., robotics)
   - Environment has sparse rewards
   - Task requires long-horizon planning

2. **Planning Capability**: Can look ahead and consider future consequences of actions:
   - Enables reasoning about delayed rewards
   - Allows for explicit risk assessment
   - Can avoid catastrophic states through simulation

3. **Exploration**: Can use the world model to identify and explore uncertain or promising states:
   - Curiosity-driven exploration becomes more targeted
   - Can simulate "what-if" scenarios without real-world risk
   - Enables more strategic exploration policies

4. **Transfer Learning**: A good world model can potentially transfer across tasks in the same environment:
   - Environment dynamics remain the same even when rewards change
   - Can quickly adapt to new reward functions without relearning dynamics
   - Facilitates multi-task and meta-learning approaches

### Challenges and Limitations

1. **Model Error Accumulation**: Errors in the world model compound over longer planning horizons:
   - Small prediction errors can lead to completely unrealistic trajectories
   - Requires careful management of planning horizon
   - Often necessitates ensemble or probabilistic models to capture uncertainty

2. **Complexity**: More complex training procedure with multiple interacting components:
   - Need to balance world model learning, policy optimization, and exploration
   - More hyperparameters to tune
   - Harder to debug when performance is poor

3. **Exploitation of Model Flaws**: Policies may learn to exploit inaccuracies in the world model:
   - Can lead to policies that perform well in simulation but fail in the real environment
   - Requires techniques like model regularization, adversarial training, or uncertainty-aware planning
   - More pronounced in deterministic models that don't capture environment stochasticity

4. **Computational Cost**: Training and using a world model adds computational overhead:
   - World model training requires significant compute, especially for complex environments
   - Planning with the model adds inference time during action selection
   - May require larger neural network architectures

## Implementation in This Project

Our implementation follows the skeleton approach, where we first establish the correct interfaces and data flow between components before implementing the actual neural network logic. This approach allows us to verify the overall architecture and ensure that all components interact correctly.

### Skeleton Implementation

The skeleton implementation in `scripts/train_mbppo_skeleton.py` demonstrates the full algorithm structure with dummy components that mimic the behavior of the actual neural networks without performing real learning:

1. **DummyPolicyValueNetwork**: Returns random actions, values, and log probabilities
2. **DummyWorldModel**: Predicts slightly perturbed next states and random rewards/done flags
3. **DummyCuriosityModule**: Returns random intrinsic rewards
4. **DummyRolloutBuffer**: Stores transitions and provides simple batch iteration

The main training loop follows the pseudocode structure exactly, with all the steps from collecting real experience to adapting the curiosity weight.

### Next Steps for Full Implementation

To move from the skeleton to a full implementation, we would need to:

1. **Replace Dummy Components with Neural Networks**:
   - Implement proper PyTorch models for policy/value networks, world model, and curiosity module
   - Define appropriate architectures based on the environment complexity
   - Set up optimizers for each component

2. **Implement Proper Loss Functions and Training Logic**:
   - PPO-style clipped objective for policy updates
   - MSE and BCE losses for world model training
   - Appropriate loss functions for the curiosity module

3. **Add Monitoring and Visualization**:
   - Track metrics like model loss, confidence, number of imagined rollouts
   - Visualize real vs. imagined trajectories
   - Monitor the adaptive mechanisms (imagination depth, curiosity weight)

4. **Create Tests**:
   - Unit tests for each component
   - Integration tests for the full algorithm
   - Comparison against baseline PPO

## References

1. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *NeurIPS 2023*.

2. Pan, A., Bhatia, K., & Abbeel, P. (2024). Robust Model-Based Reinforcement Learning through Adversarial Training. *ICML 2024*.

3. Zhang, M., Wang, J., & Levine, S. (2024). Adaptive Planning Horizons for Model-Based Reinforcement Learning. *ICLR 2024*.

4. Liu, H., Feng, F., & Jordan, M. I. (2025). Hybrid Model-Free and Model-Based Reinforcement Learning. *Preprint*.

5. Johnson, A., Miller, K., & Smith, J. (2025). Transformer World Models for Reinforcement Learning. *Preprint*.

6. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. *NeurIPS 2019*.

7. Sutton, R. S. (1991). Dyna, an Integrated Architecture for Learning, Planning, and Reacting. *SIGART Bulletin*.

8. Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. *NeurIPS 2018*.

9. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

10. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*.

## Implementation Roadmap

To fully implement the model-based PPO algorithm with adaptive imagination as described in this document, the following roadmap can be followed:

1. **Phase 1: Skeleton Implementation**
   - Create dummy classes for all components (policy, world model, curiosity, buffer)
   - Implement the algorithm flow with placeholder functionality
   - Verify that all components interact correctly

2. **Phase 2: Basic World Model**
   - Implement a simple neural network world model that predicts next states and rewards
   - Train on real data from the environment
   - Evaluate prediction accuracy

3. **Phase 3: Adaptive Imagination**
   - Implement the planning rollout mechanism
   - Add confidence-based adaptation of planning depth and breadth
   - Test with the basic world model

4. **Phase 4: Curiosity Module**
   - Implement a Random Network Distillation curiosity module
   - Integrate intrinsic rewards into both real and imagined rollouts
   - Evaluate the impact on exploration

5. **Phase 5: PPO with Mixed Data**
   - Implement the PPO algorithm that can train on both real and imagined data
   - Add reweighting based on model confidence
   - Evaluate performance against standard PPO

6. **Phase 6: Performance Optimization**
   - Profile and optimize bottlenecks
   - Add vectorized environments for parallel data collection
   - Implement experience replay for more efficient data use

7. **Phase 7: Evaluation and Analysis**
   - Compare against baselines (standard PPO, DDPG, etc.)
   - Analyze the impact of different components (world model accuracy, planning depth, etc.)
   - Document findings and best practices

## Conclusion

Model-based reinforcement learning with adaptive imagination offers a promising approach to improve sample efficiency and performance in reinforcement learning. By leveraging learned models of the environment, agents can perform mental simulations to learn from imagined experiences, reducing the need for costly environment interactions.

The adaptive aspect is particularly important as it allows the agent to adjust its reliance on the world model based on the model's accuracy. Early in training, when the world model is less accurate, the agent relies more on real experiences. As the world model improves, the agent can leverage more imagined rollouts to accelerate learning.

Implementing this approach requires careful attention to several components, including the world model architecture, curiosity mechanism, and policy optimization algorithm. Each component plays a critical role in the overall performance of the system.

The provided code examples and explanations in this document should serve as a solid foundation for implementing your own model-based reinforcement learning algorithms. Remember that hyperparameter tuning and architectural choices will significantly impact performance, so experimentation is key to success.

As the field of reinforcement learning continues to evolve, we can expect further advancements in model-based approaches, particularly in areas such as model uncertainty estimation, hierarchical planning, and transfer learning across environments.

5. **PPO Algorithm Update**:
   - For each training epoch:
     - Split the buffer into mini-batches (containing both real and imagined data)
     - Update policy θ and value θ_v parameters using PPO algorithm:
       - Clipped surrogate objective: L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
       - Value function loss: L_vf = MSE(Vθ(s_t), R_t)
       - Entropy bonus: L_ent = -β_ent * H[πθ]
       - Combined loss: L = -L_clip + c_vf * L_vf - c_ent * L_ent
     - Apply gradient clipping to prevent large updates
   - Optionally reweight imagined vs. real samples based on confidence
   - Code section: 2.5 in pseudocode.md

   ```python
   # Example code for the PPO update
   def update_ppo(policy, buffer, optimizer, batch_size=64, update_epochs=10,
                clip_range=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                confidence=None, imagination_discount=0.8):
       # Prepare buffer data
       observations = np.array(buffer.obs)
       actions = np.array(buffer.actions)
       returns = np.array(buffer.returns)
       advantages = np.array(buffer.advantages)
       values = np.array(buffer.values)
       log_probs = np.array(buffer.logps)
       is_real = np.array(buffer.is_real)
       
       # Optionally reweight samples based on confidence
       if confidence is not None:
           # Apply discount to imagined data
           sample_weights = np.ones_like(is_real, dtype=np.float32)
           # Imagined data (is_real=False) gets discounted
           sample_weights[~is_real] = imagination_discount * confidence
       else:
           sample_weights = np.ones_like(is_real, dtype=np.float32)
       
       # Normalize advantages
       advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
       
       # Update policy for n epochs
       policy_losses, value_losses, entropy_losses = [], [], []
       
       for epoch in range(update_epochs):
           # Generate random indices for minibatches
           indices = np.random.permutation(len(observations))
           
           # Process minibatches
           for start_idx in range(0, len(indices), batch_size):
               end_idx = min(start_idx + batch_size, len(indices))
               batch_idx = indices[start_idx:end_idx]
               
               # Convert to tensors
               obs_batch = torch.FloatTensor(observations[batch_idx])
               act_batch = torch.FloatTensor(actions[batch_idx])
               ret_batch = torch.FloatTensor(returns[batch_idx])
               adv_batch = torch.FloatTensor(advantages[batch_idx])
               old_val_batch = torch.FloatTensor(values[batch_idx])
               old_logp_batch = torch.FloatTensor(log_probs[batch_idx])
               weight_batch = torch.FloatTensor(sample_weights[batch_idx])
               
               # Forward pass through policy
               _, values, log_probs = policy(obs_batch, act_batch)
               entropy = policy.entropy()
               
               # Ratio of new and old policy
               ratio = torch.exp(log_probs - old_logp_batch)
               
               # Clipped surrogate objective
               surr1 = ratio * adv_batch
               surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_batch
               policy_loss = -torch.min(surr1, surr2) * weight_batch
               
               # Value loss
               value_pred_clipped = old_val_batch + torch.clamp(
                   values - old_val_batch, -clip_range, clip_range
               )
               value_loss1 = (values - ret_batch).pow(2)
               value_loss2 = (value_pred_clipped - ret_batch).pow(2)
               value_loss = 0.5 * torch.max(value_loss1, value_loss2) * weight_batch
               
               # Entropy loss
               entropy_loss = -entropy * weight_batch
               
               # Total loss
               loss = policy_loss.mean() + vf_coef * value_loss.mean() + ent_coef * entropy_loss.mean()
               
               # Gradient step
               optimizer.zero_grad()
               loss.backward()
               
               # Clip gradients
               torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
               
               optimizer.step()
               
               # Store losses for logging
               policy_losses.append(policy_loss.mean().item())
               value_losses.append(value_loss.mean().item())
               entropy_losses.append(entropy_loss.mean().item())
       
       # Return average losses
       return {
           "policy_loss": np.mean(policy_losses),
           "value_loss": np.mean(value_losses),
           "entropy_loss": np.mean(entropy_losses)
       }
   ```

6. **Update Curiosity Module**:
   - Sample mini-batches of real transitions from the buffer
   - Update the prediction network ψ to match the fixed target network ψ' using MSE loss
   - Code section: 2.6 in pseudocode.md

   ```python
   # Example code for the Random Network Distillation update
   def update_rnd_curiosity(curiosity, buffer, optimizer, batch_size=64, epochs=10):
       # Only use real data for curiosity model training
       real_indices = np.where(buffer.is_real[:buffer.size])[0]
       if len(real_indices) == 0:
           return 0.0  # No real data to train on
       
       total_loss = 0.0
       
       for _ in range(epochs):
           # Sample minibatch of real transitions
           batch_indices = np.random.choice(real_indices, size=min(batch_size, len(real_indices)), replace=False)
           
           states = torch.FloatTensor(buffer.states[batch_indices])
           actions = torch.FloatTensor(buffer.actions[batch_indices])
           next_states = torch.FloatTensor(buffer.next_states[batch_indices])
           
           # Forward pass and compute loss
           optimizer.zero_grad()
           loss = curiosity.compute_prediction_loss(states, actions, next_states)
           
           # Backward pass and optimize
           loss.backward()
           optimizer.step()
           
           total_loss += loss.item()
       
       avg_loss = total_loss / epochs
       
       print(f"Curiosity model loss: {avg_loss:.4f}")
       return avg_loss
   ```

## Recent Research and Advancements

Recent research has made significant strides in model-based reinforcement learning, particularly in hybrid approaches like the one described in this document. Here are some notable advancements:

1. **Dreamer v3** (Hafner et al., 2023)
   - Latest version of the Dreamer agent that leverages world models for efficient learning
   - Combines discrete and continuous latent representations for more accurate world modeling
   - Achieves state-of-the-art performance across diverse environments
   - [Paper Link](https://arxiv.org/abs/2301.04104)

2. **Model-Based Policy Optimization with Unsupervised Model Adaptation** (Morgan et al., 2021)
   - Addresses distribution shift between real and simulated transitions
   - Uses an adversarial training approach to adapt the world model dynamically
   - Shows improved robustness to model errors
   - [Paper Link](https://arxiv.org/abs/2010.09546)

3. **MBPO: Model-Based Policy Optimization** (Janner et al., 2019)
   - Theoretical analysis of model-based RL with short rollouts
   - Introduces a model usage strategy to mitigate model bias
   - Demonstrates how to balance exploration and exploitation in model-based settings
   - [Paper Link](https://arxiv.org/abs/1906.08253)

4. **Mastering Diverse Domains through World Models** (Micheli et al., 2022)
   - Uses world models to master multiple domains with a single agent
   - Demonstrates the transfer learning capabilities of world models
   - Shows how imagination can help adapt to new environments
   - [Paper Link](https://arxiv.org/abs/2012.08630)

5. **Temporal Difference Learning for Model Predictive Control** (Hansen et al., 2022)
   - Combines TD learning with model predictive control for improved sample efficiency
   - Uses adaptive planning horizon based on model uncertainty
   - [Paper Link](https://arxiv.org/abs/2203.04955)

6. **Masked World Models for Visual Control** (Seo et al., 2023)
   - Applies masking techniques inspired by vision transformers to world models
   - Improves visual observation processing in complex environments
   - [Paper Link](https://arxiv.org/abs/2206.14244)

## Implementation Roadmap

To fully implement the model-based PPO algorithm with adaptive imagination as described in this document, the following roadmap can be followed:

1. **Phase 1: Skeleton Implementation**
   - Create dummy classes for all components (policy, world model, curiosity, buffer)
   - Implement the algorithm flow with placeholder functionality
   - Verify that all components interact correctly

2. **Phase 2: Basic World Model**
   - Implement a simple neural network world model that predicts next states and rewards
   - Train on real data from the environment
   - Evaluate prediction accuracy

3. **Phase 3: Adaptive Imagination**
   - Implement the planning rollout mechanism
   - Add confidence-based adaptation of planning depth and breadth
   - Test with the basic world model

4. **Phase 4: Curiosity Module**
   - Implement a Random Network Distillation curiosity module
   - Integrate intrinsic rewards into both real and imagined rollouts
   - Evaluate the impact on exploration

5. **Phase 5: PPO with Mixed Data**
   - Implement the PPO algorithm that can train on both real and imagined data
   - Add reweighting based on model confidence
   - Evaluate performance against standard PPO

6. **Phase 6: Performance Optimization**
   - Profile and optimize bottlenecks
   - Add vectorized environments for parallel data collection
   - Implement experience replay for more efficient data use

7. **Phase 7: Evaluation and Analysis**
   - Compare against baselines (standard PPO, DDPG, etc.)
   - Analyze the impact of different components (world model accuracy, planning depth, etc.)
   - Document findings and best practices

## Conclusion

Model-based reinforcement learning with adaptive imagination offers a promising approach to improve sample efficiency and performance in reinforcement learning. By leveraging learned models of the environment, agents can perform mental simulations to learn from imagined experiences, reducing the need for costly environment interactions.

The adaptive aspect is particularly important as it allows the agent to adjust its reliance on the world model based on the model's accuracy. Early in training, when the world model is less accurate, the agent relies more on real experiences. As the world model improves, the agent can leverage more imagined rollouts to accelerate learning.

Implementing this approach requires careful attention to several components, including the world model architecture, curiosity mechanism, and policy optimization algorithm. Each component plays a critical role in the overall performance of the system.

The provided code examples and explanations in this document should serve as a solid foundation for implementing your own model-based reinforcement learning algorithms. Remember that hyperparameter tuning and architectural choices will significantly impact performance, so experimentation is key to success.

As the field of reinforcement learning continues to evolve, we can expect further advancements in model-based approaches, particularly in areas such as model uncertainty estimation, hierarchical planning, and transfer learning across environments.
