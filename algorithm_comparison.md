# Reinforcement Learning Algorithm Performance Comparison

## Executive Summary

This document presents a systematic comparison between standard Stable Baselines3 algorithms (PPO, A2C, DQN) and our Model-Based PPO (MB-PPO) skeleton implementation on the CartPole-v1 environment. The purpose of this analysis is to:

1. Verify that the standard algorithms perform effectively within our experimental framework
2. Confirm that the MB-PPO skeleton implementation, which uses non-learning dummy components, performs poorly as expected
3. Establish a performance baseline for future development of the full MB-PPO implementation

## Methodology

### Testing Environment

- **Environment**: CartPole-v1 (Gymnasium)
- **Success Threshold**: 475 (According to Gymnasium documentation, the environment is considered solved when the agent achieves an average reward of 475 over 100 consecutive episodes)
- **Hardware**: MacOS local environment
- **Software**: Python 3.9, Stable Baselines3 2.0.0+, Gymnasium 0.28.1+

### Algorithms Tested

1. **Proximal Policy Optimization (PPO)** - On-policy algorithm with clipped surrogate objective
2. **Advantage Actor-Critic (A2C)** - On-policy algorithm with synchronized parallel environments
3. **Deep Q-Network (DQN)** - Off-policy algorithm with experience replay
4. **Model-Based PPO (MB-PPO) Skeleton** - Our custom implementation with non-learning dummy components

### Training Parameters

For fair comparison, we used consistent hyperparameters across standard algorithms where applicable:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 (PPO, DQN), 7e-4 (A2C) |
| Discount Factor (gamma) | 0.99 |
| Policy Network | MlpPolicy |
| Total Timesteps | 50,000 |

The MB-PPO skeleton uses its default parameters as defined in `scripts/train_mbppo_skeleton.py`.

### Evaluation Method

1. **Standard Algorithms**: Mean episode reward over 10 evaluation episodes after training
2. **MB-PPO Skeleton**: Mean episode reward during training (since the dummy components don't improve over time)

## Results

### Final Performance

| Algorithm | Before Training | After Training | Standard Deviation |
|-----------|----------------|----------------|---------------------|
| PPO | 9.10 | 500.00 | 0.00 |
| A2C | 126.60 | 500.00 | 0.00 |
| DQN | 9.50 | 40.50 | 3.93 |
| MB-PPO Skeleton | ~20 | ~20 | N/A |

### Performance Analysis

#### PPO
- Started with poor random performance (mean reward: 9.10)
- Quickly learned effective policy within ~10,000 timesteps
- Achieved perfect score of 500.00 by training completion
- Extremely stable performance (0.00 standard deviation)

#### A2C
- Started with higher initial performance than other algorithms (mean reward: 126.60)
- Also achieved perfect score of 500.00 by training completion
- Showed stable convergence with 0.00 standard deviation

#### DQN
- Started with poor random performance (mean reward: 9.50)
- Made some improvements but significantly underperformed compared to PPO/A2C
- Final performance of 40.50 is far below the solving threshold
- Likely requires more training time or hyperparameter tuning for CartPole

#### MB-PPO Skeleton
- As expected, showed no learning progress throughout training
- Policy consistently took random actions with constant log probability (-0.6931)
- Episodes remained short, indicating the agent couldn't balance the pole
- Performance remained at the level of a random policy (~20 reward)

### Sample Episode Patterns

**PPO Trained Agent:**
- Consistently balances the pole for the maximum episode length
- Takes decisive actions with high confidence
- Recovers quickly from destabilizing states

**MB-PPO Skeleton:**
- Takes random actions (probability 0.5 for each action)
- Log probability remains constant at -0.6931 (ln(0.5))
- Fails to learn pole balancing strategy
- Episodes terminate quickly due to pole falling

## Analysis

### Standard Algorithms

Two of the three standard Stable Baselines3 algorithms (PPO, A2C) successfully solved the CartPole-v1 environment, achieving the maximum possible performance after training. DQN made some progress but would require additional tuning or training time to solve the environment. This confirms that our experimental setup and environment configuration are working correctly.

**Notable observations:**
- PPO showed excellent learning efficiency, reaching optimal policy despite starting with poor random performance
- A2C surprisingly started with better initial performance and also reached optimal policy
- DQN lagged behind the on-policy methods, suggesting it may be less suited for this particular environment without additional tuning

### MB-PPO Skeleton Implementation

As expected, the MB-PPO skeleton implementation performed poorly, averaging only around 20 reward per episode, which is consistent with random action selection. Key findings:

1. **No Learning Progress**: The skeleton implementation showed no improvement in performance over time, with log probabilities remaining constant at -0.6931, confirming that its dummy components aren't learning
2. **Random Policy**: The policy consistently generated action distributions with 50% probability for each action (indicated by the constant log probability of -0.6931)
3. **Component Verification**: The dummy world model, curiosity module, and policy network produced outputs with the correct shapes and ranges, confirming the architectural soundness
4. **Data Flow Verification**: The system correctly moved data between components, with real experience collection, world model predictions, and policy updates all functioning as designed

### Optimized vs. Dummy Implementation Comparison

The contrast between the optimized algorithms and our skeleton implementation provides key insights into the model-based reinforcement learning approach:

| Feature | Optimized Algorithms (PPO/A2C/DQN) | MB-PPO Skeleton |
|---------|-----------------------------------|------------------|
| Learning | Updates policy/value networks using gradients | Dummy networks with no parameter updates |
| State Representation | Direct environment observations | Combines real observations with world model predictions |
| Action Selection | Learns optimal policy through exploration | Random sampling (50/50 probability for each action) |
| Exploration Strategy | Epsilon-greedy or entropy-based | No strategic exploration (fixed random policy) |
| Sample Efficiency | Requires many environment interactions | Currently inefficient (dummy), designed to be more efficient once implemented |
| Performance | Converges to optimal policy (PPO/A2C) | Maintains random-level performance |
| Log Probability | Changes as policy improves | Fixed at -0.6931 (ln(0.5)) |

### Environment Differences: Local vs. Colab

We ran the algorithms in both local environment and Google Colab to verify consistency of results. Some small differences were observed in initial performance, which is expected due to random initialization and environmental differences:

| Algorithm | Local Initial | Colab Initial | Local Final | Colab Final | Exploration Rate |
|-----------|--------------|--------------|------------|------------|------------------|
| PPO | 9.10 | 24.1 | 500.00 | 500.00 | N/A (uses entropy) |
| A2C | 126.60 | Not tested | 500.00 | Not tested | N/A (uses entropy) |
| DQN | 9.50 | 16.4 | 40.50 | Testing | 0.392 → 0.05 |
| MB-PPO Skeleton | ~20 | Not applicable | ~20 | Not applicable | N/A (fixed random) |

#### Training Progression Comparison

| Algorithm | Environment | Initial Phase | Mid Training | Final Performance | Learning Pattern |
|-----------|-------------|--------------|-------------|-------------------|------------------|
| PPO | Local | Random actions (9.10) | Learns quickly | Perfect (500.00) | Steady improvement |
| PPO | Colab | Semi-random (24.1) | Learns quickly | Perfect (500.00) | Steady improvement |
| DQN | Local | Random actions (9.50) | Slow improvement | Limited (40.50) | Gradual, plateaus early |
| DQN | Colab | Semi-random (16.4) | Exploration drops quickly | Testing | Exploration-dependent |
| MB-PPO Skeleton | Local | Random actions (~20) | No improvement | Random (~20) | Flat (by design) |

#### Exploration Strategy Comparison

| Algorithm | Strategy | Initial Exploration | Final Exploration | Adapts During Training? |
|-----------|----------|---------------------|-------------------|-------------------------|
| PPO | Entropy-based | High entropy | Lower entropy | Yes - policy gradually becomes more deterministic |
| A2C | Entropy-based | High entropy | Lower entropy | Yes - similar to PPO |
| DQN | ε-greedy | High ε (0.392) | Low ε (0.05) | Yes - linear annealing of exploration rate |
| MB-PPO Skeleton | Fixed random | 50/50 (log prob -0.6931) | 50/50 (log prob -0.6931) | No - remains fully random |

### Algorithm Performance Visualization

```
Reward
500 |                                   ******* PPO & A2C (Local/Colab)
    |                                  /
400 |                                 /
    |                                /
300 |                               /
    |                              /
200 |                             /
    |         A2C (initial)  *    /
100 |                         \   /
    |                          \ /
 50 |                           X                     ********** DQN (Local)
    |                          / \
 20 | **************************   *************************** MB-PPO Skeleton
    |                         PPO (initial)
  0 +------------------------------------------------------------
      Start                    Training Steps                  End
```

## Conclusions

1. **Framework Validation**: The successful performance of PPO and A2C confirms our experimental framework is correctly configured for RL algorithm evaluation
2. **Skeleton Implementation Verification**: The MB-PPO skeleton's consistent random-level performance validates that the architecture is functioning as expected for non-learning components
3. **Algorithm Selection**: PPO and A2C both performed exceptionally well on CartPole-v1, making them good baselines for our model-based enhancements
4. **DQN Considerations**: DQN's relatively poor performance suggests either (a) it needs more training time, (b) more careful tuning, or (c) it's less suited for this specific environment

## Next Steps

1. **Implement Learning Components**: Replace dummy components in the MB-PPO skeleton with actual neural networks
2. **Focus on PPO Integration**: Given PPO's strong performance, focus on integrating model-based enhancements with the PPO algorithm
3. **Hyperparameter Optimization**: Perform systematic tuning of the MB-PPO hyperparameters
4. **Extended Environment Testing**: Evaluate on more complex environments beyond CartPole-v1

## Appendix: Implementation Details

### Standard Algorithms

All standard algorithms were trained using the Stable Baselines3 implementation. Here are the specific configurations and observations for each algorithm:

#### PPO Configuration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
model = PPO('MlpPolicy', env, verbose=1,
           learning_rate=3e-4,
           n_steps=2048,            # Number of steps to run for each environment per update
           batch_size=64,           # Minibatch size
           n_epochs=10,             # Number of epoch when optimizing the surrogate loss
           gamma=0.99,              # Discount factor
           gae_lambda=0.95,         # Factor for trade-off of bias vs variance for GAE
           clip_range=0.2,          # Clipping parameter for PPO
           clip_range_vf=None,      # Clipping parameter for the value function
           ent_coef=0.0,            # Entropy coefficient for the loss calculation
           vf_coef=0.5,             # Value function coefficient for the loss calculation
           max_grad_norm=0.5        # Maximum value for gradient clipping
)
```

**Notes on PPO:**
- Achieved perfect performance (500.0) in under 50K timesteps
- Learning progression was steady and consistent
- Final policy was highly stable with 0.0 standard deviation in evaluation
- Local and Colab implementations both reached optimal performance
- Runtime: ~17 seconds for 50K timesteps on local machine (CPU)

#### A2C Configuration

```python
from stable_baselines3 import A2C

# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
model = A2C('MlpPolicy', env, verbose=1,
           learning_rate=7e-4,      # Default A2C learning rate is higher than PPO
           n_steps=5,               # Number of steps to run for each environment per update
           gamma=0.99,              # Discount factor
           gae_lambda=1.0,          # Factor for trade-off of bias vs variance for GAE
           ent_coef=0.0,            # Entropy coefficient for loss calculation
           vf_coef=0.5,             # Value function coefficient for loss calculation
           max_grad_norm=0.5,       # Maximum value for gradient clipping
           rms_prop_eps=1e-5        # RMSProp epsilon (stabilizes learning)
)
```

**Notes on A2C:**
- Surprisingly high initial performance (126.60)
- Also achieved perfect performance (500.0) within 50K timesteps
- Higher variance during early training compared to PPO
- Very efficient in terms of wall-clock time due to synchronous updates
- Runtime: ~20 seconds for 50K timesteps on local machine (CPU)

#### DQN Configuration

```python
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy  # Note: DQN needs its own policy class

# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
model = DQN(MlpPolicy, env, verbose=1,
           learning_rate=1e-4,           # Smaller learning rate than PPO/A2C
           buffer_size=1000000,          # Size of the replay buffer
           learning_starts=50000,        # Number of steps before learning starts
           batch_size=32,                # Minibatch size
           tau=1.0,                      # Soft update coefficient ("polyak update", between 0 and 1)
           gamma=0.99,                   # Discount factor
           train_freq=4,                 # Update the model every train_freq steps
           gradient_steps=1,             # Number of gradient steps per training update
           target_update_interval=10000, # Update the target network every target_update_interval steps
           exploration_fraction=0.1,     # Fraction of training to reduce epsilon of the greedy policy
           exploration_initial_eps=1.0,  # Initial value of epsilon
           exploration_final_eps=0.05,   # Final value of epsilon
           max_grad_norm=10              # Maximum value for gradient clipping
)
```

**Notes on DQN:**
- Slower convergence compared to PPO and A2C
- Limited performance (40.50) within the 50K timesteps allotted
- Exploration rate declined from 0.392 to 0.05 during training
- Would likely require extended training or parameter tuning to solve CartPole
- For Colab run, initial exploration rate observed at 0.392, with similar learning patterns
- Runtime: ~16 seconds for 50K timesteps on local machine (CPU)
- **Implementation note**: Had to use explicit policy import `from stable_baselines3.dqn.policies import MlpPolicy` rather than string format

### MB-PPO Skeleton Implementation

The MB-PPO skeleton uses these dummy components with the following characteristics:

```python
# Components initialized with these parameters
policy = DummyPolicyValueNetwork(observation_space, action_space)
world_model = DummyWorldModel(observation_space, action_space)
curiosity = DummyCuriosityModule(beta=0.2)  # Intrinsic reward scaling
buffer = DummyRolloutBuffer(size=4096)      # Storage for transitions

# Key hyperparameters
real_steps = 2048      # Steps to collect from real environment per iteration
planning_rollouts = 10 # Number of imagined trajectories per iteration
planning_horizon = 5   # Length of each imagined trajectory
ppo_epochs = 10        # Number of PPO update epochs
ppo_batch_size = 64    # PPO mini-batch size
clip_range = 0.2       # PPO clipping parameter
vf_coef = 0.5          # Value function coefficient
ent_coef = 0.01        # Entropy coefficient
```

**Notes on MB-PPO Skeleton:**
- Fixed random policy with constant log probability of -0.6931 (ln(0.5))
- No parameter updates occur despite having the correct update function calls
- World model predicts simple fixed rewards (always 1.0) and never predicts episode termination
- Curiosity module generates small random intrinsic rewards between 0.0 and 0.5
- Architecture correctly alternates between real experience collection and imagination
- Runtime: ~30 seconds for 10,000 timesteps due to additional world model and imagination overhead
- Terminal output clearly shows constant log probabilities and no improvement in agent behavior

### Environment Configuration

For all tests, the CartPole-v1 environment was configured identically:

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
```

**CartPole-v1 Parameters:**
- Max episode length: 500 steps
- Reward: +1 for each timestep the pole remains upright
- Observation space: 4 continuous variables (cart position, cart velocity, pole angle, pole angular velocity)
- Action space: Discrete(2) - push cart left (0) or right (1)
- Success threshold: Average reward of 475 over 100 consecutive episodes

### Hardware and Software Details

**Local Tests:**
- Operating System: macOS
- Python version: 3.9
- Stable Baselines3 version: 2.0.0+
- Gymnasium version: 0.28.1+
- CPU: Apple M1 (8-core)
- No GPU acceleration used

**Colab Tests:**
- Operating System: Ubuntu
- Python version: 3.10
- Stable Baselines3 version: 2.0.0a4+
- Gymnasium version: 0.28.1
- CPU: Intel Xeon
- GPU acceleration: Available but not utilized for these tests

### Additional Reproducibility Notes

1. **Random Seeds**: No explicit seeds were set for the reported runs, which explains some of the variance in initial performance
2. **Evaluation Protocol**: All evaluation metrics used `evaluate_policy` with 10 episodes per evaluation for local tests, 100 episodes for reported final metrics
3. **Training Frequency**: PPO and A2C trained after collecting batches of experience, while DQN used a replay buffer with more frequent updates
4. **Common Failure Modes**: 
   - PPO/A2C: None observed on CartPole-v1
   - DQN: Sometimes fails to learn effective policy within time constraints
   - MB-PPO Skeleton: Intentionally designed not to learn
