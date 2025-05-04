
May 3, 2025 v15: Model-Based Planning with Adaptive Imagination

### 1. INITIALISATION

# Initialise policy network πθ (actor) and value network Vθ (critic)          # shared encoder optional
# Initialise world‑model  ŵφ (predicts Δstate, reward, done)                 # stochastic dynamics model
# Initialise curiosity module Cψ (e.g. RND)                                  # optional – set β_init = 0 to disable
# Initialise RolloutBuffer B  (stores both real and imagined transitions)
# Initialise optimisers:
#     - optimiser_actor_critic  (learning_rate = 3e‑4)
#     - optimiser_world_model   (learning_rate = 1e‑3)
#     - optimiser_curiosity     (learning_rate = 3e‑4)                       # if curiosity used
# Set hyper‑parameters
#     - n_real_steps          : 2048             # env steps per outer loop
#     - planning_horizon      : 5                # steps per imagined rollout
#     - min_planning_rollouts : 10
#     - max_planning_rollouts : 100
#     - confidence_thresh     : 0.05             # ↓ MSE → ↑ confidence
#     - γ                     : 0.99             # discount
#     - λ                     : 0.95             # GAE
#     - clip_range            : 0.2              # PPO clip
#     - n_epochs              : 10, batch_size : 64
#     - β_init                : 0.2, β_adapt_rate : 0.01                    # curiosity weight
#     - max_grad_norm         : 0.5

### 2. MAIN LOOP — iterate until `total_timesteps`

### 2.1  COLLECT REAL EXPERIENCE (on‑policy)
# B.reset()
# obs = env.reset()
# for step = 1 … n_real_steps:
#     with πθ in eval mode:
#         action, value, logp = πθ(obs)
#     next_obs, ext_reward, done, _ = env.step(action)
#
#     # Curiosity (optional)
#     intr_reward = β * Cψ.intrinsic_reward(obs, action, next_obs)
#
#     B.add(obs, action, ext_reward + intr_reward, value, logp,
#           ext_reward, intr_reward, done, is_real = True)
#
#     obs = next_obs if not done else env.reset()

### 2.2  UPDATE WORLD MODEL  ŵφ   &  COMPUTE CONFIDENCE
# sample (s, a, r, s′) mini‑batch from **real** part of B
# loss_model = MSE(ŵφ(s,a).state,   s′)          +
#              MSE(ŵφ(s,a).reward,  r)          +
#              BCE(ŵφ(s,a).done,    done)
# optimiser_world_model.step(∇φ loss_model)
#
# confidence = exp(−loss_model)                  # ∈(0,1], higher = better

### 2.3  ADAPTIVE IMAGINATION (PLANNING)
# num_rollouts = interpolate(confidence,
#                            low = confidence_thresh,
#                            high = 1.0,
#                            out_low = min_planning_rollouts,
#                            out_high = max_planning_rollouts)
# for rollout = 1 … num_rollouts:
#     sim_state = B.sample_recent_state()
#     for h = 1 … planning_horizon:
#         with πθ in eval mode:
#             a_sim, v_sim, logp_sim = πθ(sim_state)
#
#         ŝ_next, r̂, d̂ = ŵφ(sim_state, a_sim)          # predict
#         intr_sim  = β * Cψ.intrinsic_reward(sim_state, a_sim, ŝ_next)
#
#         B.add(sim_state, a_sim, r̂ + intr_sim, v_sim, logp_sim,
#               r̂, intr_sim, d̂, is_real = False)
#
#         sim_state = ŝ_next
#         if d̂: break

### 2.4  ADVANTAGE & RETURN COMPUTATION
# with πθ in eval mode:
#     last_value = Vθ(obs)
# B.compute_returns_and_advantages(last_value, γ, λ)     # uses combined reward

### 2.5  PPO‑STYLE POLICY / VALUE UPDATE
# for epoch = 1 … n_epochs:
#     for batch in B.iterate(batch_size):
#         ratio      = exp(πθ.logp(batch) − batch.old_logp)
#         surr1      = ratio * batch.adv
#         surr2      = clip(ratio, 1−clip_range, 1+clip_range) * batch.adv
#         policy_loss= −mean(min(surr1, surr2))
#
#         value_pred = Vθ(batch.obs)
#         value_loss = MSE(value_pred, batch.ret)
#
#         entropy    = πθ.entropy(batch)
#         entropy_loss = −mean(entropy)
#
#         total_loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss
#
#         optimiser_actor_critic.step(∇θ total_loss, clip=max_grad_norm)

### 2.6  CURIOSITY MODULE UPDATE (if β>0)
# Cψ.update(B.real_and_imagined())

### 2.7  ADAPT β  (CURIOSITY WEIGHT)
# avg_ext = moving_average(B.real_ext_rewards())
# trend   = slope(avg_ext)
# if trend < low_threshold:  β = min(1.0, β * (1 + β_adapt_rate))
# elif trend > high_threshold: β = max(0.0, β * (1 − β_adapt_rate))

### 2.8  LOGGING
# logger.record({
#     "confidence": confidence,
#     "imagined_rollouts": num_rollouts,
#     "β": β,
#     "policy_loss": policy_loss,
#     "value_loss": value_loss,
#     "model_loss": loss_model,
#     "mean_real_reward": mean(B.real_ext_rewards())
# })






## Questions
Integrating this into the existing Stable Baselines3 setup requires careful consideration because it's not a standard SB3 algorithm. Here are my initial thoughts and questions to clarify the implementation goals:

Core Algorithm Integration:

Custom Algorithm vs. Modification? The pseudocode describes a fundamentally new learning loop involving a world model and imagined rollouts. Would you prefer to:
(A) Implement a new Custom Algorithm: Create a new class (e.g., ModelBasedPPO) potentially inheriting from sb3.common.base_class.BaseAlgorithm or sb3.common.on_policy_algorithm.OnPolicyAlgorithm? This allows for a clean implementation tailored to the pseudocode but is more work.
(B) Modify Existing SB3 PPO: Try to hook the world model and imagination steps into the existing PPO's collect_rollouts and train methods? This might be faster initially but could become messy and hard to maintain.
My Recommendation: Option (A) seems cleaner given the significant differences from standard PPO.
Network Architecture: The pseudocode mentions separate policy/value networks (πθ, Vθ), a world model (ŵφ), and a curiosity module (Cψ), with an optional shared encoder.
Should we enforce separate networks as described, or can we leverage SB3's combined ActorCritic policies (like ActorCriticPolicy) and add the world model and curiosity networks separately?
For the CartPole environment (low-dimensional state), what kind of networks should we use? Simple Multi-Layer Perceptrons (MLPs) for all components seem appropriate initially.
Component Specifics:

World Model (ŵφ):
Should it predict the change in state (Δstate) or the absolute next state (s' )? Predicting the next state directly might be simpler for MLP models.
What should the loss function weights be for state, reward, and done predictions? The pseudocode uses simple MSE and BCE, implying equal weighting.
Curiosity Module (Cψ):
Is RND (Random Network Distillation) the desired method, or was it just an example? If RND, we'll need to implement its specific target/predictor network structure.
Should curiosity be enabled by default, or configurable? (The pseudocode suggests starting with β_init = 0 disables it initially).
Adaptive Imagination:
The confidence = exp(−loss_model) calculation is simple. Is this sufficient?
How exactly should the interpolate function map confidence (between confidence_thresh and 1.0) to the number of rollouts (between min_planning_rollouts and max_planning_rollouts)? Linear interpolation?
Data Buffer (RolloutBuffer B):
We need a buffer that handles both real and imagined transitions, storing associated data like is_real, intrinsic/extrinsic rewards separately. SB3's RolloutBuffer is designed for real, on-policy data. We'll likely need a custom buffer class. Does that sound right?
Adaptive Curiosity Weight (β):
The adaptation logic uses the slope of a moving_average. What window size should be used for the moving average? How should the slope be calculated (e.g., linear regression over the window)? What are suitable low_threshold and high_threshold values for the trend? These might require tuning.
Project Structure:

How should we organize the new code? I suggest:
src/algorithms/model_based_ppo/: Core algorithm logic (ModelBasedPPO class).
src/models/world_model.py: WorldModel class.
src/models/curiosity.py: CuriosityModule class (e.g., RNDModule).
src/buffers/mixed_rollout_buffer.py: Custom buffer class.
scripts/train_mbppo.py: New training script.
tests/test_mbppo.py: New test file.