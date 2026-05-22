"""
instinct_rl PPO config for TITA flat ground velocity tracking.

Key differences from G1 AMP config:
- Pure PPO (class_name="PPO", no WasabiPPO/discriminator)
- Simple MLP actor-critic (no MoE, no depth encoder)
- Fewer iterations (10K for initial testing)
"""

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class TitaPolicyCfg(InstinctRlActorCriticCfg):
    """Simple MLP actor-critic for TITA (no encoder, no MoE).

    No depth camera in Phase 1, so no encoder needed.
    """

    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"


@configclass
class TitaPPOAlgoCfg(InstinctRlPpoAlgorithmCfg):
    """Standard PPO algorithm config (no discriminator)."""

    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class TitaFlatPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    """Runner config for TITA flat velocity tracking."""

    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 100
    experiment_name = "tita_flat_velocity"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = TitaPolicyCfg()
    algorithm = TitaPPOAlgoCfg()
