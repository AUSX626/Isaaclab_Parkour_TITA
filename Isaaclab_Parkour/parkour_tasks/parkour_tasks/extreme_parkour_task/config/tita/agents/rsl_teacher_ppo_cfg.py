from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlPpoAlgorithmCfg
)
from isaaclab.utils import configclass

TITA_NUM_PROP = 39
TITA_NUM_PRIV_LATENT = 21

@configclass
class TitaParkourTeacherPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "tita_parkour"
    empirical_normalization = False
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims = [128, 64, 32],
        priv_encoder_dims = [64, 20],
        activation="elu",
        actor = ParkourRslRlActorCfg(
            class_name = "Actor",
            num_prop=TITA_NUM_PROP,
            num_priv_latent=TITA_NUM_PRIV_LATENT,
            state_history_encoder = ParkourRslRlStateHistEncoderCfg(
                class_name = "StateHistoryEncoder",
                num_prop=TITA_NUM_PROP,
                num_priv_latent=TITA_NUM_PRIV_LATENT,
            )
        )
    )
    estimator = ParkourRslRlEstimatorCfg(
            num_prop=TITA_NUM_PROP,
            num_priv_latent=TITA_NUM_PRIV_LATENT,
            hidden_dims = [128, 64]
    )
    depth_encoder = None
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate = 2.e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0.0, 0.1, 2000.0, 3000.0],
    )

