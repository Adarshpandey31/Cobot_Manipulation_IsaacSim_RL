import gymnasium as gym

from . import agents


gym.register(
    id="Template-Cobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cobot_env_cfg:CobotEnvCfg",
    },
)

gym.register(
    id="Template-Cobot-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:CobotCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CobotLiftCubePPORunnerCfg",
    },
)