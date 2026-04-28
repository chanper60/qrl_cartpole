from __future__ import annotations
import os
import time
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AutoresetMode
from torch.utils.tensorboard import SummaryWriter

from ..agents.base_agent import BaseAgent
from ..utils.replay_buffer import ReplayBuffer


def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    return max((end - start) / duration * t + start, end)


class Trainer:
    """Agent-agnostic DQN training loop.

    Gymnasium 1.3.0 notes:
    - SyncVectorEnv uses AutoresetMode.SAME_STEP so infos["final_obs"] is
      populated on every episode end (true last obs before auto-reset).
    - Episode stats move into infos["final_info"]["episode"] with mask
      infos["_final_info"].
    - done in replay buffer = terminations only (not truncations), matching
      CleanRL handle_timeout_termination=False.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        run_name: str,
        seed: int,
        output_dir: str = "runs",
    ) -> None:
        self.agent = agent
        self.env_cfg = env_cfg
        self.cfg = training_cfg
        self.run_name = run_name
        self.seed = seed

        self.runs_dir = os.path.join(output_dir, run_name)
        self.ckpt_dir = os.path.join(self.runs_dir, "checkpoints")
        self.video_dir = os.path.join(self.runs_dir, "videos")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        self.writer = SummaryWriter(self.runs_dir)

    def _make_env(self, idx: int):
        def thunk():
            capture = self.env_cfg.get("capture_video", False) and idx == 0
            env = gym.make(
                self.env_cfg["env_id"],
                render_mode="rgb_array" if capture else None,
            )
            if capture:
                env = gym.wrappers.RecordVideo(
                    env, self.video_dir,
                    episode_trigger=lambda ep: ep % 1000 == 0,
                    name_prefix="train",
                )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(self.seed + idx)
            return env
        return thunk

    def train(self, start_step: int = 0) -> None:
        """Run the training loop from start_step to total_timesteps."""
        cfg = self.cfg
        self._log_hparams(cfg)

        envs = gym.vector.SyncVectorEnv(
            [self._make_env(i) for i in range(self.env_cfg.get("num_envs", 1))],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        replay_buffer = ReplayBuffer(
            capacity=cfg["buffer_size"],
            obs_shape=envs.single_observation_space.shape,
            device=self.agent.device,
        )

        obs, _ = envs.reset(seed=self.seed)
        start_time = time.time()
        recent_returns: list[float] = []
        print_every = 1000

        for global_step in range(start_step, cfg["total_timesteps"]):
            session_steps = global_step - start_step + 1
            epsilon = _linear_schedule(
                cfg["start_epsilon"],
                cfg["end_epsilon"],
                int(cfg["exploration_fraction"] * cfg["total_timesteps"]),
                global_step,
            )

            actions = self.agent.select_action(obs, epsilon)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            final_info_mask = infos.get("_final_info")
            if final_info_mask is not None:
                final_info = infos.get("final_info", {})
                ep_data = final_info.get("episode", {})
                for i in range(envs.num_envs):
                    if final_info_mask[i] and "r" in ep_data:
                        ret    = float(ep_data["r"][i])
                        length = int(ep_data["l"][i])
                        recent_returns.append(ret)
                        self.writer.add_scalar("charts/episodic_return", ret, global_step)
                        self.writer.add_scalar("charts/episodic_length", length, global_step)
                        self.writer.add_scalar("charts/epsilon", epsilon, global_step)

            if global_step % print_every == 0 and global_step > 0:
                sps = int(session_steps / (time.time() - start_time))
                if recent_returns:
                    print(
                        f"step {global_step:>7}/{cfg['total_timesteps']}  "
                        f"return {np.mean(recent_returns):>6.1f} "
                        f"(max {np.max(recent_returns):.0f}, n={len(recent_returns)})  "
                        f"ε={epsilon:.3f}  {sps} sps",
                        flush=True,
                    )
                    recent_returns.clear()
                else:
                    print(
                        f"step {global_step:>7}/{cfg['total_timesteps']}  "
                        f"no episodes finished  ε={epsilon:.3f}  {sps} sps",
                        flush=True,
                    )

            # Store transition — done = terminations only (not truncations).
            real_next_obs = next_obs.copy()
            final_obs      = infos.get("final_obs")
            final_obs_mask = infos.get("_final_obs")
            if final_obs is not None and final_obs_mask is not None:
                for i in range(envs.num_envs):
                    if truncations[i] and final_obs_mask[i]:
                        real_next_obs[i] = final_obs[i]

            for i in range(envs.num_envs):
                replay_buffer.add(
                    obs[i], real_next_obs[i], actions[i], rewards[i], float(terminations[i])
                )

            obs = next_obs

            if global_step > cfg["learning_starts"]:
                self.agent.on_step(global_step)

                if global_step % cfg["train_frequency"] == 0:
                    batch   = replay_buffer.sample(cfg["batch_size"])
                    metrics = self.agent.update(batch)

                    if global_step % 100 == 0:
                        for name, val in metrics.items():
                            self.writer.add_scalar(f"losses/{name}", val, global_step)
                        sps = int(session_steps / (time.time() - start_time))
                        self.writer.add_scalar("charts/SPS", sps, global_step)

            if global_step > 0 and global_step % cfg["checkpoint_frequency"] == 0:
                path = os.path.join(self.ckpt_dir, f"step_{global_step:07d}.pt")
                self.agent.save(path)
                print(f"checkpoint → {path}", flush=True)

        final_path = os.path.join(self.ckpt_dir, "final.pt")
        self.agent.save(final_path)
        print(f"final model → {final_path}", flush=True)

        envs.close()

        if self.env_cfg.get("capture_video", False):
            self._record_final_episode()

        self.writer.close()

    def _record_final_episode(self) -> None:
        """Run one greedy episode and save as video."""
        env = gym.make(self.env_cfg["env_id"], render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, self.video_dir,
            episode_trigger=lambda _: True,
            name_prefix="final",
        )
        obs, _ = env.reset(seed=self.seed)
        total_reward = 0.0
        done = False
        while not done:
            action = int(self.agent.select_action(obs[np.newaxis], epsilon=0.0)[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        env.close()
        print(f"final episode return={total_reward:.1f}  video → {self.video_dir}/final-episode-0.mp4")

    def _log_hparams(self, cfg: dict) -> None:
        rows = "\n".join(f"|{k}|{v}|" for k, v in cfg.items())
        self.writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{rows}")
