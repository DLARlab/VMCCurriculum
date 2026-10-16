"""
Virtual Jump Assist Force Module
================================
Two-phase force assist for quadrupedal jumping:
  1. Jump phase: height/velocity tracking of a parabolic desired base-height curve
  2. Landing phase: damping force to cushion impact

Orientation assist (whole episode):
  3. Roll torque: PD control to drive roll deviation to zero
  4. Yaw torque:  PD control to drive yaw deviation to zero
  5. Pitch: FREE (no correction applied)

All assists share the same curriculum scale and decay together.

Drop this file into: legged_gym/legged_gym/utils/virtual_force.py
"""

import torch


class VirtualForceAssist:
    """Manages virtual jump assist: stance thrust, landing damping,
    and orientation torque correction (roll + yaw, pitch free).

    Linear forces are applied as velocity changes (dv = F/m * dt).
    Orientation corrections are applied as angular velocity changes
    (dw = effective_torque * dt, where effective_torque = Kp*error + Kd*rate
    already in rad/s^2 units, i.e. the gain has absorbed 1/I).
    """

    def __init__(self, cfg, num_envs, device):
        """
        Args:
            cfg: domain_rand config object with vf_* parameters
            num_envs: number of parallel environments
            device: torch device
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Per-robot flags and state
        self.enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.has_crouched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.trajectory_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.trajectory_timer = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.trajectory_duration = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.trajectory_start_height = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.desired_height_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.desired_velocity_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.desired_x_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.desired_vx_log = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Landing assist state
        self.landing_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.landing_timer = torch.zeros(num_envs, dtype=torch.int, device=device)

        # Curriculum
        self.curriculum_scale = torch.ones(num_envs, dtype=torch.float, device=device)
        self.adaptive_scale = 1.0
        self.success_stage = 0
        self.last_adaptive_update_iter = -1e9
        self.last_adaptive_success_rate = 0.0
        self.curriculum_history = []
        self._last_recorded_curriculum_iter = None

        # Logging
        self.stance_force_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.forward_force_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.landing_force_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.roll_torque_log = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.yaw_torque_log = torch.zeros(num_envs, dtype=torch.float, device=device)

    def state_dict(self):
        """Return curriculum state that should survive training resume."""
        return {
            "adaptive_scale": float(self.adaptive_scale),
            "success_stage": int(self.success_stage),
            "last_adaptive_update_iter": float(self.last_adaptive_update_iter),
            "last_adaptive_success_rate": float(self.last_adaptive_success_rate),
            "curriculum_scale": self.curriculum_scale.detach().cpu(),
            "curriculum_history": list(self.curriculum_history),
            "last_recorded_curriculum_iter": self._last_recorded_curriculum_iter,
        }

    def load_state_dict(self, state_dict):
        """Restore curriculum state from a training checkpoint."""
        if not state_dict:
            return

        self.adaptive_scale = float(state_dict.get("adaptive_scale", self.adaptive_scale))
        self.success_stage = int(state_dict.get("success_stage", self.success_stage))
        self.last_adaptive_update_iter = float(
            state_dict.get("last_adaptive_update_iter", self.last_adaptive_update_iter)
        )
        self.last_adaptive_success_rate = float(
            state_dict.get("last_adaptive_success_rate", self.last_adaptive_success_rate)
        )
        self.curriculum_history = list(state_dict.get("curriculum_history", self.curriculum_history))
        self._last_recorded_curriculum_iter = state_dict.get(
            "last_recorded_curriculum_iter", self._last_recorded_curriculum_iter
        )

        scale = state_dict.get("curriculum_scale")
        if scale is not None and scale.numel() == self.curriculum_scale.numel():
            self.curriculum_scale[:] = scale.to(self.device).reshape_as(self.curriculum_scale)
        else:
            self.curriculum_scale[:] = self.adaptive_scale

    def reset(self, env_ids):
        """Reset force state for environments that are being reset."""
        if len(env_ids) == 0:
            return

        cfg = self.cfg

        # Randomly assign which robots get assistance
        self.enabled[env_ids] = (
            torch.rand(len(env_ids), device=self.device) < cfg.vf_probability
        )

        # Reset state
        self.has_crouched[env_ids] = False
        self.trajectory_active[env_ids] = False
        self.trajectory_timer[env_ids] = 0.0
        self.trajectory_duration[env_ids] = 0.0
        self.trajectory_start_height[env_ids] = 0.0
        self.desired_height_log[env_ids] = 0.0
        self.desired_velocity_log[env_ids] = 0.0
        self.desired_x_log[env_ids] = 0.0
        self.desired_vx_log[env_ids] = 0.0
        self.landing_active[env_ids] = False
        self.landing_timer[env_ids] = 0
        self.stance_force_log[env_ids] = 0.0
        self.forward_force_log[env_ids] = 0.0
        self.landing_force_log[env_ids] = 0.0
        self.roll_torque_log[env_ids] = 0.0
        self.yaw_torque_log[env_ids] = 0.0

    @staticmethod
    def _quat_to_roll_yaw(quat):
        """Extract roll and yaw from quaternion (x, y, z, w).

        Pitch is NOT computed because it is left free.

        Returns:
            roll: [N] tensor, rotation around x-axis [rad]
            yaw:  [N] tensor, rotation around z-axis [rad]
        """
        qx = quat[:, 0]
        qy = quat[:, 1]
        qz = quat[:, 2]
        qw = quat[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, yaw

    def compute_and_apply(self, root_states, contacts, mid_air, was_in_flight,
                          has_jumped, commands, initial_root_states,
                          total_mass, dt, env_properties=None, box_start_margin=None):
        """Compute forces/torques and apply them as velocity changes to root_states.

        Modifies IN PLACE unless cfg.vf_apply_as_external_force is True:
          root_states[:, 7]  — x linear velocity  (trajectory tracking force)
          root_states[:, 9]  — z linear velocity  (stance + landing force)
          root_states[:, 10] — x angular velocity  (roll correction)
          root_states[:, 12] — z angular velocity  (yaw correction)
          root_states[:, 11] — y angular velocity  (pitch — NOT modified)

        Returns:
            total_force: [num_envs, 3] external base force in env/world axes
            total_torque: [num_envs, 3] external base torque in env/world axes

        Args:
            root_states: [num_envs, 13] (pos[0:3], quat[3:7], lin_vel[7:10], ang_vel[10:13])
            contacts: [num_envs, 4] bool, foot contact states
            mid_air: [num_envs] bool, all feet off ground right now
            was_in_flight: [num_envs] bool, has been airborne after settling
            has_jumped: [num_envs] bool, has completed a jump
            total_mass: scalar, robot mass [kg]
            dt: scalar, policy timestep (sim_dt × decimation) [s]
            env_properties: [num_envs, 3] (box_width, box_length, box_height) or None.
                Non-zero height selects the land-on-box trajectory; all-zero (gap/flat
                terrain) keeps the land-at-take-off-height trajectory.
            box_start_margin: scalar [m], base distance before the near box edge. Only
                read when env_properties carries a non-zero box height.
        """
        cfg = self.cfg
        x   = root_states[:, 0]   # world x position
        v_x = root_states[:, 7]   # world x linear velocity
        v_z = root_states[:, 9]   # vertical linear velocity
        h   = root_states[:, 2]   # height
        all_feet_contact = torch.all(contacts, dim=1)
        any_feet_contact = torch.any(contacts, dim=1)

        # ================================================ #
        #       PHASE 1: HEIGHT TRAJECTORY ASSIST (z-force) #
        # ================================================ #

        trigger_mode = getattr(cfg, "vf_trigger_mode", "height")
        if trigger_mode == "upward_velocity":
            trajectory_ready = (v_z > getattr(cfg, "vf_jump_trigger_vz", 0.15)) & any_feet_contact
        else:
            self.has_crouched = self.has_crouched | (h < cfg.vf_stance_h_threshold)
            trajectory_ready = self.has_crouched

        trajectory_onset = (
            self.enabled
            & (~self.trajectory_active)
            & (~was_in_flight)
            & (~has_jumped)
            & any_feet_contact
            & trajectory_ready
        )
        self.trajectory_active[trajectory_onset] = True
        self.trajectory_timer[trajectory_onset] = 0.0
        self.trajectory_start_height[trajectory_onset] = h[trajectory_onset]

        default_duration = max(float(getattr(cfg, "vf_parabola_duration", 0.6)), 1e-6)
        duration_range = getattr(cfg, "vf_parabola_duration_range", None)
        if duration_range is not None and len(duration_range) == 2:
            min_duration = float(duration_range[0])
            max_duration = float(duration_range[1])
            sampled_duration = min_duration + (max_duration - min_duration) * torch.rand(
                torch.sum(trajectory_onset).item(), device=self.device
            )
            self.trajectory_duration[trajectory_onset] = sampled_duration.clamp(min=1e-6)
        else:
            self.trajectory_duration[trajectory_onset] = default_duration

        duration = torch.where(
            self.trajectory_duration > 0.0,
            self.trajectory_duration,
            torch.full_like(self.trajectory_duration, default_duration),
        ).clamp(min=1e-6)
        phase = (self.trajectory_timer / duration).clamp(0.0, 1.0)

        # Box terrain lands the robot ON the obstacle, so the trajectory must end at box
        # height instead of back at take-off height. env_properties is all-zero on gap and
        # flat terrain, where has_box is False everywhere and every expression below
        # collapses to the previous start == end trajectory.
        if env_properties is not None:
            box_height = env_properties[:, 2].clamp(min=0.0)
        else:
            box_height = torch.zeros_like(self.trajectory_start_height)
        has_box = box_height > 0.0
        landing_base_height = float(getattr(cfg, "vf_landing_base_height", 0.0))
        trajectory_end_height = torch.where(
            has_box,
            box_height + landing_base_height,
            self.trajectory_start_height,
        )

        apex_height_cfg = float(getattr(cfg, "vf_desired_height", 0.9))
        apex_height = torch.full_like(trajectory_end_height, apex_height_cfg)
        apex_height = torch.where(
            has_box,
            torch.maximum(
                apex_height,
                torch.maximum(
                    self.trajectory_start_height,
                    trajectory_end_height + float(getattr(cfg, "vf_box_apex_clearance", 0.0)),
                ),
            ),
            apex_height,
        )
        height_delta = (apex_height - self.trajectory_start_height).clamp(min=0.0)

        # Where along the jump the apex sits. On a box it is placed over the near edge; off
        # a box it stays at 0.5, which makes the Bezier below identical to the symmetric
        # start==end curve this used before.
        travel_x = commands[:, 0]
        travel_abs = torch.abs(travel_x).clamp(min=1e-6)
        travel_sign = torch.sign(travel_x)
        travel_sign[travel_sign == 0.0] = 1.0
        near_edge_dist = torch.full_like(
            travel_abs, float(box_start_margin) if box_start_margin is not None else 0.0
        ).clamp(max=travel_abs)
        apex_dist = near_edge_dist + float(getattr(cfg, "vf_apex_box_edge_offset", 0.0))
        apex_phase = torch.where(
            has_box,
            (apex_dist / travel_abs).clamp(0.10, 0.90),
            torch.full_like(travel_abs, 0.5),
        )

        if getattr(cfg, "vf_trajectory_type", "parabola") == "bezier":
            # Quadratic Bezier through an automatically chosen apex waypoint.
            p0 = self.trajectory_start_height
            p2 = trajectory_end_height
            denom = (2.0 * (1.0 - apex_phase) * apex_phase).clamp(min=1e-6)
            p1 = (
                apex_height
                - (1.0 - apex_phase) * (1.0 - apex_phase) * p0
                - apex_phase * apex_phase * p2
            ) / denom
            one_minus_phase = 1.0 - phase
            desired_height = (
                one_minus_phase * one_minus_phase * p0
                + 2.0 * one_minus_phase * phase * p1
                + phase * phase * p2
            )
            desired_velocity = 2.0 * (
                one_minus_phase * (p1 - p0)
                + phase * (p2 - p1)
            ) / duration
            desired_acc = 2.0 * (p2 - 2.0 * p1 + p0) / (duration * duration)
        else:
            rise = trajectory_end_height - self.trajectory_start_height
            desired_height = self.trajectory_start_height + rise * phase + 4.0 * height_delta * phase * (1.0 - phase)
            desired_velocity = rise / duration + 4.0 * height_delta * (1.0 - 2.0 * phase) / duration
            desired_acc = -8.0 * height_delta / (duration * duration)

        height_error = desired_height - h
        velocity_error = desired_velocity - v_z
        if getattr(cfg, "vf_force_mode", "trajectory_pd") == "weight_error":
            force_kp = getattr(cfg, "vf_height_force_kp", cfg.vf_height_kp)
            force_kd = getattr(cfg, "vf_height_force_kd", cfg.vf_height_kd)
            assist_force = total_mass * 9.81 + force_kp * height_error + force_kd * velocity_error
        else:
            assist_acc = desired_acc + 9.81 + cfg.vf_height_kp * height_error + cfg.vf_height_kd * velocity_error
            assist_force = total_mass * assist_acc
        assist_force = assist_force.clamp(0.0, cfg.vf_stance_max)

        # Forward trajectory, same Bezier treatment. Off a box the control point sits at the
        # midpoint, for which the quadratic Bezier reduces exactly to the linear
        # x0 + commands * phase (and desired_vx to commands / duration) used previously.
        x0 = initial_root_states[:, 0]
        x2 = initial_root_states[:, 0] + commands[:, 0]
        apex_x = x0 + travel_sign * apex_dist.clamp(max=travel_abs)
        denom_x = (2.0 * (1.0 - apex_phase) * apex_phase).clamp(min=1e-6)
        x1 = torch.where(
            has_box,
            (
                apex_x
                - (1.0 - apex_phase) * (1.0 - apex_phase) * x0
                - apex_phase * apex_phase * x2
            ) / denom_x,
            x0 + float(getattr(cfg, "vf_bezier_x_midpoint", 0.5)) * commands[:, 0],
        )
        one_minus_phase = 1.0 - phase
        desired_x = (
            one_minus_phase * one_minus_phase * x0
            + 2.0 * one_minus_phase * phase * x1
            + phase * phase * x2
        )
        desired_vx = 2.0 * (
            one_minus_phase * (x1 - x0)
            + phase * (x2 - x1)
        ) / duration
        x_error = desired_x - x
        vx_error = desired_vx - v_x
        x_force_kp = getattr(cfg, "vf_x_force_kp", 0.0)
        x_force_kd = getattr(cfg, "vf_x_force_kd", 0.0)
        x_force_max = getattr(cfg, "vf_x_force_max", 0.0)
        assist_force_x = x_force_kp * x_error + x_force_kd * vx_error
        assist_force_x = assist_force_x.clamp(-x_force_max, x_force_max)

        trajectory_eligible = (
            self.enabled
            & self.trajectory_active
            & (~has_jumped)
        )
        assist_force[~trajectory_eligible] = 0.0
        assist_force_x[~trajectory_eligible] = 0.0

        self.trajectory_timer[trajectory_eligible] += dt
        trajectory_done = self.trajectory_timer >= duration
        self.trajectory_active[trajectory_done | has_jumped] = False

        # ================================================ #
        #          PHASE 2: LANDING ASSIST (z-force)       #
        # ================================================ #

        landing_onset = was_in_flight & any_feet_contact & (~has_jumped) & (~self.landing_active)
        self.landing_active[landing_onset] = True
        self.landing_timer[landing_onset] = 0

        self.landing_timer[self.landing_active] += 1

        expired = self.landing_timer > cfg.vf_landing_duration
        self.landing_active[expired] = False

        landing_eligible = (
            self.enabled
            & self.landing_active
            & (~has_jumped)
            & (v_z < 0)
        )

        landing_force = -cfg.vf_landing_kd * v_z
        landing_force = landing_force.clamp(0.0, cfg.vf_landing_max)
        landing_force[~landing_eligible] = 0.0

        # ================================================ #
        #   PHASE 3: ORIENTATION ASSIST (roll + yaw)       #
        #   Pitch is FREE — index 11 is never touched.     #
        # ================================================ #

        roll_cmd = torch.zeros(self.num_envs, device=self.device)
        yaw_cmd  = torch.zeros(self.num_envs, device=self.device)

        torque_assist_enabled = (
            getattr(cfg, 'vf_torque_assist', False) and self.enabled.any()
        )

        if torque_assist_enabled:
            quat = root_states[:, 3:7]          # (x, y, z, w)
            roll, yaw = self._quat_to_roll_yaw(quat)

            wx = root_states[:, 10]             # roll rate  (ωx)
            # root_states[:, 11] is ωy (pitch) — left free
            wz = root_states[:, 12]             # yaw rate   (ωz)

            # PD law for roll/yaw orientation assist.
            roll_cmd = -(cfg.vf_torque_kp_roll * roll + cfg.vf_torque_kd_roll * wx)
            yaw_cmd  = -(cfg.vf_torque_kp_yaw  * yaw  + cfg.vf_torque_kd_yaw  * wz)

            roll_cmd = roll_cmd.clamp(-cfg.vf_torque_max_roll, cfg.vf_torque_max_roll)
            yaw_cmd  = yaw_cmd.clamp( -cfg.vf_torque_max_yaw,  cfg.vf_torque_max_yaw)

            # Only apply to enabled robots
            roll_cmd[~self.enabled] = 0.0
            yaw_cmd[~self.enabled]  = 0.0

        # ================================================ #
        #              COMBINE AND APPLY                   #
        # ================================================ #

        # Apply the same curriculum decay to all linear assistant forces.
        assist_scale = self.curriculum_scale
        total_force_z = (assist_force + landing_force) * assist_scale
        total_force_x = assist_force_x * assist_scale
        total_torque_roll = roll_cmd * assist_scale
        total_torque_yaw = yaw_cmd * assist_scale

        if not getattr(cfg, "vf_apply_as_external_force", False):
            # Linear: convert force → Δv
            root_states[:, 7] += total_force_x / total_mass * dt
            root_states[:, 9] += total_force_z / total_mass * dt

            # Angular: integrate decayed roll/yaw command. Pitch index 11 is untouched.
            root_states[:, 10] += total_torque_roll * dt
            root_states[:, 12] += total_torque_yaw * dt

        # Store applied force components so logs reflect curriculum decay.
        self.stance_force_log = (assist_force * assist_scale).clone()
        self.forward_force_log = total_force_x.clone()
        self.landing_force_log = (landing_force * assist_scale).clone()
        self.desired_height_log = desired_height.clone()
        self.desired_velocity_log = desired_velocity.clone()
        self.desired_x_log = desired_x.clone()
        self.desired_vx_log = desired_vx.clone()
        self.roll_torque_log = total_torque_roll.abs().clone()
        self.yaw_torque_log  = total_torque_yaw.abs().clone()

        total_force = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        total_force[:, 0] = total_force_x
        total_force[:, 2] = total_force_z
        total_torque = torch.zeros_like(total_force)
        total_torque[:, 0] = total_torque_roll
        total_torque[:, 2] = total_torque_yaw
        return total_force, total_torque

    def update_curriculum(self, common_step_counter, max_episode_length, success_history=None):
        """Update assistance scale.

        Linear mode decays by iteration. Success-gated mode holds the current
        force coefficient until recent success history reaches a threshold, then
        decays the coefficient and starts collecting a fresh success window.
        """
        if not self.cfg.vf_curriculum:
            return

        cfg = self.cfg
        steps_per_iter = getattr(cfg, "vf_curriculum_steps_per_iter", max_episode_length)
        current_iter = common_step_counter / steps_per_iter
        mode = getattr(cfg, "vf_curriculum_mode", "linear")

        if current_iter < cfg.vf_curriculum_decay_start:
            linear_scale = 1.0
        elif current_iter >= cfg.vf_curriculum_decay_end:
            linear_scale = 0.0
        else:
            progress = (
                (current_iter - cfg.vf_curriculum_decay_start) /
                (cfg.vf_curriculum_decay_end - cfg.vf_curriculum_decay_start)
            )
            linear_scale = 1.0 - progress

        if mode in ("adaptive", "success_gated"):
            linear_cap = linear_scale if mode == "adaptive" else 1.0
            if success_history is None:
                self._set_curriculum_scale(
                    min(self.adaptive_scale, linear_cap), current_iter, linear_cap
                )
                return

            valid_envs = torch.all(success_history >= 0.0, dim=1)
            num_valid = int(torch.sum(valid_envs).item())
            min_valid = getattr(cfg, "vf_adaptive_min_valid_envs", max(1, self.num_envs // 8))
            if num_valid < min_valid:
                self._set_curriculum_scale(
                    min(self.adaptive_scale, linear_cap), current_iter, linear_cap
                )
                return

            success_rate = torch.mean(success_history[valid_envs].float()).item()
            self.last_adaptive_success_rate = success_rate

            check_interval = getattr(cfg, "vf_adaptive_check_interval", 20)
            threshold = getattr(cfg, "vf_adaptive_success_threshold", 0.8)
            decay_step = getattr(cfg, "vf_adaptive_decay_step", 0.1)
            can_update = current_iter - self.last_adaptive_update_iter >= check_interval
            if success_rate >= threshold and can_update and self.adaptive_scale > 0.0:
                self.adaptive_scale = max(0.0, self.adaptive_scale - decay_step)
                self.success_stage += 1
                self.last_adaptive_update_iter = current_iter
                if mode == "success_gated":
                    success_history[:] = -1.0

            self._set_curriculum_scale(
                min(self.adaptive_scale, linear_cap), current_iter, linear_cap
            )
            return

        self._set_curriculum_scale(linear_scale, current_iter, linear_scale)

    def _set_curriculum_scale(self, scale, current_iter, linear_scale):
        self.curriculum_scale[:] = scale
        self._record_curriculum_scale(scale, current_iter, linear_scale)

    def _record_curriculum_scale(self, scale, current_iter, linear_scale):
        iter_idx = int(current_iter)
        record = {
            "iter": iter_idx,
            "current_iter": float(current_iter),
            "curriculum_scale": float(scale),
            "linear_scale": float(linear_scale),
            "adaptive_scale": float(self.adaptive_scale),
            "adaptive_success_rate": float(self.last_adaptive_success_rate),
            "success_stage": int(self.success_stage),
        }
        if self._last_recorded_curriculum_iter == iter_idx and self.curriculum_history:
            self.curriculum_history[-1] = record
        else:
            self.curriculum_history.append(record)
            self._last_recorded_curriculum_iter = iter_idx

    def get_curriculum_history(self):
        return list(self.curriculum_history)

    def get_curriculum_summary_lines(self, interval=100):
        if not self.curriculum_history:
            return ["Virtual force curriculum: no scale history recorded."]

        selected = []
        seen = set()
        for idx, record in enumerate(self.curriculum_history):
            prev = self.curriculum_history[idx - 1] if idx > 0 else None
            changed = prev is None or record["curriculum_scale"] != prev["curriculum_scale"]
            should_print = (
                record["iter"] % interval == 0
                or changed
                or idx == len(self.curriculum_history) - 1
            )
            if should_print and record["iter"] not in seen:
                selected.append(record)
                seen.add(record["iter"])

        lines = ["Virtual force curriculum decay factor history:"]
        lines.append("iter | effective | linear_cap | adaptive | success_rate | stage")
        for record in selected:
            lines.append(
                f"{record['iter']:4d} | {record['curriculum_scale']:.4f}    | "
                f"{record['linear_scale']:.4f}     | {record['adaptive_scale']:.4f}   | "
                f"{record['adaptive_success_rate']:.4f}       | {record['success_stage']}"
            )
        return lines

    def get_log_dict(self):
        """Return logging metrics."""
        return {
            "virtual_force/curriculum_scale":    self.curriculum_scale.mean().item(),
            "virtual_force/stance_force_mean":   self.stance_force_log.mean().item(),
            "virtual_force/forward_force_mean":  self.forward_force_log.mean().item(),
            "virtual_force/landing_force_mean":  self.landing_force_log.mean().item(),
            "virtual_force/desired_height_mean": self.desired_height_log.mean().item(),
            "virtual_force/desired_vel_mean":    self.desired_velocity_log.mean().item(),
            "virtual_force/desired_x_mean":      self.desired_x_log.mean().item(),
            "virtual_force/desired_vx_mean":     self.desired_vx_log.mean().item(),
            "virtual_force/num_assisted":        self.enabled.sum().item(),
            "virtual_force/adaptive_success_rate": self.last_adaptive_success_rate,
            "virtual_force/adaptive_scale":      self.adaptive_scale,
            "virtual_force/success_stage":        self.success_stage,
            "virtual_force/roll_torque_mean":    self.roll_torque_log.mean().item(),
            "virtual_force/yaw_torque_mean":     self.yaw_torque_log.mean().item(),
            "virtual_force/roll_dw_mean":        self.roll_torque_log.mean().item(),
            "virtual_force/yaw_dw_mean":         self.yaw_torque_log.mean().item(),
        }
