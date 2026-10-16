"""
============================================================
Evaluation Script for Stage I (v3): Strict Success Criteria
============================================================

A successful jump requires ALL of the following:
  1. has_jumped = True (completed flight + landing)
  2. NOT terminated (no body/thigh/calf contact with ground, not fallen)
  3. Base height > 0.25m at episode end (standing, not collapsed)
  4. Orientation error < 0.5 rad at episode end (not tipped over)
  5. Landing error < 0.15m (landed near takeoff position)

One episode per environment. No resets mid-evaluation.

Usage:
    python eval_stage1.py --task go2_gap --load_run <folder> --run_name RSI --headless --num_envs 4096
    python eval_stage1.py --task go2_gap --run_name compare
============================================================
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym.torch_utils import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from legged_gym.utils.math import quat_distance, wrap_to_pi


def evaluate(args, method_name):
    torch.manual_seed(0)
    np.random.seed(0)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Clean evaluation — no randomization
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4096)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_robot_pos = False
    env_cfg.domain_rand.randomize_robot_vel = False
    env_cfg.domain_rand.randomize_robot_ori = False
    env_cfg.domain_rand.randomize_dof_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_PD_gains = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_upwards = False
    env_cfg.domain_rand.push_towards_goal = False
    env_cfg.domain_rand.randomize_gravity = False
    env_cfg.domain_rand.sim_latency = False
    env_cfg.domain_rand.randomize_has_jumped = False
    env_cfg.domain_rand.pos_vel_random_prob = 0.0

    if hasattr(env_cfg.domain_rand, 'virtual_force_assist'):
        env_cfg.domain_rand.virtual_force_assist = False
    if hasattr(env_cfg.domain_rand, 'vf_curriculum'):
        env_cfg.domain_rand.vf_curriculum = False
    if hasattr(env_cfg.domain_rand, 'vf_probability'):
        env_cfg.domain_rand.vf_probability = 0.0
    print("Evaluation mode: virtual assistant force OFF")

    # Minimal terrain
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.num_zero_height_terrains = 4
    env_cfg.terrain.terrain_difficulty_height_range = [0.0, 0.0]
    env_cfg.terrain.make_terrain_uneven = False
    env_cfg.terrain.border_size = 2

    env_cfg.commands.curriculum = False
    env_cfg.env.continuous_jumping = False
    env_cfg.env.episode_length_s = 4

    # Create env and load policy
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.num_steps_per_env = 1
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ========================================
    # Run ONE episode per environment
    # ========================================
    num_envs = env.num_envs
    env.reset()
    obs = env.get_observations()

    # Per-robot tracking
    max_height = torch.full((num_envs,), env.base_init_state[2].item(), device=env.device)
    has_jumped = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    body_contact_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    low_height_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    orientation_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    landing_error_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    post_landing_drift_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    action_rate_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    gravity_acc_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    first_failure_reason = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    penalized_contact_seen = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    penalized_contact_steps = torch.zeros(num_envs, dtype=torch.int, device=env.device)
    flight_steps = torch.zeros(num_envs, dtype=torch.int, device=env.device)
    total_energy = torch.zeros(num_envs, device=env.device)
    peak_torque = torch.zeros(num_envs, device=env.device)
    total_reward_log = []

    # Trajectory storage for plotting
    base_heights_all = []

    for step in range(int(env.max_episode_length)):
        actions = policy(obs.detach())
        if step == 0:
            actions[:] = 0.0

        obs, _, rews, dones, infos = env.step(actions.detach())
        total_reward_log.append(rews.mean().item())

        # Track max height (during real flight only)
        in_flight = env.mid_air & (~env.has_jumped)
        max_height[in_flight] = torch.max(max_height[in_flight], env.root_states[in_flight, 2])

        # Track flight time
        flight_steps[env.mid_air & (~env.has_jumped)] += 1

        # Track if jumped
        has_jumped = has_jumped | env.has_jumped

        # Track termination (non-timeout)
        non_timeout_reset = env.reset_buf.bool() & ~env.time_out_buf.bool()
        terminated = terminated | non_timeout_reset

        body_contact_step = getattr(env, "termination_reason_body_contact", torch.zeros_like(non_timeout_reset)) & non_timeout_reset
        low_height_step = getattr(env, "termination_reason_low_height", torch.zeros_like(non_timeout_reset)) & non_timeout_reset
        orientation_step = getattr(env, "termination_reason_orientation", torch.zeros_like(non_timeout_reset)) & non_timeout_reset
        landing_error_step = getattr(env, "termination_reason_landing_error", torch.zeros_like(non_timeout_reset)) & non_timeout_reset
        post_landing_drift_step = getattr(env, "termination_reason_post_landing_drift", torch.zeros_like(non_timeout_reset)) & non_timeout_reset
        action_rate_step = getattr(env, "termination_reason_action_rate", torch.zeros_like(non_timeout_reset)) & non_timeout_reset
        gravity_acc_step = getattr(env, "termination_reason_gravity_acc", torch.zeros_like(non_timeout_reset)) & non_timeout_reset

        body_contact_terminated |= body_contact_step
        low_height_terminated |= low_height_step
        orientation_terminated |= orientation_step
        landing_error_terminated |= landing_error_step
        post_landing_drift_terminated |= post_landing_drift_step
        action_rate_terminated |= action_rate_step
        gravity_acc_terminated |= gravity_acc_step

        new_failure = non_timeout_reset & (first_failure_reason < 0)
        first_failure_reason[new_failure] = 0
        reason_steps = [
            body_contact_step,
            low_height_step,
            orientation_step,
            landing_error_step,
            post_landing_drift_step,
            action_rate_step,
            gravity_acc_step,
        ]
        for reason_id, reason_step in enumerate(reason_steps, start=1):
            idx = new_failure & reason_step & (first_failure_reason == 0)
            first_failure_reason[idx] = reason_id

        if hasattr(env, "penalised_contact_indices") and len(env.penalised_contact_indices) > 0:
            penalized_contact_step = torch.any(
                torch.linalg.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1,
                dim=1,
            )
            penalized_contact_seen |= penalized_contact_step
            penalized_contact_steps += penalized_contact_step.int()

        # Track energy
        total_energy += torch.sum(torch.abs(env.torques * env.dof_vel), dim=1) * env.dt

        # Track peak torque
        current_peak = torch.max(torch.abs(env.torques), dim=1)[0]
        peak_torque = torch.max(peak_torque, current_peak)

        # Store trajectory
        base_heights_all.append(env.root_states[:, 2].cpu().numpy().copy())

    # ========================================
    # Compute final state at episode end
    # ========================================
    final_height = env.root_states[:, 2].cpu()
    final_ori_error = env.ori_error.cpu()

    # Landing xy error (distance from takeoff position, not a stability/contact metric)
    landing_error = torch.zeros(num_envs, device=env.device)
    jumped_valid = has_jumped & (~torch.isnan(env.landing_poses[:, 0]))
    if jumped_valid.any():
        landing_error[jumped_valid] = torch.linalg.norm(
            env.landing_poses[jumped_valid, :2] - env.initial_root_states[jumped_valid, :2],
            dim=1
        )

    # ========================================
    # Apply STRICT success criteria
    # ========================================
    # Criterion 1: completed a jump
    c1_jumped = has_jumped.cpu()

    # Criterion 2: not terminated (no body collision, not fallen)
    c2_not_terminated = ~terminated.cpu()

    desired_jump_height = getattr(env.cfg.rewards, "desired_jump_height", 0.9)
    height_error_threshold = 0.3

    # Criterion 3: reached near target jump height
    c3_height = torch.abs(max_height.cpu() - desired_jump_height) < height_error_threshold

    # Criterion 4: standing at episode end (height > 0.25m)
    c3_standing = final_height > 0.25

    # Criterion 5: orientation OK (error < 0.5 rad)
    c4_orientation = final_ori_error < 0.5

    # Criterion 6: landed near takeoff in xy (error < 0.15m)
    c5_landing = landing_error.cpu() < 0.15

    # Full success: ALL criteria met
    full_success = c1_jumped & c2_not_terminated & c3_height & c3_standing & c4_orientation & c5_landing

    # Convert to numpy
    max_height_np = max_height.cpu().numpy()
    has_jumped_np = has_jumped.cpu().numpy()
    terminated_np = terminated.cpu().numpy()
    landing_error_np = landing_error.cpu().numpy()
    flight_time_np = (flight_steps.float() * env.dt).cpu().numpy()
    energy_np = total_energy.cpu().numpy()
    peak_torque_np = peak_torque.cpu().numpy()
    full_success_np = full_success.numpy()
    final_height_np = final_height.numpy()
    final_ori_error_np = final_ori_error.numpy()

    # ========================================
    # Print summary
    # ========================================
    N = num_envs
    print("=" * 70)
    print(f"  STAGE I EVALUATION (STRICT) — {method_name}")
    print("=" * 70)
    print(f"  Total robots: {N}")
    print()
    print(f"  --- Individual Criteria ---")
    print(f"  C1: Jumped (has_jumped):        {c1_jumped.sum().item():5d} / {N}  ({100*c1_jumped.float().mean():.1f}%)")
    print(f"  C2: Not terminated:             {c2_not_terminated.sum().item():5d} / {N}  ({100*c2_not_terminated.float().mean():.1f}%)")
    print(f"  C3: Height (|h-{desired_jump_height:.2f}|<{height_error_threshold:.2f}m): {c3_height.sum().item():5d} / {N}  ({100*c3_height.float().mean():.1f}%)")
    print(f"  C4: Standing (h>0.25m):         {c3_standing.sum().item():5d} / {N}  ({100*c3_standing.float().mean():.1f}%)")
    print(f"  C5: Orientation (<0.5rad):       {c4_orientation.sum().item():5d} / {N}  ({100*c4_orientation.float().mean():.1f}%)")
    print(f"  C6: Landing XY (<0.15m):        {c5_landing.sum().item():5d} / {N}  ({100*c5_landing.float().mean():.1f}%)")
    print()
    print(f"  --- Combined Success (ALL criteria) ---")
    print(f"  FULL SUCCESS:                   {full_success.sum().item():5d} / {N}  ({100*full_success.float().mean():.1f}%)")
    print()
    print(f"  --- Termination Diagnostics ---")
    print(f"  Body/thigh/calf contact:        {body_contact_terminated.sum().item():5d} / {N}  ({100*body_contact_terminated.float().mean():.1f}%)")
    print(f"  Low base height:                {low_height_terminated.sum().item():5d} / {N}  ({100*low_height_terminated.float().mean():.1f}%)")
    print(f"  Orientation error:              {orientation_terminated.sum().item():5d} / {N}  ({100*orientation_terminated.float().mean():.1f}%)")
    print(f"  Landing task error:             {landing_error_terminated.sum().item():5d} / {N}  ({100*landing_error_terminated.float().mean():.1f}%)")
    print(f"  Post-landing drift:             {post_landing_drift_terminated.sum().item():5d} / {N}  ({100*post_landing_drift_terminated.float().mean():.1f}%)")
    print(f"  Action-rate reset:              {action_rate_terminated.sum().item():5d} / {N}  ({100*action_rate_terminated.float().mean():.1f}%)")
    print(f"  Downward accel reset:           {gravity_acc_terminated.sum().item():5d} / {N}  ({100*gravity_acc_terminated.float().mean():.1f}%)")
    known_terminated = (
        body_contact_terminated
        | low_height_terminated
        | orientation_terminated
        | landing_error_terminated
        | post_landing_drift_terminated
        | action_rate_terminated
        | gravity_acc_terminated
    )
    unknown_terminated = terminated & ~known_terminated
    print(f"  Unknown non-timeout reset:      {unknown_terminated.sum().item():5d} / {N}  ({100*unknown_terminated.float().mean():.1f}%)")
    print()
    print(f"  --- First Failure Reason (disjoint) ---")
    first_reason_names = [
        ("Unknown reset", 0),
        ("Body/thigh/calf contact", 1),
        ("Low base height", 2),
        ("Orientation error", 3),
        ("Landing task error", 4),
        ("Post-landing drift", 5),
        ("Action-rate reset", 6),
        ("Downward accel reset", 7),
    ]
    for label, reason_id in first_reason_names:
        count = (first_failure_reason == reason_id).sum().item()
        print(f"  {label:<28} {count:5d} / {N}  ({100*count/N:.1f}%)")
    print()
    print(f"  --- Physical Contact Diagnostics (not necessarily reset) ---")
    print(f"  Thigh/calf/hip contact seen:    {penalized_contact_seen.sum().item():5d} / {N}  ({100*penalized_contact_seen.float().mean():.1f}%)")
    print(f"  Mean thigh/calf/hip contact steps per robot: {penalized_contact_steps.float().mean().item():.2f}")
    print()

    reward_term_means = {}
    reward_term_episode_means = {}
    if hasattr(env, "reward_logs") and len(env.reward_logs) > 0:
        print(f"  --- Mean Reward Terms ---")
        print(f"  {'term':<32} {'mean/step':>14} {'mean/episode':>16}")
        print(f"  {'-' * 64}")

        for name in env.reward_names:
            logs = env.reward_logs.get(name, [])
            if len(logs) > 0:
                term_log = torch.stack(logs)
                mean_step = term_log.mean().item()
                mean_episode = term_log.sum().item()
            else:
                mean_step = 0.0
                mean_episode = 0.0

            reward_term_means[name] = mean_step
            reward_term_episode_means[name] = mean_episode
            print(f"  {name:<32} {mean_step:14.6f} {mean_episode:16.6f}")

        total_mean_step = float(np.mean(total_reward_log)) if len(total_reward_log) > 0 else 0.0
        total_mean_episode = float(np.sum(total_reward_log)) if len(total_reward_log) > 0 else 0.0
        print(f"  {'-' * 64}")
        print(f"  {'total':<32} {total_mean_step:14.6f} {total_mean_episode:16.6f}")
        print()

    # Height breakdown among full successes
    if full_success.sum() > 0:
        s_heights = max_height_np[full_success_np]
        print(f"  --- Among Successful Jumps ---")
        print(f"  Mean max height:     {np.mean(s_heights):.3f} m")
        print(f"  Std max height:      {np.std(s_heights):.3f} m")
        print(f"  Median max height:   {np.median(s_heights):.3f} m")
        print(f"  Min / Max height:    {np.min(s_heights):.3f} / {np.max(s_heights):.3f} m")
        print()
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, desired_jump_height]:
            rate = 100 * np.mean(s_heights > threshold)
            count = np.sum(s_heights > threshold)
            print(f"  Reached > {threshold:.1f}m:       {count:5d} / {full_success.sum().item()}  ({rate:.1f}%)")
        print()
        print(f"  Mean flight time:    {np.mean(flight_time_np[full_success_np]):.3f} s")
        print(f"  Mean landing error:  {np.mean(landing_error_np[full_success_np]):.4f} m")
        print(f"  Mean energy:         {np.mean(energy_np[full_success_np]):.2f} J")
        print(f"  Mean peak torque:    {np.mean(peak_torque_np[full_success_np]):.2f} Nm")
    print("=" * 70)

    # ========================================
    # Save results
    # ========================================
    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', f'eval_stage1_{method_name}.npz')
    np.savez(save_path,
             max_heights=max_height_np,
             has_jumped=has_jumped_np,
             terminated=terminated_np,
             full_success=full_success_np,
             flight_times=flight_time_np,
             landing_errors=landing_error_np,
             energy=energy_np,
             peak_torques=peak_torque_np,
             final_heights=final_height_np,
             final_ori_errors=final_ori_error_np,
             reward_names=np.array(list(reward_term_means.keys())),
             reward_mean_per_step=np.array(list(reward_term_means.values())),
             reward_mean_per_episode=np.array(list(reward_term_episode_means.values())),
             c1=c1_jumped.numpy(), c2=c2_not_terminated.numpy(),
             c3=c3_height.numpy(), c4=c3_standing.numpy(),
             c5=c4_orientation.numpy(), c6=c5_landing.numpy())
    print(f"Results saved to: {save_path}")

    # ========================================
    # Generate figures
    # ========================================
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.grid'] = True

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Stage I Evaluation (Strict) — {method_name}', fontsize=16, fontweight='bold')

    # --- 1: Height distribution (full success only) --- #
    ax = axes[0, 0]
    if full_success.sum() > 0:
        ax.hist(s_heights, bins=40, range=(0.3, 1.1), color='steelblue',
                edgecolor='white', alpha=0.8, label=f'Success ({len(s_heights)})')
    failed_heights = max_height_np[~full_success_np & has_jumped_np]
    if len(failed_heights) > 0:
        ax.hist(failed_heights, bins=40, range=(0.3, 1.1), color='salmon',
                edgecolor='white', alpha=0.5, label=f'Failed ({len(failed_heights)})')
    ax.axvline(x=desired_jump_height, color='darkred', linestyle='--', linewidth=2,
               label=f'{desired_jump_height:.2f}m target')
    ax.set_xlabel('Max Height (m)')
    ax.set_ylabel('Count')
    ax.set_title('Height Distribution')
    ax.legend(fontsize=10)

    # --- 2: Success rate vs threshold (strict) --- #
    ax = axes[0, 1]
    thresholds = np.arange(0.3, 1.01, 0.05)
    if full_success.sum() > 0:
        strict_rates = [100 * np.mean(s_heights > t) for t in thresholds]
        ax.plot(thresholds, strict_rates, 'o-', color='steelblue', linewidth=2,
                markersize=6, label='Strict (jumped + stable)')
    # Also show loose rate for comparison
    all_heights = max_height_np[has_jumped_np]
    if len(all_heights) > 0:
        loose_rates = [100 * np.mean(all_heights > t) for t in thresholds]
        ax.plot(thresholds, loose_rates, 's--', color='gray', linewidth=1.5,
                markersize=4, alpha=0.5, label='Loose (jumped only)')
    ax.set_xlabel('Height Threshold (m)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate vs Height')
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=10)

    # --- 3: Criteria breakdown bar chart --- #
    ax = axes[0, 2]
    criteria_names = ['C1\nJumped', 'C2\nNo crash', 'C3\nHeight', 'C4\nStanding', 'C5\nOrientation', 'C6\nLanding\nXY', 'ALL\nSuccess']
    criteria_values = [
        100 * c1_jumped.float().mean().item(),
        100 * c2_not_terminated.float().mean().item(),
        100 * c3_height.float().mean().item(),
        100 * c3_standing.float().mean().item(),
        100 * c4_orientation.float().mean().item(),
        100 * c5_landing.float().mean().item(),
        100 * full_success.float().mean().item(),
    ]
    bar_colors = ['#64B5F6', '#64B5F6', '#64B5F6', '#64B5F6', '#64B5F6', '#64B5F6', '#2196F3']
    ax.bar(criteria_names, criteria_values, color=bar_colors, edgecolor='white')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Success Criteria Breakdown')
    ax.set_ylim([0, 105])
    for i, v in enumerate(criteria_values):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    # --- 4: Trajectory plot --- #
    ax = axes[1, 0]
    if len(base_heights_all) > 0:
        heights_array = np.array(base_heights_all)
        dt = env.dt * env.cfg.control.decimation
        time_axis = np.arange(heights_array.shape[0]) * dt

        # Plot up to 5 successful and 5 failed trajectories
        success_indices = np.where(full_success_np)[0]
        fail_indices = np.where(has_jumped_np & ~full_success_np)[0]

        for idx in success_indices[:5]:
            ax.plot(time_axis, heights_array[:, idx], color='steelblue', alpha=0.7, linewidth=1.5)
        for idx in fail_indices[:5]:
            ax.plot(time_axis, heights_array[:, idx], color='salmon', alpha=0.5, linewidth=1.0)

        ax.axhline(y=0.20, color='blue', linestyle=':', alpha=0.5, label='Squat (0.20m)')
        ax.axhline(y=desired_jump_height, color='red', linestyle=':', alpha=0.5,
                   label=f'Flight ({desired_jump_height:.2f}m)')
        ax.plot([], [], color='steelblue', linewidth=2, label='Success')
        ax.plot([], [], color='salmon', linewidth=2, label='Failed')
        ax.legend(fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Base Height (m)')
    ax.set_title('Jump Trajectories')

    # --- 5: Landing error vs height --- #
    ax = axes[1, 1]
    if has_jumped_np.sum() > 0:
        colors_scatter = np.where(full_success_np[has_jumped_np], 'steelblue', 'salmon')
        ax.scatter(max_height_np[has_jumped_np], landing_error_np[has_jumped_np],
                   c=colors_scatter, s=10, alpha=0.4)
        ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Landing threshold (0.15m)')
        ax.legend()
    ax.set_xlabel('Max Height (m)')
    ax.set_ylabel('Landing Error (m)')
    ax.set_title('Landing Accuracy vs Jump Height')

    # --- 6: Energy vs height --- #
    ax = axes[1, 2]
    if full_success.sum() > 0:
        ax.scatter(max_height_np[full_success_np], energy_np[full_success_np],
                   s=10, alpha=0.4, color='steelblue', label='Success')
    if (has_jumped_np & ~full_success_np).sum() > 0:
        ax.scatter(max_height_np[has_jumped_np & ~full_success_np],
                   energy_np[has_jumped_np & ~full_success_np],
                   s=10, alpha=0.3, color='salmon', label='Failed')
    ax.set_xlabel('Max Height (m)')
    ax.set_ylabel('Total Energy (J)')
    ax.set_title('Energy vs Jump Height')
    ax.legend()

    plt.tight_layout()
    save_fig_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', f'eval_stage1_{method_name}.png')
    plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_fig_path}")
    plt.show()


def compare_methods():
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs')
    npz_files = sorted([f for f in os.listdir(log_dir)
                        if f.startswith('eval_stage1_') and f.endswith('.npz')])

    if len(npz_files) == 0:
        print("No evaluation results found.")
        return

    print(f"Found: {npz_files}")

    method_colors = ['#2196F3', '#FF5722', '#9E9E9E', '#4CAF50', '#9C27B0']
    datasets = []
    for f in npz_files:
        data = np.load(os.path.join(log_dir, f))
        label = f.replace('eval_stage1_', '').replace('.npz', '')
        datasets.append((label, data))

    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stage I: Method Comparison (Strict Criteria)', fontsize=16, fontweight='bold')

    # --- 1: Strict success rate vs height threshold --- #
    ax = axes[0, 0]
    thresholds = np.arange(0.3, 1.01, 0.05)
    for i, (label, data) in enumerate(datasets):
        success = data['full_success'].astype(bool)
        if success.sum() > 0:
            heights = data['max_heights'][success]
            rates = [100 * np.mean(heights > t) for t in thresholds]
        else:
            rates = [0] * len(thresholds)
        ax.plot(thresholds, rates, 'o-', color=method_colors[i % len(method_colors)],
                linewidth=2, markersize=5, label=label)
    ax.set_xlabel('Height Threshold (m)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Strict Success Rate vs Height')
    ax.legend()
    ax.set_ylim([-5, 105])

    # --- 2: Criteria breakdown comparison --- #
    ax = axes[0, 1]
    criteria_short = ['Jumped', 'No crash', 'Height', 'Standing', 'Orientation', 'Landing', 'ALL']
    x = np.arange(len(criteria_short))
    width = 0.8 / max(len(datasets), 1)
    for i, (label, data) in enumerate(datasets):
        values = [
            100 * np.mean(data['c1']),
            100 * np.mean(data['c2']),
            100 * np.mean(data['c3']),
            100 * np.mean(data['c4']),
            100 * np.mean(data['c5']),
            100 * np.mean(data['c6']) if 'c6' in data else 100 * np.mean(data['c5']),
            100 * np.mean(data['full_success']),
        ]
        ax.bar(x + i * width - 0.4 + width / 2, values, width,
               label=label, color=method_colors[i % len(method_colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_short, fontsize=11)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Criteria Breakdown')
    ax.legend()
    ax.set_ylim([0, 110])

    # --- 3: Height box plot (strict success only) --- #
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    for label, data in datasets:
        success = data['full_success'].astype(bool)
        if success.sum() > 0:
            box_data.append(data['max_heights'][success])
        else:
            box_data.append([0])
        box_labels.append(label)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(method_colors[i % len(method_colors)])
        patch.set_alpha(0.6)
    ax.set_ylabel('Max Height (m)')
    ax.set_title('Jump Height (Strict Success Only)')
    ax.axhline(y=1.00, color='darkred', linestyle='--', alpha=0.5)

    # --- 4: Summary metrics table --- #
    ax = axes[1, 1]
    ax.axis('off')
    metric_names = ['Full Success Rate (%)', 'Mean Height (m)', 'Mean Flight Time (s)',
                    'Mean Landing Error (m)', 'Mean Energy (J)', 'Mean Peak Torque (Nm)']
    table_data = []
    col_labels = []
    for label, data in datasets:
        col_labels.append(label)
        success = data['full_success'].astype(bool)
        if success.sum() > 0:
            row = [
                f"{100 * np.mean(success):.1f}",
                f"{np.mean(data['max_heights'][success]):.3f}",
                f"{np.mean(data['flight_times'][success]):.3f}",
                f"{np.mean(data['landing_errors'][success]):.4f}",
                f"{np.mean(data['energy'][success]):.1f}",
                f"{np.mean(data['peak_torques'][success]):.1f}",
            ]
        else:
            row = ["0.0", "—", "—", "—", "—", "—"]
        table_data.append(row)

    table = ax.table(cellText=list(zip(*table_data)),
                     rowLabels=metric_names,
                     colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)
    ax.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(log_dir, 'eval_stage1_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    args = get_args()
    method_name = args.run_name if args.run_name else "unknown"
    if method_name == "compare":
        compare_methods()
    else:
        evaluate(args, method_name)
