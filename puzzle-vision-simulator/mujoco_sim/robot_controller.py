"""Jacobian arm control and electromagnetic attachment state machine."""

from __future__ import annotations

import math

import mujoco
import numpy as np


class ArmController:
    """Damped-least-squares controller for the copied six-axis PIPER-L."""
    JOINTS = tuple(f"joint{i}" for i in range(1, 7))
    # PIPER-L zero/transport pose: links folded close to the base.
    HOME = np.array([0.0, 0.001, -0.001, 0.0, 0.0, 0.0])
    LOWER = np.array([-2.618, 0.001, -2.696, -1.831, -1.219, -3.139])
    UPPER = np.array([2.618, 3.139, -0.001, 1.831, 1.219, 3.139])

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, piece_count: int):
        self.model, self.data = model, data
        self.site_id = model.site("magnet_site").id
        self.joint_ids = [model.joint(name).id for name in self.JOINTS]
        self.qpos_ids = [model.jnt_qposadr[j] for j in self.joint_ids]
        self.dof_ids = [model.jnt_dofadr[j] for j in self.joint_ids]
        self.ctrl = self.HOME.copy()
        self.piece_bodies = [model.body(f"piece_{i}").id for i in range(piece_count)]
        self.eq_ids = [model.equality(f"magnet_piece_{i}").id
                       for i in range(piece_count)]
        self.attached_body = None
        self.attached_index = None
        self.relative_pos = None
        self.relative_quat = None
        self.attached_yaw = None
        self.attached_tool_yaw = None
        self.pickup_body_xy = None
        # Placed fragments no longer use their conservative cylindrical
        # collision proxies, so preserve their final vision-planned free-joint
        # poses explicitly while the arm continues moving.
        self.locked_piece_poses: dict[int, np.ndarray] = {}
        self.orientation_error = math.inf
        self._yaw_reference = None
        data.ctrl[:] = self.ctrl
        data.qpos[self.qpos_ids] = self.ctrl
        mujoco.mj_forward(model, data)

    @property
    def magnet_position(self):
        return self.data.site_xpos[self.site_id].copy()

    def step_toward(self, target_xyz, target_yaw=0.0, gain=.45):
        """Numerical PIPER-L IK; joint 6 supplies the requested piece rotation."""
        target_xyz = np.asarray(target_xyz, dtype=float)
        position_error = target_xyz - self.magnet_position
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jac = jacp[:, self.dof_ids]
        damping = 2.5e-3
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + damping * np.eye(3), position_error * gain)
        dq = np.clip(dq, -.025, .025)
        # Preserve the wrist rotation as an independent delta while position
        # IK uses the first five joints.
        dq[5] = np.clip(target_yaw - self.ctrl[5], -.018, .018)
        self.ctrl += dq
        self.ctrl = np.clip(self.ctrl, self.LOWER, self.UPPER)
        self.data.ctrl[:] = self.ctrl
        self.orientation_error = abs(target_yaw - self.data.qpos[self.qpos_ids[5]])
        return float(np.linalg.norm(position_error))

    def solve_ik(self, target_xyz, target_yaw=0.0, max_iterations=500):
        """Solve the hover position without moving the displayed robot."""
        target_xyz = np.asarray(target_xyz, dtype=float)
        work = mujoco.MjData(self.model)
        q = self.ctrl.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # joint6's positive axis produces negative world yaw when the magnetic
        # face points down, so command the opposite sign of the planned piece
        # rotation. Magnet and attached piece then rotate together correctly.
        q[5] = np.clip(-target_yaw, self.LOWER[5], self.UPPER[5])
        position_solved = False
        for _ in range(max_iterations):
            work.qpos[self.qpos_ids] = q
            mujoco.mj_forward(self.model, work)
            position_error = target_xyz - work.site_xpos[self.site_id]
            if np.linalg.norm(position_error) < .0015:
                position_solved = True
                break
            mujoco.mj_jacSite(
                self.model, work, jacp, jacr, self.site_id)
            jac = jacp[:, self.dof_ids[:5]]
            error = position_error
            damping = 1.5e-3
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + damping * np.eye(3), error * .55)
            q[:5] += np.clip(dq, -.045, .045)
            q = np.clip(q, self.LOWER, self.UPPER)
        if not position_solved:
            final_error = np.linalg.norm(
                target_xyz - work.site_xpos[self.site_id])
            if final_error > .008:
                raise RuntimeError(
                    f"PIPER-L 逆解失败，位置误差 "
                    f"{final_error * 1000:.1f} mm")

        # Refine the solution so the electromagnet's +Z face normal points
        # vertically down. Starting from the position solution avoids the
        # wrong IK branch and keeps the whole lower face parallel to the piece.
        target_normal = np.array([0.0, 0.0, -1.0])
        for _ in range(350):
            work.qpos[self.qpos_ids] = q
            mujoco.mj_forward(self.model, work)
            position_error = target_xyz - work.site_xpos[self.site_id]
            current_normal = work.site_xmat[self.site_id].reshape(3, 3)[:, 2]
            normal_error = np.cross(current_normal, target_normal)
            if (np.linalg.norm(position_error) < .0015
                    and np.linalg.norm(normal_error) < .012):
                return q
            mujoco.mj_jacSite(
                self.model, work, jacp, jacr, self.site_id)
            jac = np.vstack((
                jacp[:, self.dof_ids[:5]],
                jacr[:, self.dof_ids[:5]] * .15))
            error = np.r_[position_error, normal_error * .15]
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + 2e-3 * np.eye(6), error * .30)
            q[:5] += np.clip(dq, -.025, .025)
            q = np.clip(q, self.LOWER, self.UPPER)
        final_error = np.linalg.norm(
            target_xyz - work.site_xpos[self.site_id])
        if final_error > .008 or np.linalg.norm(normal_error) > .06:
            raise RuntimeError(
                f"PIPER-L 垂直吸附面逆解失败，位置误差 "
                f"{final_error * 1000:.1f} mm")
        return q

    def command_joint_position(self, joint_position):
        self.ctrl = np.clip(
            np.asarray(joint_position, dtype=float), self.LOWER, self.UPPER)
        self.data.ctrl[:] = self.ctrl
        self.physics_step()

    def attach_nearest(self, max_distance=.060):
        """Magnet sensor selects only a physically nearby, not-yet-moved body."""
        magnet = self.magnet_position
        candidates = []
        for index, body_id in enumerate(self.piece_bodies):
            distance = np.linalg.norm(self.data.xpos[body_id] - magnet)
            candidates.append((distance, index, body_id))
        distance, index, body_id = min(candidates)
        if distance > max_distance:
            return None
        # Camera planning assumes a static tabletop scene. Waiting fragments
        # are pose-locked so their conservative cylindrical proxies cannot
        # silently push them away while the arm travels. The selected body is
        # released from that initial lock only when the magnet reaches it.
        self.locked_piece_poses.pop(index, None)
        collision_id = self.model.geom(f"piece_{index}_collision").id
        self.model.geom_contype[collision_id] = 0
        self.model.geom_conaffinity[collision_id] = 0
        self.pickup_body_xy = self.data.xpos[body_id, :2].copy()
        magnet_body = self.model.body("magnet").id
        magnet_quat_inv = np.empty(4)
        relative_quat = np.empty(4)
        mujoco.mju_negQuat(
            magnet_quat_inv, self.data.xquat[magnet_body])
        mujoco.mju_mulQuat(
            relative_quat, magnet_quat_inv, self.data.xquat[body_id])
        self.attached_body = body_id
        self.attached_index = index
        self.relative_pos = np.array([0.0, 0.0, .007])
        self.relative_quat = relative_quat
        self.follow_attachment()
        return body_id

    def follow_attachment(self):
        """Stable kinematic magnetic constraint for a free thin-metal piece."""
        if self.attached_body is None:
            return
        magnet_body = self.model.body("magnet").id
        magnet_rot = self.data.xmat[magnet_body].reshape(3, 3)
        world_pos = (
            self.data.xpos[magnet_body] + magnet_rot @ self.relative_pos)
        world_quat = np.empty(4)
        mujoco.mju_mulQuat(
            world_quat, self.data.xquat[magnet_body], self.relative_quat)
        joint_id = self.model.joint(f"piece_{self.attached_index}_free").id
        qadr = self.model.jnt_qposadr[joint_id]
        dadr = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qadr:qadr + 3] = world_pos
        self.data.qpos[qadr + 3:qadr + 7] = world_quat
        self.data.qvel[dadr:dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def physics_step(self):
        # Position-controlled educational simulation: advance the thin pieces
        # physically, then lock the arm to the commanded servo positions. This
        # avoids mesh/contact inertia affecting the visual trajectory.
        mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.qpos_ids] = self.ctrl
        self.data.qvel[self.dof_ids] = 0
        for index, pose in self.locked_piece_poses.items():
            joint_id = self.model.joint(f"piece_{index}_free").id
            qadr = self.model.jnt_qposadr[joint_id]
            dadr = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qadr:qadr + 7] = pose
            self.data.qvel[dadr:dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.follow_attachment()

    def lock_piece_pose(self, index: int):
        """Keep one completed fragment fixed at its planned table pose."""
        joint_id = self.model.joint(f"piece_{index}_free").id
        qadr = self.model.jnt_qposadr[joint_id]
        self.locked_piece_poses[index] = self.data.qpos[
            qadr:qadr + 7].copy()

    def clear_piece_locks(self):
        """Release completed-pose locks before generating a new scene."""
        self.locked_piece_poses.clear()

    def release(self, target_xy=None, target_piece_yaw=None, drop=False):
        if self.attached_body is not None:
            # Thin puzzle pieces are placed flat on the paper. Remove the tiny
            # residual roll/pitch left by actuator tolerance before disabling
            # the magnetic constraint, preventing a tilted collision proxy
            # from tunneling through the 2 mm paper.
            joint_id = self.model.joint(
                f"piece_{self.attached_index}_free").id
            qadr = self.model.jnt_qposadr[joint_id]
            dadr = self.model.jnt_dofadr[joint_id]
            if target_xy is not None:
                self.data.qpos[qadr:qadr + 2] = target_xy
            if not drop or target_piece_yaw is not None:
                rotation = self.data.xmat[
                    self.attached_body].reshape(3, 3)
                yaw = (math.atan2(rotation[1, 0], rotation[0, 0])
                       if target_piece_yaw is None else target_piece_yaw)
                self.data.qpos[qadr + 3:qadr + 7] = [
                    math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            if not drop:
                self.data.qpos[qadr + 2] = .009
            self.data.qvel[dadr:dadr + 6] = 0
        self.attached_body = None
        self.attached_index = None
        self.relative_pos = None
        self.relative_quat = None
        self.attached_yaw = None
        self.attached_tool_yaw = None
        # pickup_body_xy intentionally remains available to the executor
        # immediately after release for final visual-origin compensation.
        mujoco.mj_forward(self.model, self.data)

    def command_home(self):
        self.ctrl = self.HOME.copy()
        self.data.ctrl[:] = self.ctrl

    def reset_home(self):
        """Atomically align state and controller; used after scene resets."""
        self.ctrl = self.HOME.copy()
        self.data.ctrl[:] = self.ctrl
        self.data.qpos[self.qpos_ids] = self.ctrl
        self.data.qvel[self.dof_ids] = 0
        mujoco.mj_forward(self.model, self.data)
