"""Process bridge between the original vision GUI and an independent MuJoCo window."""

from __future__ import annotations

import math
import multiprocessing as mp
import time

import cv2
import mujoco
import numpy as np

from mujoco_sim.robot_controller import ArmController
from mujoco_sim.run_sim import render_camera
from mujoco_sim.scene_builder import build_scene_xml, refresh_piece_geometry
from mujoco_sim.vision_planner import plan_from_camera_rgb, plans_from_solution
from puzzle_sim import resolve_cut_mode


def _child(connection, speed):
    """Own all MuJoCo/GL objects in a separate process."""
    from mujoco import viewer as mj_viewer

    viewer = renderer = None
    model = data = controller = plans = None
    piece_count = 0
    current_key = None

    def close_scene():
        nonlocal viewer, renderer
        if viewer is not None:
            viewer.close()
            viewer = None
        if renderer is not None:
            renderer.close()
            renderer = None

    def sync_delay():
        if viewer is not None and viewer.is_running():
            viewer.sync()
        time.sleep(.0015 / max(.1, float(speed.value)))

    def move(target, yaw, piece=-1, translation_range=None):
        """IK first, then execute a zero-velocity/acceleration quintic path."""
        start_xyz = controller.magnet_position
        q_start = controller.ctrl.copy()
        q_goal = controller.solve_ik(target, yaw)
        cartesian_distance = float(
            np.linalg.norm(np.asarray(target) - start_xyz))
        joint_distance = float(np.max(np.abs(q_goal - q_start)))
        # Nominal limits are intentionally conservative for a readable demo.
        duration = max(.75, cartesian_distance / .10, joint_distance / .65)
        samples = max(24, int(duration * 60))
        # Keep UI traffic near 30 Hz in wall-clock time even at high playback
        # speed. Sending every simulated frame at 4x can fill the pipe and
        # block the robot process immediately before placement.
        emit_stride = max(2, int(math.ceil(2 * float(speed.value))))
        for step in range(samples + 1):
            u = step / samples
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            controller.command_joint_position(
                q_start + (q_goal - q_start) * blend)
            if viewer is not None and viewer.is_running():
                viewer.sync()
            if (translation_range is not None and
                    (step % emit_stride == 0 or step == samples)):
                lo, hi = translation_range
                connection.send((
                    "motion_pose", piece,
                    float(lo + (hi - lo) * blend), 0.0))
            time.sleep((1.0 / 60.0) / max(.1, float(speed.value)))

    def rotate_attached_piece(desired_yaw, piece):
        """Rotate joint6 until the attached piece reaches its world yaw."""
        for correction_pass in range(2):
            body_id = controller.attached_body
            rotation = data.xmat[body_id].reshape(3, 3)
            current_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            error = math.atan2(
                math.sin(desired_yaw - current_yaw),
                math.cos(desired_yaw - current_yaw))
            if abs(error) < math.radians(.08):
                break
            q_start = controller.ctrl.copy()
            q_goal = q_start.copy()
            # With the magnetic face downward, negative joint6 is positive
            # world yaw. The magnet and piece share one rigid transform.
            q_goal[5] = np.clip(
                q_start[5] - error,
                controller.LOWER[5], controller.UPPER[5])
            duration = max(.65, abs(error) / .65)
            samples = max(40, int(duration * 60))
            emit_stride = max(
                1, int(math.ceil(2 * float(speed.value))))
            for step in range(samples + 1):
                u = step / samples
                blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                controller.command_joint_position(
                    q_start + (q_goal - q_start) * blend)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                if step % emit_stride == 0 or step == samples:
                    connection.send((
                        "motion_pose", piece, 1.0, float(blend)))
                time.sleep(
                    (1 / 60) / max(.1, float(speed.value)))

    def move_to_folded_home():
        """Return to the exact startup pose with a smooth quintic trajectory."""
        q_start = controller.ctrl.copy()
        q_goal = controller.HOME.copy()
        joint_distance = float(np.max(np.abs(q_goal - q_start)))
        # The task is already complete here. Use a faster but still zero-
        # velocity/acceleration quintic return so the last placement does not
        # appear to hang for several seconds.
        duration = max(1.1, joint_distance / .90)
        samples = max(66, int(duration * 60))
        connection.send(("stage", -1, "机械臂折叠复位"))
        for step in range(samples + 1):
            u = step / samples
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            controller.command_joint_position(
                q_start + (q_goal - q_start) * blend)
            if viewer is not None and viewer.is_running():
                viewer.sync()
            time.sleep((1 / 60) / max(.1, float(speed.value)))
        # Remove all numerical residue so end state equals startup exactly.
        controller.reset_home()
        if viewer is not None and viewer.is_running():
            viewer.sync()

    try:
        while True:
            command = connection.recv()
            if command[0] == "close":
                break
            try:
                if command[0] == "generate":
                    seed, requested_count = int(command[1]), int(command[2])
                    material_mode = command[3] if len(command) > 3 else "color"
                    cut_mode = command[4] if len(command) > 4 else "sequential"
                    key = (seed, requested_count, material_mode, cut_mode)
                    # Reusing an unchanged scene keeps the native MuJoCo
                    # window alive instead of flashing off and on.
                    if renderer is None:
                        piece_count = 4
                        model = mujoco.MjModel.from_xml_string(
                            build_scene_xml(seed, 4, material_mode, cut_mode))
                        data = mujoco.MjData(model)
                        controller = ArmController(model, data, 4)
                        for _ in range(30):
                            controller.physics_step()
                        renderer = mujoco.Renderer(
                            model, height=1200, width=900)
                        plans = None
                        viewer = mj_viewer.launch_passive(model, data)
                        # Start with a useful angle but leave mjCAMERA_FREE so
                        # the user can rotate, pan and zoom normally.
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.lookat[:] = [-.03, 0, .12]
                        viewer.cam.distance = .82
                        viewer.cam.azimuth = 135
                        viewer.cam.elevation = -32
                        viewer.sync()
                    if key != current_key:
                        controller.release()
                        controller.clear_piece_locks()
                        mujoco.mj_resetData(model, data)
                        controller.reset_home()
                        refresh_piece_geometry(
                            model, data, seed, requested_count,
                            material_mode, cut_mode)
                        # Freeze the camera-observed scene until each fragment
                        # is deliberately picked. Without this, conservative
                        # cylindrical collision proxies can intersect even
                        # when the actual polygons have legal clearance and
                        # drift before the robot reaches them.
                        for piece_index in range(requested_count):
                            controller.lock_piece_pose(piece_index)
                        for mesh_index in range(4):
                            for mesh_name in (
                                    f"piece_mesh_{mesh_index}",
                                    f"piece_border_mesh_{mesh_index}"):
                                mesh_id = model.mesh(mesh_name).id
                                mujoco.mjr_uploadMesh(
                                    model, renderer._mjr_context, mesh_id)
                                viewer.update_mesh(mesh_id)
                        plans = None
                        current_key = key
                        viewer.sync()
                    rgb = render_camera(model, data, renderer)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # Same exposure correction as the pure-vision planner.
                    # Without it the saturated MuJoCo blue can miss the
                    # original GUI's color threshold.
                    bgr = np.clip(
                        bgr.astype(np.float32) * .85, 0, 255).astype(np.uint8)
                    connection.send(("generated", bgr))
                elif command[0] == "detect":
                    if renderer is None:
                        raise RuntimeError("请先生成 MuJoCo 场景")
                    rgb = render_camera(model, data, renderer)
                    solver_mode = resolve_cut_mode(
                        seed, requested_count, cut_mode)
                    plans = plan_from_camera_rgb(rgb, solver_mode)[0]
                    connection.send(("detected", len(plans)))
                elif command[0] == "stitch":
                    if renderer is None:
                        raise RuntimeError("请先生成 MuJoCo 场景")
                    if len(command) >= 5:
                        # These arrays came from the camera image displayed by
                        # the GUI, not from scene-generation ground truth.
                        # Reuse them so both windows execute one visual plan.
                        plans = plans_from_solution(
                            command[1], command[2],
                            int(command[3]), int(command[4]))
                    elif plans is None:
                        rgb = render_camera(model, data, renderer)
                        solver_mode = resolve_cut_mode(
                            seed, requested_count, cut_mode)
                        plans = plan_from_camera_rgb(rgb, solver_mode)[0]
                    moved = set()
                    for order, plan in enumerate(plans):
                        is_last = order == len(plans) - 1
                        above_pick = np.r_[plan.pickup_xy, .075]
                        # Piece top is about z=8 mm. The magnetic face stops
                        # exactly 10 mm above it before energizing.
                        pick = np.r_[plan.pickup_xy, .018]
                        above_target = np.r_[plan.target_xy, .075]
                        place = np.r_[plan.target_xy, .019]
                        connection.send(("stage", order, "移动到碎片上方"))
                        move(above_pick, 0)
                        connection.send(("stage", order, "下降并吸附"))
                        move(pick, 0)
                        body_id = controller.attach_nearest()
                        if body_id is None or body_id in moved:
                            raise RuntimeError(f"P{order} 电磁吸附失败")
                        moved.add(body_id)
                        initial_rot = data.xmat[body_id].reshape(3, 3)
                        initial_yaw = math.atan2(
                            initial_rot[1, 0], initial_rot[0, 0])
                        c_delta = math.cos(plan.target_yaw)
                        s_delta = math.sin(plan.target_yaw)
                        delta_rotation = np.array([
                            [c_delta, -s_delta],
                            [s_delta, c_delta]])
                        # The visual polygon's vertex mean is a grasping
                        # reference, not necessarily the MuJoCo free-body
                        # origin (fan/triangle pieces can differ by millimetres).
                        # Rotate that measured origin offset with the fragment.
                        desired_body_xy = (
                            plan.target_xy
                            + delta_rotation @ (
                                controller.pickup_body_xy
                                - plan.pickup_xy))
                        # Keep joint6 fixed after energizing: lift and transfer
                        # first, with no piece rotation.
                        move(above_pick, 0)
                        move(above_target, 0, order, (0.0, 1.0))
                        connection.send(("stage", order, "目标上方第六轴对齐"))
                        desired_piece_yaw = initial_yaw + plan.target_yaw
                        rotate_attached_piece(desired_piece_yaw, order)
                        # Descend without changing joint6, then make a small
                        # final closed-loop correction before power-off.
                        held_rotation = -controller.ctrl[5]
                        move(place, held_rotation)
                        rotate_attached_piece(desired_piece_yaw, order)
                        released_index = controller.attached_index
                        controller.release(drop=True)
                        free_joint = model.joint(
                            f"piece_{released_index}_free").id
                        qadr = model.jnt_qposadr[free_joint]
                        dadr = model.jnt_dofadr[free_joint]
                        collision_id = model.geom(
                            f"piece_{released_index}_collision").id
                        # The collision proxy is a conservative cylinder. It
                        # is useful while picking, but cylinders belonging to
                        # tightly tessellated fragments overlap at the target
                        # and would push already placed pieces away. Once the
                        # magnet is off, smoothly settle the measured body
                        # origin to the vision-planned pose and make that
                        # placed fragment kinematic for the rest of this run.
                        model.geom_contype[collision_id] = 0
                        model.geom_conaffinity[collision_id] = 0
                        start_xy = data.qpos[qadr:qadr + 2].copy()
                        start_z = float(data.qpos[qadr + 2])
                        released_rot = data.xmat[body_id].reshape(3, 3)
                        start_yaw = math.atan2(
                            released_rot[1, 0], released_rot[0, 0])
                        yaw_delta = math.atan2(
                            math.sin(desired_piece_yaw - start_yaw),
                            math.cos(desired_piece_yaw - start_yaw))
                        connection.send((
                            "stage", order,
                            "释放并快速确认落位" if is_last else "释放并确认落位"))
                        settle_frames = 10 if is_last else 18
                        for settle_frame in range(settle_frames):
                            u = (settle_frame + 1) / settle_frames
                            # Quintic smoothstep: zero speed/acceleration at
                            # both ends, so correction is visible but silky.
                            blend = u * u * u * (
                                u * (u * 6.0 - 15.0) + 10.0)
                            xy = (
                                start_xy * (1.0 - blend)
                                + desired_body_xy * blend)
                            z = start_z * (1.0 - blend) + .007 * blend
                            yaw = start_yaw + yaw_delta * blend
                            data.qpos[qadr:qadr + 7] = [
                                xy[0], xy[1], z,
                                math.cos(yaw / 2.0), 0.0, 0.0,
                                math.sin(yaw / 2.0)]
                            data.qvel[dadr:dadr + 6] = 0
                            mujoco.mj_forward(model, data)
                            if viewer is not None and viewer.is_running():
                                viewer.sync()
                            time.sleep(
                                (1 / 60) / max(.1, float(speed.value)))
                        controller.lock_piece_pose(released_index)
                        connection.send(("motion_pose", order, 1.0, 1.0))
                        connection.send((
                            "stage", order,
                            "末块完成，抬升并折叠复位"
                            if is_last else "抬升前往下一块"))
                        move(above_target, 0)
                    move_to_folded_home()
                    connection.send(("done",))
            except Exception as exc:
                # A bad frame or unreachable pose must not kill/recreate the
                # native viewer. Report it and keep accepting UI commands.
                connection.send(("error", str(exc)))
    except EOFError:
        pass
    finally:
        close_scene()
        connection.close()


class MuJoCoBridge:
    def __init__(self):
        context = mp.get_context("spawn")
        self.parent, child = context.Pipe()
        self.speed = context.Value("d", 1.0)
        self.process = context.Process(
            target=_child, args=(child, self.speed), daemon=True)
        self.process.start()

    def send(self, *message):
        if self.process.is_alive():
            self.parent.send(message)

    def poll(self):
        events = []
        while self.parent.poll():
            events.append(self.parent.recv())
        return events

    def set_speed(self, value):
        self.speed.value = float(value)

    def close(self):
        if self.process.is_alive():
            try:
                self.parent.send(("close",))
                self.process.join(timeout=1.5)
            finally:
                if self.process.is_alive():
                    self.process.terminate()
        self.parent.close()
