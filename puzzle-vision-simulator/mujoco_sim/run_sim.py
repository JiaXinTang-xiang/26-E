#!/usr/bin/env python3
"""Full MuJoCo puzzle simulation: camera -> vision -> magnet pick and place."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from mujoco_sim.robot_controller import ArmController
from mujoco_sim.scene_builder import build_scene_xml
from mujoco_sim.vision_planner import plan_from_camera_rgb


def settle(model, data, steps=500):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def render_camera(model, data, renderer):
    # The algorithmic camera receives the paper/pieces only. The robot is
    # group 1 and deliberately hidden here, matching a triggered image taken
    # before motion; the UI's scene camera still renders the complete robot.
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0
    option.sitegroup[:] = 0
    renderer.update_scene(data, camera="overhead", scene_option=option)
    return renderer.render().copy()


def move_segment(model, data, controller, target, yaw, viewer=None,
                 max_steps=1800, tolerance=.007):
    stable = 0
    for _ in range(max_steps):
        error = controller.step_toward(target, yaw)
        controller.physics_step()
        if viewer is not None:
            viewer.sync()
        if error < tolerance and controller.orientation_error < .025:
            stable += 1
            if stable > 45:
                return True
        else:
            stable = 0
    return False


def run(seed=7, piece_count=4, headless=False, save_dir=Path("output/mujoco")):
    xml = build_scene_xml(seed, piece_count)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    controller = ArmController(model, data, piece_count)
    settle(model, data)

    renderer = mujoco.Renderer(model, height=1200, width=900)
    rgb = render_camera(model, data, renderer)
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / "camera_input.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Strict boundary: the planner receives only the rendered RGB image.
    plans, pieces, transforms, matches, detected_bgr = plan_from_camera_rgb(rgb)
    cv2.imwrite(str(save_dir / "vision_input.png"), detected_bgr)
    print(f"视觉检测 {len(plans)} 块，匹配 {len(matches)} 对内部边")

    viewer_ctx = None
    if not headless:
        from mujoco import viewer as mj_viewer
        viewer_ctx = mj_viewer.launch_passive(model, data)

    moved_bodies = set()
    placement_records = []
    try:
        for order, plan in enumerate(plans):
            print(f"[{order + 1}/{len(plans)}] P{plan.visual_id}: "
                  f"pickup={plan.pickup_xy.round(4)}, "
                  f"target={plan.target_xy.round(4)}")
            above_pick = np.r_[plan.pickup_xy, .085]
            pick = np.r_[plan.pickup_xy, .023]
            above_target = np.r_[plan.target_xy, .085]
            place = np.r_[plan.target_xy, .024]

            move_segment(model, data, controller, above_pick, 0, viewer_ctx)
            move_segment(model, data, controller, pick, 0, viewer_ctx)
            body_id = controller.attach_nearest()
            if body_id is None or body_id in moved_bodies:
                raise RuntimeError(f"P{plan.visual_id} 吸附失败：磁铁下方没有可用碎片")
            moved_bodies.add(body_id)
            initial_rot = data.xmat[body_id].reshape(3, 3)
            initial_yaw = math.atan2(initial_rot[1, 0], initial_rot[0, 0])
            placement_records.append((
                plan.visual_id, body_id, plan.target_xy.copy(),
                initial_yaw, plan.target_yaw))
            for _ in range(80):
                controller.physics_step()
                if viewer_ctx:
                    viewer_ctx.sync()

            move_segment(model, data, controller, above_pick, plan.target_yaw, viewer_ctx)
            move_segment(model, data, controller, above_target, plan.target_yaw, viewer_ctx)
            move_segment(model, data, controller, place, plan.target_yaw, viewer_ctx)
            controller.release(
                plan.target_xy, initial_yaw + plan.target_yaw)
            for _ in range(180):
                controller.physics_step()
                if viewer_ctx:
                    viewer_ctx.sync()
            move_segment(model, data, controller, above_target, 0, viewer_ctx)

        # Retract the arm before the final overhead inspection image.
        controller.command_home()
        for _ in range(1200):
            controller.physics_step()
            if viewer_ctx:
                viewer_ctx.sync()

        final_rgb = render_camera(model, data, renderer)
        cv2.imwrite(str(save_dir / "final_result.png"),
                    cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        for visual_id, body_id, target_xy, initial_yaw, planned_delta in placement_records:
            actual_xy = data.xpos[body_id, :2]
            error_mm = np.linalg.norm(actual_xy - target_xy) * 1000
            final_rot = data.xmat[body_id].reshape(3, 3)
            final_yaw = math.atan2(final_rot[1, 0], final_rot[0, 0])
            actual_delta = math.atan2(
                math.sin(final_yaw - initial_yaw),
                math.cos(final_yaw - initial_yaw))
            yaw_error = math.degrees(math.atan2(
                math.sin(actual_delta - planned_delta),
                math.cos(actual_delta - planned_delta)))
            print(f"P{visual_id} 放置中心误差: {error_mm:.2f} mm，"
                  f"旋转误差: {yaw_error:.2f} deg，"
                  f"高度: {data.xpos[body_id, 2] * 1000:.2f} mm")
        print(f"全流程完成，结果保存到 {save_dir.resolve()}")
        if viewer_ctx is not None:
            print("仿真完成，关闭 MuJoCo 窗口退出。")
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                time.sleep(.02)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pieces", type=int, choices=range(2, 5), default=4)
    parser.add_argument("--headless", action="store_true", help="不打开窗口，适合自动测试")
    parser.add_argument("--output", type=Path, default=Path("output/mujoco"))
    args = parser.parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    run(args.seed, args.pieces, args.headless, args.output)


if __name__ == "__main__":
    main()
