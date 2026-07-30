"""Build a self-contained MuJoCo scene with randomized polygon pieces."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzle_sim import random_cut  # noqa: E402


PAPER_W = 0.210
PAPER_H = 0.297
TARGET_W = 0.100
TARGET_H = 0.060
TARGET_CENTER = np.array([0.0, -0.082])
PIECE_Z = 0.010
PIXEL_TO_METER = TARGET_W / 400.0
CAMERA_FOVY = 26.3812214244
PIECE_CLEARANCE = 0.006


def _five_vertex_polygon(polygon: np.ndarray) -> np.ndarray:
    """Add collinear boundary points so every mutable mesh has five vertices."""
    points = [p.copy() for p in polygon]
    while len(points) < 5:
        # Split a non-root edge so the fixed triangle fan remains non-degenerate.
        candidates = range(1, len(points) - 1) if len(points) > 2 else range(len(points))
        index = max(
            candidates,
            key=lambda i: np.linalg.norm(points[(i + 1) % len(points)] - points[i]))
        points.insert(index + 1, (points[index] + points[(index + 1) % len(points)]) / 2)
    return np.asarray(points)


def _mesh_vertices(polygon: np.ndarray, thickness: float = 0.002) -> np.ndarray:
    """Return the fixed-size, centered 3-D vertex array for one piece."""
    local = (polygon - polygon.mean(axis=0)) * PIXEL_TO_METER
    # Keep the boundary order: taking a convex hull here would visually fill
    # the notch of a legal concave pentagon. Generated concave pieces place the
    # notch so the existing fixed five-vertex fan triangulation remains inside.
    local = _five_vertex_polygon(local)
    n = 5
    half = thickness / 2
    return np.asarray(
        [(x, y, -half) for x, y in local]
        + [(x, y, half) for x, y in local], dtype=float)


def _mesh_texcoords(polygon: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    card_xy = vertices[:, :2] / PIXEL_TO_METER + polygon.mean(axis=0)
    return np.c_[card_xy[:, 0] / 400.0, 1.0 - card_xy[:, 1] / 240.0]


def _border_polygon(polygon: np.ndarray) -> np.ndarray:
    center = polygon.mean(axis=0)
    return center + (polygon - center) * 1.025


def _mesh_xml(name: str, polygon: np.ndarray, thickness: float = 0.002) -> str:
    """Create a fixed-topology closed mesh that can be updated in place."""
    vertices = _mesh_vertices(polygon, thickness)
    n = 5
    faces: list[tuple[int, int, int]] = []
    for i in range(1, n - 1):
        faces.append((0, i + 1, i))
        faces.append((n, n + i, n + i + 1))
    for i in range(n):
        j = (i + 1) % n
        faces.extend([(i, j, n + j), (i, n + j, n + i)])
    vertex_text = " ".join(f"{v:.7f}" for xyz in vertices for v in xyz)
    # Preserve the original uncut card coordinates as UVs, so all independently
    # moving meshes still show their correct portion of one complete Joker.
    texcoords = _mesh_texcoords(polygon, vertices)
    texcoord_text = " ".join(f"{v:.7f}" for uv in texcoords for v in uv)
    face_text = " ".join(str(v) for face in faces for v in face)
    return (f'<mesh name="{name}" vertex="{vertex_text}" '
            f'texcoord="{texcoord_text}" face="{face_text}"/>')


def _sample_placements(polygons, rng: np.random.Generator):
    """Place generated pieces in the upper half without overlaps."""
    placements = [None] * len(polygons)
    occupied = []
    order = sorted(range(len(polygons)),
                   key=lambda i: abs(cv2.contourArea(
                       (polygons[i] * PIXEL_TO_METER).astype(np.float32))),
                   reverse=True)
    for index in order:
        polygon = (polygons[index] - polygons[index].mean(axis=0)) * PIXEL_TO_METER
        for _ in range(3000):
            angle = rng.uniform(-math.pi, math.pi)
            c, s = math.cos(angle), math.sin(angle)
            rotated = polygon @ np.array([[c, -s], [s, c]]).T
            low, high = rotated.min(axis=0), rotated.max(axis=0)
            x_min, x_max = -.095 - low[0], .095 - high[0]
            y_min, y_max = .018 - low[1], .135 - high[1]
            if x_min >= x_max or y_min >= y_max:
                continue
            x, y = rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)
            candidate = cv2.convexHull(
                (rotated + [x, y]).astype(np.float32))
            def safely_separated(old):
                if cv2.intersectConvexConvex(candidate, old)[0] > 1e-8:
                    return False
                a = candidate.reshape(-1, 2)
                b = old.reshape(-1, 2)
                # Minimum vertex-to-boundary distance is exact for disjoint
                # convex polygons and enforces a visible grasping clearance.
                distances = [
                    abs(cv2.pointPolygonTest(old, tuple(map(float, p)), True))
                    for p in a
                ] + [
                    abs(cv2.pointPolygonTest(
                        candidate, tuple(map(float, p)), True))
                    for p in b
                ]
                return min(distances) >= PIECE_CLEARANCE

            if all(safely_separated(old) for old in occupied):
                radius = np.max(np.linalg.norm(polygon, axis=1))
                placements[index] = (x, y, angle, radius)
                occupied.append(candidate)
                break
        else:
            raise RuntimeError("MuJoCo 上半区无法无重叠摆放碎片，请更换随机种子")
    return placements


def generate_scene_geometry(seed: int, piece_count: int,
                            cut_mode: str = "sequential"):
    rng = np.random.default_rng(seed)
    polygons = random_cut(rng, piece_count, cut_mode)
    return polygons, _sample_placements(polygons, rng)


def refresh_piece_geometry(model, data, seed: int, piece_count: int,
                           material_mode: str = "color",
                           cut_mode: str = "sequential"):
    """Refresh fixed piece slots without recompiling or replacing the viewer."""
    import mujoco

    polygons, placements = generate_scene_geometry(
        seed, piece_count, cut_mode)
    for index in range(4):
        joint_id = model.joint(f"piece_{index}_free").id
        qpos_address = model.jnt_qposadr[joint_id]
        visual_id = model.geom(f"piece_{index}_geom").id
        border_id = model.geom(f"piece_{index}_border").id
        collision_id = model.geom(f"piece_{index}_collision").id
        if index < piece_count:
            polygon = polygons[index]
            x, y, angle, radius = placements[index]
            for mesh_name, mesh_polygon in (
                    (f"piece_mesh_{index}", polygon),
                    (f"piece_border_mesh_{index}", _border_polygon(polygon))):
                mesh_id = model.mesh(mesh_name).id
                vertex_address = model.mesh_vertadr[mesh_id]
                vertices = _mesh_vertices(mesh_polygon)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.mesh_quat[mesh_id])
                rotation = rotation.reshape(3, 3)
                # MuJoCo stores mesh vertices in its principal-axis frame.
                model.mesh_vert[vertex_address:vertex_address + 10] = (
                    vertices - model.mesh_pos[mesh_id]) @ rotation
                texcoord_address = model.mesh_texcoordadr[mesh_id]
                if texcoord_address >= 0:
                    model.mesh_texcoord[
                        texcoord_address:texcoord_address + 10] = (
                            _mesh_texcoords(mesh_polygon, vertices))
            data.qpos[qpos_address:qpos_address + 7] = [
                x, y, PIECE_Z,
                math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
            model.geom_rgba[visual_id] = [.96, .96, .96, 1]
            model.geom_matid[visual_id] = model.material(
                "joker_piece" if material_mode == "joker"
                else "solid_piece").id
            model.geom_rgba[border_id, 3] = (
                0.0 if material_mode == "joker" else 1.0)
            model.geom_size[collision_id, 0] = max(.008, radius * .42)
            model.geom_contype[collision_id] = 1
            model.geom_conaffinity[collision_id] = 2
        else:
            data.qpos[qpos_address:qpos_address + 7] = [
                2 + index, 2, PIECE_Z, 1, 0, 0, 0]
            model.geom_rgba[visual_id, 3] = 0
            model.geom_rgba[border_id, 3] = 0
            model.geom_contype[collision_id] = 0
            model.geom_conaffinity[collision_id] = 0
        dof_address = model.jnt_dofadr[joint_id]
        data.qvel[dof_address:dof_address + 6] = 0
    mujoco.mj_forward(model, data)
    return polygons, placements


def build_scene_xml(seed: int = 7, piece_count: int = 4,
                    material_mode: str = "color",
                    cut_mode: str = "sequential") -> str:
    """Return MJCF. Generated geometry never leaves this scene boundary."""
    polygons, placements = generate_scene_geometry(seed, piece_count, cut_mode)

    mesh_assets = "\n".join(
        _mesh_xml(name, mesh_polygon)
        for i, polygon in enumerate(polygons)
        for name, mesh_polygon in (
            (f"piece_mesh_{i}", polygon),
            (f"piece_border_mesh_{i}", _border_polygon(polygon)))
    )
    piper_mesh_dir = ROOT / "mujoco_sim" / "assets" / "piper_l" / "meshes"
    joker_path = (
        ROOT / "mujoco_sim" / "assets" / "cards"
        / "big_joker_rounded.png")
    piece_material = "joker_piece" if material_mode == "joker" else "solid_piece"
    border_alpha = 0 if material_mode == "joker" else 1
    piper_assets = "\n".join(
        f'<mesh name="piper_{name.lower()}" file="{piper_mesh_dir / (name + ".STL")}"/>'
        for name in ("base_link", "Link1", "Link2", "Link3",
                     "Link4", "Link5", "Link6")
    )
    piece_bodies = []
    welds = []
    for i, ((x, y, angle, radius), polygon) in enumerate(zip(placements, polygons)):
        edge_a = polygon[1] - polygon[0]
        edge_b = polygon[2] - polygon[0]
        triangle_twice_area = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
        mass = max(0.008, abs(triangle_twice_area) * 2e-7)
        collision_radius = max(0.008, radius * .42)
        piece_bodies.append(f"""
        <body name="piece_{i}" pos="{x:.6f} {y:.6f} {PIECE_Z:.6f}"
              quat="{math.cos(angle/2):.7f} 0 0 {math.sin(angle/2):.7f}">
          <freejoint name="piece_{i}_free"/>
          <geom name="piece_{i}_geom" type="mesh" mesh="piece_mesh_{i}"
                mass="0.00001" material="{piece_material}"
                contype="0" conaffinity="0"/>
          <geom name="piece_{i}_border" type="mesh"
                mesh="piece_border_mesh_{i}" pos="0 0 -0.0002"
                mass="0.00001" material="piece_border"
                rgba="0.96 0.96 0.96 {border_alpha}"
                contype="0" conaffinity="0"/>
          <geom name="piece_{i}_collision" type="cylinder"
                size="{collision_radius:.6f} 0.001"
                mass="{mass:.5f}" rgba="0 0 0 0"
                contype="1" conaffinity="2"
                friction="1.2 0.02 0.002" solref="0.004 1"/>
        </body>""")
        welds.append(
            f'<weld name="magnet_piece_{i}" body1="magnet" body2="piece_{i}" '
            'active="false" solref="0.002 1" solimp="0.95 0.99 0.001"/>'
        )

    return f"""
<mujoco model="2026_e_puzzle_robot">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">
    <flag contact="enable"/>
  </option>
  <visual>
    <global offwidth="900" offheight="1200"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.75 0.75 0.75"/>
  </visual>
  <asset>
    <material name="paper" rgba="0.002 0.002 0.002 1"
              specular="0" shininess="0" reflectance="0"/>
    <material name="table" rgba="0.18 0.20 0.22 1"/>
    <texture name="joker_card" type="2d" file="{joker_path}"/>
    <material name="solid_piece" rgba="0.92 0.92 0.92 1"/>
    <material name="piece_border" rgba="0.96 0.96 0.96 1"/>
    <material name="joker_piece" texture="joker_card"
              rgba="0.86 0.86 0.86 1"/>
    {mesh_assets}
    {piper_assets}
  </asset>
  <worldbody>
    <light pos="0 0 0.8" dir="0 0 -1" directional="true"/>
    <geom name="table" type="plane" size="0.65 0.65 0.02" material="table"
          contype="2" conaffinity="1"/>
    <geom name="a4_paper" type="box" pos="0 0 0.005"
          size="{PAPER_W/2} {PAPER_H/2} 0.001" material="paper"
          contype="2" conaffinity="1" friction="1.0 0.01 0.001"/>
    <geom name="divider" type="box" pos="0 0 0.007"
          size="{PAPER_W/2} 0.0008 0.0003" rgba="0.08 0.08 0.08 1"
          contype="0" conaffinity="0"/>

    <!-- 10 cm x 6 cm target outline in the lower A4 half -->
    <geom type="box" pos="0 {-0.082-TARGET_H/2:.6f} 0.0075"
          size="{TARGET_W/2} 0.0007 0.00035" rgba="0.85 0.15 0.10 1"
          contype="0" conaffinity="0"/>
    <geom type="box" pos="0 {-0.082+TARGET_H/2:.6f} 0.0075"
          size="{TARGET_W/2} 0.0007 0.00035" rgba="0.85 0.15 0.10 1"
          contype="0" conaffinity="0"/>
    <geom type="box" pos="{-TARGET_W/2:.6f} -0.082 0.0075"
          size="0.0007 {TARGET_H/2} 0.00035" rgba="0.85 0.15 0.10 1"
          contype="0" conaffinity="0"/>
    <geom type="box" pos="{TARGET_W/2:.6f} -0.082 0.0075"
          size="0.0007 {TARGET_H/2} 0.00035" rgba="0.85 0.15 0.10 1"
          contype="0" conaffinity="0"/>

    <!-- Physical camera housing plus two simulated optical views. -->
    <body name="camera_rig" pos="0.135 0 0">
      <geom type="cylinder" pos="0 0 0.33" size="0.006 0.33"
            rgba="0.18 0.20 0.22 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.03 0 0.65" size="0.04 0.025 0.018"
            rgba="0.08 0.10 0.12 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.03 0 0.628" quat="0.7071 0.7071 0 0"
            size="0.010 0.008" rgba="0.05 0.08 0.12 1"
            contype="0" conaffinity="0"/>
    </body>
    <camera name="overhead" pos="0 0 0.650" quat="1 0 0 0"
            fovy="{CAMERA_FOVY:.10f}"/>
    <camera name="scene_view" pos="0.48 -0.52 0.46"
            xyaxes="0.735 0.678 0 -0.365 0.396 0.843" fovy="42"/>

    <!-- PIPER-L: copied from the control station URDF and its validated MJCF. -->
    <body name="piper_base" pos="-0.285 0 0.008">
      <geom type="mesh" mesh="piper_base_link" rgba=".72 .74 .76 1" group="1"
            contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0.123">
        <joint name="joint1" axis="0 0 1" range="-2.618 2.618"/>
        <geom type="mesh" mesh="piper_link1" rgba=".82 .84 .86 1" group="1"
              contype="0" conaffinity="0"/>
        <body name="link2" quat="-0.0616195 0.0616141 0.704419 0.704416">
          <joint name="joint2" axis="0 0 1" range="0 3.14"/>
          <geom type="mesh" mesh="piper_link2" rgba=".82 .84 .86 1" group="1"
                contype="0" conaffinity="0"/>
          <body name="link3" pos="0.33823 0 0"
                quat="0.608799 0 0 -0.793324">
            <joint name="joint3" axis="0 0 1" range="-2.697 0"/>
            <geom type="mesh" mesh="piper_link3" rgba=".82 .84 .86 1" group="1"
                  contype="0" conaffinity="0"/>
            <body name="link4" pos="0 -0.3215 0"
                  quat="0.704964 0.704966 0.0549928 0.0549926">
              <joint name="joint4" axis="0 0 1" range="-1.832 1.832"/>
              <geom type="mesh" mesh="piper_link4" rgba=".82 .84 .86 1" group="1"
                    contype="0" conaffinity="0"/>
              <body name="link5"
                    quat="0.706432 -0.706435 -0.0308455 -0.0308456">
                <joint name="joint5" axis="0 0 -1" range="-1.22 1.22"/>
                <geom type="mesh" mesh="piper_link5" rgba=".82 .84 .86 1" group="1"
                      contype="0" conaffinity="0"/>
                <body name="link6" pos="0 -0.091 0"
                      quat="0.707105 0.707108 0 0">
                  <joint name="joint6" axis="0 0 1" range="-3.14 3.14"/>
                  <geom type="mesh" mesh="piper_link6" rgba=".82 .84 .86 1" group="1"
                        contype="0" conaffinity="0"/>
                  <!-- Unmodified electromagnet body. TCP is its real lower
                       face (+Z points downward in the grasp posture). -->
                  <body name="magnet" pos="0 0 0.006">
                    <geom name="magnet_geom" type="cylinder"
                          size="0.018 0.006" rgba="0.72 0.22 0.08 1"
                          group="1"
                          contype="0" conaffinity="0"/>
                    <!-- Asymmetric index mark makes joint6/tool rotation
                         visible; it is part of the magnet rigid body. -->
                    <geom name="magnet_index" type="box"
                          pos="0.014 0 0" size="0.003 0.004 0.0061"
                          rgba="1 0.82 0.12 1" group="1"
                          contype="0" conaffinity="0"/>
                    <site name="magnet_site" pos="0 0 0.006" size="0.006"
                          rgba="1 0.75 0.05 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    {''.join(piece_bodies)}
  </worldbody>
  <actuator>
    <position name="a_joint1" joint="joint1" kp="100" kv="20"/>
    <position name="a_joint2" joint="joint2" kp="100" kv="20"/>
    <position name="a_joint3" joint="joint3" kp="100" kv="20"/>
    <position name="a_joint4" joint="joint4" kp="80" kv="15"/>
    <position name="a_joint5" joint="joint5" kp="80" kv="15"/>
    <position name="a_joint6" joint="joint6" kp="60" kv="10"/>
  </actuator>
  <equality>
    {''.join(welds)}
  </equality>
</mujoco>
"""
