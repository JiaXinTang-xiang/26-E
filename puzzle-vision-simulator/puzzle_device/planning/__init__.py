"""Puzzle assembly and motion planning."""

from .assembly import (
    AssemblyConfig,
    AssemblyPlan,
    legacy_4_0_config,
    relaxed_card_config,
    solve_assembly,
    solve_self_assembly,
    solve_textured_assembly,
)
from .movement import (
    build_movement_plan,
    draw_assembly_preview,
    draw_card_candidate_gallery,
    target_rectangle_pixels,
)
from .composite_card import solve_composite_card_assembly
from .transfer import build_transfer_plan, draw_transfer_preview
from .execution import ExecutionTask, build_execution_tasks

__all__ = [
    "AssemblyConfig",
    "AssemblyPlan",
    "legacy_4_0_config",
    "relaxed_card_config",
    "build_movement_plan",
    "draw_assembly_preview",
    "draw_card_candidate_gallery",
    "ExecutionTask",
    "build_execution_tasks",
    "solve_assembly",
    "solve_self_assembly",
    "solve_textured_assembly",
    "solve_composite_card_assembly",
    "target_rectangle_pixels",
    "build_transfer_plan",
    "draw_transfer_preview",
]
