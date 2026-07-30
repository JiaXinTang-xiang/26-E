"""Puzzle assembly and motion planning."""

from .assembly import AssemblyConfig, AssemblyPlan, solve_assembly
from .movement import build_movement_plan, draw_assembly_preview, target_rectangle_pixels
from .execution import ExecutionTask, build_execution_tasks

__all__ = [
    "AssemblyConfig",
    "AssemblyPlan",
    "build_movement_plan",
    "draw_assembly_preview",
    "ExecutionTask",
    "build_execution_tasks",
    "solve_assembly",
    "target_rectangle_pixels",
]
