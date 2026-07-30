# Architecture

The repository contains only code and documentation for the puzzle device.
The competition PDF and the `24-Tic-tac-toe` project remain sibling reference
folders and are not imported by this project.

```text
apps/                         User-facing entry points
puzzle_device/vision/         Camera frame processing and piece measurements
puzzle_device/simulation/     Synthetic scenes for repeatable development
puzzle_device/planning/       Assembly and motion planning
puzzle_device/calibration/    Camera-to-gantry coordinate conversion
configs/                      Configuration templates and local setup location
data/                         Captured images, backgrounds, and calibration data
tests/                        Unit and regression tests
docs/                         Architecture, setup, and experiments
output/                       Generated debug images and run results
```

## Coordinate Conventions

- Camera pixels: origin at the top-left; X right; Y down.
- Work area: millimetres after camera calibration; origin and directions are
  defined by the gantry homing setup.
- Gantry: absolute X/Y pulse coordinates after homing.

The calibration module will own camera-pixel to gantry-pulse mappings. Vision
and puzzle planning must not encode motor steps, serial ports, or machine limits.
