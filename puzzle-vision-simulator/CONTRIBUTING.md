# Development Rules

Keep each concern in its own module. This prevents camera, motion, and puzzle
logic from becoming coupled while the hardware is still changing.

| Additions | Location |
| --- | --- |
| Camera acquisition, undistortion, segmentation | `puzzle_device/vision/` |
| Synthetic images and simulation-only checks | `puzzle_device/simulation/` |
| Edge matching, assembly, target poses, motion ordering | `puzzle_device/planning/` |
| Pixel-to-pulse calibration and validation | `puzzle_device/calibration/` |
| Tkinter/OpenCV interfaces and CLI launchers | `apps/` |
| Captured images and calibration measurements | `data/` |
| Hardware-specific settings and persisted matrices | `configs/local/` |

Rules for new code:

1. Vision returns image-space measurements only; it does not send serial data.
2. Calibration owns all pixel-to-millimetre and pixel-to-pulse conversion.
3. Planning consumes calibrated geometry and produces device-independent moves.
4. Hardware communication is added as a separate adapter after the message
   protocol and controller state feedback are fixed.
5. Every bug fix that changes geometry should include a regression test in
   `tests/`.

6. The temporary legacy 17-byte pick-and-place encoder lives in
   `puzzle_device/calibration/gantry_protocol.py`. Do not place serial handling
   inside a vision or calibration-math module.
