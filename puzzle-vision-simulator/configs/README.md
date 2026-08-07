# Configurations

Store portable defaults under this directory. Machine-specific calibration and
runtime tuning belong under `configs/local/`; that directory is intentionally
ignored and must be created on each deployment target.

Copy the default files as a starting point, then generate local values for the
camera, A4 position, and gantry zero on the target.

`vision_detection.json` contains the committed default parameters for piece
segmentation, outer-edge extraction, contour filtering, and polygon corner
approximation. The live detection GUI saves machine-specific tuning to
`configs/local/vision_detection.json`; automatic recognition loads the local
file first and falls back to the committed default.
