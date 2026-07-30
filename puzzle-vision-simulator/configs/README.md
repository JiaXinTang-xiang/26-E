# Configurations

Store machine-specific files here, but do not commit local serial-port settings,
camera intrinsics, or camera-to-gantry calibration results. Start from templates
when they are added; keep local values under `configs/local/`.

`vision_detection.json` contains the committed default parameters for piece
segmentation, outer-edge extraction, contour filtering, and polygon corner
approximation. The live detection GUI saves machine-specific tuning to
`configs/local/vision_detection.json`; later automatic recognition should load
the local file first and fall back to the committed default.
