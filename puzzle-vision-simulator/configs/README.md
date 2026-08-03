# Configurations

Store committed defaults under this directory. The three stable deployment files
under `configs/local/` are also versioned because the Jetson target needs the
same calibrated machine profile as the PC:

- `local/a4_roi.json`
- `local/calibration.json`
- `local/vision_detection.json`

Temporary click drafts and debug settings remain ignored. If the camera, A4
position, or gantry zero changes, regenerate the local profile on the target.

`vision_detection.json` contains the committed default parameters for piece
segmentation, outer-edge extraction, contour filtering, and polygon corner
approximation. The live detection GUI saves machine-specific tuning to
`configs/local/vision_detection.json`; later automatic recognition should load
the local file first and fall back to the committed default.
