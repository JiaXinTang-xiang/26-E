# Data Layout

- `raw/`: untouched photos captured from the camera.
- `background/`: empty work-area images used for background subtraction.
- `annotations/`: manually verified piece and assembly labels.
- `calibration/`: exported calibration-point tables and validation measurements.

These files are intentionally ignored by Git because they are machine- and
experiment-specific. Keep a small, non-sensitive sample only when a regression
test requires it.
