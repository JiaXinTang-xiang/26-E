# Data Layout

- `raw/`: untouched photos captured from the camera.
- `background/`: empty work-area images used for background subtraction.
- `annotations/`: manually verified piece and assembly labels.
- `calibration/`: exported calibration-point tables and validation measurements.

Captured experiment folders and runtime backgrounds remain machine-specific and
are intentionally ignored. Create `data/local/empty_work_area.png` locally when
background-difference recognition is enabled on a deployment target.
