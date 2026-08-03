# Data Layout

- `raw/`: untouched photos captured from the camera.
- `background/`: empty work-area images used for background subtraction.
- `annotations/`: manually verified piece and assembly labels.
- `calibration/`: exported calibration-point tables and validation measurements.

Captured experiment folders remain machine-specific and may be ignored. The
runtime background at `data/local/empty_work_area.png` is versioned because
background-difference recognition needs it on the Jetson target.
