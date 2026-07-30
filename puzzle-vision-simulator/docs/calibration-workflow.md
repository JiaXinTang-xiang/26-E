# Manual Camera-to-Gantry Calibration

The first hardware workflow uses manual clicks rather than automatic tool-tip
detection. The operator moves the gantry to known X/Y pulse positions, clicks
the visible suction-head centre in the camera frame, then fits a pixel-to-pulse
homography from the recorded pairs.

Use at least 16 points distributed across the complete A4 work area. Keep a
separate set of five points for validation. Capture points with the suction head
at its normal pickup height to avoid perspective error caused by tool height.

The calibration application will later use the existing two-position pickup and
place command: each command stops at its source and destination so the operator
can record two point pairs.

## Current GUI

Run `python -m apps.manual_calibration_gui`. For every source/target task:

1. Enter source and target X/Y absolute pulses.
2. Optionally send the compatible legacy pick-and-place frame.
3. When the controller holds at the source point, arm source recording and
   click the visible suction-head centre.
4. Repeat for the destination point.
5. Fit after at least 16 samples, inspect errors, remove bad clicks, and save.

The current legacy controller does not stop at source and destination long
enough for manual clicks. Use its manual movement mode or temporarily add pauses
until `SRC_READY` and `DST_READY` responses are implemented.

The GUI rotates camera frames 180 degrees by default before display and point
recording. Apply the same rotation in the real-time vision pipeline before using
the exported matrix. The saved metadata records the selected rotation.
