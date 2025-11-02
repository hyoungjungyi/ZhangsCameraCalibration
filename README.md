# Zhang's Camera Calibration

This repository implements **Zhang’s camera calibration method** in Python, including intrinsic and extrinsic parameter estimation, reprojection error analysis, and visualization of results. Additionally, it contains a comparison with OpenCV's built-in `cv2.calibrateCamera`.

---

## Features

- Compute homography matrices (H) for each calibration image.
- Estimate intrinsic matrix (K) and extrinsic parameters (R, T).
- Optional non-linear refinement to reduce reprojection error.
- Reprojection error calculation and visualization:
  - Scatter plots of detected vs reprojected points.
  - Overlay of points on original images.
- Comparison between Zhang’s method and OpenCV calibration.
- Handles images where corner detection fails by removing them automatically.

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

Install dependencies using:

```bash
pip install opencv-python numpy matplotlib

```

---

## Usage

1. **Prepare images**  
   Place your chessboard calibration images in a folder, e.g., `images/`.

2. **Set chessboard parameters**  
   Edit the script to set your chessboard size:

```python
pattern_size = (columns, rows)  # e.g., (13, 9)
square_size = 2.0               # size of one square in your chosen unit
```

---

## Outputs

- **H_list**: List of homography matrices.
- **R_list**, **T_list**: Lists of rotation matrices and translation vectors.
- **P_list**: List of projection matrices.
- Reprojection error metrics and plots.
- Visualizations saved to disk.

---

## Results

- **Mean Reprojection Error (Zhang’s Method)**: ~10.67 px
- **Mean Reprojection Error (OpenCV)**: ~5.89 px
- Visualizations show detected corners (red) vs reprojected points (blue).

**Example for `IMG_1901.jpg`:**

| Method         | Rotation Matrix R                                                        | Translation Vector t  | Mean Error |
| -------------- | ------------------------------------------------------------------------ | --------------------- | ---------- |
| Zhang's Method | [[0.8128,0.0229,0.5821],[0.2114,0.9195,-0.3314],[-0.5428,0.3924,0.7425]] | [-10.63,-15.55,40.04] | 10.67 px   |
| OpenCV         | [[0.7967,0.0226,0.6040],[0.2196,0.9202,-0.3242],[-0.5631,0.3909,0.7281]] | [-10.07,-16.24,39.90] | 5.89 px    |

---

## Notes

- Zhang’s method without non-linear refinement may produce higher reprojection errors due to noise.
- OpenCV automatically handles lens distortion and non-linear optimization.
- For improved accuracy, implement optional non-linear calibration refinement.
- Images where `cv2.findChessboardCorners` fails are automatically removed to improve stability.
- Minor deviations in rotation matrices are adjusted using SVD to enforce orthogonality.

---

## References

- Zhang, Z. “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2000.
- OpenCV Documentation: [`cv2.findChessboardCorners`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
