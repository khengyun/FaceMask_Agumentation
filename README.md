# FaceMask_Augmentation

## Introduction

The `FaceMask_Augmentation` tool is designed to help users create and adjust facial masks for image augmentation. It enables interactive marking and adjustment of points on facial images to generate custom masks. After running the `CreateMaskPTS.py` script, you can use the `FaceMasking.py` script to visualize the generated masks on live camera feed.


## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Pillow (PIL)

Install the required dependencies using the following command:

```bash
pip install opencv-python numpy pillow
```

## Usage
1. Clone the Repository:
```bash
git clone https://github.com/khengyun/FaceMask_Augmentation.git
```
2. Navigate to the Project Directory:
```bash
cd FaceMask_Augmentation
```
3. Run the Face Mask Augmentation Tool:

    - To create and adjust masks for images in the mask_images directory:
    ```bash
    python CreateMaskPTS.py
    ```
    - Follow the on-screen instructions to mark and adjust points on each image. Press ESC to finish the adjustment.

4. Review Adjustments:

    - The adjusted mask points are stored in the `mask_pts.pkl` file.

5. Run the Face Masking Script:

    - After generating mask points, use the FaceMasking.py script to visualize the masks on a live camera feed.
    ```bash
    python FaceMasking.py
    ```
    - Adjust the camera index if needed (`cam = cv2.VideoCapture(0)`).
    - Press 'ESC' to exit the live camera feed.

6. Further Customization:

    - You can customize the script by adjusting parameters and file paths directly in the `FaceMasking.py` file.

## Additional Functionality

### Bulk Adjustment using Shift Key
-  While adjusting points, pressing the Shift key allows you to drag all points simultaneously.

### Face Masking Script Details (`FaceMasking.py`):
 - The script uses the `MTCNNFaceDetector` for face detection, replacing the previous `FaceDetector`.
 - Adjust the camera index in the script (`cam = cv2.VideoCapture(0)`) based on your setup.

