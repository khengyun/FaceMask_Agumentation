# FaceMask_Augmentation

## Introduction

The `FaceMask_Augmentation` tool is designed to help users create and adjust facial masks for image augmentation. It enables interactive marking and adjustment of points on facial images to generate custom masks.

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

5. Further Customization:

    - You can customize the script by adjusting parameters and file paths directly in the `CreateMaskPTS.py` file.

## Additional Functionality
### Resize and Center Images
- Images are automatically resized and centered within a target size of (800, 600). You can adjust the target size in the resize_and_center function within the script.

- Shift Key for Bulk Adjustment
While adjusting points, pressing the Shift key allows you to drag all points simultaneously.
