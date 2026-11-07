# OpenCV Image Processing Laboratory

A hands-on Python/OpenCV portfolio project demonstrating common image-processing operations on an original **synthetic** image.

## Operations
Gaussian blur, histogram equalization, Canny edge detection, Otsu thresholding, morphological opening, and six-panel visualization.

## Run
```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m cv_lab.demo
```
The demo saves `outputs/processing_grid.png` and does not require downloading images.

## Notes
- Functions validate grayscale, uint8 inputs and return numpy arrays.
- These techniques can be useful building blocks but are not universally beneficial; effects depend on image modality, noise, and downstream task.
- No biomedical data or unpublished research is included.

## License
MIT.
