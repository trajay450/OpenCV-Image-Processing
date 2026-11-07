"""Reusable image processing functions with clear inputs and outputs."""
import cv2
import numpy as np

def validate_gray(gray):
    if not isinstance(gray,np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Expected 2D grayscale uint8 numpy image.')

def gaussian_blur(gray, kernel=5):
    validate_gray(gray)
    if kernel < 1 or kernel % 2 == 0: raise ValueError('kernel must be positive and odd')
    return cv2.GaussianBlur(gray,(kernel,kernel),0)

def histogram_equalization(gray):
    validate_gray(gray)
    return cv2.equalizeHist(gray)

def canny_edges(gray, low=50, high=150):
    validate_gray(gray)
    if not (0 <= low < high): raise ValueError('Expected 0 <= low < high')
    return cv2.Canny(gray,low,high)

def threshold_otsu(gray):
    validate_gray(gray)
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

def morphological_open(binary,kernel=3):
    validate_gray(binary)
    if kernel < 1: raise ValueError('kernel must be positive')
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,kernel))
    return cv2.morphologyEx(binary,cv2.MORPH_OPEN,k)
