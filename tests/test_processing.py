import numpy as np
import pytest
from cv_lab.demo import make_sample
from cv_lab.processing import gaussian_blur,histogram_equalization,canny_edges,threshold_otsu,morphological_open

def test_outputs():
    im=make_sample()
    for fn in [gaussian_blur,histogram_equalization,canny_edges,threshold_otsu,morphological_open]:
        out=fn(im)
        assert out.shape==im.shape and out.dtype==np.uint8

def test_bad_input():
    with pytest.raises(ValueError): gaussian_blur(np.zeros((8,8,3),dtype=np.uint8))
    with pytest.raises(ValueError): gaussian_blur(np.zeros((8,8),dtype=np.uint8),kernel=4)

def test_reproducible():
    assert np.array_equal(make_sample(),make_sample())
