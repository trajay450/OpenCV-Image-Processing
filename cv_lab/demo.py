from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .processing import gaussian_blur,histogram_equalization,canny_edges,threshold_otsu,morphological_open

def make_sample(seed=42):
    rng=np.random.default_rng(seed)
    image=np.full((256,256),45,dtype=np.uint8)
    cv2.circle(image,(80,95),45,180,-1)
    cv2.rectangle(image,(135,135),(215,220),235,-1)
    noise=rng.normal(0,14,image.shape)
    return np.clip(image.astype(float)+noise,0,255).astype(np.uint8)

def main():
    original=make_sample()
    blurred=gaussian_blur(original)
    mask=threshold_otsu(blurred)
    results={'Original':original,'Gaussian blur':blurred,
             'Histogram equalization':histogram_equalization(original),
             'Canny edges':canny_edges(blurred),
             'Otsu threshold':mask,
             'Morphological opening':morphological_open(mask)}
    fig,axes=plt.subplots(2,3,figsize=(9,6))
    for ax,(label,image) in zip(axes.flat,results.items()):
        ax.imshow(image,cmap='gray',vmin=0,vmax=255)
        ax.set_title(label); ax.axis('off')
    fig.tight_layout()
    out=Path('outputs');out.mkdir(exist_ok=True)
    fig.savefig(out/'processing_grid.png',dpi=160);plt.close(fig)
    print(f'Saved {out / "processing_grid.png"}')

if __name__=='__main__':main()
