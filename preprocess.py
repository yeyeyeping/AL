import pathlib
import multiprocessing
import numpy as np
from scipy.ndimage import zoom
from skimage.io import imread,imsave



target_size = 256

def to_png(src_img,src_mask, target_folder):
    img,mask = imread(src_img),imread(src_mask)
    h,w,_ = img.shape
    zoomed_img = zoom(img, (target_size/h, target_size/w,1),order=1, mode="nearest")
    zoomed_mask = zoom(mask, (target_size/h, target_size/w),order=0, mode="nearest")

    # zoomed_img  = zoomed_img / 255.0
    zoomed_mask = zoomed_mask == 255

    out_img,out_mask = target_folder/src_img.name,target_folder/src_mask.name
    imsave(out_img, zoomed_img.astype(np.uint8),check_contrast=False)
    imsave(out_mask,zoomed_mask.astype(np.uint8),check_contrast=False)

def to_numpy(src_img,src_mask, target_folder):
    img,mask = imread(src_img),imread(src_mask)
    h,w,_ = img.shape
    zoomed_img = zoom(img, (target_size/h, target_size/w,1),order=1, mode="nearest")
    zoomed_mask = zoom(mask, (target_size/h, target_size/w),order=0, mode="nearest")

    # zoomed_img  = zoomed_img / 255.0
    zoomed_mask = zoomed_mask == 255

    out_img,out_mask = target_folder/"images"/src_img.name[:-4],target_folder/"mask"/src_img.name[:-4]
    
    np.save(out_img, zoomed_img.astype(np.uint8))
    np.save(out_mask,zoomed_mask.astype(np.uint8))

l = list(pathlib.Path("/home/yeep/project/py/deeplearning/AL-ACDC/data/ISICPreprocessed").rglob("*.jpg"))

param = [[p, p.parent.parent/"mask"/(p.name[:-4]+"_segmentation.jpg"), pathlib.Path("/home/yeep/project/py/deeplearning/AL-ACDC/data/PNG256ISIC")/p.parent.parent.name] for p in l]

if __name__ == "__main__":
    with multiprocessing.get_context("spawn").Pool(5) as p:
        r = p.starmap_async(to_png,param)
        r.get()
