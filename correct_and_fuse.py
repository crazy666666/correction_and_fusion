import cv2
import numpy as np
from filter_adaptive_weights import Gaussian_weight,W_k
from fast_guided_filtering import Filter
# from gamma_adaptive_weights import gamma_weight


def correct_exposure(I,ksize):

    G_kernel = Gaussian_weight(sigma=3,size=ksize)
    w_k = W_k(cv2.cvtColor(I,cv2.COLOR_BGR2GRAY),G_kernel)
    w_k = np.repeat(w_k[..., None], 3, axis=-1)
    img = I.astype(float) / 255.
    guideFilter_img = Filter(img,(ksize,ksize),0.031,0.5,w_k=w_k)
    L_refined = np.clip(guideFilter_img, 1e-3, 1) ** (np.log10(0.5) / np.log10(np.mean(guideFilter_img)))
    im_corrected = img / L_refined
    img_en = np.clip(im_corrected * 255, 0, 255).astype("uint8")

    return img_en


def fuse_images(img, img_positive, img_reverse,bc = 1.0, bs = 1.0, be = 1.0):

    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [img, img_positive, img_reverse]
    fused_images = merge_mertens.process(images)

    return fused_images

def positive_reverse_correction(img,ksize):

    img_positive = correct_exposure(img,ksize)
    img_reverse = 255 - correct_exposure(255-img,ksize)
    img_all_corrected = fuse_images(img, img_positive, img_reverse)

    return np.clip(img_all_corrected * 255, 0, 255).astype("uint8")


if __name__ == '__main__':
    img = cv2.imread('../1200_1200.bmp', 1)
    img_corrected = positive_reverse_correction(img, 7)