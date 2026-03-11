import cv2 as cv
import plantcv.plantcv as pcv

def bgr2rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def get_original_img(img_bgr, mode):
    if mode == 'plot':
        return bgr2rgb(img_bgr)
    return img_bgr

def get_mask(img):
    gaussian_blur = pcv.gaussian_blur(img, (3, 3))

    gray_lab_a = pcv.rgb2gray_lab(gaussian_blur, 'a')
    gray_lab_b = pcv.rgb2gray_lab(gaussian_blur, 'b')

    thresh_a = pcv.threshold.otsu(gray_lab_a, 'dark')
    thresh_b = pcv.threshold.otsu(gray_lab_b, 'light')

    thresh = pcv.logical_or(thresh_a, thresh_b)

    kernel = pcv.get_kernel((7, 7), 'rectangle')
    mask = pcv.closing(thresh, kernel)

    return mask

def apply_mask(img, mask, mode):
    masked = pcv.apply_mask(img, mask, 'white')
    if mode == 'plot':
        return bgr2rgb(masked)
    return masked


def get_roi(img_bgr, mask, mode):
    dilated_mask = pcv.dilate(mask, ksize=7, i=1)
    edges = pcv.logical_xor(mask, dilated_mask)

    img_copy = img_bgr.copy()

    for i in range(img_bgr.shape[0]):
        for j in range(img_bgr.shape[1]):
            if edges[i][j]:
                img_copy[i][j] = [255, 0, 0] # blue edge

    if mode == 'plot':
        return bgr2rgb(img_copy)
    return img_copy

def get_analysis(img, mask, mode):
    analysis = pcv.analyze.size(img, mask, label='plant')
    if mode == 'plot':
        return bgr2rgb(analysis)
    return analysis

def get_pseudolandmarks(img, mask, mode):
    top_points, bottom_points, vertical_center_points = pcv.homology.x_axis_pseudolandmarks(
    img, mask)

    landmarks = [
        (top_points, (255, 0, 0)),
        (vertical_center_points, (0, 0, 255)),
        (bottom_points, (255, 0, 255))
    ]

    img_copy = img.copy()

    for points, color in landmarks:
        for point in points:
            x, y = point[0].astype(int)
            cv.circle(img=img_copy, center=(x, y), radius=3, color=color, thickness=-1)
    
    if mode == 'plot':
        return bgr2rgb(img_copy)
    
    return img_copy

def get_image_transformations(img_bgr, mode):
    original = get_original_img(img_bgr, mode)
    mask = get_mask(img_bgr)
    masked = apply_mask(img_bgr, mask, mode)
    roi = get_roi(img_bgr, mask, mode)
    analysis = get_analysis(img_bgr, mask, mode)
    landmarks = get_pseudolandmarks(img_bgr, mask, mode)

    return {
        "original": original,
        "blur": mask,
        "mask": masked,
        "roi": roi,
        "analyse": analysis,
        "landmarks": landmarks
    }