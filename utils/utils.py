import os
import cv2
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from skimage.color import label2rgb
from skimage.segmentation import slic, felzenszwalb
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from .structure import HierarchicalGrouping
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_list_images(path_images):
    filename_list = []
    for roots, dirs_, filenames in os.walk(path_images):
        for filename in filenames:
            if filename.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
                filename_list.append(os.path.join(roots, filename))
    return filename_list


def resize_crop(image, size=1080):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(size * h / w), size
        else:
            h, w = size, int(size * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 12) * 12, (w // 12) * 12
    image = image[:h, :w, :]
    return image


def write_batch_image(image, save_dir, name, n):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = image.numpy()
    fused_dir = os.path.join(save_dir, name)
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            image[k] = (image[k] + 1) * 127.5
            fused_image[i].append(image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image.astype(np.uint8))


def selective_adacolor(batch_image, seg_num=200, power=1, num_job=2, use_parallel=False):
    if not use_parallel:
        batch_out = []
        for image in batch_image:
            batch_out.append(color_ss_map(image))
    else:
        batch_out = Parallel(n_jobs=num_job)(
            delayed(color_ss_map)(np.asarray(image), seg_num, power) for image in batch_image)
    return np.asarray(batch_out)


def process_slic(image, seg_num=200, sigma=1.2):
    seg_label = slic(image, n_segments=seg_num, sigma=sigma,
                     compactness=10, convert2lab=True)
    image = label2rgb(seg_label, image, kind='avg')
    return image


def simple_superpixel(batch_image, seg_num=200, sigma=1.2, use_parallel=False, num_job=2):
    if not use_parallel:
        batch_out = []
        for image in batch_image:
            batch_out.append(process_slic(image, seg_num, sigma))
    else:
        batch_out = Parallel(n_jobs=num_job)(
            delayed(process_slic)(image, seg_num, sigma) for image in batch_image)
    return np.asarray(batch_out)


def color_ss_map(image, seg_num=200, power=1,
                 color_space='Lab', k=10, sim_strategy='CTSF'):
    img_seg = felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='avg')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping
    while S.num_regions() > seg_num:
        i, j = S.get_highest_similarity()
        S.merge_region(i, j)
        S.remove_similarities(i, j)
        S.calculate_similarity_for_new_region()

    image = label2rgb(S.img_seg, image, kind='avg')
    image = (image + 1) / 2
    image = image ** power
    image = image / np.max(image)
    image = image * 2 - 1

    return image


def switch_color_space(img, target):
    """
        RGB to target color space conversion.
        I: the intensity (grey scale), Lab, rgI: the rg channels of
        normalized RGB plus intensity, HSV, H: the Hue channel H from HSV
    """

    if target == 'HSV':
        return rgb2hsv(img)

    elif target == 'Lab':
        return rgb2lab(img)

    elif target == 'I':
        return rgb2gray(img)

    elif target == 'rgb':
        img = img / np.sum(img, axis=0)
        return img

    elif target == 'rgI':
        img = img / np.sum(img, axis=0)
        img[:, :, 2] = rgb2gray(img)
        return img

    elif target == 'H':
        return rgb2hsv(img)[:, :, 0]

    else:
        raise "{} is not suported.".format(target)


def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = tf.split(image1, num_or_size_splits=3, axis=3)
    b2, g2, r2 = tf.split(image2, num_or_size_splits=3, axis=3)
    if mode == 'uniform':
        b_weight = tf.random.uniform(shape=[1], minval=0.014, maxval=0.214)
        g_weight = tf.random.uniform(shape=[1], minval=0.487, maxval=0.687)
        r_weight = tf.random.uniform(shape=[1], minval=0.199, maxval=0.399)
    else:  # norm
        b_weight = tf.random.normal(shape=[1], mean=0.114, stddev=0.1)
        g_weight = tf.random.normal(shape=[1], mean=0.587, stddev=0.1)
        r_weight = tf.random.normal(shape=[1], mean=0.299, stddev=0.1)

    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    output2 = (b_weight * b2 + g_weight * g2 + r_weight * r2) / (b_weight + g_weight + r_weight)
    return output1, output2
