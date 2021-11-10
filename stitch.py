import math
from itertools import compress
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def prepare_image_for_matching(img: np.ndarray, roi: tuple = None, downscale_factor=4, kernel_size=3) -> np.ndarray:
    """
    Prepares image for keypoint matching by cropping, shrinking and blurring image
    :param img: Image in BGR format
    :param roi: Region of interest to crop image to in format x0, y0, x1, y1
    :param downscale_factor: Factor to downscale image
    :param kernel_size: Size of kernel for blurring
    :return: Grayed, downscaled and blurred image
    """
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[roi[1]:roi[3], roi[0]:roi[2]] if roi is not None else img
    img = cv2.resize(img, (int(img.shape[1] // downscale_factor), int(img.shape[0] // downscale_factor)))
    img = cv2.medianBlur(img, kernel_size)
    
    return img


def get_matches(descs1: np.ndarray, descs2: np.ndarray, k=2, filter_by_distance=True, ratio_thresh=0.7, increase_by=0.1, filter_by_cosine=True,
                pts1: np.ndarray = None, pts2: np.ndarray = None) -> list:
    """
    Matches keypoints and returns (filtered) matches using a heuristical matcher
    :param descs1: Descriptors of keypoints of first image
    :param descs2: Descriptors of keypoints of second image
    :param ratio_thresh: Ratio of threshold to filter matches. The smaller the more matches will be filtered
    :param k: Count of best matches. Used for `knnMatch` function
    :param filter_by_distance: Whether to filter matches based on distance
    :param filter_by_cosine: Whether to filter matches based on cosine similarity of corresponding point pairs
    :param pts1: Points of first image
    :param pts2: Points of second image
    :param increase_by: Value to increase `ratio_thresh` by until keypoints are found or 0.99 is reached
    :return: (Filtered) matches
    """
    assert not filter_by_cosine or (pts1 is not None and pts2 is not None), 'If filtering by cosine similarity points need to be given'
    
    matcher = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    matches = list(matcher.knnMatch(descs1, descs2, k=k))
    
    # Filter good matches based on distance and given threshold
    if filter_by_distance:
        tries, good_matches = 0, []
        
        # Succesively increaser ratio_thresh to have at least 4 matches which are needed to calculate homography
        while len(good_matches) < 4 and ratio_thresh + (tries * increase_by) < 0.99:
            good_matches = [m for m in matches if len(m) > 1 and m[0].distance < min(ratio_thresh + (tries * increase_by), 0.99) * m[1].distance]
            tries += 1
        
        matches = good_matches
    
    # Filters matches based on cosine similarity of their corresponding point pairs
    if filter_by_cosine:
        matched_pts1 = pts1[[match[0].queryIdx for match in matches]]
        matched_pts2 = pts2[[match[0].trainIdx for match in matches]]
        
        # Calculate angles between all pairs of matched points
        angles = [abs(math.atan2(pt1[0] - pt2[0], pt1[1] - pt2[1])) for pt1, pt2 in zip(matched_pts1, matched_pts2)]
        
        # Calculate Interquartile Range
        q75, q25 = np.percentile(angles, [75, 25])
        iqr = q75 - q25
        
        # Create mask and filter matched points
        mask = (angles > q25 - 1.5 * iqr) & (angles < q75 + 1.5 * iqr)
        
        matches = list(compress(matches, mask))
    
    return matches


def plot_matches(img1: np.ndarray, img2: np.ndarray, pts1: list, pts2: list, matches: list, keypoint=False):
    """
    Plots the matches of two images
    :param img1: First image
    :param img2: Second image
    :param pts1: First list of either cv2.KeyPoints or tuples of coordinates
    :param pts2: Second list of either cv2.KeyPoints or tuples of coordinates
    :param matches: List of matches
    :param keypoint: Whether the points are of type cv2.KeyPoint or standard tuples of integer coordinates
    """
    
    kps1, kps2 = [[cv2.KeyPoint(*pt, 50) for pt in pts] for pts in [pts1, pts2]] if not keypoint else (pts1, pts2)
    
    vis = cv2.drawMatchesKnn(img1, kps1, img2, kps2, matches, None, flags=2)
    plt.imshow(vis)
    plt.show()


def get_points_and_descriptors(img: np.ndarray, roi: tuple = None, downscale_factor=4, kernel_size=3) -> (list, np.ndarray):
    """
    Determines points and descriptors in an image using SIFT algorithm
    :param img: Image to find keypoints in
    :param roi: Region of interest in image to find keypoints in
    :param downscale_factor: Factor to downscale image to make SIFT more robust
    :param kernel_size: Size of blurring kernel to make SIFT more robust
    :return: Keypoints and descriptors
    """
    
    detector = cv2.xfeatures2d.SIFT_create()
    
    img_prepd = prepare_image_for_matching(img, roi, downscale_factor, kernel_size)
    kps, descs = detector.detectAndCompute(img_prepd, None)
    
    return list(kps), descs


def get_matching_points(img1: np.ndarray, img2: np.ndarray, roi1: tuple = None, roi2: tuple = None, downscale_factor=4, kernel_size=3, k=2, ratio_thresh=0.7,
                        increase_by=0.1, plot=False) -> (np.ndarray, np.ndarray):
    """
    Using points and descriptors retrieved by a keypoint matcher (here: AKAZE) generate bounding boxes on a new image
    For robustness and accuracy, for each labeled object only nearby keypoints are considered and outliers are removed
    :param img1: Reference image
    :param img2: New image to compare
    :param roi1: Region of interest to crop image to in format x0, y0, x1, y1
    :param roi2: Region of interest to crop image to in format x0, y0, x1, y1
    :param downscale_factor: Scaling factor to resize image for better feature extraction
    :param kernel_size: Size of kernel for blurring
    :param ratio_thresh: Threshold to filter out bad matches [0, 1] (the higher the more matches will be considered)
    :param increase_by: Value to increase `ratio_thresh` by until matches are found or 0.99 is reached
    :param k: Count of best matches. Used for filtering matches
    :param plot: Whether to plot the matched pairs of keypoints
    :return: Matched points for both images
    """
    
    (kps1, descs1), (kps2, descs2) = [get_points_and_descriptors(img, roi, downscale_factor, kernel_size) for img, roi in zip([img1, img2], [roi1, roi2])]
    
    # Get actual points for original image by transforming back the keypoints
    pts1, pts2 = [np.array([kp.pt for kp in kps]) * downscale_factor + (roi[:2] if roi is not None else (0, 0)) for kps, roi in zip([kps1, kps2], [roi1, roi2])]
    
    matches = get_matches(descs1, descs2, k, True, ratio_thresh, increase_by, True, pts1, pts2)
    
    # Get actual matched points using matches
    matched_pts1 = pts1[[match[0].queryIdx for match in matches]]
    matched_pts2 = pts2[[match[0].trainIdx for match in matches]]
    
    if plot:
        plot_matches(img1, img2, pts1, pts2, matches, keypoint=False)
    
    return matched_pts1, matched_pts2


def stitch_images(img1: np.ndarray, img2: np.ndarray, homography: np.ndarray, overlay=False) -> (np.ndarray, list):
    """
    Stitches two images by warping second image to first image and adding first image to the warped image
    :param img1: First image
    :param img2: Second image to be warped on the other image
    :param homography: Calculated homography
    :param overlay: Whether to overlay image in background at black areas
    :return: Stitched image and used translation
    """
    
    pts1, pts2 = [np.float32([[[0, 0]], [[0, height]], [[width, height]], [[width, 0]]]) for height, width in [img1.shape[:2], img2.shape[:2]]]
    pts2_transformed = cv2.perspectiveTransform(pts2, homography)
    pts = np.concatenate((pts1, pts2_transformed), axis=0)
    
    xmin, ymin = np.int32(pts.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    homography_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    stitched_img = cv2.warpPerspective(img2, homography_translation.dot(homography), (xmax - xmin, ymax - ymin))
    
    if overlay:
        mask = np.any(stitched_img[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]] != (0, 0, 0), axis=-1) & np.any(img1 != (0, 0, 0), axis=-1)
        
        stitched_img[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]] += img1
        stitched_img[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]][mask] = img1[mask]
    else:
        stitched_img[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]] = img1
    
    return stitched_img, t


def create_panorama(img_paths: list, rois: list = None, downscale_factor=4, kernel_size=3, k=2, ratio_thresh=0.7, increase_by=0.1, ransac_thresh=4) -> (
        np.ndarray, list):
    """
    Creates a panorama by subsequently stitching neighbouring images
    Image paths must be sorted but can be from right to left or the other way around
    :param img_paths: List of sorted image paths
    :param rois: List of Region of Interests for each image. Can be None for an image when whole image is region of interest
    :param downscale_factor: Factor to downscale image for more robust keypoint detection
    :param kernel_size: Size of kernel for blurring image for more robust keypoint detection. Number must be odd. The higher the more blurry
    :param ratio_thresh: Ratio of threshold to filter matches. The lower the more matches will be filtered out
    :param increase_by: Value to increase `ratio_thresh` by until matches are found or 0.99 is reached
    :param k: Count of best matches. Used for getting matches
    :param ransac_thresh: Threshold for RANSAC algorithm to find homography
    :return: Panorama of multiple stitched images and last translation
    """
    if rois is None:
        rois = [None for _ in img_paths]
    
    stitched_img, last_t = None, None
    
    # Create generator with pairs of subsequent items
    gen = zip(*[[lst[i:i + 2] for i in range(len(lst) - 1)] for lst in [img_paths, rois]])
    
    for img_paths, rois in tqdm(gen, total=len(img_paths) - 1):
        img1, img2 = [cv2.imread(path) for path in img_paths]
        
        # Stitch subsequent images
        matched_pts1, matched_pts2 = get_matching_points(img2, img1, *rois, downscale_factor, kernel_size, k, ratio_thresh, increase_by, plot=True)
        homography, _ = cv2.findHomography(matched_pts2 + ([0, 0] if last_t is None else last_t), matched_pts1, cv2.RANSAC,
                                           ransacReprojThreshold=ransac_thresh)
        stitched_img, last_t = stitch_images(img2, img1 if stitched_img is None else stitched_img, homography)
    
    return stitched_img, last_t


def create_centered_panorama(img_paths: list, rois: list = None, downscale_factor=4, kernel_size=3, k=2, ratio_thresh=0.7, increase_by=0.1,
                             ransac_thresh=4) -> np.ndarray:
    """
    Creates a centered panorama by subsequently stitching neighbouring images for both halfs of the images and then combining those two panoramas
    Image paths must be sorted but can be from right to left or the other way around
    :param img_paths: List of sorted image paths
    :param rois: List of Region of Interests for each image. Can be None for an image when whole image is region of interest
    :param downscale_factor: Factor to downscale image for more robust keypoint detection
    :param kernel_size: Size of kernel for blurring image for more robust keypoint detection. Number must be odd. The higher the more blurry
    :param k: Count of best matches. Used for getting matches
    :param ratio_thresh: Ratio of threshold to filter matches. The lower the more matches will be filtered out
    :param increase_by: Value to increase `ratio_thresh` by until matches are found or 0.99 is reached
    :param ransac_thresh: Threshold for RANSAC algorithm to find homography
    :return: Panorama of multiple stitched images
    """
    if rois is None:
        rois = [None for _ in img_paths]
    
    if len(img_paths) > 3:
        center_idx = len(img_paths) // 2
        
        # Create panoramas for both halfs of the images
        panorama1, last_t1 = create_panorama(img_paths[:center_idx], rois[:center_idx], downscale_factor, kernel_size, k, ratio_thresh, increase_by,
                                             ransac_thresh)
        panorama2, last_t2 = create_panorama(img_paths[center_idx:][::-1], rois[center_idx:][::-1], downscale_factor, kernel_size, k, ratio_thresh, increase_by,
                                             ransac_thresh)
        
        # Read in images at both ends of the panoramas
        img1, img2 = cv2.imread(img_paths[center_idx - 1]), cv2.imread(img_paths[center_idx])
        
        # Stitch both panoramas
        matched_pts1, matched_pts2 = get_matching_points(img2, img1, rois[center_idx - 1], rois[center_idx], downscale_factor, kernel_size, k, ratio_thresh,
                                                         increase_by)
        homography, _ = cv2.findHomography(matched_pts2 + ([0, 0] if last_t1 is None else last_t1), matched_pts1 + ([0, 0] if last_t2 is None else last_t2),
                                           cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        stitched_img, _ = stitch_images(panorama2, panorama1, homography, overlay=True)
    
    # Fall back to standard panorama stitching for less than 4 images
    else:
        stitched_img, _ = create_panorama(img_paths, rois, downscale_factor, kernel_size, k, ratio_thresh, increase_by, ransac_thresh)
    
    return stitched_img
