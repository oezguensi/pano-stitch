import argparse
from cv2 import cv2
from matplotlib import pyplot as plt

from stitch import create_centered_panorama


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt-path', type=str, required=True,
                        help='.txt file containing sorted paths to images and additional Region of interests seperated by comma')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save image to')
    
    parser.add_argument('--downscale-factor', type=int, default=4, help='Factor to downscale image for more robust keypoint detection')
    parser.add_argument('--kernel-size', type=int, default=3,
                        help='Size of kernel for blurring image for more robust keypoint detection. Number must be odd. The higher the more blurry')
    parser.add_argument('--ratio-thresh', type=float, default=0.7, help='Ratio of threshold to filter matches. The lower the more matches will be filtered out')
    parser.add_argument('--k', type=int, default=2, help='Count of best matches. Used for getting matches')
    parser.add_argument('--ransac-thresh', type=int, default=4, help='Threshold for RANSAC algorithm to find homography')
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    with open(args.txt_path, 'r') as f:
        rows = [row.rstrip('\r\n') for row in f.readlines()]
    img_paths = [row if ',' not in row else row.split(',')[0] for row in rows]
    rois = [None if ',' not in row else eval(','.join(row.split(',')[1:])) for row in rows]
    
    panorama = create_centered_panorama(img_paths, rois, **{k: v for k, v in vars(args).items() if k not in ['save_path', 'txt_path']})
    
    if args.save_path is None:
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        cv2.imwrite(args.save_path, panorama)


if __name__ == '__main__':
    main()
