import cv2
import time
from correct_and_fuse import positive_reverse_correction
from adaptive_histogram_equalization import Histogram_equalization
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('imgpath', type=str, help='图像路径')
parser.add_argument('filter_size', type=int, help='滤波核大小（奇数）')
parser.add_argument('save_path', type=int, help='增强图像保存路径')
args = parser.parse_args()

start = time.time()
img = cv2.imread(args.imgpath, 1)
img_corrected = positive_reverse_correction(img, args.filter_size)
last_img = Histogram_equalization(img_corrected)
cv2.imwrite(args.save_path,last_img)

end = time.time()

print('running time：', end - start)