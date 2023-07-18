#coding:utf-8
import xlwt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import numpy as np
import cv2
import os

# 灰度级压缩
def reduce_intensity_levels(img, level):
    img = cv2.copyTo(img, None)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            si = img[x, y]
            ni = int(level * si / 255 - 0.5)
            img[x, y] = ni
    return img

def tztq(img):
    img_gray = reduce_intensity_levels(img,16)
    g = graycomatrix(img_gray,[4],[0,np.pi/4,np.pi/2,np.pi*3/4],levels=16,normed=False)
    nengliang = graycoprops(g,'ASM')
    nengliang_jz = np.mean(nengliang)
    nengliang_jz = format(nengliang_jz,'.4f')
    nengliang_bzc = np.std(nengliang,ddof = 1)
    nengliang_bzc = format(nengliang_bzc, '.4f')
    shang = graycoprops(g,'entroy')
    shang_jz = np.mean(shang)
    shang_jz = format(shang_jz,'.4f')
    shang_bzc = np.std(shang,ddof = 1)
    shang_bzc = format(shang_bzc, '.4f')
    guanxingju = graycoprops(g,'contrast')
    guanxingju_jz = np.mean(guanxingju)
    guanxingju_jz = format(guanxingju_jz, '.4f')
    guanxingju_bzc = np.std(guanxingju,ddof = 1)
    guanxingju_bzc = format(guanxingju_bzc, '.4f')
    xiangguanxing = graycoprops(g,'correlation')
    xiangguanxing_jz = np.mean(xiangguanxing)
    xiangguanxing_jz = format(xiangguanxing_jz, '.4f')
    xiangguanxing_bzc = np.std(xiangguanxing,ddof = 1)
    xiangguanxing_bzc = format(xiangguanxing_bzc, '.4f')

    tz = nengliang_jz,nengliang_bzc,shang_jz,shang_bzc,guanxingju_jz,guanxingju_bzc,xiangguanxing_jz,xiangguanxing_bzc
    return tz

# dir = 'F:/datasets/pmdataset/yb1/littlepaper/me/glb0839/01'
# paths = os.listdir(dir)
#
# workbook = xlwt.Workbook(encoding='utf-8')
# worksheet = workbook.add_sheet('paths')
#
# n = 0
# for path in paths:
#     abpath = dir + '/' + path
#     img = cv2.imread(abpath,0)
#     a,b,c,d,e,f,g,h = tztq(img)
#     worksheet1.write(n, 0, abpath)
#     worksheet.write(n, 1, a)
#     worksheet.write(n, 2, b)
#     worksheet.write(n, 3, c)
#     worksheet.write(n, 4, d)
#     worksheet.write(n, 5, e)
#     worksheet.write(n, 6, f)
#     worksheet.write(n, 7, g)
#     worksheet.write(n, 8, h)
#     n += 1
#
# workbook.save("F:/datasets/pmdataset/yb1/littlepaper/334.xls")

dir = 'F:/datasets/pmdataset/yb1/littlepaper/me/blb0839/01'
dir2 = 'F:/datasets/pmdataset/yb1/littlepaper/me/blb0839/02'
dir4 = 'F:/datasets/pmdataset/yb1/littlepaper/me/blb0839/04'
dir8 = 'F:/datasets/pmdataset/yb1/littlepaper/me/blb0839/08'
paths1 = os.listdir(dir)
paths2 = os.listdir(dir2)
paths4 = os.listdir(dir4)
paths8 = os.listdir(dir8)

workbook = xlwt.Workbook(encoding='utf-8')
worksheet1 = workbook.add_sheet('1')
worksheet2 = workbook.add_sheet('2')
worksheet4 = workbook.add_sheet('4')
worksheet8 = workbook.add_sheet('8')

n = 0
for path in paths1:
    abpath = dir + '/' + path
    img = cv2.imread(abpath,0)
    a,b,c,d,e,f,g,h = tztq(img)
    worksheet1.write(n, 0, abpath)
    worksheet1.write(n, 1, a)
    worksheet1.write(n, 2, b)
    worksheet1.write(n, 3, c)
    worksheet1.write(n, 4, d)
    worksheet1.write(n, 5, e)
    worksheet1.write(n, 6, f)
    worksheet1.write(n, 7, g)
    worksheet1.write(n, 8, h)
    n += 1

n = 0
for path in paths2:
    abpath = dir2 + '/' + path
    img = cv2.imread(abpath,0)
    a,b,c,d,e,f,g,h = tztq(img)
    worksheet2.write(n, 0, abpath)
    worksheet2.write(n, 1, a)
    worksheet2.write(n, 2, b)
    worksheet2.write(n, 3, c)
    worksheet2.write(n, 4, d)
    worksheet2.write(n, 5, e)
    worksheet2.write(n, 6, f)
    worksheet2.write(n, 7, g)
    worksheet2.write(n, 8, h)
    n += 1

n = 0
for path in paths4:
    abpath = dir4 + '/' + path
    img = cv2.imread(abpath,0)
    a,b,c,d,e,f,g,h = tztq(img)
    worksheet4.write(n, 0, abpath)
    worksheet4.write(n, 1, a)
    worksheet4.write(n, 2, b)
    worksheet4.write(n, 3, c)
    worksheet4.write(n, 4, d)
    worksheet4.write(n, 5, e)
    worksheet4.write(n, 6, f)
    worksheet4.write(n, 7, g)
    worksheet4.write(n, 8, h)
    n += 1

n = 0
for path in paths8:
    abpath = dir8 + '/' + path
    img = cv2.imread(abpath,0)
    a,b,c,d,e,f,g,h = tztq(img)
    worksheet8.write(n, 0, abpath)
    worksheet8.write(n, 1, a)
    worksheet8.write(n, 2, b)
    worksheet8.write(n, 3, c)
    worksheet8.write(n, 4, d)
    worksheet8.write(n, 5, e)
    worksheet8.write(n, 6, f)
    worksheet8.write(n, 7, g)
    worksheet8.write(n, 8, h)
    n += 1

workbook.save("F:/datasets/pmdataset/yb1/littlepaper/blb111.xls")
