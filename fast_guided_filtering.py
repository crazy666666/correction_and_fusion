import cv2


def Filter(img, winSize, eps, s,w_k):

    h, w = img.shape[:2]
    size = (int(round(w * s)), int(round(h * s)))
    small_I = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))
    mean_small_I = cv2.blur(small_I, small_winSize)
    mean_small_p = cv2.blur(small_p, small_winSize)
    mean_small_II = cv2.blur(small_I * small_I, small_winSize)
    mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)
    var_small_I = mean_small_II - mean_small_I * mean_small_I  # 方差公式
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
    q = w_k * mean_a * img + mean_b
    return q