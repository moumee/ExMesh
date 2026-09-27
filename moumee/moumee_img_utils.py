import cv2
import numpy as np

def compute_sobel_magnitude(img):
    
    img = np.asarray(img, dtype=np.float32)
    
    img_smoothed = cv2.GaussianBlur(
        img, (5, 5), sigmaX=1.0, sigmaY=1.0
    )
    
    gx = cv2.Sobel(img_smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_smoothed, cv2.CV_32F, 0, 1, ksize=3)
    
    return np.sqrt(np.sum(gx**2 + gy**2, axis=-1))


def compute_global_percentile(magnitude_maps, valid_masks, percentile=95):
    """
    magnitude_maps: 시점별 Sobel 크기 배열 (H, W)의 리스트
    valid_masks: 시점별 유효 픽셀 bool 배열 (H, W)의 리스트
                 T_k >= 0으로 계산
    반환: 전체 유효 픽셀의 공통 백분위수
    """
    if len(magnitude_maps) != len(valid_masks):
        raise ValueError("영상과 마스크 개수가 다릅니다.")

    valid_values = []

    for magnitude, mask in zip(magnitude_maps, valid_masks):
        values = magnitude[mask]

        if values.size > 0:
            valid_values.append(values)

    if not valid_values:
        raise ValueError("메시가 보이는 유효 픽셀이 없습니다.")

    all_values = np.concatenate(valid_values)
    return float(np.percentile(all_values, percentile))


def normalize_magnitude(magnitude, reference, eps=1e-8):
    """
    magnitude: 정규화 전 Sobel 크기 (H, W)
    reference: 모든 시점에서 구한 공통 기준값
    반환: [0, 1] 범위로 정규화한 중요도 (H, W)
    """
    return np.clip(magnitude / max(reference, eps), 0.0, 1.0)
