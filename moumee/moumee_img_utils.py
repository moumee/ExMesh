import cv2
import numpy as np

def compute_sobel_magnitude(img, sigma=1.0, gaussian_kernel=13, sobel_kernel=3):
    """Compute RGB Sobel magnitude; defaults use the selected scan24 filter setup."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if gaussian_kernel < 1 or gaussian_kernel % 2 != 1:
        raise ValueError("gaussian_kernel must be a positive odd integer")
    if sobel_kernel not in (1, 3, 5, 7):
        raise ValueError("sobel_kernel must be 1, 3, 5 or 7")
    
    img = np.asarray(img, dtype=np.float32)
    
    img_smoothed = cv2.GaussianBlur(
        img, (gaussian_kernel, gaussian_kernel), sigmaX=sigma, sigmaY=sigma
    )
    
    gx = cv2.Sobel(img_smoothed, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    gy = cv2.Sobel(img_smoothed, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    
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


def aggregate_face_importance(view_samples, num_faces):
    """Average normalized pixels per face, then equally over visible views.

    Each sample is (visible_face_ids, normalized_pixel_scores), both 1D.
    This is the aggregation used by ExMesh training and experiment scripts.
    """
    if not isinstance(num_faces, (int, np.integer)) or num_faces < 1:
        raise ValueError("num_faces must be a positive integer")
    sums = np.zeros(num_faces, dtype=np.float64)
    views = np.zeros(num_faces, dtype=np.int32)
    for ids, scores in view_samples:
        ids, scores = np.asarray(ids), np.asarray(scores)
        if ids.ndim != 1 or scores.shape != ids.shape:
            raise ValueError("Visible face IDs and scores must be matching 1D arrays")
        if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= num_faces)):
            raise ValueError("Visible face ID is outside the original mesh")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Normalized pixel scores must be finite values in [0, 1]")
        counts = np.bincount(ids, minlength=num_faces)
        score_sum = np.bincount(ids, weights=scores, minlength=num_faces)
        visible = counts > 0
        sums[visible] += score_sum[visible] / counts[visible]
        views[visible] += 1
    importance = np.divide(sums, views, out=np.zeros_like(sums), where=views > 0)
    return importance.astype(np.float32), views


def compute_face_importance(magnitude_maps, face_id_maps, num_faces, percentile=95):
    """Original ExMesh score protocol: visible-pixel P95 and per-view averaging.

    Face maps come from the caller's renderer, with background ID -1.
    Return (float32 score per original face, global percentile reference).
    """
    if len(magnitude_maps) != len(face_id_maps) or not magnitude_maps:
        raise ValueError("One magnitude map and face ID map are required per view")
    valid_masks = []
    for magnitude, ids in zip(magnitude_maps, face_id_maps):
        magnitude, ids = np.asarray(magnitude), np.asarray(ids)
        if magnitude.ndim != 2 or ids.shape != magnitude.shape:
            raise ValueError("Magnitude and face ID maps must have matching (H, W) shapes")
        if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < -1) | (ids >= num_faces)):
            raise ValueError("Face map must use original face IDs and -1 for background")
        valid = ids >= 0
        if not np.all(np.isfinite(magnitude[valid])) or np.any(magnitude[valid] < 0):
            raise ValueError("Visible Sobel magnitudes must be finite and non-negative")
        valid_masks.append(valid)
    q = compute_global_percentile(magnitude_maps, valid_masks, percentile=percentile)
    samples = (
        (ids[valid], normalize_magnitude(magnitude[valid], q))
        for magnitude, ids, valid in zip(magnitude_maps, face_id_maps, valid_masks)
    )
    importance, _ = aggregate_face_importance(samples, num_faces)
    return importance, q
