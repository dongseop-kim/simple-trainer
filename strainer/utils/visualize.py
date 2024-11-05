import cv2
import numpy as np
import shapely
import torch
from shapely import Polygon


def normalized_polygon(polygon: Polygon, width: int, height: int) -> Polygon:
    return shapely.affinity.scale(polygon, xfact=100/width, yfact=100/height, origin=(0, 0))


def wkt_for_db(polygon: Polygon) -> str:
    return shapely.to_wkt(polygon, rounding_precision=3, trim=True)


def probability_map_to_shapely_polygons(prob_map: torch.Tensor | np.ndarray, finding: str,
                                        threshold: float) -> list[dict[str, float | Polygon]]:
    """
    확률 맵을 입력받아 각 contour를 Shapely Polygon으로 변환하고, 
    각 폴리곤 영역 내의 확률 통계 정보와 함께 반환합니다.

    :param prob_map: 2D numpy array, 0~1 사이의 확률값을 가진 맵
    :param thresholds: list, contour를 추출할 확률 임계값 리스트
    :return: list of lists of dicts, 각 임계값에 대한 딕셔너리 리스트
             각 딕셔너리는 'max_prob', 'min_prob', 'mean_prob', 'std_prob', 'polygon' 키를 가짐
    """
    # 확률 맵을 0~255 사이의 정수값으로 변환
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()

    prob_map = np.squeeze(prob_map)
    if prob_map.ndim != 2:
        raise ValueError("Probability map must be 2D array")

    prob_map_int = np.where(prob_map < 0, 0, prob_map * 255).astype(np.uint8)
    height, width = prob_map_int.shape

    all_polygon_data = []

    _, binary = cv2.threshold(prob_map_int, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Contour 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contour를 Shapely Polygon으로 변환하고 확률 통계 계산
    polygon_data = []
    for contour in contours:
        if len(contour) >= 3:  # 최소 3개의 점이 필요함
            polygon = Polygon(contour.squeeze())
            if polygon.is_valid:
                # 폴리곤 영역에 해당하는 마스크 생성
                mask = np.zeros(prob_map.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 1, -1)

                # 마스크를 사용하여 폴리곤 영역 내의 확률값 추출
                masked_prob_map = prob_map * mask
                prob_values = masked_prob_map[mask == 1]
                norm_polygon = normalized_polygon(polygon, width, height)
                polygon_data.append({'max_prob': round(np.max(prob_values), 4),
                                     'min_prob': round(np.min(prob_values), 4),
                                     'mean_prob': round(np.mean(prob_values), 4),
                                     'std_prob': round(np.std(prob_values), 4),
                                     'polygon': wkt_for_db(norm_polygon),
                                     'finding': finding})

        all_polygon_data.append(polygon_data)

    return all_polygon_data
