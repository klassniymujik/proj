"""
HomographyMapper — использует готовые гомографии из calibration-6p.txt
вместо параметрической модели камеры.
Не трогает существующий mapper.py.
"""
import numpy as np

# Ground plane гомографии из calibration-6p.txt
# H * X_image = X_topview (358x360 пикселей)
_HOMOGRAPHIES = {
    0: np.array([
        [0.176138,  0.647589, -63.412272],
        [-0.180912, 0.622446,  -0.125533],
        [-0.000002, 0.001756,   0.102316]
    ]),
    1: np.array([
        [0.177291,  0.004724,  31.224545],
        [0.169895,  0.661935, -79.781865],
        [-0.000028, 0.001888,   0.054634]
    ]),
    2: np.array([
        [-0.104843, 0.099275, 50.734500],
        [ 0.107082, 0.102216,  7.822562],
        [-0.000054, 0.001922, -0.068053]
    ]),
    3: np.array([
        [-0.142865, 0.553150, -17.395045],
        [-0.125726, 0.039770,  75.937144],
        [-0.000011, 0.001780,   0.015675]
    ]),
}

class HomographyMapper:
    """
    Mapper на основе готовой гомографии из файла калибровки EPFL.
    Top view размер: 358x360 пикселей = реальная комната ~4x4м.
    """
    def __init__(self, cam_id: int):
        if cam_id not in _HOMOGRAPHIES:
            raise ValueError(f"Нет гомографии для камеры {cam_id}")
        self.H = _HOMOGRAPHIES[cam_id]
        self.cam_id = cam_id

    def pixel_to_world(self, u, v):
        p = np.array([u, v, 1.0])
        w = self.H @ p
        if abs(w[2]) < 1e-9:
            return None
        wx, wy = float(w[0]/w[2]), float(w[1]/w[2])
        # фильтруем точки за пределами top view
        if not (0 <= wx <= 358 and 0 <= wy <= 360):
            return None
        return wx, wy


def get_homography_mapper(cam_id: int) -> HomographyMapper:
    """Фабричная функция — аналог create_mapper_from_config."""
    return HomographyMapper(cam_id)