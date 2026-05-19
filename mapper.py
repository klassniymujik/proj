import numpy as np
import math


class Camera:
    """
    Модель камеры с внутренними и внешними параметрами.

    position: (x, y, z) — положение камеры в мировых координатах
    yaw:   поворот по оси Z (вправо/влево), градусы
    pitch: наклон вниз/вверх, градусы
    fov_h: горизонтальный угол обзора, градусы
    width, height: разрешение изображения
    """

    def __init__(self, position, yaw=0.0, pitch=-60.0,
                 fov_h=90.0, width=1280, height=720):
        self.position = np.array(position, dtype=float)
        self.yaw = yaw
        self.pitch = pitch
        self.fov_h = fov_h
        self.width = width
        self.height = height

        # Вычисляем фокусные расстояния из угла обзора
        self.fx = (width / 2) / math.tan(math.radians(fov_h / 2))
        self.fy = self.fx  # квадратные пиксели
        self.cx = width / 2
        self.cy = height / 2

        # Матрица поворота из yaw + pitch
        self.rotation = self._build_rotation(yaw, pitch)

    def _build_rotation(self, yaw_deg, pitch_deg):
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        # Поворот вокруг Z (yaw)
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0,              0,             1]
        ])

        # Поворот вокруг X (pitch — наклон вниз)
        Rx = np.array([
            [1, 0,               0              ],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch),  math.cos(pitch)]
        ])

        return Rz @ Rx

    def get_frustum_corners(self, near=0.5, far=8.0):
        """
        Возвращает 8 углов фрустума (усечённой пирамиды видимости) в мировых координатах.
        Используется для визуализации зоны обзора в 3D-редакторе.
        """
        corners_img = [
            (0, 0), (self.width, 0),
            (self.width, self.height), (0, self.height)
        ]

        def img_to_ray(u, v):
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
            ray_cam = np.array([x, y, 1.0])
            ray_world = self.rotation @ ray_cam
            return ray_world / np.linalg.norm(ray_world)

        rays = [img_to_ray(u, v) for u, v in corners_img]
        C = self.position

        near_pts = [C + near * r for r in rays]
        far_pts  = [C + far  * r for r in rays]

        return near_pts + far_pts


class SpatialMapper:
    """
    Пространственное сопоставление: обратная проекция точки изображения
    на плоскость пола (z = 0) в мировых координатах.
    """

    def __init__(self, camera: Camera):
        self.camera = camera

    def pixel_to_world(self, u, v):
        """
        u, v — координаты точки на изображении (нижняя точка bbox).
        Возвращает (wx, wy) — положение на полу, или None если проекция невозможна.
        """
        # Нормализованные координаты в системе камеры
        x = (u - self.camera.cx) / self.camera.fx
        y = (v - self.camera.cy) / self.camera.fy

        # Луч в системе координат камеры
        ray_camera = np.array([x, y, 1.0])

        # Луч в мировых координатах
        ray_world = self.camera.rotation @ ray_camera

        C = self.camera.position

        # Пересечение луча с плоскостью z = 0 (пол)
        if abs(ray_world[2]) < 1e-6:
            return None

        t = -C[2] / ray_world[2]

        if t < 0:
            return None

        P = C + t * ray_world

        return float(P[0]), float(P[1])


def create_mapper_from_config(cam_config: dict) -> SpatialMapper:
    """
    Создаёт SpatialMapper из конфигурации камеры (dict от фронтенда).

    Ожидаемые поля:
      x, y       — положение на полу (z задаётся через height)
      height     — высота установки камеры (default 3.0)
      yaw        — поворот по горизонтали в градусах (default 0)
      pitch      — наклон вниз в градусах (default -60)
      fov        — горизонтальный угол обзора (default 90)
      img_width  — ширина кадра (default 1280)
      img_height — высота кадра (default 720)
    """
    cam = Camera(
        position=(
            cam_config.get("x", 0),
            cam_config.get("y", 0),
            cam_config.get("height", 3.0)
        ),
        yaw=cam_config.get("yaw", 0.0),
        pitch=cam_config.get("pitch", -60.0),
        fov_h=cam_config.get("fov", 90.0),
        width=cam_config.get("img_width", 1280),
        height=cam_config.get("img_height", 720),
    )
    return SpatialMapper(cam)