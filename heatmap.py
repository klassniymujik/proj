import numpy as np


class Heatmap:
    """
    Тепловая карта активности посетителей на плане помещения.

    width, height  — размер зала в метрах
    resolution     — метров на ячейку (чем меньше, тем детальнее)
    decay          — коэффициент затухания (0..1): насколько быстро
                     старые данные «стареют» при каждом вызове update()
    """

    def __init__(self, width=20, height=20, resolution=0.5, decay=0.995):
        self.resolution = resolution
        self.decay = decay
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.grid_w = max(1, int(width / resolution))
        self.grid_h = max(1, int(height / resolution))

        self.map = np.zeros((self.grid_h, self.grid_w), dtype=float)

    def update(self, points):
        """
        points — список dict с ключами x, y (мировые координаты).
        """
        # Затухание старых значений
        self.map *= self.decay

        for p in points:
            gx = int((p["x"] - self.origin_x) / self.resolution)
            gy = int((p["y"] - self.origin_y) / self.resolution)

            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                self.map[gy, gx] += 1.0

    def get(self):
        """Возвращает тепловую карту в виде 2D-списка (для JSON)."""
        return self.map.tolist()

    def get_normalized(self):
        """Нормализованная карта [0..1]."""
        max_val = self.map.max()
        if max_val == 0:
            return self.map.tolist()
        return (self.map / max_val).tolist()

    def reset(self):
        self.map = np.zeros((self.grid_h, self.grid_w), dtype=float)

    def resize(self, width, height):
        """Пересоздаёт карту под новый размер зала."""
        self.grid_w = max(1, int(width / self.resolution))
        self.grid_h = max(1, int(height / self.resolution))
        self.map = np.zeros((self.grid_h, self.grid_w), dtype=float)