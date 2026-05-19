import cv2
import numpy as np


def annotate_frame(frame, objects_px):
    out = frame.copy()
    for obj in objects_px:
        x1, y1, x2, y2 = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]

        # confirmed_id is None означает что трек ещё в tentative фазе
        # (не накопил достаточно descriptors для confirmation). Рисуем
        # его тонким серым контуром без подписи — пусть пользователь видит
        # что детекция есть, но система ещё не уверена кто это.
        cid = obj.get("confirmed_id")
        if cid is None:
            # Tentative: тонкий серый bbox, без label, без точки
            cv2.rectangle(out, (x1, y1), (x2, y2), (130, 130, 130), 1)
            continue

        # Confirmed: полноценная разметка с подписью
        color = (200, 80, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        label = f"G#{cid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cx = (x1 + x2) // 2
        cy = y2
        cv2.circle(out, (cx, cy), 3, (0, 150, 255), -1)
    return out


def project_point(mapper, px, py, frame_w, frame_h, scene_w, scene_h):
    def norm_to_scene(nx, ny):
        return (nx / max(1.0, frame_w)) * scene_w, (ny / max(1.0, frame_h)) * scene_h

    margin = 1.0
    if mapper is not None:
        wxwy = mapper.pixel_to_world(px, py)
        if wxwy is not None:
            wx, wy = wxwy
            if -margin <= wx <= (scene_w + margin) and -margin <= wy <= (scene_h + margin):
                return min(max(wx, 0.0), scene_w), min(max(wy, 0.0), scene_h)
    wx, wy = norm_to_scene(px, py)
    return min(max(wx, 0.0), scene_w), min(max(wy, 0.0), scene_h)