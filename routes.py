from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import cv2

from backend.db import storage
from backend.video_analytics.reid import get_manager

router = APIRouter()

connections: list[WebSocket] = []

CAMERAS: list[dict] = []
SCENE: dict = {"width": 20, "height": 20}
LATEST_POINTS: list[dict] = []
LATEST_HEATMAP: list[list] = []
LATEST_STATS: dict = {"visitor_count": 0, "active_tracks": 0}
LATEST_FRAME: list = []
HOMOGRAPHIES: dict[str, list] = {}


class HomographyPointsModel(BaseModel):
    cam_a: int
    cam_b: int
    points_a: list[list[float]]
    points_b: list[list[float]]


class SceneConfig(BaseModel):
    width: float = 20
    height: float = 20


class CameraConfig(BaseModel):
    x: float = 0
    y: float = 0
    height: float = 3.0
    yaw: float = 0.0
    pitch: float = -60.0
    fov: float = 90.0
    img_width: int = 1280
    img_height: int = 720
    label: Optional[str] = ""
    address: Optional[str] = ""


@router.get("/scene")
def get_scene():
    return SCENE


@router.post("/scene")
def set_scene(config: SceneConfig):
    global SCENE
    SCENE = config.dict()
    storage.save_scene(config.width, config.height)
    return {"status": "ok", "scene": SCENE}


@router.get("/cameras")
def get_cameras():
    return {"cameras": CAMERAS}


@router.get("/cameras/list")
def list_cameras_for_video():
    """Список камер с адресами для страницы видео."""
    return {
        "cameras": [
            {
                "id": i,
                "label": c.get("label") or f"Камера {i + 1}",
                "address": c.get("address", ""),
            }
            for i, c in enumerate(CAMERAS)
        ]
    }


@router.post("/cameras")
def set_cameras(cameras: List[CameraConfig]):
    CAMERAS.clear()
    CAMERAS.extend([c.dict() for c in cameras])
    storage.save_cameras(CAMERAS)
    print(f"[routes] Сохранено {len(CAMERAS)} камер")
    # Перезапускаем все пайплайны под новый список камер.
    try:
        from backend.main import restart_all_video_loops
        restart_all_video_loops()
    except Exception as e:
        print(f"[routes] restart_all_video_loops warn: {e}")
    return {"status": "ok", "count": len(CAMERAS)}


@router.get("/points")
def get_points():
    return {"points": LATEST_POINTS}


@router.get("/heatmap")
def get_heatmap():
    return {"heatmap": LATEST_HEATMAP}


@router.get("/stats")
def get_stats():
    return LATEST_STATS


@router.get("/reid/debug")
def get_reid_debug():
    """Диагностика качества cross-camera ReID."""
    return get_manager().debug_stats()


@router.get("/history")
def get_history(limit: int = 50):
    return {"history": storage.load_snapshots(limit)}


@router.post("/reset")
def reset_stats():
    LATEST_STATS["visitor_count"] = 0
    LATEST_STATS["active_tracks"] = 0
    LATEST_POINTS.clear()
    LATEST_HEATMAP.clear()
    try:
        from backend.main import reset_all_stats
        reset_all_stats()
    except Exception:
        pass
    return {"status": "reset"}


@router.post("/set_source")
def set_source(body: dict):
    from backend.main import restart_video_loop
    src = body.get("source", "video.mp4")
    restart_video_loop(src)
    return {"status": "ok", "source": src}


@router.post("/homography/set")
def set_homography_points(data: HomographyPointsModel):
    import numpy as np
    if len(data.points_a) < 4 or len(data.points_b) < 4:
        return {"status": "error", "message": "Need at least 4 point pairs"}
    if len(data.points_a) != len(data.points_b):
        return {"status": "error", "message": "Point count mismatch"}

    pts_a = np.float32(data.points_a)
    pts_b = np.float32(data.points_b)
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)

    if H is None:
        return {"status": "error", "message": "Failed to compute homography"}

    key = f"{data.cam_a}_{data.cam_b}"
    HOMOGRAPHIES[key] = {
        "H": H.tolist(),
        "points_a": data.points_a,
        "points_b": data.points_b,
        "cam_a": data.cam_a,
        "cam_b": data.cam_b,
    }

    from backend.video_analytics.reid import get_manager
    get_manager().set_homography(data.cam_a, data.cam_b, H)

    storage.save_homography(data.cam_a, data.cam_b, data.points_a, data.points_b)

    return {"status": "ok", "key": key, "reprojection_error": _reproj_error(pts_a, pts_b, H)}


@router.get("/homography/get")
def get_homography(cam_a: int, cam_b: int):
    key = f"{cam_a}_{cam_b}"
    if key in HOMOGRAPHIES:
        h = HOMOGRAPHIES[key]
        return {"status": "ok", "points_a": h["points_a"], "points_b": h["points_b"]}
    return {"status": "none"}


class CalibHideModel(BaseModel):
    hide: bool


@router.post("/calib/hide_bboxes")
def set_calib_hide_bboxes(data: CalibHideModel):
    from backend.main import _calib_hide_bboxes
    import backend.main as main_mod
    main_mod._calib_hide_bboxes = data.hide
    return {"status": "ok", "hide": data.hide}


def _reproj_error(pts_a, pts_b, H):
    import numpy as np
    pts_a_h = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    projected = (H @ pts_a_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    err = np.sqrt(np.sum((projected - pts_b) ** 2, axis=1))
    return float(np.mean(err))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    print(f"[ws] подключён, всего: {len(connections)}")
    try:
        # receive() блокируется до получения сообщения или разрыва соединения.
        # Это единственный надёжный способ обнаруживать отключение клиента.
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in connections:
            connections.remove(websocket)
        print(f"[ws] отключён, осталось: {len(connections)}")