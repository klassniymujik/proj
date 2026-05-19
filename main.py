import asyncio
import json
import os
import threading

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(2)

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api import routes as api_routes
from backend.api.routes import CAMERAS, SCENE, router
from backend.core.batch import (
    batch_loop, JPEG_QUALITY, WS_MAX_W, WARMUP_ITERS,
)
from backend.core.render import annotate_frame
from backend.core.state import (
    set_main_loop, broadcast_current_state,
    clear_camera_states, reset_camera_tracks,
)
from backend.db import storage
from backend.video_analytics.reid import get_manager, preload_reid
from backend.video_analytics.pipeline import preload_detector

app = FastAPI(title="Retail Analytics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)

@app.get("/")
def root():
    try:
        if storage.load_cameras():
            return RedirectResponse(url="/video.html")
    except Exception:
        pass
    return RedirectResponse(url="/editor.html")

_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if not os.path.isdir(_frontend_dir):
    _frontend_dir = "frontend"

try:
    app.mount("/", StaticFiles(directory=_frontend_dir), name="frontend")
except Exception as e:
    print(f"[warn] StaticFiles: {e}")

_batch_stop = threading.Event()
_batch_thread: threading.Thread | None = None


def stop_all_video_loops():
    _batch_stop.set()
    if _batch_thread is not None and _batch_thread.is_alive():
        _batch_thread.join(timeout=5.0)
    clear_camera_states()
    broadcast_current_state()


def restart_all_video_loops():
    stop_all_video_loops()
    _batch_stop.clear()

    cam_cfgs = []
    for idx, cam in enumerate(CAMERAS):
        src = (cam.get("address") or "").strip()
        if not src:
            continue
        cam_cfgs.append((idx, cam, src))

    if not cam_cfgs:
        fallback_cam = {
            "x": 10.0, "y": 10.0, "height": 3.0,
            "yaw": 0.0, "pitch": -60.0, "fov": 90.0,
            "img_width": 1280, "img_height": 720,
            "label": "Fallback",
        }
        cam_cfgs.append((0, fallback_cam, "video.mp4"))

    global _batch_thread
    _batch_thread = threading.Thread(target=batch_loop, args=(cam_cfgs, _batch_stop), daemon=True)
    _batch_thread.start()
    print(f"[batch] Starting: {len(cam_cfgs)} cameras")


def restart_video_loop(source: str = "video.mp4"):
    global _batch_thread
    source = source or "video.mp4"
    if CAMERAS:
        CAMERAS[0]["address"] = source
        restart_all_video_loops()
    else:
        stop_all_video_loops()
        fallback_cam = {
            "x": 10.0, "y": 10.0, "height": 3.0,
            "yaw": 0.0, "pitch": -60.0, "fov": 90.0,
            "img_width": 1280, "img_height": 720,
            "label": "Fallback",
        }
        cam_cfgs = [(0, fallback_cam, source)]
        _batch_stop.clear()
        _batch_thread = threading.Thread(target=batch_loop, args=(cam_cfgs, _batch_stop), daemon=True)
        _batch_thread.start()


def reset_all_stats():
    get_manager().reset()
    reset_camera_tracks()
    api_routes.LATEST_POINTS = []
    api_routes.LATEST_HEATMAP = []
    api_routes.LATEST_STATS = {"visitor_count": 0, "active_tracks": 0}
    broadcast_current_state()


@app.on_event("startup")
async def start_video():
    set_main_loop(asyncio.get_running_loop())

    storage.init_db()
    scene = storage.load_scene()
    if scene:
        SCENE["width"] = float(scene.get("width", 20))
        SCENE["height"] = float(scene.get("height", 20))
        api_routes.SCENE["width"] = SCENE["width"]
        api_routes.SCENE["height"] = SCENE["height"]
    saved = storage.load_cameras()
    if saved:
        CAMERAS.clear()
        CAMERAS.extend(saved)
        print(f"[startup] Loaded {len(saved)} cameras")

    homographies = storage.load_homographies()
    if homographies:
        for h in homographies:
            pts_a = np.float32(json.loads(h["points_a"]))
            pts_b = np.float32(json.loads(h["points_b"]))
            H, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
            if H is not None:
                get_manager().set_homography(h["cam_a"], h["cam_b"], H)
                print(f"[startup] Homography cam {h['cam_a']}->{h['cam_b']} loaded")

    preload_detector()
    preload_reid()

    dummy_frame = np.random.randint(0, 255, (288, 360, 3), dtype=np.uint8)
    dummy_objs = [{"global_id": 1, "reid_matched": True, "bbox": [30, 20, 90, 200]}]
    _ = annotate_frame(dummy_frame, dummy_objs)
    small = cv2.resize(dummy_frame, (WS_MAX_W, 360), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    from backend.core.batch import _encode_pool
    def _enc_noop():
        s = cv2.resize(dummy_frame, (WS_MAX_W, 360), interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    futs = [_encode_pool.submit(_enc_noop) for _ in range(4)]
    for f in futs:
        f.result()
    print("[startup] cv2+encode+pool warmup done")

    restart_all_video_loops()
    print("[startup] Ready")