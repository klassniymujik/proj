import asyncio
import threading
import time
from typing import Any

import numpy as np

from backend.api import routes as api_routes
from backend.api.routes import connections
from backend.video_analytics.reid import get_manager

_main_loop: asyncio.AbstractEventLoop | None = None
_camera_states: dict[int, dict[str, Any]] = {}
_state_lock = threading.Lock()

_reid_stats_cache: dict = {}
_reid_stats_ts: float = 0.0


def set_main_loop(loop):
    global _main_loop
    _main_loop = loop


async def broadcast(data: dict):
    dead = []
    for ws in list(connections):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connections:
            connections.remove(ws)


def broadcast_from_thread(data: dict):
    if _main_loop is None or not _main_loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(broadcast(data), _main_loop)


def get_camera_states():
    with _state_lock:
        return dict(_camera_states)


def update_camera_state(cam_id, state: dict):
    with _state_lock:
        _camera_states[cam_id] = state


def clear_camera_states():
    with _state_lock:
        _camera_states.clear()


def reset_camera_tracks():
    with _state_lock:
        for st in _camera_states.values():
            st["points"] = []
            st["stats"]["active_tracks"] = 0
            if "heatmap_raw" in st:
                h = np.array(st["heatmap_raw"], dtype=float)
                h[:] = 0
                st["heatmap_raw"] = h.tolist()


def build_aggregate_payload():
    global _reid_stats_cache, _reid_stats_ts

    now = time.time()
    if now - _reid_stats_ts > 1.0:
        _reid_stats_cache = get_manager().debug_stats()
        _reid_stats_ts = now
    reid_stats = _reid_stats_cache
    rc = reid_stats.get("counters", {})
    global_unique = rc.get("total_unique", 0)

    with _state_lock:
        cam_items = sorted(_camera_states.items(), key=lambda x: x[0])
        cameras_payload = []
        total_active = 0
        all_points = []
        heatmap_sum = None

        for cam_id, st in cam_items:
            points = st.get("points", [])
            stats = st.get("stats", {"visitor_count": 0, "active_tracks": 0})
            hm = st.get("heatmap_raw")
            total_active += int(stats.get("active_tracks", 0))
            all_points.extend(points)
            if hm is not None:
                if heatmap_sum is None:
                    heatmap_sum = np.array(hm, dtype=float)
                else:
                    heatmap_sum += np.array(hm, dtype=float)
            cameras_payload.append({
                "id": cam_id,
                "label": st.get("label", f"Камера {cam_id + 1}"),
                "address": st.get("source", ""),
                "frame": st.get("frame", ""),
                "points": list(points),
                "stats": dict(stats),
            })

    if heatmap_sum is None:
        agg_heatmap = []
    else:
        max_val = float(np.max(heatmap_sum))
        agg_heatmap = (heatmap_sum / max_val).tolist() if max_val > 0 else heatmap_sum.tolist()

    agg_stats = {
        "visitor_count": global_unique,
        "active_tracks": total_active,
        "reid": {
            "homography_matches": rc.get("homography_matches", 0),
            "same_camera_reentries": rc.get("same_camera_reentries", 0),
            "avg_match_distance": reid_stats.get("distance_stats", {}).get("avg"),
        },
    }

    api_routes.LATEST_POINTS = all_points
    api_routes.LATEST_HEATMAP = agg_heatmap
    api_routes.LATEST_STATS = agg_stats
    api_routes.LATEST_FRAME = cameras_payload[0]["frame"] if cameras_payload else ""

    return {
        "cameras": cameras_payload,
        "points": all_points,
        "heatmap": agg_heatmap,
        "stats": agg_stats,
        "frame": cameras_payload[0]["frame"] if cameras_payload else "",
    }


def broadcast_current_state():
    broadcast_from_thread(build_aggregate_payload())
