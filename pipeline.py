import time

import numpy as np
import supervision as sv
from ultralytics import YOLO

YOLO_IMGSZ = 640
CAP_FRAME_W = 360
CAP_FRAME_H = 288
YOLO_CONF = 0.483
YOLO_NMS_IOU = 0.45

TRACK_ACTIVATION_THRESHOLD = 0.358
LOST_TRACK_BUFFER = 82
MINIMUM_MATCHING_THRESHOLD = 0.919
MINIMUM_CONSECUTIVE_FRAMES = 2
FRAME_RATE = 25

DEBUG_YOLO = False
DEBUG_TRACK = True
_debug_yolo_n = 0
_debug_yolo_t0 = 0.0
_debug_yolo_prev = {}
_debug_yolo_miss = 0
_debug_yolo_iou_low = 0
_debug_n = 0
_debug_lost = 0
_debug_new = 0
_debug_last_tid = {}
_debug_t0 = 0.0
_shared_model = None
_detect_device = None
_detect_half = False


def _auto_device():
    global _detect_device, _detect_half
    if _detect_device is not None:
        return
    try:
        import torch
        if torch.cuda.is_available():
            _detect_device = 0
            _detect_half = True
            torch.backends.cudnn.benchmark = True
        else:
            _detect_device = "cpu"
            _detect_half = False
    except ImportError:
        _detect_device = "cpu"
        _detect_half = False
    print(f"[detector] device={_detect_device}, half={_detect_half}, imgsz={YOLO_IMGSZ}, nms_iou={YOLO_NMS_IOU}")


def preload_detector():
    _get_shared_model()
    print("[detector] Pre-loaded")


def _get_shared_model():
    global _shared_model
    if _shared_model is None:
        _auto_device()
        _shared_model = YOLO("yolo26n_people_tuned.pt")
        warmup_frames = [np.random.randint(0, 255, (CAP_FRAME_H, CAP_FRAME_W, 3), dtype=np.uint8) for _ in range(4)]
        for _ in range(3):
            _shared_model(warmup_frames, verbose=False, imgsz=YOLO_IMGSZ,
                          device=_detect_device, half=_detect_half, classes=[0],
                          iou=YOLO_NMS_IOU)
        if _detect_device != "cpu":
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
        print("[detector] YOLO warmed up (4 frames x3 iters)")
    return _shared_model


def _box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter)


def batch_detect(frames, conf=None):
    model = _get_shared_model()
    results = model(frames, verbose=False, conf=conf or YOLO_CONF, imgsz=YOLO_IMGSZ,
                    device=_detect_device, half=_detect_half, classes=[0],
                    iou=YOLO_NMS_IOU, agnostic_nms=True)

    all_objects = []
    for ri, res in enumerate(results):
        objects = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf_arr = res.boxes.conf.cpu().numpy()
            objects.append((xyxy, conf_arr))
        all_objects.append(objects)

    if DEBUG_YOLO:
        global _debug_yolo_n, _debug_yolo_t0, _debug_yolo_prev, _debug_yolo_miss, _debug_yolo_iou_low
        _debug_yolo_n += 1
        for ri, dets in enumerate(all_objects):
            if not dets:
                _debug_yolo_miss += 1
                continue
            xyxy, conf_arr = dets[0]
            cur = [(tuple(xyxy[j].tolist()), float(conf_arr[j])) for j in range(len(xyxy))]
            prev = _debug_yolo_prev.get(ri)
            if prev is not None:
                for cb, cc in cur:
                    best_iou = max((_box_iou(cb, pb) for pb, _ in prev), default=0.0)
                    if best_iou < 0.5 and cc > 0.35:
                        _debug_yolo_iou_low += 1
            _debug_yolo_prev[ri] = cur
        now = time.time()
        if now - _debug_yolo_t0 >= 2.0:
            counts = {}
            for ri, dets in enumerate(all_objects):
                n = len(dets[0][0]) if dets else 0
                if n > 0: counts[ri] = n
            avg_conf = {}
            for ri, dets in enumerate(all_objects):
                if dets and len(dets[0]) >= 2:
                    ca = dets[0][1]
                    if len(ca) > 0:
                        avg_conf[ri] = f"{ca.min():.2f}-{ca.max():.2f}"
            print(f"[yolo] {_debug_yolo_n}fr | dets={counts} | conf={avg_conf} | "
                  f"miss={_debug_yolo_miss} | iou_low={_debug_yolo_iou_low}")
            _debug_yolo_t0 = now

    return all_objects


def make_tracker():
    return sv.ByteTrack(
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
        minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
        frame_rate=FRAME_RATE,
    )


def track_objects(tracker, det_list, timestamp, cam_id_hint=0):
    objects = []
    if not det_list:
        return objects
    xyxy, conf_arr = det_list[0]
    n_det = len(xyxy)
    sv_det = sv.Detections(
        xyxy=xyxy,
        confidence=conf_arr,
        class_id=np.zeros(len(xyxy), dtype=int),
    )
    tracks = tracker.update_with_detections(sv_det)

    if tracks.tracker_id is not None and len(tracks.tracker_id) > 0:
        for i in range(len(tracks.xyxy)):
            x1, y1, x2, y2 = tracks.xyxy[i]
            tid = int(tracks.tracker_id[i])
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            conf = float(tracks.confidence[i]) if tracks.confidence is not None else 0.0
            objects.append({
                "id": tid,
                "x": cx,
                "y": cy,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "timestamp": timestamp,
                "confidence": conf,
            })

    if DEBUG_TRACK:
        global _debug_n, _debug_lost, _debug_new, _debug_last_tid, _debug_t0
        _debug_n += 1
        if n_det > 0 and len(objects) == 0:
            _debug_lost += 1
        for o in objects:
            cam_tids = _debug_last_tid.setdefault(cam_id_hint, {})
            tid = o["id"]
            if tid not in cam_tids:
                _debug_new += 1
                cam_tids[tid] = True
        now = time.time()
        if now - _debug_t0 >= 2.0:
            total_tids = sum(len(v) for v in _debug_last_tid.values())
            print(f"[track] {_debug_n}fr | lost_det={_debug_lost} | "
                  f"new_ids={_debug_new} | total_unique_tids={total_tids} | "
                  f"per_cam={dict((k,len(v)) for k,v in _debug_last_tid.items())}")
            _debug_t0 = now

    return objects
