import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from backend.analytics.heatmap import Heatmap
from backend.api.routes import SCENE
from backend.core.render import annotate_frame, project_point
from backend.core.state import update_camera_state, broadcast_current_state
from backend.mapping.mapper import create_mapper_from_config
from backend.video_analytics.reid import get_manager, extract_descriptors_multi, REID_MIN_CONF
from backend.video_analytics.pipeline import (
    batch_detect, make_tracker, track_objects,
)

USE_HOMOGRAPHY_MAPPER = False

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
JPEG_QUALITY = 50
WS_MAX_W = 480
WARMUP_ITERS = 3
DESC_INTERVAL = 1            # Больше не используется в _prepare_camera —
                             # descriptor извлекается для каждого трека всегда.
                             # Оставлен для backward compatibility.
REID_DEADLOCK_FRAMES = 0     # Больше не используется в _prepare_camera —
                             # фильтрация по confidence убрана полностью,
                             # как в test_reid_homography.
CALIB_HIDE_BBOXES = False
DEDUP_IOU = 1.01             # >1.0 эффективно отключает dedup. В тесте этого нет —
                             # оригинальные ByteTrack outputs идут в ReID напрямую.
                             # Восстанавливаем эквивалент тестового пайплайна.

_encode_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="enc")


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _deduplicate_tracks(objects_px):
    """
    Merge duplicate ByteTrack tracks for the same person.
    At 360x288 YOLO can produce overlapping detections → ByteTrack
    assigns separate local_ids. If IoU > DEDUP_IOU, keep the track
    with higher confidence (ReidTrack: filter overlapping detections).
    """
    if len(objects_px) <= 1:
        return objects_px
    objects_px.sort(key=lambda o: o["confidence"], reverse=True)
    keep = []
    for obj in objects_px:
        merged = False
        for kept in keep:
            if _iou(obj["bbox"], kept["bbox"]) > DEDUP_IOU:
                merged = True
                break
        if not merged:
            keep.append(obj)

    return keep


def _prepare_camera(cam_id, frame, det_list, tracker, mapper,
                    track_frame_ctr, last_desc_cache, sw, sh):
    timestamp = time.time()
    objects_px = track_objects(tracker, det_list, timestamp, cam_id)
    objects_px = _deduplicate_tracks(objects_px)

    frame_h, frame_w = frame.shape[:2]
    objects_world_tmp = []
    for obj in objects_px:
        px = float(obj.get("x", 0))
        py = float(obj.get("y", 0))
        wx, wy = project_point(mapper, px, py, frame_w, frame_h, sw, sh)
        objects_world_tmp.append({
            "local_id": int(obj["id"]),
            "bbox": obj["bbox"],
            "wx": wx,
            "wy": wy,
            "confidence": float(obj.get("confidence", 0.0)),
        })

    gallery_keys = get_manager().gallery_keys()
    need_desc = []
    for i, obj in enumerate(objects_world_tmp):
        lid = obj["local_id"]
        # Descriptor извлекается для КАЖДОГО трека без фильтрации.
        # ReID manager сам решит что делать (фильтрует по REID_MIN_CONF
        # внутри assign()). Любая фильтрация ЗДЕСЬ означала бы потерю
        # descriptors для tentative треков, которые могли бы пригодиться
        # при confirmation.
        need_desc.append(i)
        track_frame_ctr[lid] = track_frame_ctr.get(lid, 0) + 1

    active_lids = {obj["local_id"] for obj in objects_world_tmp}
    for lid in list(track_frame_ctr):
        if lid not in active_lids:
            del track_frame_ctr[lid]
    for (ec, elid) in list(last_desc_cache):
        if ec != cam_id or elid not in active_lids:
            if ec == cam_id:
                del last_desc_cache[(ec, elid)]

    bboxes_to_run = [objects_world_tmp[i]["bbox"] for i in need_desc]

    desc_map = {}
    for i in range(len(objects_world_tmp)):
        if i in need_desc:
            continue
        lid = objects_world_tmp[i]["local_id"]
        desc_map[i] = last_desc_cache.get((cam_id, lid))

    return objects_world_tmp, need_desc, bboxes_to_run, desc_map


def _run_reid_for_camera(cam_id, objects_world_tmp, need_desc, desc_map,
                          last_desc_cache, reid_mgr, advance_frame=False):
    """
    Синхронная часть: ReID assign + end_frame.
    Должна вызываться ПОСЛЕДОВАТЕЛЬНО для всех камер в одном потоке,
    иначе race conditions на _frame_count, _vote_history, _consolidation_map.

    Возвращает objects_world (готовый к рендерингу).
    """
    # Обновляем кеш дескрипторов. last_desc_cache фактически не используется
    # при DESC_INTERVAL=1 (descriptor извлекается каждый кадр), оставлен для
    # backward compat. REID_MIN_CONF — порог фильтра низкоконфидентных
    # детекций; ниже него descriptor не сохраняем.
    for pos, idx in enumerate(need_desc):
        d = desc_map.get(idx)
        lid = objects_world_tmp[idx]["local_id"]
        conf = objects_world_tmp[idx].get("confidence", 0.0)
        if d is not None and conf >= REID_MIN_CONF:
            last_desc_cache[(cam_id, lid)] = d

    active_lids = {obj["local_id"] for obj in objects_world_tmp}
    objects_world = []
    if reid_mgr is not None:
        for i, obj in enumerate(objects_world_tmp):
            lid = obj["local_id"]
            desc = desc_map.get(i)
            conf = obj.get("confidence", 0.0)
            # assign() возвращает confirmed_id (int) если трек уже созрел,
            # либо None если он ещё tentative (накапливается).
            cid = reid_mgr.assign(cam_id, lid, desc, obj["wx"], obj["wy"],
                                  confidence=conf,
                                  active_local_ids=active_lids, bbox=obj["bbox"])
            objects_world.append({
                "id": f"{cam_id}:{lid}",        # для совместимости с фронтендом
                "track_id": lid,
                "global_id": cid,               # None для tentative
                "confirmed_id": cid,            # явное поле для analytics.js
                "camera_id": cam_id,
                "x": obj["wx"],
                "y": obj["wy"],
                "bbox": obj["bbox"],
            })
        reid_mgr.end_frame(cam_id, active_lids, advance_frame=advance_frame)
    else:
        for obj in objects_world_tmp:
            lid = obj["local_id"]
            objects_world.append({
                "id": f"{cam_id}:{lid}",
                "track_id": lid,
                "global_id": lid,
                "confirmed_id": lid,
                "camera_id": cam_id,
                "x": obj["wx"],
                "y": obj["wy"],
                "bbox": obj["bbox"],
            })

    return objects_world


def _encode_and_publish(cam_id, frame, objects_world, cam_cfg, source,
                        heatmap, total_unique):
    """
    Асинхронная часть: annotate frame + JPEG encode + heatmap update + publish state.
    Безопасно гонять параллельно — все операции работают с per-camera данными
    (heatmap, frame), а update_camera_state имеет свой внутренний лок.

    total_unique передаётся как параметр (а не дёргается из reid_mgr) чтобы
    не брать лок на ReID manager из этого потока.
    """
    if CALIB_HIDE_BBOXES:
        ann = frame
        quality = 80
    else:
        frame_h, frame_w = frame.shape[:2]
        if frame_w > WS_MAX_W:
            scale = WS_MAX_W / frame_w
            small = cv2.resize(frame, (WS_MAX_W, int(frame_h * scale)),
                               interpolation=cv2.INTER_AREA)
            scaled = []
            for obj in objects_world:
                so = dict(obj)
                so["bbox"] = [v * scale for v in obj["bbox"]]
                scaled.append(so)
            ann = annotate_frame(small, scaled)
        else:
            ann = annotate_frame(frame, objects_world)
        quality = JPEG_QUALITY
    _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, quality])
    frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    # В heatmap пишем ТОЛЬКО confirmed треки. Tentative (без confirmed_id)
    # не должны попадать в heatmap — иначе heatmap будет показывать
    # "мусорные" точки от треков-однодневок (мелких ByteTrack glitches).
    confirmed_for_heatmap = [
        o for o in objects_world if o.get("confirmed_id") is not None
    ]
    heatmap.update(confirmed_for_heatmap)

    # Считаем сколько confirmed людей видно на этой камере (tentative не
    # считаем — они "ещё не созрели"). Это число показывается на UI
    # в "в кадре" per-camera.
    confirmed_in_frame = len(confirmed_for_heatmap)

    cam_stats = {
        "visitor_count": total_unique,
        "active_tracks": confirmed_in_frame,
    }
    cam_label = cam_cfg.get("label") or f"Камера {cam_id + 1}"

    update_camera_state(cam_id, {
        "id": cam_id,
        "label": cam_label,
        "source": source,
        "frame": frame_b64,
        "points": objects_world,
        "stats": cam_stats,
        "heatmap_raw": heatmap.get(),
    })


def batch_loop(cam_cfgs, stop):
    reid_mgr = get_manager()
    sw = float(SCENE.get("width", 20))
    sh = float(SCENE.get("height", 20))

    n = len(cam_cfgs)
    caps = []
    trackers = []
    mappers = []
    heatmaps = []
    track_ctrs = []
    desc_caches = []

    for idx, (cam_id, cam_cfg, source) in enumerate(cam_cfgs):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[camera {cam_id}] ERROR opening {source}")
            caps.append(None)
        else:
            caps.append(cap)
        trackers.append(make_tracker())
        try:
            if USE_HOMOGRAPHY_MAPPER:
                from backend.mapping.homography_mapper import get_homography_mapper
                mappers.append(get_homography_mapper(cam_id))
            else:
                mappers.append(create_mapper_from_config(cam_cfg) if cam_cfg else None)
        except Exception as e:
            print(f"[camera {cam_id}] mapper error: {e}")
            mappers.append(None)
        heatmaps.append(Heatmap(width=sw, height=sh, resolution=0.5))
        track_ctrs.append({})
        desc_caches.append({})

    if reid_mgr is not None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            dummy_warmup_frame = np.random.randint(0, 255, (288, 360, 3), dtype=np.uint8)
            for ncrops in range(1, 13):
                dummy_warmup_bboxes = [[30+i*25, 20+i*15, 80+i*25, 180+i*15] for i in range(ncrops)]
                for _ in range(2):
                    extract_descriptors_multi([(dummy_warmup_frame, dummy_warmup_bboxes)])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("[batch] OSNet warmup in batch thread done (1-12 crops)")
        except Exception as e:
            print(f"[batch] OSNet warmup error: {e}")
    _frame_skip = 0
    _iter = 0

    try:
      while not stop.is_set():
        t0 = time.time()
        frames = []
        valid_indices = []

        for idx in range(n):
            cap = caps[idx]
            if cap is None:
                frames.append(None)
                continue
            for _ in range(_frame_skip):
                if not cap.grab():
                    break
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    frames.append(None)
                    continue
                trackers[idx] = make_tracker()
                track_ctrs[idx] = {}
                desc_caches[idx] = {}
            frames.append(frame)
            valid_indices.append(idx)

        if not valid_indices:
            continue

        t_read = time.time()
        valid_frames = [frames[i] for i in valid_indices]
        all_dets = batch_detect(valid_frames)
        t_detect = time.time()

        preps = []
        for vi, cam_idx in enumerate(valid_indices):
            cam_id, cam_cfg, source = cam_cfgs[cam_idx]
            owt, nd, bb, dm = _prepare_camera(
                cam_id, frames[cam_idx], all_dets[vi],
                trackers[cam_idx], mappers[cam_idx],
                track_ctrs[cam_idx], desc_caches[cam_idx], sw, sh,
            )
            preps.append((cam_idx, owt, nd, bb, dm))

        fbp_list = []
        for item in preps:
            cam_idx, owt, nd, bb, dm = item
            if bb:
                fbp_list.append((frames[cam_idx], bb))

        t_prep = time.time()
        multi_descs = extract_descriptors_multi(fbp_list) if fbp_list else []
        t_reid = time.time()

        fbp_i = 0
        # ─────────────────────────────────────────────────────────────
        # ФАЗА 1: ReID synchronously, sequentially for all cameras.
        # ─────────────────────────────────────────────────────────────
        # Это критично для корректности ReID state. Все assign() и end_frame()
        # должны идти последовательно в одном потоке — иначе race conditions
        # на _frame_count, _vote_history, _consolidation_map.
        #
        # Сначала собираем desc_map'ы для всех камер (тоже последовательно,
        # хотя это легковесно), затем гоняем ReID синхронно.
        per_cam_results = []  # [(cam_idx, objects_world), ...]
        for item in preps:
            cam_idx, owt, nd, bb, dm = item
            if bb:
                descs = multi_descs[fbp_i]
                fbp_i += 1
                for pos, idx in enumerate(nd):
                    dm[idx] = descs[pos]

            cam_id, cam_cfg, source = cam_cfgs[cam_idx]
            objects_world = _run_reid_for_camera(
                cam_id, owt, nd, dm,
                desc_caches[cam_idx], reid_mgr,
                advance_frame=False,
            )
            per_cam_results.append((cam_idx, objects_world))

        if reid_mgr is not None:
            reid_mgr.advance_frame()

        # Снимок total_unique ОДИН раз после всех assign/end_frame,
        # чтобы все камеры показывали одно и то же значение.
        total_unique = reid_mgr.total_unique if reid_mgr is not None else 0

        # ─────────────────────────────────────────────────────────────
        # ФАЗА 2: encode/annotate/heatmap parallel for all cameras.
        # ─────────────────────────────────────────────────────────────
        # Эти операции работают с per-camera данными (heatmap, frame) и не
        # трогают ReID state, поэтому безопасно гонять параллельно.
        encode_futures = []
        for cam_idx, objects_world in per_cam_results:
            cam_id, cam_cfg, source = cam_cfgs[cam_idx]
            encode_futures.append(_encode_pool.submit(
                _encode_and_publish,
                cam_id, frames[cam_idx], objects_world,
                cam_cfg, source, heatmaps[cam_idx], total_unique,
            ))

        for f in encode_futures:
            try:
                f.result()
            except Exception as exc:
                print(f"[encode] error: {exc}")

        t_encode = time.time()

        _iter += 1
        if _iter <= WARMUP_ITERS:
            print(f"[batch] warmup {_iter}/{WARMUP_ITERS} — "
                  f"read={(t_read-t0)*1000:.0f}ms "
                  f"det={(t_detect-t_read)*1000:.0f}ms "
                  f"reid={(t_reid-t_prep)*1000:.0f}ms "
                  f"enc={(t_encode-t_reid)*1000:.0f}ms")
            continue

        broadcast_current_state()

        elapsed = time.time() - t0
        # _frame_skip отключён: в тесте обрабатывается каждый кадр.
        # Адаптивный skip создаёт нерегулярные пропуски когда GPU не успевает,
        # из-за чего ByteTrack теряет треки и ReID создаёт лишние ID.
        # Если CPU/GPU не справляются — FPS просто будет ниже TARGET_FPS,
        # но качество ReID останется стабильным.
        _frame_skip = 0
        remaining = FRAME_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)
    except Exception as e:
        import traceback
        print(f"[batch] FATAL: {e}")
        traceback.print_exc()

    for cap in caps:
        if cap is not None:
            cap.release()
    print("[batch] Stopped")
