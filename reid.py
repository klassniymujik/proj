"""
reid.py — Cross-camera Re-Identification with two-layer track lifecycle.

ARCHITECTURE
============
Two distinct concepts:

1. TentativeTrack — one per live (camera, ByteTrack local_id).
   Born when ByteTrack first reports a (cam, lid) pair.
   Accumulates descriptors, positions, frame timestamps while alive.
   Dies when ByteTrack stops reporting that (cam, lid).
   Has confirmed_id: Optional[int] — None until "promoted".

2. ConfirmedTrack — one per unique person ever seen.
   Created when a TentativeTrack accumulates CONFIRMATION_FRAMES descriptors.
   At creation, runs bank-vs-bank matching against DORMANT confirmed tracks
   (people who were active recently but no tentative track currently holds
   them). If match found → revive that confirmed_id. Else → new id.
   ConfirmedTrack.status ∈ {ACTIVE, DORMANT}.
   ACTIVE = at least one TentativeTrack currently bound to it.
   DORMANT = no live tentative track. Has dormant_since_frame timestamp.
   Removed permanently when DORMANT > DORMANT_TTL_FRAMES (30 sec at 25fps).

PUBLIC contract
===============
- assign(cam, lid, desc, wx, wy, confidence, active_local_ids, bbox)
  → int or None. Returns confirmed_id if track is confirmed, None if tentative.
- end_frame(cam, active_lids): kills tentative tracks not in active_lids,
  transitions confirmed tracks to DORMANT if no tentative holds them,
  runs GC.
- total_unique: count of confirmed tracks ever created.
- active_track_count: confirmed tracks currently ACTIVE.

What this DOESN'T have anymore (vs old architecture):
- Anchor rebuild, ghost anchors, voting windows.
- Offline boundary-cos consolidation.
- Spatial veto (homography projection unused).
- Track splitting.

The simplification is intentional: this architecture is designed for
real-time trajectory tracking and heatmap rendering, where stability of
confirmed_id matters more than chasing best evaluation accuracy.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque, Counter
from typing import Optional

import cv2
import numpy as np

# ── OSNet model loading ──────────────────────────────────────────────────────
_reid_model = None
_reid_device = None
_reid_half = False
_reid_lock = threading.Lock()

MIN_BBOX_SIZE = 10
BBOX_PAD_RATIO = 0.1
_INPUT_H = 256
_INPUT_W = 128
_MEAN_T = None
_STD_T = None


def preload_reid():
    model = _get_reid_model()
    if model is None or _reid_device is None:
        return
    try:
        import torch
        dummy = np.random.randint(0, 255, (288, 360, 3), dtype=np.uint8)
        for nc in range(1, 13):
            bb = [[30+i*25, 20+i*15, 80+i*25, 180+i*15] for i in range(nc)]
            for _ in range(2):
                extract_descriptors_multi([(dummy, bb)])
        if _reid_device.type == "cuda":
            torch.cuda.synchronize()
        print("[reid] Warmup done (1-12 crops x2, flip TTA)")
    except Exception as e:
        print(f"[reid] preload error: {e}")


def _get_reid_model():
    global _reid_model, _reid_device, _reid_half, _MEAN_T, _STD_T
    if _reid_model is not None:
        return _reid_model
    with _reid_lock:
        if _reid_model is not None:
            return _reid_model
        try:
            import torch
            import torchreid

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"[reid] Loading OSNet-ain-x1.0 on {device}")

            model = torchreid.models.build_model(
                name="osnet_ain_x1_0", num_classes=4101, pretrained=False)

            wp = os.path.join(os.path.dirname(__file__), '..', '..',
                              'reid_weights',
                              'osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
            if os.path.exists(wp):
                state = torch.load(wp, map_location=device)
                new_state = {k.replace("module.", "", 1): v for k, v in state.items()}
                new_state = {k: v for k, v in new_state.items() if not k.startswith("classifier.")}
                result = model.load_state_dict(new_state, strict=False)
                print(f"[reid] Weights: {wp}")
                print(f"[reid] missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
            else:
                print(f"[reid] WARNING: no weights ({wp})")

            model.eval().to(device)

            _reid_device = device
            if device.type == "cuda":
                _reid_half = True
                model.half()
            _reid_model = model
            _MEAN_T = torch.tensor([0.485, 0.456, 0.406],
                                   device=device).view(1, 3, 1, 1)
            _STD_T = torch.tensor([0.229, 0.224, 0.225],
                                  device=device).view(1, 3, 1, 1)
            if _reid_half:
                _MEAN_T = _MEAN_T.half()
                _STD_T = _STD_T.half()

            with torch.no_grad():
                d = torch.randn(1, 3, _INPUT_H, _INPUT_W, device=device)
                if _reid_half:
                    d = d.half()
                _ = model(d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[reid] OSNet ready (half={_reid_half})")
        except Exception as e:
            print(f"[reid] ERROR: {e}")
            _reid_model = None
    return _reid_model


def _preprocess(crop_bgr):
    import torch
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_INPUT_W, _INPUT_H))
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0
                         ).permute(2, 0, 1).unsqueeze(0).to(_reid_device)
    if _reid_half:
        t = t.half()
    return (t - _MEAN_T) / _STD_T


def _pad_bbox(bbox, h, w):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * BBOX_PAD_RATIO), int(bh * BBOX_PAD_RATIO)
    return max(0, x1-px), max(0, y1-py), min(w, x2+px), min(h, y2+py)


def _avg_flip(a, b):
    v = (a + b) * 0.5
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def extract_descriptors_batch(frame, bboxes):
    if not bboxes:
        return []
    model = _get_reid_model()
    h, w = frame.shape[:2]
    vi, crops = [], []
    for i, bb in enumerate(bboxes):
        x1, y1, x2, y2 = _pad_bbox(bb, h, w)
        if (x2-x1) >= MIN_BBOX_SIZE and (y2-y1) >= MIN_BBOX_SIZE:
            vi.append(i)
            crops.append(frame[y1:y2, x1:x2])
    res = [None] * len(bboxes)
    if not crops or model is None:
        return res
    try:
        import torch
        ts = []
        for c in crops:
            ts.append(_preprocess(c))
            ts.append(_preprocess(cv2.flip(c, 1)))
        with torch.no_grad():
            embs = model(torch.cat(ts, 0)).detach().cpu().numpy().astype(np.float32)
        for i, si in enumerate(vi):
            res[si] = _avg_flip(embs[2*i], embs[2*i+1])
    except Exception as e:
        print(f"[reid] batch error: {e}")
    return res


def extract_descriptors_multi(frame_bbox_pairs):
    if not frame_bbox_pairs:
        return []
    model = _get_reid_model()
    crops, pairs = [], []
    for fi, (frame, bboxes) in enumerate(frame_bbox_pairs):
        h, w = frame.shape[:2]
        for bi, bb in enumerate(bboxes):
            x1, y1, x2, y2 = _pad_bbox(bb, h, w)
            if (x2-x1) >= MIN_BBOX_SIZE and (y2-y1) >= MIN_BBOX_SIZE:
                c = frame[y1:y2, x1:x2]
                crops.append(c)
                crops.append(cv2.flip(c, 1))
                pairs.append((fi, bi))
    res = [[None]*len(bbs) for _, bbs in frame_bbox_pairs]
    if not crops or model is None:
        return res
    try:
        import torch
        ts = [_preprocess(c) for c in crops]
        with torch.no_grad():
            embs = model(torch.cat(ts, 0)).detach().cpu().numpy().astype(np.float32)
        for i, (fi, bi) in enumerate(pairs):
            res[fi][bi] = _avg_flip(embs[2*i], embs[2*i+1])
    except Exception as e:
        print(f"[reid] multi error: {e}")
    return res


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v



# ── Параметры новой архитектуры ──────────────────────────────────────────────

# Конфиденс детекции ниже которого мы НЕ обрабатываем descriptor.
# Низкоконфидентные детекции часто содержат части других людей или фона,
# и засоряют bank. Тем не менее, tentative tracks для них всё равно
# создаются — просто их descriptors игнорируются.
REID_MIN_CONF = 0.50

# Сколько descriptors нужно tentative track'у накопить, прежде чем
# мы попытаемся его "промоутить" в ConfirmedTrack. При DESC_INTERVAL=1
# (descriptor каждый кадр) и 25 fps это даёт ~0.6 секунды "слепой зоны"
# для каждого нового человека на видео и на heatmap.
# Увеличить → надёжнее матчинг к dormant, но дольше слепая зона.
# Уменьшить → быстрее появление, но менее надёжный bank-vs-bank match.
# С per-query top-1 matching (точнее чем top-K mean) достаточно 15.
CONFIRMATION_FRAMES = 15

# Сколько кадров confirmed track может быть в DORMANT состоянии прежде чем
# его удалим окончательно. При 25 fps: 750 кадров ≈ 30 секунд.
# Это и есть твоё требование "вернувшийся через 30 сек считается тем же".
DORMANT_TTL_FRAMES = 750

# Сколько descriptors хранить в банке confirmed track. Когда переполняется,
# выкидываем самые старые. 50 даёт хорошую статистику для bank-vs-bank
# матчинга, не съедает много памяти (~100KB на confirmed track).
BANK_MAX_SIZE = 50

# Порог cos distance для bank-vs-bank матчинга при confirmation против
# [Устаревшие пороги per-query top-1 — заменены на centroid-based matching]
# См. MATCH_CENTROID_ACTIVE и MATCH_CENTROID_DORMANT ниже.

# Максимальное расстояние (метры) между позициями tentative и ACTIVE
# confirmed track для разрешения ACTIVE match. Один и тот же человек
# на разных камерах должен проецироваться в близкие мировые координаты;
# разные люди — далеко. Это главный барьер против ложных слияний,
# т.к. OSNet-ain не всегда различает людей визуально.
SPATIAL_GATE_ACTIVE = 5.0

# Grace period: сколько кадров confirmed track может быть без живого
# tentative прежде чем перейдёт в DORMANT. Без этого ByteTrack теряет
# трек на 1–2 кадра → immediate DORMANT → при повторном обнаружении
# новый tentative → новая промоция → шанс ошибки. Grace period
# уменьшает частоту ACTIVE↔DORMANT циклов.
DORMANT_GRACE_FRAMES = 30

# Порог косинусного расстояния нового дескриптора до центроида банка.
# Если dist > порога — дескриптор не добавляется (защита от загрязнения банка
# при ложном слиянии). OSNet same-person ≈ 0.03–0.18, different ≈ 0.10–0.30+.
BANK_PURITY_THRESHOLD = 0.25

# ── Centroid-based matching (первичная метрика) ──────────────────────────────
# Косинусное расстояние между центроидом tentative и центроидом candidate
# используется как ОСНОВНАЯ метрика вместо per-query top-1 mean.
# Centroid стабильнее: same-person ≈ 0.02–0.15, different ≈ 0.20–0.50+.
# Per-query top-1 был обманчиво мягким (2–3/15 случайных совпадений → 0.05–0.15
# для разных людей).
MATCH_CENTROID_ACTIVE = 0.18
MATCH_CENTROID_DORMANT = 0.28

# Максимальное пространственное расстояние (метры) для ACTIVE merge.
# Жёсткий gate: один и тот же человек не может "телепортировать".
SPATIAL_GATE_ACTIVE_MERGE = 2.0

# Максимальное пространственное расстояние (метры) для DORMANT revive.
# Чуть мягче чем ACTIVE (2.0m) — человек мог отойти и вернуться,
# но 2.5m всё ещё физически разумно. Блокирует "телепортацию".
SPATIAL_GATE_DORMANT_MERGE = 2.5

# Сколько кадров после последнего наблюдения CID на своей камере мы считаем
# его "присутствующим" и запрещаем ACTIVE cross-camera match. При 25 fps
# 10 кадров = 0.4 сек — защищает от ложного слияния при кратковременной
# потере трека ByteTrack'ом (1–2 кадра). Если CID реально присутствует на
# камере A, cross-camera match с камеры B почти наверняка ложный.
ACTIVE_PRESENCE_GRACE_FRAMES = 10

# Включить debug-логирование решений в _promote_tentative.
DEBUG_PROMOTION = True

# Каждые сколько кадров запускаем GC удаления старых DORMANT tracks.
GC_INTERVAL_FRAMES = 100

# Статусы для ConfirmedTrack
STATUS_ACTIVE = "active"
STATUS_DORMANT = "dormant"


def _bbox_area(bbox):
    """Площадь bbox в пикселях. bbox = [x1, y1, x2, y2]."""
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


# ── Внутренние структуры ─────────────────────────────────────────────────────

class TentativeTrack:
    """Live (cam, local_id) тандем. Накапливает descriptors пока ByteTrack
    отдаёт этот local_id. Имеет confirmed_id когда его "промоутили"."""
    __slots__ = ("cam", "lid", "birth_frame", "last_frame",
                 "descs", "last_wx", "last_wy", "confirmed_id")

    def __init__(self, cam, lid, frame):
        self.cam = cam
        self.lid = lid
        self.birth_frame = frame
        self.last_frame = frame
        # Используем list для descriptors (не deque) — нам нужен access ко всему
        # для bank-vs-bank при confirmation.
        self.descs = []
        self.last_wx = 0.0
        self.last_wy = 0.0
        self.confirmed_id = None  # None пока не созрел


class ConfirmedTrack:
    """Уникальный человек. Один confirmed_id на одного человека на всю сессию.
    Никогда не меняет ID после создания."""
    __slots__ = ("cid", "status", "bank", "centroid", "dormant_since_frame",
                 "pending_dormant_since_frame",
                 "last_wx", "last_wy", "last_seen_frame")

    def __init__(self, cid, descs, wx, wy, frame):
        self.cid = cid
        self.status = STATUS_ACTIVE
        self.bank = list(descs[-BANK_MAX_SIZE:])
        if self.bank:
            stacked = np.stack([_norm(d) for d in self.bank]).astype(np.float32)
            self.centroid = stacked.mean(axis=0)
            n = float(np.linalg.norm(self.centroid))
            self.centroid = self.centroid / n if n > 0 else self.centroid
        else:
            self.centroid = None
        self.dormant_since_frame = None
        self.pending_dormant_since_frame = None
        self.last_wx = wx
        self.last_wy = wy
        self.last_seen_frame = frame

    def add_descs(self, descs, wx, wy, frame):
        """Добавить descriptors в банк с purity gate. Возвращает (n_added, n_rejected, max_reject_dist)."""
        n_added = 0
        n_rejected = 0
        max_reject_dist = 0.0
        for d in descs:
            d_norm = _norm(d)
            if self.centroid is not None and len(self.bank) >= 3:
                cos_dist = 1.0 - float(np.dot(d_norm, self.centroid))
                if cos_dist > BANK_PURITY_THRESHOLD:
                    n_rejected += 1
                    if cos_dist > max_reject_dist:
                        max_reject_dist = cos_dist
                    continue
            self.bank.append(d)
            n_added += 1
            if self.centroid is not None:
                old_n = len(self.bank) - 1
                new_n = len(self.bank)
                self.centroid = (self.centroid * old_n + d_norm) / new_n
                cn = float(np.linalg.norm(self.centroid))
                self.centroid = self.centroid / cn if cn > 0 else self.centroid
            else:
                self.centroid = d_norm.copy()
        if len(self.bank) > BANK_MAX_SIZE:
            del self.bank[:len(self.bank) - BANK_MAX_SIZE]
            self._recompute_centroid()
        self.last_wx = wx
        self.last_wy = wy
        self.last_seen_frame = frame
        return n_added, n_rejected, max_reject_dist

    def _recompute_centroid(self):
        if not self.bank:
            self.centroid = None
            return
        stacked = np.stack([_norm(d) for d in self.bank]).astype(np.float32)
        self.centroid = stacked.mean(axis=0)
        n = float(np.linalg.norm(self.centroid))
        self.centroid = self.centroid / n if n > 0 else self.centroid

    def go_dormant(self, frame):
        self.status = STATUS_DORMANT
        self.dormant_since_frame = frame
        self.pending_dormant_since_frame = None

    def revive(self, frame):
        self.status = STATUS_ACTIVE
        self.dormant_since_frame = None
        self.pending_dormant_since_frame = None
        self.last_seen_frame = frame


# ── Главный manager ──────────────────────────────────────────────────────────

class GlobalReIDManager:
    """
    Хранит все TentativeTrack и ConfirmedTrack, выдаёт confirmed_id'ы.

    Threading: один публичный лок `_lk` для всех state mutations. Snapshot
    operations берут лок коротко. assign() и end_frame() предполагаются
    последовательным вызовом из одного потока (см. batch.py sequential ReID).
    """

    def __init__(self):
        self._lk = threading.Lock()

        self._tentative = {}      # (cam, lid) -> TentativeTrack
        self._confirmed = {}      # cid -> ConfirmedTrack
        self._cid_seq = 0         # счётчик для новых cid'ов

        self._frame = 0           # глобальный frame counter
        self._last_gc_frame = 0

        # Счётчики для метрик
        self._n_promotions = 0           # сколько tentative → confirmed
        self._n_revivals = 0             # сколько раз matched к DORMANT
        self._n_new_confirmed = 0        # сколько раз создали новый cid
        self._n_total_confirmed_ever = 0 # = visitor_count

        self._purity_rej_per_frame = 0
        self._purity_rej_merge = 0
        self._purity_max_dist = 0.0
        self._purity_summary_interval = 100
        self._last_purity_frame = 0

    # ────────────────────────────────────────────────────────────────────────
    # Публичный API (то что batch.py / main.py вызывают)
    # ────────────────────────────────────────────────────────────────────────

    def assign(self, cam, lid, desc, wx, wy, confidence=1.0,
               active_local_ids=None, bbox=None):
        """
        Главный метод. Вызывается для каждой детекции на каждом кадре.
        Возвращает confirmed_id (int) или None если трек ещё tentative.

        Параметры active_local_ids принимается для совместимости с прежним
        API но не используется. bbox используется для quality gate.
        """
        with self._lk:
            key = (cam, lid)
            tent = self._tentative.get(key)
            if tent is None:
                tent = TentativeTrack(cam, lid, self._frame)
                self._tentative[key] = tent

            tent.last_frame = self._frame
            tent.last_wx = float(wx) if wx is not None else 0.0
            tent.last_wy = float(wy) if wy is not None else 0.0

            # Descriptor добавляем в банк tentative только если confidence
            # достаточный. Низкоконфидентные детекции часто содержат части
            # других людей.
            if desc is not None and confidence >= REID_MIN_CONF:
                tent.descs.append(desc)

            # Если уже confirmed — просто добавляем descriptors в банк
            # confirmed track и возвращаем cid.
            if tent.confirmed_id is not None:
                cid = tent.confirmed_id
                ct = self._confirmed.get(cid)
                if ct is not None:
                    if desc is not None and confidence >= REID_MIN_CONF:
                        n_add, n_rej, max_rd = ct.add_descs(
                            [desc], tent.last_wx, tent.last_wy, self._frame)
                        if n_rej > 0:
                            self._purity_rej_per_frame += n_rej
                            self._purity_max_dist = max(
                                self._purity_max_dist, max_rd)
                    else:
                        ct.last_wx = tent.last_wx
                        ct.last_wy = tent.last_wy
                        ct.last_seen_frame = self._frame
                    if ct.pending_dormant_since_frame is not None:
                        ct.pending_dormant_since_frame = None
                return cid

            # Не confirmed. Проверяем готов ли промоутить.
            if len(tent.descs) >= CONFIRMATION_FRAMES:
                cid = self._promote_tentative(tent)
                return cid

            # Всё ещё tentative — возвращаем None.
            return None

    def end_frame(self, cam, active_lids):
        """
        Вызывается после всех assign() для камеры в этом кадре.

        1. Tentative tracks которые не в active_lids — удаляем. Если у них был
           confirmed_id — помечаем confirmed track как pending_dormant (если
           ни один другой tentative его не "держит"). После grace period —
           окончательный переход в DORMANT.
        2. Проверяем pending_dormant tracks — истёк ли grace period.
        3. Возможно запускаем GC старых DORMANT tracks.
        4. Инкрементируем frame counter (но только для cam=0, чтобы он
           продвигался один раз за итерацию multi-camera цикла).
        """
        with self._lk:
            active_lids = set(active_lids) if active_lids else set()

            # Найти все tentative tracks этой камеры которые умерли
            dead_keys = []
            for key, tent in self._tentative.items():
                if key[0] != cam:
                    continue
                if key[1] not in active_lids:
                    dead_keys.append(key)

            for key in dead_keys:
                tent = self._tentative.pop(key)
                if tent.confirmed_id is None:
                    continue
                cid = tent.confirmed_id
                still_active = any(
                    t.confirmed_id == cid for t in self._tentative.values()
                )
                if not still_active:
                    ct = self._confirmed.get(cid)
                    if ct is not None and ct.status == STATUS_ACTIVE:
                        ct.pending_dormant_since_frame = self._frame

            # Проверяем все pending_dormant: если grace period истёк —
            # переводим в DORMANT. Если снова появился tentative — отменяем.
            for ct in self._confirmed.values():
                if ct.status != STATUS_ACTIVE:
                    continue
                if ct.pending_dormant_since_frame is None:
                    continue
                # Есть ли живой tentative с этим cid?
                has_tent = any(
                    t.confirmed_id == ct.cid for t in self._tentative.values()
                )
                if has_tent:
                    ct.pending_dormant_since_frame = None
                elif (self._frame - ct.pending_dormant_since_frame
                      >= DORMANT_GRACE_FRAMES):
                    ct.go_dormant(self._frame)

            # GC по таймеру.
            if self._frame - self._last_gc_frame >= GC_INTERVAL_FRAMES:
                self._gc_dormant()
                self._last_gc_frame = self._frame

            # Frame counter продвигаем только для cam=0, чтобы он
            # инкрементировался один раз за итерацию multi-camera пайплайна.
            if cam == 0:
                self._frame += 1
                if (self._frame - self._last_purity_frame
                        >= self._purity_summary_interval):
                    total_rej = (self._purity_rej_per_frame
                                 + self._purity_rej_merge)
                    if total_rej > 0:
                        print(f"[bank-purity summary] frames "
                              f"{self._last_purity_frame}-{self._frame}: "
                              f"per-frame_rej={self._purity_rej_per_frame} "
                              f"merge_rej={self._purity_rej_merge} "
                              f"max_dist={self._purity_max_dist:.3f}")
                    self._purity_rej_per_frame = 0
                    self._purity_rej_merge = 0
                    self._purity_max_dist = 0.0
                    self._last_purity_frame = self._frame

    @property
    def total_unique(self):
        """Сколько уникальных людей мы видели за всю сессию."""
        with self._lk:
            return self._n_total_confirmed_ever

    @property
    def active_track_count(self):
        """Сколько confirmed_id сейчас в ACTIVE статусе."""
        with self._lk:
            return sum(1 for ct in self._confirmed.values()
                       if ct.status == STATUS_ACTIVE)

    def gallery_keys(self):
        """Используется batch.py для is_known логики. Возвращает (cam, lid)
        пары которые у нас уже есть как tentative."""
        with self._lk:
            return set(self._tentative.keys())

    def reset(self):
        """Полный сброс. Вызывается из POST /reset."""
        with self._lk:
            self._tentative.clear()
            self._confirmed.clear()
            self._cid_seq = 0
            self._frame = 0
            self._last_gc_frame = 0
            self._n_promotions = 0
            self._n_revivals = 0
            self._n_new_confirmed = 0
            self._n_total_confirmed_ever = 0

    def set_homography(self, a, b, H):
        """Сохранён для backward compat. Homography в новой архитектуре
        не используется (мы не делаем spatial vetos). No-op."""
        pass

    def debug_stats(self):
        """Для GET /stats и подобных endpoints."""
        with self._lk:
            return {
                "counters": {
                    "frame": self._frame,
                    "tentative_tracks": len(self._tentative),
                    "confirmed_active": sum(
                        1 for ct in self._confirmed.values()
                        if ct.status == STATUS_ACTIVE
                    ),
                    "confirmed_dormant": sum(
                        1 for ct in self._confirmed.values()
                        if ct.status == STATUS_DORMANT
                    ),
                    "total_unique": self._n_total_confirmed_ever,
                    "promotions": self._n_promotions,
                    "revivals": self._n_revivals,
                    "new_confirmed": self._n_new_confirmed,
                }
            }

    # ────────────────────────────────────────────────────────────────────────
    # Внутренние методы (вызываются под self._lk)
    # ────────────────────────────────────────────────────────────────────────

    def _promote_tentative(self, tent):
        """
        Tentative накопил CONFIRMATION_FRAMES descriptors. Время решать
        кто он: новый человек, возрождение DORMANT, или слияние с ACTIVE
        (тот же человек одновременно виден на другой камере).

        Возвращает confirmed_id.
        """
        self._n_promotions += 1

        cam_by_cid = {}
        max_frame_by_cid = {}
        for t in self._tentative.values():
            if t.confirmed_id is not None:
                cam_by_cid.setdefault(t.confirmed_id, set()).add(t.cam)
                prev = max_frame_by_cid.get(t.confirmed_id, 0)
                if t.last_frame > prev:
                    max_frame_by_cid[t.confirmed_id] = t.last_frame

        active_cids = {ct.cid for ct in self._confirmed.values()
                       if ct.status == STATUS_ACTIVE}

        candidates = []
        n_presence_rejected = 0
        for ct in self._confirmed.values():
            if ct.cid not in active_cids and ct.status == STATUS_DORMANT:
                candidates.append(ct)
            elif ct.cid in active_cids:
                cams_with_cid = cam_by_cid.get(ct.cid, set())
                if tent.cam in cams_with_cid:
                    continue
                last_seen_on_other = max_frame_by_cid.get(ct.cid, 0)
                if (self._frame - last_seen_on_other) < ACTIVE_PRESENCE_GRACE_FRAMES:
                    n_presence_rejected += 1
                    continue
                dx = tent.last_wx - ct.last_wx
                dy = tent.last_wy - ct.last_wy
                spatial_dist = math.sqrt(dx * dx + dy * dy)
                if spatial_dist <= SPATIAL_GATE_ACTIVE:
                    candidates.append(ct)

        # Считаем сколько ACTIVE отсечено spatial gate (для debug)
        n_spatial_rejected = 0
        active_cids_for_gate = {ct.cid for ct in self._confirmed.values()
                                if ct.status == STATUS_ACTIVE}
        for ct_cid in active_cids_for_gate:
            cams_with = cam_by_cid.get(ct_cid, set())
            if tent.cam in cams_with:
                continue
            ct = self._confirmed[ct_cid]
            dx = tent.last_wx - ct.last_wx
            dy = tent.last_wy - ct.last_wy
            sd = math.sqrt(dx * dx + dy * dy)
            if sd > SPATIAL_GATE_ACTIVE:
                n_spatial_rejected += 1

        best_cid = None
        best_cctr = 1.0
        best_threshold = 1.0
        all_dists = []

        if candidates and tent.descs:
            tent_bank = np.stack(tent.descs).astype(np.float32)
            tent_bank = tent_bank / np.maximum(
                np.linalg.norm(tent_bank, axis=1, keepdims=True), 1e-9
            )
            tent_centroid = tent_bank.mean(axis=0)
            tn = float(np.linalg.norm(tent_centroid))
            if tn > 0:
                tent_centroid = tent_centroid / tn
            for ct in candidates:
                if ct.centroid is None:
                    continue
                cctr = 1.0 - float(np.dot(tent_centroid, ct.centroid))
                threshold = (MATCH_CENTROID_ACTIVE
                             if ct.status == STATUS_ACTIVE
                             else MATCH_CENTROID_DORMANT)
                reject_reason = None
                if cctr >= threshold:
                    reject_reason = f"cctr={cctr:.3f}>={threshold:.2f}"
                if reject_reason is None:
                    dx = tent.last_wx - ct.last_wx
                    dy = tent.last_wy - ct.last_wy
                    sd = math.sqrt(dx * dx + dy * dy)
                    gate = (SPATIAL_GATE_ACTIVE_MERGE
                            if ct.status == STATUS_ACTIVE
                            else SPATIAL_GATE_DORMANT_MERGE)
                    if sd > gate:
                        reject_reason = f"sdist={sd:.1f}m>{gate}m"
                tag = "A" if ct.status == STATUS_ACTIVE else "D"
                all_dists.append((ct.cid, cctr, tag, reject_reason))
                if reject_reason is None and cctr < best_cctr:
                    best_cctr = cctr
                    best_cid = ct.cid
                    best_threshold = threshold

        if DEBUG_PROMOTION:
            tent_size = len(tent.descs)
            n_dormant = sum(1 for _, _, t, _ in all_dists if t == "D")
            n_active = sum(1 for _, _, t, _ in all_dists if t == "A")
            n_rejected = sum(1 for _, _, _, v in all_dists if v is not None)
            tent_pos = f"({tent.last_wx:.1f},{tent.last_wy:.1f})"
            if all_dists:
                top3 = sorted(all_dists, key=lambda x: x[1])[:3]
                top3_str = ", ".join(
                    f"cid={c}:{d:.3f}({t})"
                    + (f" REJ({v})" if v else "")
                    for c, d, t, v in top3)
                if best_cid is not None:
                    ct = self._confirmed[best_cid]
                    best_pos = f"({ct.last_wx:.1f},{ct.last_wy:.1f})"
                    best_sdist = math.sqrt(
                        (tent.last_wx - ct.last_wx) ** 2 +
                        (tent.last_wy - ct.last_wy) ** 2)
                    action = "MERGE" if ct.status == STATUS_ACTIVE else "REVIVE"
                    decision = (f"{action}→cid={best_cid} "
                               f"pos={tent_pos}→{best_pos} "
                               f"sdist={best_sdist:.1f}m")
                else:
                    best_rej = ""
                    if top3 and top3[0][3] is not None:
                        best_rej = f" rej={top3[0][3]}"
                    decision = (f"NEW pos={tent_pos} "
                               f"(best cctr {best_cctr:.3f}{best_rej})")
                print(f"[promote] tent=(cam={tent.cam},lid={tent.lid}) "
                      f"descs={tent_size} "
                      f"cands(d={n_dormant},a={n_active},"
                      f"sp_rej={n_spatial_rejected},"
                      f"pr_rej={n_presence_rejected},"
                      f"rej={n_rejected}) "
                      f"top3={{{top3_str}}} → {decision}")
            else:
                print(f"[promote] tent=(cam={tent.cam},lid={tent.lid}) "
                      f"descs={tent_size} pos={tent_pos} "
                      f"no_cands(sp_rej={n_spatial_rejected},"
                      f"pr_rej={n_presence_rejected}) → NEW")

        if best_cid is not None:
            ct = self._confirmed[best_cid]
            if ct.status == STATUS_DORMANT:
                ct.revive(self._frame)
                self._n_revivals += 1
            elif ct.status == STATUS_ACTIVE:
                ct.pending_dormant_since_frame = None
            n_add, n_rej, max_rd = ct.add_descs(
                tent.descs, tent.last_wx, tent.last_wy, self._frame)
            if n_rej > 0:
                self._purity_rej_merge += n_rej
                self._purity_max_dist = max(
                    self._purity_max_dist, max_rd)
            tent.confirmed_id = best_cid
            return best_cid

        # Новый человек.
        self._cid_seq += 1
        new_cid = self._cid_seq
        ct = ConfirmedTrack(new_cid, tent.descs,
                            tent.last_wx, tent.last_wy, self._frame)
        self._confirmed[new_cid] = ct
        tent.confirmed_id = new_cid
        self._n_new_confirmed += 1
        self._n_total_confirmed_ever += 1
        return new_cid

    def _gc_dormant(self):
        """Удалить DORMANT confirmed tracks старше TTL."""
        to_remove = []
        for cid, ct in self._confirmed.items():
            if ct.status != STATUS_DORMANT:
                continue
            if ct.dormant_since_frame is None:
                continue
            age = self._frame - ct.dormant_since_frame
            if age >= DORMANT_TTL_FRAMES:
                to_remove.append(cid)
        for cid in to_remove:
            del self._confirmed[cid]


# ── Singleton ────────────────────────────────────────────────────────────────
_mgr = GlobalReIDManager()


def get_manager():
    return _mgr