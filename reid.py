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
            _reid_model = model
            _MEAN_T = torch.tensor([0.485, 0.456, 0.406],
                                   device=device).view(1, 3, 1, 1)
            _STD_T = torch.tensor([0.229, 0.224, 0.225],
                                  device=device).view(1, 3, 1, 1)

            with torch.no_grad():
                d = torch.randn(1, 3, _INPUT_H, _INPUT_W, device=device)
                _ = model(d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[reid] OSNet ready")
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
# (descriptor каждый кадр) и 25 fps это даёт ~1.2 секунды "слепой зоны"
# для каждого нового человека на видео и на heatmap.
# Увеличить → надёжнее матчинг к dormant, но дольше слепая зона.
# Уменьшить → быстрее появление, но менее надёжный bank-vs-bank match.
CONFIRMATION_FRAMES = 30

# Сколько кадров confirmed track может быть в DORMANT состоянии прежде чем
# его удалим окончательно. При 25 fps: 750 кадров ≈ 30 секунд.
# Это и есть твоё требование "вернувшийся через 30 сек считается тем же".
DORMANT_TTL_FRAMES = 750

# Сколько descriptors хранить в банке confirmed track. Когда переполняется,
# выкидываем самые старые. 50 даёт хорошую статистику для bank-vs-bank
# матчинга, не съедает много памяти (~100KB на confirmed track).
BANK_MAX_SIZE = 50

# Порог cos distance для bank-vs-bank матчинга. Кандидат принимается
# только если best_dist < MATCH_BANK_VS_BANK.
# Используется top-K mean similarity (берём топ-K лучших пар descriptors).
#
# Порог УЖЕСТОЧЁН до 0.30 (с 0.40). Логика: при большом числе кандидатов
# (10-30 confirmed cid'ов) вероятность случайного матча на distance 0.30-0.40
# слишком высока — это приводит к обменам ID между разными людьми. Лучше
# создать лишний cid (он со временем сольётся через DORMANT revival если
# тот же человек реально вернётся) чем "склеить" двух разных людей.
MATCH_BANK_VS_BANK = 0.30

# CONFIDENT-уровень: если best_dist < этого порога, матч принимается
# БЕЗ gap check. Логика: distance 0.15 это очень уверенный матч сам по
# себе (sim 0.85+), не имеет значения что рядом могут быть другие
# кандидаты с близкими distances — мы матчим лучшего.
#
# Это критично для сцен с несколькими похожими людьми (одинаковая одежда,
# общая обстановка): их descriptors close together, gap может быть малым,
# но best_dist всё равно clearly разный (типа 0.07 vs 0.10).
#
# Без этого мы попадаем в зону "все близко, ни один не выбран" и
# плодим cid каждый раз когда tentative промоутится.
MATCH_CONFIDENT = 0.15

# Минимальная разница между лучшим и вторым кандидатом для принятия
# revival/merge КОГДА best_dist в зоне неопределённости (0.15-0.40).
# Защита от random matching когда несколько кандидатов одинаково плохие.
MATCH_GAP = 0.08

# Top-K — сколько лучших similarity pairs использовать для усреднения.
MATCH_TOP_K = 20

# ── Bank protection (защита от загрязнения descriptors после ID swap) ────────
#
# Когда ByteTrack swap'нул двух людей, наш confirmed track продолжает
# получать descriptors но уже **другого** человека. Без защиты bank
# постепенно загрязняется и последующие матчи становятся ненадёжными.
#
# Hysteresis: перед добавлением descriptor в bank confirmed track,
# проверяем max cosine similarity с уже существующими в bank.
# Если sim < BANK_INTRUSION_SIM — descriptor совсем не похож на bank'а
# хозяина → подозрительно (вероятно swap). НЕ добавляем.
#
# Порог 0.5 (= distance 0.5) консервативный: реальные descriptors одного
# человека дают max sim 0.85+ против своего bank, разные люди обычно
# < 0.40. Зазор 0.5 безопасный.
BANK_INTRUSION_SIM = 0.50

# ── Soft sticky (cam, lid) → cid memory ───────────────────────────────────────
#
# ByteTrack часто теряет local_id и сразу же пересоздаёт его с тем же
# значением для того же физического человека (классический сценарий
# occlusion). Без sticky наша система начинает заново матчинг этого
# tentative, и может ассоциировать его с другим cid (особенно когда
# в комнате есть люди в похожей одежде).
#
# Sticky запоминает (cam, lid) → cid с timestamp на STICKY_TTL_FRAMES.
# Когда тот же (cam, lid) появляется снова в окне TTL — мы помечаем
# tentative как "кандидат на sticky-revival" и ВЕРИФИЦИРУЕМ через ReID:
# когда накопится STICKY_VERIFY_FRAMES descriptors, сравниваем с bank
# старого cid, и если distance < STICKY_THRESHOLD — ассоциируем.
# Иначе отпускаем sticky (это другой человек после ID swap).
#
# Эта мягкая стратегия защищает от обоих сценариев:
# - ByteTrack пересоздал тот же lid для того же человека → sticky cid
# - ByteTrack swap'нул двух людей → ReID видит другие descriptors → отпуск
STICKY_TTL_FRAMES = 100  # ~4 сек @ 25fps; для коротких потерь lid'а
STICKY_VERIFY_FRAMES = 5  # сколько descriptors накопить перед верификацией
STICKY_THRESHOLD = 0.20  # max distance чтобы считать sticky подтверждённым

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
                 "descs", "last_wx", "last_wy", "confirmed_id",
                 "sticky_candidate_cid", "sticky_resolved")

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
        # Sticky verification: если sticky_map содержал запись для (cam, lid),
        # храним cid сюда. После STICKY_VERIFY_FRAMES descriptors верифицируем
        # через ReID. После одной попытки (успех или провал) sticky_resolved
        # становится True и больше не пробуем.
        self.sticky_candidate_cid = None
        self.sticky_resolved = False


class ConfirmedTrack:
    """Уникальный человек. Один confirmed_id на одного человека на всю сессию.
    Никогда не меняет ID после создания."""
    __slots__ = ("cid", "status", "bank", "dormant_since_frame",
                 "last_wx", "last_wy", "last_seen_frame")

    def __init__(self, cid, descs, wx, wy, frame):
        self.cid = cid
        self.status = STATUS_ACTIVE
        # Bank — компактный список L2-norm descriptors. Когда переполняется,
        # выкидываем старые (FIFO).
        self.bank = list(descs[-BANK_MAX_SIZE:])
        self.dormant_since_frame = None
        self.last_wx = wx
        self.last_wy = wy
        self.last_seen_frame = frame

    def add_descs(self, descs, wx, wy, frame):
        """Добавить descriptors из tentative track в банк. Применяет
        проверку BANK_INTRUSION_SIM: descriptor добавляется только если
        он достаточно похож хотя бы на один из существующих в bank
        (защита от загрязнения после ID swap)."""
        rejected = 0
        for d in descs:
            if len(self.bank) > 0:
                # Считаем max cosine similarity descriptor против bank
                d_arr = np.asarray(d, dtype=np.float32)
                d_norm = max(float(np.linalg.norm(d_arr)), 1e-9)
                d_arr = d_arr / d_norm
                bank_arr = np.stack(self.bank).astype(np.float32)
                bank_norms = np.maximum(
                    np.linalg.norm(bank_arr, axis=1, keepdims=True), 1e-9
                )
                bank_arr = bank_arr / bank_norms
                max_sim = float((bank_arr @ d_arr).max())
                if max_sim < BANK_INTRUSION_SIM:
                    # Descriptor подозрительно непохож на свой bank.
                    # Возможно ByteTrack swap'нул людей, descriptor
                    # принадлежит другому. Не добавляем.
                    rejected += 1
                    continue
            self.bank.append(d)
        if len(self.bank) > BANK_MAX_SIZE:
            # Выкидываем самые старые
            del self.bank[:len(self.bank) - BANK_MAX_SIZE]
        self.last_wx = wx
        self.last_wy = wy
        self.last_seen_frame = frame
        # Возвращаем сколько отклонили — для логирования/диагностики
        return rejected

    def go_dormant(self, frame):
        self.status = STATUS_DORMANT
        self.dormant_since_frame = frame

    def revive(self, frame):
        self.status = STATUS_ACTIVE
        self.dormant_since_frame = None
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

        # Sticky memory: (cam, lid) → (cid, frame_when_lost)
        # Используется когда ByteTrack пересоздаёт тот же lid для того же
        # физического человека. См. STICKY_TTL_FRAMES и _try_sticky_verify.
        self._sticky_map = {}

        # Счётчики для метрик
        self._n_promotions = 0           # сколько tentative → confirmed
        self._n_revivals = 0             # сколько раз matched к DORMANT
        self._n_new_confirmed = 0        # сколько раз создали новый cid
        self._n_total_confirmed_ever = 0 # = visitor_count
        self._n_sticky_hits = 0          # сколько раз sticky подтвердился
        self._n_sticky_misses = 0        # сколько раз sticky отвергнут ReID
        self._n_bank_intrusions_rejected = 0  # отклонённые descriptors

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
                # Проверяем sticky_map: был ли у этой (cam, lid) пары
                # cid в недавнем прошлом?
                sticky_entry = self._sticky_map.get(key)
                if sticky_entry is not None:
                    sticky_cid, frame_when_lost = sticky_entry
                    age = self._frame - frame_when_lost
                    if age < STICKY_TTL_FRAMES:
                        # Только если этот cid всё ещё существует
                        # (не был полностью удалён по DORMANT_TTL)
                        if sticky_cid in self._confirmed:
                            tent.sticky_candidate_cid = sticky_cid
                    # Удаляем запись либо потому что используем, либо
                    # потому что устарела
                    self._sticky_map.pop(key, None)

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
                        rejected = ct.add_descs([desc], tent.last_wx,
                                                tent.last_wy, self._frame)
                        self._n_bank_intrusions_rejected += rejected
                    else:
                        # Без descriptor — только обновляем "last seen"
                        ct.last_wx = tent.last_wx
                        ct.last_wy = tent.last_wy
                        ct.last_seen_frame = self._frame
                return cid

            # Не confirmed. Если есть sticky-кандидат — пробуем верифицировать.
            # После STICKY_VERIFY_FRAMES descriptors сравниваем bank tentative
            # с bank старого cid; если distance < STICKY_THRESHOLD —
            # подтверждаем sticky. Иначе отпускаем sticky и идём обычным
            # путём.
            if (tent.sticky_candidate_cid is not None
                    and not tent.sticky_resolved
                    and len(tent.descs) >= STICKY_VERIFY_FRAMES):
                sticky_cid = self._try_sticky_verify(tent)
                if sticky_cid is not None:
                    return sticky_cid
                # sticky_resolved уже True после _try_sticky_verify

            # Проверяем готов ли обычный promote.
            if len(tent.descs) >= CONFIRMATION_FRAMES:
                cid = self._promote_tentative(tent)
                return cid

            # Всё ещё tentative — возвращаем None.
            return None

    def end_frame(self, cam, active_lids):
        """
        Вызывается после всех assign() для камеры в этом кадре.

        1. Tentative tracks которые не в active_lids — удаляем. Если у них был
           confirmed_id — confirmed track переходит в DORMANT (если ни один
           другой tentative его не "держит").
        2. Возможно запускаем GC старых DORMANT tracks.
        3. Инкрементируем frame counter (но только для cam=0, чтобы он
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
                    # Не дожил до confirmation — просто выбрасываем.
                    continue
                # Был confirmed. Записываем в sticky_map: если ByteTrack
                # пересоздаст тот же (cam, lid) в течение STICKY_TTL_FRAMES,
                # мы попробуем восстановить этот cid (с верификацией ReID).
                self._sticky_map[key] = (tent.confirmed_id, self._frame)
                # Проверим — есть ли ещё живые tentative
                # с тем же confirmed_id? Если нет — переводим в DORMANT.
                cid = tent.confirmed_id
                still_active = any(
                    t.confirmed_id == cid for t in self._tentative.values()
                )
                if not still_active:
                    ct = self._confirmed.get(cid)
                    if ct is not None:
                        ct.go_dormant(self._frame)

            # GC sticky_map: удалить устаревшие записи (старше STICKY_TTL)
            stale_sticky = [
                k for k, (_, f) in self._sticky_map.items()
                if self._frame - f >= STICKY_TTL_FRAMES
            ]
            for k in stale_sticky:
                self._sticky_map.pop(k, None)

            # GC по таймеру.
            if self._frame - self._last_gc_frame >= GC_INTERVAL_FRAMES:
                self._gc_dormant()
                self._last_gc_frame = self._frame

            # Frame counter продвигаем только для cam=0, чтобы он
            # инкрементировался один раз за итерацию multi-camera пайплайна.
            if cam == 0:
                self._frame += 1

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
                    "sticky_hits": self._n_sticky_hits,
                    "sticky_misses": self._n_sticky_misses,
                    "sticky_pending": len(self._sticky_map),
                    "bank_intrusions_rejected": self._n_bank_intrusions_rejected,
                }
            }

    # ────────────────────────────────────────────────────────────────────────
    # Внутренние методы (вызываются под self._lk)
    # ────────────────────────────────────────────────────────────────────────

    def _try_sticky_verify(self, tent):
        """
        Soft sticky verification.

        У tentative есть sticky_candidate_cid — был ассоциирован с этим cid
        раньше через ту же (cam, lid) пару. Проверяем через ReID что это
        действительно тот же человек:
        - сравниваем bank tentative с bank сохранённого cid
        - top-K mean distance
        - если < STICKY_THRESHOLD — подтверждаем (sticky-revival)
        - иначе отпускаем sticky (это другой человек после ID swap)

        Возвращает cid если sticky подтверждён, иначе None.
        В обоих случаях устанавливает tent.sticky_resolved = True чтобы
        не повторять верификацию (одной попытки достаточно).
        """
        tent.sticky_resolved = True
        sticky_cid = tent.sticky_candidate_cid
        tent.sticky_candidate_cid = None  # очищаем независимо от результата

        ct = self._confirmed.get(sticky_cid)
        if ct is None or not ct.bank or not tent.descs:
            return None

        # Bank-vs-bank distance
        tent_bank = np.stack(tent.descs).astype(np.float32)
        tent_bank = tent_bank / np.maximum(
            np.linalg.norm(tent_bank, axis=1, keepdims=True), 1e-9
        )
        cand_bank = np.stack(ct.bank).astype(np.float32)
        cand_bank = cand_bank / np.maximum(
            np.linalg.norm(cand_bank, axis=1, keepdims=True), 1e-9
        )
        sim = tent_bank @ cand_bank.T
        flat = sim.flatten()
        k = min(MATCH_TOP_K, len(flat))
        top_k_mean = float(np.sort(flat)[-k:].mean())
        dist = 1.0 - top_k_mean

        # Twin rule: если этот cid сейчас ACTIVE на нашей же камере
        # (через другой tentative) — не делаем sticky, это бы создало
        # конфликт. Это редкий случай но защищает от состояния
        # "два tentative на одной камере ассоциированы с одним cid".
        if ct.status == STATUS_ACTIVE:
            for t in self._tentative.values():
                if (t.cam == tent.cam and t is not tent
                        and t.confirmed_id == sticky_cid):
                    if DEBUG_PROMOTION:
                        print(f"[sticky] tent=(cam={tent.cam},lid={tent.lid}) "
                              f"sticky_cid={sticky_cid} dist={dist:.3f} "
                              f"→ REJECTED (twin on same cam)")
                    self._n_sticky_misses += 1
                    return None

        if dist < STICKY_THRESHOLD:
            # Подтверждено — sticky revival
            if DEBUG_PROMOTION:
                print(f"[sticky] tent=(cam={tent.cam},lid={tent.lid}) "
                      f"descs={len(tent.descs)} sticky_cid={sticky_cid} "
                      f"dist={dist:.3f} → CONFIRMED")
            self._n_sticky_hits += 1
            if ct.status == STATUS_DORMANT:
                ct.revive(self._frame)
                self._n_revivals += 1
            rejected = ct.add_descs(tent.descs, tent.last_wx, tent.last_wy,
                                    self._frame)
            self._n_bank_intrusions_rejected += rejected
            tent.confirmed_id = sticky_cid
            return sticky_cid
        else:
            # ReID не подтверждает — другой человек, пускаем обычный pipeline
            if DEBUG_PROMOTION:
                print(f"[sticky] tent=(cam={tent.cam},lid={tent.lid}) "
                      f"descs={len(tent.descs)} sticky_cid={sticky_cid} "
                      f"dist={dist:.3f} → REJECTED (ReID mismatch)")
            self._n_sticky_misses += 1
            return None

    def _promote_tentative(self, tent):
        """
        Tentative накопил CONFIRMATION_FRAMES descriptors. Время решать
        кто он: новый человек, или revival DORMANT, или unification с
        ACTIVE-на-другой-камере (cross-camera matching).

        Возвращает confirmed_id.

        Стратегия матчинга (3 типа кандидатов):

        1) ACTIVE на ДРУГИХ камерах (НЕ на cam tent.cam): один и тот же
           человек может одновременно быть виден несколькими камерами.
           ByteTrack на каждой камере независимо создаёт local_id, и наша
           задача — связать их в один cid. БЕЗ этого мы получаем
           фрагментацию одного человека на разные G# для каждой камеры.

           ВАЖНО: ACTIVE на ТЕКУЩЕЙ tent.cam ИСКЛЮЧЕНЫ (twin rule):
           один ByteTrack local_id уже отвечает за этого человека на
           этой камере; новый local_id здесь — это ДРУГОЙ человек.

        2) DORMANT (везде): человек ушёл из всех камер и вернулся в
           течение DORMANT_TTL_FRAMES. Классический re-identification.

        3) Новый cid: если ни один кандидат не подходит достаточно
           уверенно (см. gap check ниже).

        Gap check: revival происходит ТОЛЬКО если разница между лучшим
        и вторым кандидатом >= MATCH_GAP. Это защищает от случайного
        матчинга когда у нас несколько похожих кандидатов с близкими
        distances (например 0.17 / 0.19 / 0.22 — все ниже порога, но
        неуверенно). Без gap check мы случайно выбираем одного из них,
        что приводит к перепутыванию людей и загрязнению banks.
        """
        self._n_promotions += 1

        # Текущие ACTIVE cid'ы по камерам — для twin rule.
        active_cids_on_my_cam = set()
        for ct in self._confirmed.values():
            if ct.status != STATUS_ACTIVE:
                continue
            # Проверим есть ли у этого cid tentative track на нашей камере.
            for t in self._tentative.values():
                if t.cam == tent.cam and t.confirmed_id == ct.cid:
                    active_cids_on_my_cam.add(ct.cid)
                    break

        # Кандидаты для матчинга:
        # - ACTIVE на ДРУГИХ камерах (не на нашей)
        # - DORMANT везде
        candidates = []
        for ct in self._confirmed.values():
            if ct.cid in active_cids_on_my_cam:
                continue  # twin rule: уже представлен на нашей камере
            if ct.status == STATUS_ACTIVE or ct.status == STATUS_DORMANT:
                candidates.append(ct)

        all_dists = []  # для логирования (cid, dist, status)

        if candidates and tent.descs:
            tent_bank = np.stack(tent.descs).astype(np.float32)
            tent_bank = tent_bank / np.maximum(
                np.linalg.norm(tent_bank, axis=1, keepdims=True), 1e-9
            )
            for ct in candidates:
                if not ct.bank:
                    continue
                cand_bank = np.stack(ct.bank).astype(np.float32)
                cand_bank = cand_bank / np.maximum(
                    np.linalg.norm(cand_bank, axis=1, keepdims=True), 1e-9
                )
                # Cosine similarity matrix → flat sorted top-K mean
                sim = tent_bank @ cand_bank.T  # (M, N)
                flat = sim.flatten()
                k = min(MATCH_TOP_K, len(flat))
                top_k_mean = float(np.sort(flat)[-k:].mean())
                dist = 1.0 - top_k_mean
                all_dists.append((ct.cid, dist, ct.status))

        # Сортируем кандидатов по distance (меньше = лучше)
        all_dists.sort(key=lambda x: x[1])

        best_cid = None
        decision_reason = ""

        if all_dists:
            best = all_dists[0]
            best_dist = best[1]
            best_status = best[2]

            # Двухуровневая логика принятия решения:
            #
            # 1) CONFIDENT zone (best_dist < MATCH_CONFIDENT=0.15):
            #    Матч уверенный сам по себе. Принимаем БЕЗ gap check.
            #    Это критично для multi-person сцен где descriptors похожих
            #    людей близки — gap может быть мал, но best всё равно
            #    значимо лучше остальных.
            #
            # 2) UNCERTAIN zone (MATCH_CONFIDENT <= best_dist < MATCH_BANK_VS_BANK):
            #    Матч приемлемый, но требуем gap >= MATCH_GAP чтобы
            #    отличить от случайных совпадений.
            #
            # 3) REJECT (best_dist >= MATCH_BANK_VS_BANK):
            #    Слишком далеко — создаём NEW.

            if best_dist < MATCH_CONFIDENT:
                # CONFIDENT: принимаем без gap check
                best_cid = best[0]
                prefix = ("REVIVE" if best_status == STATUS_DORMANT
                          else "MERGE-ACTIVE")
                decision_reason = f"{prefix}→cid={best_cid} (confident)"
            elif best_dist < MATCH_BANK_VS_BANK:
                # UNCERTAIN: gap check обязателен
                if len(all_dists) >= 2:
                    second_dist = all_dists[1][1]
                    gap = second_dist - best_dist
                    if gap < MATCH_GAP:
                        decision_reason = (f"NEW (uncertain: gap={gap:.3f}<"
                                           f"{MATCH_GAP}, best={best_dist:.3f})")
                    else:
                        best_cid = best[0]
                        prefix = ("REVIVE" if best_status == STATUS_DORMANT
                                  else "MERGE-ACTIVE")
                        decision_reason = f"{prefix}→cid={best_cid} (gap={gap:.3f})"
                else:
                    # Один кандидат и он в uncertain zone — рискнём матчем
                    best_cid = best[0]
                    prefix = ("REVIVE" if best_status == STATUS_DORMANT
                              else "MERGE-ACTIVE")
                    decision_reason = f"{prefix}→cid={best_cid} (single uncertain)"
            else:
                decision_reason = (f"NEW (best dist {best_dist:.3f}>="
                                   f"{MATCH_BANK_VS_BANK})")
        else:
            decision_reason = "NEW (no candidates)"

        # Debug-логирование
        if DEBUG_PROMOTION:
            tent_size = len(tent.descs)
            candidate_count = len(candidates)
            if candidate_count > 0 and all_dists:
                top3 = all_dists[:3]
                top3_str = ", ".join(
                    f"cid={c}({'A' if s == STATUS_ACTIVE else 'D'}):{d:.3f}"
                    for c, d, s in top3
                )
                print(f"[promote] tent=(cam={tent.cam},lid={tent.lid}) "
                      f"descs={tent_size} candidates={candidate_count} "
                      f"top3={{{top3_str}}} → {decision_reason}")
            else:
                print(f"[promote] tent=(cam={tent.cam},lid={tent.lid}) "
                      f"descs={tent_size} no_candidates → {decision_reason}")

        if best_cid is not None:
            # Revival (DORMANT) или Merge (ACTIVE на другой камере)
            ct = self._confirmed[best_cid]
            if ct.status == STATUS_DORMANT:
                ct.revive(self._frame)
                self._n_revivals += 1
            # Для ACTIVE: уже active, просто добавляем descriptors
            rejected = ct.add_descs(tent.descs, tent.last_wx, tent.last_wy,
                                    self._frame)
            self._n_bank_intrusions_rejected += rejected
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