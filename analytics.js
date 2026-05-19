const API = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws";
const CALIB_KEY = "rv_analytics_calib_by_camera";
const HEAT_ROWS = 60;
const HEAT_COLS = 60;
const MAX_TRAIL_POINTS = 120;

class AnalyticsPage {
  constructor() {
    this.el = {
      heatmapCanvas: document.getElementById("heatmap-canvas"),
      heatmapWrap: document.getElementById("heatmap-wrap"),
      historyCanvas: document.getElementById("history-canvas"),
      historyHint: document.getElementById("history-hint"),
      zonesList: document.getElementById("zones-list"),
      wsDot: document.getElementById("ws-dot"),
      wsLabel: document.getElementById("ws-label"),
      total: document.getElementById("a-total"),
      active: document.getElementById("a-active"),
      footerCoords: document.getElementById("footer-coords"),
      calibCameraList: document.getElementById("calib-camera-list"),
      btnHmReset: document.getElementById("btn-hm-reset"),
      btnResetStats: document.getElementById("btn-reset-stats"),
      btnHmExport: document.getElementById("btn-hm-export"),
      calibReset: document.getElementById("calib-reset"),
    };
    this.historyCtx = this.el.historyCanvas.getContext("2d");

    this.sceneW = 20;
    this.sceneH = 20;
    this.wsRetry = 0;
    this.hadLivePoints = false;

    this.heatAccum = [];
    this.calibByCamera = {};
    this.activeCalibCamera = null;
    this.calibCameras = [];

    // Trail хранится per-confirmed_id (не per-(camera, local_id) как было!).
    // Это значит один человек = одна траектория, даже если он переходит
    // между камерами или ByteTrack потерял его на пару кадров.
    this.trackTrail = new Map();
    this.sceneCameraObjects = new Map();

    this.setup3D();
  }

  setup3D() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.el.heatmapCanvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.scene3d = new THREE.Scene();
    this.scene3d.background = new THREE.Color(0x0d0f14);
    this.scene3d.fog = new THREE.FogExp2(0x0d0f14, 0.015);
    this.scene3d.add(new THREE.AmbientLight(0xffffff, 0.6));
    this.scene3d.add(new THREE.HemisphereLight(0x7aa2ff, 0x1a1f2b, 0.5));
    const sun = new THREE.DirectionalLight(0xffffff, 0.7);
    sun.position.set(12, 20, 10);
    this.scene3d.add(sun);
    this.scene3d.add(new THREE.GridHelper(120, 120, 0x2f364a, 0x1f2536));

    this.cam3d = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
    this.cam3d.position.set(10, 16, 14);
    this.cam3d.lookAt(10, 0, 10);

    this.floorMesh = null;
    this.heatmapPivot = null;
    this.roomGroup = new THREE.Group();
    this.trajectoryGroup = new THREE.Group();
    this.scene3d.add(this.roomGroup);
    this.scene3d.add(this.trajectoryGroup);

    this.hmTextureCanvas = document.createElement("canvas");
    this.hmTextureCtx = this.hmTextureCanvas.getContext("2d");
    this.hmTexture = null;

    this.orbit = this.buildOrbitControls();
    window.addEventListener("resize", () => this.resize3D());
    this.resize3D();
  }

  buildOrbitControls() {
    let rmb = false;
    let lx = 0;
    let ly = 0;
    let theta = Math.PI / 4;
    let phi = Math.PI / 3.2;
    let radius = 24;
    const target = new THREE.Vector3(10, 0, 10);

    const update = () => {
      this.cam3d.position.set(
        target.x + radius * Math.sin(phi) * Math.sin(theta),
        target.y + radius * Math.cos(phi),
        target.z + radius * Math.sin(phi) * Math.cos(theta)
      );
      this.cam3d.lookAt(target);
    };
    update();

    this.el.heatmapCanvas.addEventListener("contextmenu", e => e.preventDefault());
    this.el.heatmapCanvas.addEventListener("mousedown", e => {
      if (e.button === 2) rmb = true;
      lx = e.clientX;
      ly = e.clientY;
    });
    window.addEventListener("mouseup", e => {
      if (e.button === 2) rmb = false;
    });
    window.addEventListener("mousemove", e => {
      if (!rmb) return;
      const dx = e.clientX - lx;
      const dy = e.clientY - ly;
      lx = e.clientX;
      ly = e.clientY;
      theta -= dx * 0.005;
      phi = Math.max(0.05, Math.min(Math.PI / 2.05, phi + dy * 0.005));
      update();
    });
    this.el.heatmapCanvas.addEventListener("wheel", e => {
      radius = Math.max(6, Math.min(120, radius + e.deltaY * 0.04));
      update();
    }, { passive: true });

    return {
      reset: (w, h) => {
        theta = Math.PI / 4;
        phi = Math.PI / 3.2;
        radius = Math.max(w, h) * 1.2;
        target.set(w / 2, 0, h / 2);
        update();
      },
    };
  }

  resize3D() {
    const w = this.el.heatmapWrap.clientWidth || 600;
    const h = this.el.heatmapWrap.clientHeight || 420;
    this.renderer.setSize(w, h, false);
    this.cam3d.aspect = w / h;
    this.cam3d.updateProjectionMatrix();
  }

  defaultCalib() {
    return { tx: 0, tz: 0, rot: 0, sx: 1, sz: 1 };
  }

  getCalib(cameraId) {
    if (cameraId === null || cameraId === undefined) return this.defaultCalib();
    return { ...this.defaultCalib(), ...(this.calibByCamera[String(cameraId)] || {}) };
  }

  setCalib(cameraId, value) {
    if (cameraId === null || cameraId === undefined) return;
    this.calibByCamera[String(cameraId)] = value;
    localStorage.setItem(CALIB_KEY, JSON.stringify(this.calibByCamera));
  }

  initHeatAccumulator() {
    this.heatAccum = Array.from({ length: HEAT_ROWS }, () => Array(HEAT_COLS).fill(0));
  }

  createFloor() {
    if (this.floorMesh) this.scene3d.remove(this.floorMesh);
    if (this.heatmapPivot) this.scene3d.remove(this.heatmapPivot);

    this.floorMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(this.sceneW, this.sceneH),
      new THREE.MeshStandardMaterial({ color: 0x1a2233, roughness: 0.9, metalness: 0.1, side: THREE.DoubleSide })
    );
    this.floorMesh.rotation.x = -Math.PI / 2;
    this.floorMesh.position.set(this.sceneW / 2, 0, this.sceneH / 2);
    this.scene3d.add(this.floorMesh);

    this.hmTextureCanvas.width = 160;
    this.hmTextureCanvas.height = 160;
    this.hmTexture = new THREE.CanvasTexture(this.hmTextureCanvas);
    this.hmTexture.needsUpdate = true;

    this.heatmapPivot = new THREE.Group();
    this.heatmapPivot.position.set(this.sceneW / 2, 0.02, this.sceneH / 2);
    this.scene3d.add(this.heatmapPivot);

    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(this.sceneW, this.sceneH),
      new THREE.MeshBasicMaterial({
        map: this.hmTexture,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      })
    );
    mesh.rotation.x = -Math.PI / 2;
    this.heatmapPivot.add(mesh);
    this.orbit.reset(this.sceneW, this.sceneH);
  }

  clearRoom() {
    this.scene3d.remove(this.roomGroup);
    this.roomGroup = new THREE.Group();
    this.scene3d.add(this.roomGroup);
  }

  addBox(obj, color) {
    const w = obj.w || obj.length || 1;
    const d = obj.d || 0.2;
    const h = obj.h || 1;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w, h, d),
      new THREE.MeshStandardMaterial({ color, roughness: 0.75 })
    );
    mesh.position.set(obj.x || 0, h / 2, obj.z || 0);
    mesh.rotation.y = THREE.MathUtils.degToRad(obj.rot || 0);
    this.roomGroup.add(mesh);
  }

  buildCameraMarker(camData) {
    const h = camData.height || 3;
    const group = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.18, 0.45), new THREE.MeshStandardMaterial({ color: 0x0055cc }));
    body.position.y = h;
    group.add(body);
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.03, h, 6),
      new THREE.MeshStandardMaterial({ color: 0x334455 })
    );
    pole.position.y = h / 2;
    group.add(pole);
    const fov = camData.fov || 90;
    const radius = h * Math.tan(THREE.MathUtils.degToRad(fov / 2)) * 0.9;
    const ring = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 32),
      new THREE.MeshBasicMaterial({ color: 0x0055cc, transparent: true, opacity: 0.25, side: THREE.DoubleSide })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.01;
    group.add(ring);
    const arrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.08, 0.5, 6),
      new THREE.MeshBasicMaterial({ color: 0x0099ff })
    );
    arrow.rotation.x = Math.PI / 2;
    arrow.position.set(0, h, 0.55);
    group.add(arrow);
    group.position.set(camData.x || 0, 0, camData.z || 0);
    group.rotation.y = THREE.MathUtils.degToRad(-(camData.yaw || 0));
    return { group, body, ring };
  }

  syncSceneCameras(cameras) {
    const keys = new Set(cameras.map((_, i) => String(i)));
    this.sceneCameraObjects.forEach((obj, key) => {
      if (!keys.has(key)) {
        this.scene3d.remove(obj.group);
        this.sceneCameraObjects.delete(key);
      }
    });
    cameras.forEach((camData, i) => {
      const key = String(i);
      if (this.sceneCameraObjects.has(key)) return;
      const marker = this.buildCameraMarker(camData);
      this.scene3d.add(marker.group);
      this.sceneCameraObjects.set(key, marker);
    });
    this.highlightActiveSceneCamera();
  }

  highlightActiveSceneCamera() {
    this.sceneCameraObjects.forEach(({ body, ring }, key) => {
      const isActive = key === String(this.activeCalibCamera);
      body.material.color.setHex(isActive ? 0x00e5a0 : 0x0055cc);
      body.material.emissive = new THREE.Color(isActive ? 0x00e5a0 : 0x000000);
      body.material.emissiveIntensity = isActive ? 0.5 : 0;
      ring.material.color.setHex(isActive ? 0x00e5a0 : 0x0055cc);
      ring.material.opacity = isActive ? 0.5 : 0.18;
    });
  }

  loadRoomFromEditor() {
    this.clearRoom();
    let sceneData = [];
    try {
      sceneData = JSON.parse(localStorage.getItem("rv_scene") || "[]");
    } catch (_) {}
    sceneData.forEach(obj => {
      if (obj.type === "wall") this.addBox(obj, 0x3a4055);
      if (obj.type === "shelf") this.addBox(obj, 0x1a3a6e);
      if (obj.type === "counter") this.addBox(obj, 0x5a2a1a);
    });
    this.syncSceneCameras(sceneData.filter(obj => obj.type === "camera"));
  }

  applyCalibToPoint(x, z, cameraId) {
    const c = this.getCalib(cameraId);
    const cx = this.sceneW / 2;
    const cz = this.sceneH / 2;
    const lx = (x - cx) * c.sx;
    const lz = (z - cz) * c.sz;
    const a = THREE.MathUtils.degToRad(c.rot);
    const rx = lx * Math.cos(a) - lz * Math.sin(a);
    const rz = lx * Math.sin(a) + lz * Math.cos(a);
    return { x: rx + cx + c.tx, z: rz + cz + c.tz };
  }

  applyDecayAndAccumulate(camerasPayload) {
    for (let r = 0; r < HEAT_ROWS; r++) {
      for (let c = 0; c < HEAT_COLS; c++) this.heatAccum[r][c] *= 0.97;
    }

    // Группируем confirmed points по cid через все камеры.
    // Используем ОПОРНУЮ КАМЕРУ из trail (если она есть) — это та же
    // стратегия что и для trails, гарантирует что один cid даёт вклад
    // в ОДНУ клетку heatmap'а из ОДНОЙ калибровки (не плавает между
    // разными world-coord'ами разных камер).
    //
    // Если trail для cid ещё нет (только промоутился) — fallback на
    // среднее, чтобы heatmap всё равно реагировал.
    const aggPerCid = new Map();  // cid → { perCam: Map<camKey, {x,z}> }
    (camerasPayload || []).forEach(cam => {
      const camKey = String(cam.id);
      (cam.points || []).forEach(p => {
        if (p.confirmed_id == null) return;  // tentative — пропускаем
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
        const cp = this.applyCalibToPoint(p.x, p.y, p.camera_id);
        if (!Number.isFinite(cp.x) || !Number.isFinite(cp.z)) return;
        let agg = aggPerCid.get(p.confirmed_id);
        if (!agg) {
          agg = { perCam: new Map() };
          aggPerCid.set(p.confirmed_id, agg);
        }
        agg.perCam.set(camKey, { x: cp.x, z: cp.z });
      });
    });

    aggPerCid.forEach((agg, cid) => {
      // Выбираем позицию опорной камеры если такая есть в trail.
      const trail = this.trackTrail.get(cid);
      let pos = null;
      if (trail && trail.refCam !== null && agg.perCam.has(trail.refCam)) {
        pos = agg.perCam.get(trail.refCam);
      } else {
        // Fallback: среднее по всем камерам (для cid у которого ещё
        // нет trail или его опорная сейчас не видит).
        let xSum = 0, zSum = 0, n = 0;
        agg.perCam.forEach(v => { xSum += v.x; zSum += v.z; n += 1; });
        if (n > 0) pos = { x: xSum / n, z: zSum / n };
      }
      if (!pos) return;

      const col = Math.floor((pos.x / Math.max(this.sceneW, 0.001)) * HEAT_COLS);
      const row = Math.floor((pos.z / Math.max(this.sceneH, 0.001)) * HEAT_ROWS);
      if (row >= 0 && row < HEAT_ROWS && col >= 0 && col < HEAT_COLS) {
        this.heatAccum[row][col] += 1.0;
      }
    });
  }

  drawHeatmapTextureFromAccum() {
    const W = this.hmTextureCanvas.width;
    const H = this.hmTextureCanvas.height;
    this.hmTextureCtx.clearRect(0, 0, W, H);
    this.hmTextureCtx.fillStyle = "rgba(20,24,34,0.2)";
    this.hmTextureCtx.fillRect(0, 0, W, H);

    let maxVal = 0;
    for (let r = 0; r < HEAT_ROWS; r++) {
      for (let c = 0; c < HEAT_COLS; c++) maxVal = Math.max(maxVal, this.heatAccum[r][c]);
    }
    if (maxVal <= 0) {
      this.hmTexture.needsUpdate = true;
      this.updateZonesList([]);
      return;
    }

    const normalized = Array.from({ length: HEAT_ROWS }, () => Array(HEAT_COLS).fill(0));
    const cw = W / HEAT_COLS;
    const ch = H / HEAT_ROWS;
    for (let r = 0; r < HEAT_ROWS; r++) {
      for (let c = 0; c < HEAT_COLS; c++) {
        const v = Math.min(1, this.heatAccum[r][c] / maxVal);
        normalized[r][c] = v;
        if (v < 0.01) continue;
        this.hmTextureCtx.fillStyle = `hsla(${(1 - v) * 240},100%,50%,${(v * 0.85).toFixed(3)})`;
        this.hmTextureCtx.fillRect(c * cw, r * ch, cw + 1, ch + 1);
      }
    }
    this.hmTexture.needsUpdate = true;
    this.updateZonesList(normalized);
  }

  disposeTrail(id, trail) {
    this.trajectoryGroup.remove(trail.line);
    this.trajectoryGroup.remove(trail.dot);
    trail.line.geometry.dispose();
    trail.line.material.dispose();
    trail.dot.geometry.dispose();
    trail.dot.material.dispose();
    this.trackTrail.delete(id);
  }

  updateTrackTrails(points) {
    if (points.length > 0) this.hadLivePoints = true;
    const activeCam = this.activeCalibCamera === null ? null : String(this.activeCalibCamera);
    const activeCamLabel = this.calibCameras.find(c => String(c.id) === activeCam)?.label || (activeCam !== null ? `Камера ${Number(activeCam) + 1}` : "—");

    // Считаем confirmed треки в фокусе (для статуса в футере)
    const confirmedPoints = points.filter(p => p.confirmed_id != null);
    const activeCount = activeCam === null
      ? confirmedPoints.length
      : confirmedPoints.filter(p => String(p.camera_id) === activeCam).length;
    this.el.footerCoords.textContent = this.hadLivePoints
      ? `Калибровка: ${activeCamLabel} · треков в фокусе: ${activeCount} · всего live: ${this.trackTrail.size}`
      : "Нет live-координат (проверь источник/камеру)";

    // Группируем точки по confirmed_id. У одного человека может быть
    // несколько детекций с разных камер. Стратегия стабилизации:
    //
    // 1) ОПОРНАЯ КАМЕРА: для каждого cid выбираем одну "опорную" камеру
    //    и используем ТОЛЬКО её world-coords. Это убирает скачки между
    //    разными калибровками. Опорная сохраняется в trail.refCam пока
    //    эта камера видит cid. Если перестала — переключаемся на любую
    //    другую видящую (предпочитая активную для focus filter).
    //
    // 2) EMA СГЛАЖИВАНИЕ: новая позиция в trail = alpha * raw + (1-alpha) * prev.
    //    Убивает мелкий jitter (calibration noise, bbox shifting) и
    //    смягчает резкий скачок при смене опорной камеры.
    const TRAIL_SMOOTHING_ALPHA = 0.35;

    // Сначала собираем: для каждого cid — { perCam: Map<camKey, {x,z}>, cams: Set }
    const perCidPoints = new Map();
    points.forEach(p => {
      if (p.confirmed_id == null) return;  // tentative — не рисуем
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
      const cp = this.applyCalibToPoint(p.x, p.y, p.camera_id);
      if (!Number.isFinite(cp.x) || !Number.isFinite(cp.z)) return;
      let info = perCidPoints.get(p.confirmed_id);
      if (!info) {
        info = { perCam: new Map(), cams: new Set() };
        perCidPoints.set(p.confirmed_id, info);
      }
      const camKey = String(p.camera_id);
      info.perCam.set(camKey, { x: cp.x, z: cp.z });
      info.cams.add(camKey);
    });

    const seen = new Set();
    perCidPoints.forEach((info, cid) => {
      seen.add(cid);
      let trail = this.trackTrail.get(cid);
      if (!trail) {
        trail = {
          points: [],
          stale: 0,
          refCam: null,           // опорная камера для этого cid
          smoothedX: null,        // EMA-сглаженная позиция
          smoothedZ: null,
          line: new THREE.Line(
            new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({ color: 0x00e5a0, transparent: true, opacity: 0.95 })
          ),
          dot: new THREE.Mesh(
            new THREE.SphereGeometry(0.12, 8, 8),
            new THREE.MeshBasicMaterial({ color: 0xffd166 })
          ),
        };
        this.trajectoryGroup.add(trail.line);
        this.trajectoryGroup.add(trail.dot);
        this.trackTrail.set(cid, trail);
      }
      trail.stale = 0;

      // Выбор опорной камеры:
      // - если предыдущая опорная всё ещё видит cid — оставляем её
      // - иначе предпочитаем активную (фокусную) камеру
      // - иначе любую видящую (детерминированно — наименьший camera_id)
      if (trail.refCam !== null && !info.perCam.has(trail.refCam)) {
        trail.refCam = null;  // потеряли опорную, надо переключиться
      }
      if (trail.refCam === null) {
        if (activeCam !== null && info.perCam.has(activeCam)) {
          trail.refCam = activeCam;
        } else {
          // Берём минимальный camera_id для стабильности
          let minCam = null;
          info.perCam.forEach((_, k) => {
            if (minCam === null || k < minCam) minCam = k;
          });
          trail.refCam = minCam;
        }
      }

      const refPos = info.perCam.get(trail.refCam);
      const rawX = refPos.x;
      const rawZ = refPos.z;

      // EMA сглаживание
      if (trail.smoothedX === null) {
        trail.smoothedX = rawX;
        trail.smoothedZ = rawZ;
      } else {
        trail.smoothedX = TRAIL_SMOOTHING_ALPHA * rawX + (1 - TRAIL_SMOOTHING_ALPHA) * trail.smoothedX;
        trail.smoothedZ = TRAIL_SMOOTHING_ALPHA * rawZ + (1 - TRAIL_SMOOTHING_ALPHA) * trail.smoothedZ;
      }

      const sx = trail.smoothedX;
      const sz = trail.smoothedZ;

      trail.cameraId = trail.refCam;
      trail.points.push(new THREE.Vector3(sx, 0.08, sz));
      if (trail.points.length > MAX_TRAIL_POINTS) trail.points.shift();
      trail.line.geometry.setFromPoints(trail.points);
      trail.dot.position.set(sx, 0.1, sz);
      // Восстанавливаем видимость dot — если он был скрыт после stale
      trail.dot.visible = true;

      // Focus filter: трек "в фокусе" если виден на активной камере.
      const inFocus = activeCam === null || info.cams.has(activeCam);
      trail.line.material.color.setHex(inFocus ? 0x00e5a0 : 0x4b5c88);
      trail.line.material.opacity = inFocus ? 0.95 : 0.2;
      trail.dot.material.color.setHex(inFocus ? 0xffd166 : 0x6e7896);
    });

    // Stale handling: trail который не видели в этом кадре стареет.
    // После 60 кадров удаляется полностью. До этого:
    // - DOT сразу скрываем (иначе он "висит" где человек был, давая
    //   ложное впечатление лишних людей в комнате)
    // - LINE остаётся как затухающий хвост (визуальный contextual cue)
    this.trackTrail.forEach((trail, cid) => {
      if (seen.has(cid)) return;
      trail.stale += 1;
      trail.dot.visible = false;  // прячем dot отсутствующего трека
      // Линия плавно угасает по мере роста stale (0 → 60)
      const fade = Math.max(0, 1 - trail.stale / 60);
      trail.line.material.opacity = 0.4 * fade;
      if (trail.stale >= 60) this.disposeTrail(cid, trail);
    });
  }

  drawHistory(data) {
    const W = this.el.historyCanvas.width;
    const H = this.el.historyCanvas.height;
    this.historyCtx.clearRect(0, 0, W, H);
    this.historyCtx.fillStyle = "#13161e";
    this.historyCtx.fillRect(0, 0, W, H);

    if (!data || data.length < 2) {
      this.historyCtx.fillStyle = "#6b7080";
      this.historyCtx.font = "10px monospace";
      this.historyCtx.textAlign = "center";
      this.historyCtx.fillText("Недостаточно данных", W / 2, H / 2);
      return;
    }

    const sorted = [...data].reverse();
    const maxV = Math.max(...sorted.map(d => d.visitor_count), 1);
    const pL = 28, pR = 6, pT = 8, pB = 18;
    const cW = W - pL - pR;
    const cH = H - pT - pB;

    this.historyCtx.strokeStyle = "#252936";
    this.historyCtx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pT + cH - (i / 4) * cH;
      this.historyCtx.beginPath();
      this.historyCtx.moveTo(pL, y);
      this.historyCtx.lineTo(pL + cW, y);
      this.historyCtx.stroke();
      this.historyCtx.fillStyle = "#6b7080";
      this.historyCtx.font = "8px monospace";
      this.historyCtx.textAlign = "right";
      this.historyCtx.fillText(Math.round((i / 4) * maxV), pL - 3, y + 3);
    }

    this.historyCtx.beginPath();
    this.historyCtx.strokeStyle = "#00e5a0";
    this.historyCtx.lineWidth = 1.5;
    sorted.forEach((d, i) => {
      const px = pL + (i / (sorted.length - 1)) * cW;
      const py = pT + cH - (d.visitor_count / maxV) * cH;
      i === 0 ? this.historyCtx.moveTo(px, py) : this.historyCtx.lineTo(px, py);
    });
    this.historyCtx.stroke();
    this.historyCtx.lineTo(pL + cW, pT + cH);
    this.historyCtx.lineTo(pL, pT + cH);
    this.historyCtx.closePath();
    this.historyCtx.fillStyle = "rgba(0,229,160,0.08)";
    this.historyCtx.fill();

    const fmt = t => {
      const d = new Date(t * 1000);
      return `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
    };
    this.historyCtx.fillStyle = "#6b7080";
    this.historyCtx.font = "8px monospace";
    this.historyCtx.textAlign = "left";
    this.historyCtx.fillText(fmt(sorted[0].ts), pL, H - 3);
    this.historyCtx.textAlign = "right";
    this.historyCtx.fillText(fmt(sorted[sorted.length - 1].ts), pL + cW, H - 3);
    this.el.historyHint.textContent = `${sorted.length} точек · макс ${maxV} чел.`;
  }

  async loadHistory() {
    try {
      const res = await fetch(`${API}/history?limit=60`);
      const data = await res.json();
      this.drawHistory(data.history || []);
    } catch (_) {
      this.el.historyHint.textContent = "Нет данных";
    }
  }

  updateZonesList(data) {
    if (!data?.length || !data[0]?.length) {
      this.el.zonesList.innerHTML = '<div class="hint">Данных пока нет</div>';
      return;
    }
    const rows = data.length;
    const cols = data[0].length;
    const hot = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r][c] || 0;
        if (v > 0.05) hot.push({ r, c, v });
      }
    }
    hot.sort((a, b) => b.v - a.v);
    const top = hot.slice(0, 6);
    if (!top.length) {
      this.el.zonesList.innerHTML = '<div class="hint">Данных пока нет</div>';
      return;
    }
    this.el.zonesList.innerHTML = "";
    top.forEach((z, i) => {
      const wx = ((z.c + 0.5) / cols) * this.sceneW;
      const wy = ((z.r + 0.5) / rows) * this.sceneH;
      const row = document.createElement("div");
      row.className = "track-item";
      row.textContent = `${i + 1}. (${wx.toFixed(1)}, ${wy.toFixed(1)}) м · ${(z.v * 100).toFixed(0)}%`;
      this.el.zonesList.appendChild(row);
    });
  }

  syncCalibrationCameraOptions(camerasPayload) {
    const cameras = camerasPayload || [];
    const newIds = cameras.map(c => String(c.id)).join(",");
    const oldIds = this.calibCameras.map(c => String(c.id)).join(",");
    const needRebuild = newIds !== oldIds;
    this.calibCameras = cameras;
    if (this.activeCalibCamera === null && cameras.length > 0) {
      this.activeCalibCamera = String(cameras[0].id);
    }
    if (needRebuild) this.renderCalibCameraList();
    else this.updateCalibActiveClass();
  }

  updateCalibActiveClass() {
    this.el.calibCameraList.querySelectorAll(".calib-cam-btn").forEach(btn => {
      btn.classList.toggle("calib-cam-btn--active", btn.dataset.camKey === String(this.activeCalibCamera));
    });
  }

  syncCalibInputs() {
    const c = this.getCalib(this.activeCalibCamera);
    ["tx", "tz", "rot", "sx", "sz"].forEach(k => {
      const input = document.getElementById(`calib-${k}`);
      if (input) input.value = String(c[k]);
    });
  }

  renderCalibCameraList() {
    this.el.calibCameraList.innerHTML = "";
    if (this.calibCameras.length === 0) {
      this.el.calibCameraList.innerHTML = '<div class="hint">Нет камер</div>';
      this.syncCalibInputs();
      return;
    }
    this.calibCameras.forEach(cam => {
      const camKey = String(cam.id);
      const btn = document.createElement("button");
      btn.className = "calib-cam-btn" + (camKey === String(this.activeCalibCamera) ? " calib-cam-btn--active" : "");
      btn.dataset.camKey = camKey;
      btn.textContent = cam.label || `Камера ${cam.id + 1}`;
      btn.title = `ID: ${cam.id}`;
      btn.addEventListener("click", () => {
        this.activeCalibCamera = camKey;
        this.updateCalibActiveClass();
        this.syncCalibInputs();
        this.highlightActiveSceneCamera();
        this.drawHeatmapTextureFromAccum();
      });
      this.el.calibCameraList.appendChild(btn);
    });
    this.syncCalibInputs();
  }

  bindCalibrationUI() {
    try {
      const raw = localStorage.getItem(CALIB_KEY);
      if (raw) this.calibByCamera = JSON.parse(raw) || {};
    } catch (_) {}

    ["tx", "tz", "rot", "sx", "sz"].forEach(k => {
      const input = document.getElementById(`calib-${k}`);
      if (!input) return;
      input.addEventListener("input", () => {
        const c = this.getCalib(this.activeCalibCamera);
        c[k] = Number(input.value);
        this.setCalib(this.activeCalibCamera, c);
      });
    });

    this.el.calibReset.addEventListener("click", () => {
      this.setCalib(this.activeCalibCamera, this.defaultCalib());
      this.syncCalibInputs();
    });
    this.renderCalibCameraList();
  }

  connectWS() {
    this.el.wsDot.className = "status-dot connecting";
    this.el.wsLabel.textContent = "Подключение...";
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      this.el.wsDot.className = "status-dot connected";
      this.el.wsLabel.textContent = "Подключено";
      this.wsRetry = 0;
    };

    ws.onmessage = e => {
      const data = JSON.parse(e.data);
      // ВАЖНО: updateTrackTrails ДО applyDecayAndAccumulate.
      // Trails устанавливают опорную камеру (refCam) для каждого cid,
      // и heatmap использует её, чтобы аккумулировать в ту же клетку
      // (стабильно, без скачков между калибровками).
      if (Array.isArray(data.points)) this.updateTrackTrails(data.points);
      if (Array.isArray(data.cameras)) {
        this.syncCalibrationCameraOptions(data.cameras);
        this.applyDecayAndAccumulate(data.cameras);
        this.drawHeatmapTextureFromAccum();
      } else if (Array.isArray(data.heatmap)) {
        const rows = data.heatmap.length;
        const cols = rows ? data.heatmap[0].length : 0;
        if (rows && cols) {
          this.heatAccum = Array.from({ length: rows }, (_, r) =>
            Array.from({ length: cols }, (_, c) => data.heatmap[r][c] || 0)
          );
          this.drawHeatmapTextureFromAccum();
        }
      }
      if (data.stats) {
        this.el.total.textContent = data.stats.visitor_count ?? "—";
        this.el.active.textContent = data.stats.active_tracks ?? "—";
      }
    };

    ws.onclose = () => {
      this.el.wsDot.className = "status-dot error";
      this.el.wsLabel.textContent = "Нет соединения";
      setTimeout(() => this.connectWS(), Math.min(5000, 1000 * ++this.wsRetry));
    };
  }

  bindActionButtons() {
    this.el.btnHmReset.addEventListener("click", () => {
      fetch(`${API}/reset`, { method: "POST" }).catch(console.error);
      this.initHeatAccumulator();
      this.drawHeatmapTextureFromAccum();
      this.trackTrail.forEach((trail, id) => this.disposeTrail(id, trail));
      this.trackTrail.clear();
    });

    this.el.btnResetStats.addEventListener("click", () => {
      fetch(`${API}/reset`, { method: "POST" }).catch(console.error);
      this.el.total.textContent = "0";
      this.el.active.textContent = "0";
    });

    this.el.btnHmExport.addEventListener("click", async () => {
      const res = await fetch(`${API}/history?limit=1000`);
      const data = await res.json();
      const rows = (data.history || []).map(h => `${new Date(h.ts * 1000).toISOString()},${h.visitor_count},${h.active_tracks}`);
      const csv = ["timestamp,visitor_count,active_tracks", ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "analytics.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  async loadSceneConfig() {
    try {
      const res = await fetch(`${API}/scene`);
      const scene = await res.json();
      this.sceneW = Number(scene.width) || 20;
      this.sceneH = Number(scene.height) || 20;
    } catch (_) {
      this.sceneW = 20;
      this.sceneH = 20;
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene3d, this.cam3d);
  }

  async init() {
    await this.loadSceneConfig();
    this.bindCalibrationUI();
    this.bindActionButtons();
    this.initHeatAccumulator();
    this.createFloor();
    this.loadRoomFromEditor();
    this.drawHeatmapTextureFromAccum();
    this.loadHistory();
    setInterval(() => this.loadHistory(), 30000);
    this.connectWS();
    this.animate();
  }
}

new AnalyticsPage().init();
