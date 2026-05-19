/* video.js — страница просмотра видеопотока с аннотацией */

const API = "http://127.0.0.1:8000";
const overlayLbl = document.getElementById("video-overlay-label");
const videoGrid = document.getElementById("video-grid");
let paused = false;

// ================================================================
// Список камер из БД
// ================================================================
let cameras = [];       // [{id, label, address}]
let activeCamId = null; // id активной камеры в списке

function renderCameraList() {
  const list = document.getElementById("camera-list");
  if (!cameras.length) {
    list.innerHTML = '<div class="hint">Камеры не найдены</div>';
    return;
  }
  list.innerHTML = "";
  cameras.forEach(cam => {
    const el = document.createElement("div");
    el.className = "camera-item" + (cam.id === activeCamId ? " active" : "");
    el.title = cam.address || "(адрес не задан)";
    el.innerHTML = `
      <span class="camera-item-dot"></span>
      <span class="camera-item-label">${cam.label}</span>
      <span class="camera-item-addr">${cam.address || "—"}</span>
    `;
    el.addEventListener("click", () => {
      activeCamId = cam.id;
      renderCameraList();
      document.getElementById("footer-coords").textContent = `Фокус: ${cam.label}`;
    });
    list.appendChild(el);
  });
}

async function loadCamerasFromDB() {
  const btn = document.getElementById("btn-load-cameras");
  btn.textContent = "Загрузка...";
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/cameras/list`);
    const data = await res.json();
    cameras = data.cameras || [];
    renderCameraList();
    if (cameras.length) {
      document.getElementById("footer-coords").textContent =
        `Загружено ${cameras.length} камер`;
    }
  } catch (e) {
    document.getElementById("camera-list").innerHTML =
      '<div class="hint" style="color:var(--warn)">Ошибка загрузки</div>';
  } finally {
    btn.textContent = "↓ Загрузить из БД";
    btn.disabled = false;
  }
}

function applySource(src) {
  paused = false;
  overlayLbl.textContent = "Переключение в single-source режим...";
  overlayLbl.style.display = "block";
  fetch(`${API}/set_source`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: src }),
  }).catch(console.error);
}

document.getElementById("btn-load-cameras").addEventListener("click", loadCamerasFromDB);

let _camCards = [];

function renderMultiCamera(camerasPayload) {
  if (paused) return;
  if (!Array.isArray(camerasPayload) || camerasPayload.length === 0) {
    overlayLbl.textContent = "Нет активных потоков";
    overlayLbl.style.display = "block";
    videoGrid.innerHTML = "";
    _camCards = [];
    return;
  }

  overlayLbl.style.display = "none";

  if (_camCards.length !== camerasPayload.length) {
    videoGrid.innerHTML = "";
    _camCards = [];
    calibCanvasEls = [];
    camerasPayload.forEach(cam => {
      const card = document.createElement("div");
      card.style.border = "1px solid var(--border)";
      card.style.borderRadius = "8px";
      card.style.background = "var(--bg2)";
      card.style.overflow = "hidden";
      card.style.minHeight = "220px";
      card.className = "cam-card";
      card.style.display = "flex";
      card.style.flexDirection = "column";

      const head = document.createElement("div");
      head.style.display = "flex";
      head.style.justifyContent = "space-between";
      head.style.alignItems = "center";
      head.style.padding = "6px 10px";
      head.style.fontFamily = "var(--font)";
      head.style.fontSize = "11px";
      head.style.borderBottom = "1px solid var(--border)";
      card.appendChild(head);

      const img = document.createElement("img");
      img.style.width = "100%";
      img.style.height = "100%";
      img.style.objectFit = "contain";
      img.style.background = "#05070c";
      img.alt = cam.label || `Камера ${cam.id}`;
      card.appendChild(img);

      videoGrid.appendChild(card);
      _camCards.push({ card, head, img });
    });
  }

  camerasPayload.forEach((cam, i) => {
    const { head, img } = _camCards[i];
    head.innerHTML = `<span>${cam.label || `Камера ${cam.id}`}</span><span>${cam.stats?.active_tracks ?? 0} в кадре</span>`;
    if (cam.frame && cam.frame.length > 100)
      img.src = "data:image/jpeg;base64," + cam.frame;
  });

  if (calibMode) {
    requestAnimationFrame(() => { attachCalibClickHandlers(); });
  }
}

// ================================================================
// WebSocket
// ================================================================
const wsDot   = document.getElementById("ws-dot");
const wsLabel = document.getElementById("ws-label");
let wsRetry   = 0;
let lastFrameTs = Date.now();

function connectWS() {
  wsDot.className = "status-dot connecting";
  wsLabel.textContent = "Подключение...";

  const ws = new WebSocket("ws://127.0.0.1:8000/ws");

  ws.onopen = () => {
    wsDot.className = "status-dot connected";
    wsLabel.textContent = "Подключено";
    wsRetry = 0;
  };

  ws.onmessage = e => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    if (Array.isArray(data.cameras)) {
      const now = Date.now();
      const dt  = now - lastFrameTs;
      lastFrameTs = now;
      renderMultiCamera(data.cameras);

      const fps = Math.round(1000 / Math.max(1, dt));
      document.getElementById("v-fps").textContent     = fps + " кадр/с";
      document.getElementById("v-latency").textContent = dt + " мс";
    }

    const stats = data.stats || {};
    document.getElementById("v-active").textContent = stats.active_tracks ?? "—";
    document.getElementById("v-total").textContent  = stats.visitor_count  ?? "—";

    const tracks = data.points || [];
    const list   = document.getElementById("track-list");
    list.innerHTML = "";
    tracks.slice(0, 20).forEach(p => {
      const el = document.createElement("div");
      el.className = "track-item";
      const camText = typeof p.camera_id === "number" ? `C${p.camera_id + 1}` : "C?";
      const localId = p.track_id ?? p.id;
      // confirmed_id явное поле от backend'а; для backward compat также
      // смотрим global_id (в новой архитектуре они совпадают).
      const cid = p.confirmed_id ?? p.global_id ?? null;
      if (cid !== null) {
        el.textContent = `[${camText}] G#${cid} (L#${localId}) \u2192 (${p.x.toFixed(1)}, \u202f${p.y.toFixed(1)}) м`;
      } else {
        // Tentative — ещё не подтверждён. Показываем многоточием.
        el.textContent = `[${camText}] L#${localId} \u2026 \u2192 (${p.x.toFixed(1)}, \u202f${p.y.toFixed(1)}) м`;
        el.style.opacity = "0.55";
      }
      list.appendChild(el);
    });
    if (tracks.length > 20) {
      const more = document.createElement("div");
      more.className = "hint";
      more.textContent = `...ещё ${tracks.length - 20}`;
      list.appendChild(more);
    }
  };

  ws.onclose = () => {
    wsDot.className = "status-dot error";
    wsLabel.textContent = "Нет соединения";
    overlayLbl.textContent = "Соединение потеряно. Переподключение...";
    overlayLbl.style.display = "block";
    setTimeout(connectWS, Math.min(5000, 1000 * ++wsRetry));
  };
}

connectWS();

// ================================================================
// Кнопки управления
// ================================================================
document.getElementById("btn-play").addEventListener("click", () => {
  paused = false;
  overlayLbl.style.display = "none";
});

document.getElementById("btn-pause").addEventListener("click", () => {
  paused = true;
  overlayLbl.textContent = "\u23f8 Пауза";
  overlayLbl.style.display = "block";
});

// ================================================================
// Homography calibration — пошаговый wizard
// ================================================================
let calibMode = false;
let calibStep = "idle"; // "idle" | "pickA" | "pickB" | "done"
let calibPairs = [];    // [{a: [x,y], b: [x,y]}]
let calibCanvasEls = [];
let calibCurrentPair = null; // [x,y] выбранная точка на камере A, ждёт B
let calibCamA = 0;
let calibCamB = 1;

const CALIB_COLORS = ["#ff6b35","#00ccff","#44ff44","#ff44ff","#ffff44","#ff8888","#88ffff","#ff88ff"];

function updateCalibPairSelect() {
  const sel = document.getElementById("calib-pair-select");
  if (!sel) return;
  sel.innerHTML = "";
  for (let i = 0; i < cameras.length; i++) {
    for (let j = i + 1; j < cameras.length; j++) {
      const opt = document.createElement("option");
      opt.value = i + "," + j;
      opt.textContent = "Кам " + (i+1) + " ↔ Кам " + (j+1);
      if (i === calibCamA && j === calibCamB) opt.selected = true;
      sel.appendChild(opt);
    }
  }
  sel.onchange = () => {
    const [a, b] = sel.value.split(",").map(Number);
    calibCamA = a;
    calibCamB = b;
    calibPairs = [];
    calibCurrentPair = null;
    calibStep = "pickA";
    calibHint();
    drawCalib();
  };
}

function calibHint() {
  const el = document.getElementById("calib-hint");
  if (!el) return;
  const n = calibPairs.length;
  const need = Math.max(0, 4 - n);
  const camALabel = "Кам " + (calibCamA + 1);
  const camBLabel = "Кам " + (calibCamB + 1);
  if (calibStep === "pickA") {
    el.innerHTML = `<b>Шаг ${n+1}.</b> Кликни на заметную точку на <span style="color:#ff6b35">${camALabel}</span><br>Нужно ещё ${need} пар(ы). Всего: ${n}/4+`;
  } else if (calibStep === "pickB") {
    el.innerHTML = `<b>Шаг ${n+1}.</b> Теперь кликни на <b>ту же точку</b> на <span style="color:#00ccff">${camBLabel}</span><br><span style="color:var(--text2)">Точка ${n+1} на ${camALabel} уже выбрана</span>`;
  } else if (calibStep === "done") {
    el.innerHTML = `<span style="color:#44ff44">Готово!</span> ${n} пар точек. Можно сохранить.`;
  } else {
    el.innerHTML = "";
  }
}

function toggleCalibMode() {
  calibMode = !calibMode;
  const btn = document.getElementById("btn-calib");
  const panel = document.getElementById("calib-panel");
  if (calibMode) {
    btn.textContent = "✕ Отмена";
    btn.style.background = "var(--warn)";
    calibStep = "pickA";
    calibPairs = [];
    calibCurrentPair = null;
    panel.style.display = "block";
    updateCalibPairSelect();
    calibHint();
    attachCalibClickHandlers();
    fetch(`${API}/calib/hide_bboxes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hide: true }),
    }).catch(console.error);
  } else {
    btn.textContent = "Калибровка";
    btn.style.background = "";
    calibStep = "idle";
    calibCurrentPair = null;
    panel.style.display = "none";
    detachCalibClickHandlers();
    document.querySelectorAll(".calib-overlay").forEach(c => c.remove());
    fetch(`${API}/calib/hide_bboxes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hide: false }),
    }).catch(console.error);
  }
}

function attachCalibClickHandlers() {
  detachCalibClickHandlers();
  const cards = videoGrid.querySelectorAll(".cam-card");
  if (cards.length < 2) {
    console.log("[calib] Need 2 camera cards, found:", cards.length);
    return;
  }
  cards.forEach((card, idx) => {
    const img = card.querySelector("img");
    if (!img) return;
    const handler = e => {
      e.preventDefault();
      e.stopPropagation();
      if (!calibMode) return;

      const rect = img.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const px = Math.round((e.clientX - rect.left) * scaleX);
      const py = Math.round((e.clientY - rect.top) * scaleY);

      console.log("[calib] click on cam", idx, "px", px, py, "step", calibStep);

      if (calibStep === "pickA" && idx === calibCamA) {
        calibCurrentPair = [px, py];
        calibStep = "pickB";
        calibHint();
        drawCalib();
      } else if (calibStep === "pickB" && idx === calibCamB) {
        calibPairs.push({ a: calibCurrentPair, b: [px, py] });
        calibCurrentPair = null;
        calibStep = calibPairs.length >= 4 ? "done" : "pickA";
        calibHint();
        drawCalib();
      }
    };
    img.style.cursor = "crosshair";
    img.addEventListener("click", handler);
    img._calibHandler = handler;
    calibCanvasEls.push(img);
  });
  drawCalib();
  console.log("[calib] handlers attached, cards:", cards.length);
}

function detachCalibClickHandlers() {
  calibCanvasEls.forEach(img => {
    if (img._calibHandler) {
      img.removeEventListener("click", img._calibHandler);
      img._calibHandler = null;
    }
    img.style.cursor = "";
  });
  calibCanvasEls = [];
}

function drawCalib() {
  const cards = videoGrid.querySelectorAll(".cam-card");
  cards.forEach((card, idx) => {
    let cvs = card.querySelector("canvas.calib-overlay");
    const img = card.querySelector("img");
    if (!img) return;

    if (!calibMode) {
      if (cvs) cvs.remove();
      return;
    }

    if (!cvs) {
      cvs = document.createElement("canvas");
      cvs.className = "calib-overlay";
      cvs.style.position = "absolute";
      cvs.style.top = card.querySelector("div")?.offsetHeight + "px" || "30px";
      cvs.style.left = "0";
      cvs.style.width = "100%";
      cvs.style.height = "calc(100% - 30px)";
      cvs.style.pointerEvents = "none";
      cvs.style.zIndex = "5";
      card.style.position = "relative";
      card.appendChild(cvs);
    }

    cvs.width = img.naturalWidth || 1280;
    cvs.height = img.naturalHeight || 720;
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, cvs.width, cvs.height);

    // Draw camera label
    ctx.fillStyle = idx === calibCamA ? "rgba(255,107,53,0.8)" : "rgba(0,204,255,0.8)";
    ctx.fillRect(8, 8, 90, 26);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("Кам " + (idx + 1), 14, 26);

    if ((calibStep === "pickA" && idx === calibCamA) || (calibStep === "pickB" && idx === calibCamB)) {
      ctx.strokeStyle = idx === calibCamA ? "#ff6b35" : "#00ccff";
      ctx.lineWidth = 3;
      ctx.setLineDash([10, 5]);
      ctx.strokeRect(2, 2, cvs.width - 4, cvs.height - 4);
      ctx.setLineDash([]);
    }

    // Draw paired points
    calibPairs.forEach((pair, i) => {
      const color = CALIB_COLORS[i % CALIB_COLORS.length];
      const pt = idx === calibCamA ? pair.a : pair.b;
      drawPoint(ctx, pt, i + 1, color);
    });

    if (calibStep === "pickB" && idx === calibCamA && calibCurrentPair) {
      drawPoint(ctx, calibCurrentPair, calibPairs.length + 1, "#ff6b35");
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.font = "bold 18px sans-serif";
      ctx.fillText("→ Кам " + (calibCamB + 1) + "?", calibCurrentPair[0] + 16, calibCurrentPair[1] + 6);
    }
  });
}

function drawPoint(ctx, pt, num, color) {
  ctx.beginPath();
  ctx.arc(pt[0], pt[1], 12, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = "#fff";
  ctx.font = "bold 13px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(num, pt[0], pt[1]);
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

async function submitCalibPoints() {
  if (calibPairs.length < 4) {
    alert("Нужно минимум 4 пары точек. Сейчас: " + calibPairs.length);
    return;
  }
  const pointsA = calibPairs.map(p => p.a);
  const pointsB = calibPairs.map(p => p.b);
  console.log("[calib] submitting pointsA:", pointsA, "pointsB:", pointsB);
  try {
    const res = await fetch(`${API}/homography/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cam_a: calibCamA,
        cam_b: calibCamB,
        points_a: pointsA,
        points_b: pointsB,
      }),
    });
    const data = await res.json();
    console.log("[calib] response:", data);
    if (data.status === "ok") {
      alert("Калибровка выполнена! Ошибка проекции: " + (data.reprojection_error?.toFixed(1) || "?") + " px\nЧем меньше — тем точнее.");
      toggleCalibMode();
    } else {
      alert("Ошибка: " + (data.message || "unknown"));
    }
  } catch (e) {
    console.error("[calib] fetch error:", e);
    alert("Сетевая ошибка: " + e.message);
  }
}

function undoCalibPoint() {
  if (calibStep === "pickB" && calibCurrentPair) {
    calibCurrentPair = null;
    calibStep = "pickA";
  } else if (calibPairs.length > 0) {
    calibPairs.pop();
    calibStep = "pickA";
  }
  if (calibPairs.length < 4 && calibStep === "done") {
    calibStep = "pickA";
  }
  calibHint();
  drawCalib();
}

function clearCalibPoints() {
  calibPairs = [];
  calibCurrentPair = null;
  calibStep = "pickA";
  calibHint();
  drawCalib();
}

document.getElementById("btn-source-apply").addEventListener("click", () => {
  const src = document.getElementById("video-source").value.trim();
  if (!src) return;
  activeCamId = null;
  renderCameraList();
  applySource(src);
});

document.getElementById("btn-calib").addEventListener("click", toggleCalibMode);
document.getElementById("btn-calib-clear").addEventListener("click", clearCalibPoints);
document.getElementById("btn-calib-submit").addEventListener("click", submitCalibPoints);
document.getElementById("btn-calib-undo").addEventListener("click", undoCalibPoint);