/* offline.js — подключается последним на каждой странице.
   1. Регистрирует Service Worker (sw.js).
   2. Следит за online/offline и показывает баннер.
   3. При офлайне блокирует кнопки с классом btn-primary / btn-secondary
      (кроме тех, что помечены data-offline-ok="true").
*/

// ── 1. Регистрация Service Worker ────────────────────────────
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}

// ── 2. Баннер и состояние ────────────────────────────────────
const _banner = document.getElementById("offline-banner");

// Кнопки которые взаимодействуют с сервером — блокируем при офлайне.
// Кнопки с data-offline-ok="true" оставляем (например, пауза, сброс вида).
function _setServerButtons(disabled) {
  document.querySelectorAll(
    ".btn-primary, .btn-secondary, .add-btn, #btn-source-apply, #btn-load-cameras"
  ).forEach(btn => {
    if (btn.dataset.offlineOk === "true") return;
    btn.disabled = disabled;
    btn.style.opacity = disabled ? "0.35" : "";
    btn.style.cursor  = disabled ? "not-allowed" : "";
  });
}

function _onOnline() {
  if (_banner) _banner.style.display = "none";
  _setServerButtons(false);
}

function _onOffline() {
  if (_banner) _banner.style.display = "block";
  _setServerButtons(true);
}

// ── 3. Определяем доступность сервера (не просто интернета) ──
// navigator.onLine ненадёжен в LAN-сценарии: интернет есть,
// а наш FastAPI недоступен. Поэтому пингуем /stats раз в 3 с.

let _serverAlive = true;
let _checking    = false;

async function _checkServer() {
  if (_checking) return;
  _checking = true;
  try {
    const res = await fetch("/stats", {
      method: "GET",
      cache:  "no-store",
      signal: AbortSignal.timeout(2000),
    });
    if (!_serverAlive && res.ok) {
      _serverAlive = true;
      _onOnline();
    }
  } catch {
    if (_serverAlive) {
      _serverAlive = false;
      _onOffline();
    }
  } finally {
    _checking = false;
  }
}

setInterval(_checkServer, 3000);
_checkServer(); // сразу при загрузке

// Стандартные события на случай полного отключения сети
window.addEventListener("online",  _checkServer);
window.addEventListener("offline", _onOffline);

// ── 4. Кнопки навигации между страницами — всегда работают ──
// (ссылки <a href=...> не затрагиваем вообще, они не <button>)