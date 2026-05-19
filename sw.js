/* sw.js — Service Worker для RetailVision
   Кэширует всю статику при первом посещении.
   При недоступности сервера отдаёт из кэша.
*/

const CACHE = "rv-static-v3";

// Всё что нужно закэшировать при установке
const PRECACHE = [
  "/editor.html",
  "/video.html",
  "/analytics.html",
  "/style.css",
  "/editor.js",
  "/video.js",
  "/analytics.js",
  "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;800&display=swap",
  "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js",
];

// ── Установка: кэшируем статику ──────────────────────────────
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => {
      // Грузим по одному — чтобы ошибка одного ресурса не блокировала всё
      return Promise.allSettled(
        PRECACHE.map(url => cache.add(url).catch(() => {}))
      );
    }).then(() => self.skipWaiting())
  );
});

// ── Активация: удаляем старые кэши ───────────────────────────
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// ── Fetch: стратегия по типу запроса ─────────────────────────
self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);

  // API-запросы (REST + WS) — только сеть, никогда не кэшируем.
  // При ошибке возвращаем JSON-заглушку с признаком офлайна.
  const isAPI = url.pathname.startsWith("/ws") ||
                url.pathname.startsWith("/cameras") ||
                url.pathname.startsWith("/points") ||
                url.pathname.startsWith("/heatmap") ||
                url.pathname.startsWith("/stats") ||
                url.pathname.startsWith("/history") ||
                url.pathname.startsWith("/reset") ||
                url.pathname.startsWith("/scene") ||
                url.pathname.startsWith("/set_source") ||
                url.pathname.startsWith("/homography") ||
                event.request.method !== "GET";

  if (isAPI) {
    // Для API просто пропускаем в сеть — ошибку обработает сам JS
    event.respondWith(
      fetch(event.request).catch(() =>
        new Response(
          JSON.stringify({ offline: true, error: "Сервер недоступен" }),
          { status: 503, headers: { "Content-Type": "application/json" } }
        )
      )
    );
    return;
  }

  // Для локальной статики используем network-first, чтобы обновления JS/CSS
  // применялись при обычной перезагрузке без Ctrl+F5.
  const isLocalStatic =
    url.origin === self.location.origin &&
    (url.pathname.endsWith(".js") || url.pathname.endsWith(".css") || url.pathname.endsWith(".html"));

  if (isLocalStatic) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          if (response && response.status === 200) {
            const clone = response.clone();
            caches.open(CACHE).then(cache => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Остальная статика — стратегия «кэш первый, сеть как обновление»
  event.respondWith(
    caches.match(event.request).then(cached => {
      const networkFetch = fetch(event.request).then(response => {
        // Обновляем кэш свежей версией
        if (response && response.status === 200 && response.type !== "opaque") {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => cached); // нет сети — вернём что было в кэше

      // Есть кэш → сразу отдаём, фоном обновляем
      return cached || networkFetch;
    })
  );
});