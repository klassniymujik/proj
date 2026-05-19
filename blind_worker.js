/* blind_worker.js — Web Worker для вычисления слепых зон.
   Работает без Three.js: собственный ray-AABB raycast.

   Принимает postMessage({ cam, shadowBoxes, wallBoxes })
   Возвращает postMessage({ positions, indices })
*/

// ── Ray-AABB пересечение ────────────────────────────────────
// box: { minX, minY, minZ, maxX, maxY, maxZ }
// ray: { ox,oy,oz, dx,dy,dz }  (направление нормализовано)
// Возвращает дистанцию до входа или Infinity если нет пересечения
function rayAABB(ox, oy, oz, dx, dy, dz, box) {
  const invDx = dx === 0 ? Infinity : 1 / dx;
  const invDy = dy === 0 ? Infinity : 1 / dy;
  const invDz = dz === 0 ? Infinity : 1 / dz;

  const tx1 = (box.minX - ox) * invDx;
  const tx2 = (box.maxX - ox) * invDx;
  const ty1 = (box.minY - oy) * invDy;
  const ty2 = (box.maxY - oy) * invDy;
  const tz1 = (box.minZ - oz) * invDz;
  const tz2 = (box.maxZ - oz) * invDz;

  const tmin = Math.max(Math.min(tx1, tx2), Math.min(ty1, ty2), Math.min(tz1, tz2));
  const tmax = Math.min(Math.max(tx1, tx2), Math.max(ty1, ty2), Math.max(tz1, tz2));

  if (tmax < 0 || tmin > tmax) return Infinity;
  return tmin < 0 ? tmax : tmin;
}

// Ближайшее пересечение луча с массивом боксов
// Возвращает { dist, idx } или null
function nearestHit(ox, oy, oz, dx, dy, dz, boxes) {
  let best = Infinity, bestIdx = -1;
  for (let i = 0; i < boxes.length; i++) {
    const t = rayAABB(ox, oy, oz, dx, dy, dz, boxes[i]);
    if (t < best) { best = t; bestIdx = i; }
  }
  return best < Infinity ? { dist: best, idx: bestIdx } : null;
}

// ── FOV-проверка ────────────────────────────────────────────
function isInCameraFOV(cam, px, pz) {
  const tx = px - cam.x, ty = -cam.height, tz = pz - cam.z;
  const len = Math.sqrt(tx*tx + ty*ty + tz*tz);
  if (len === 0) return false;
  const nx = tx/len, ny = ty/len, nz = tz/len;

  // Центральное направление взгляда (та же формула что в editor.js)
  const yR = cam.yaw   * Math.PI / 180;
  const pR = cam.pitch * Math.PI / 180;
  const cosP = Math.cos(pR), sinP = Math.sin(pR);
  const cosY = Math.cos(yR), sinY = Math.sin(yR);
  const cdx = -sinY * cosP, cdy = -sinP, cdz = -cosY * cosP;
  const cl  = Math.sqrt(cdx*cdx + cdy*cdy + cdz*cdz);

  const dot = (nx*cdx + ny*cdy + nz*cdz) / cl;
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;

  const fovH    = cam.fov || 90;
  const fovV    = 2 * Math.atan(Math.tan(fovH / 2 * Math.PI / 180) * (cam.ih / cam.iw)) * 180 / Math.PI;
  const halfDiag = Math.sqrt((fovH/2)*(fovH/2) + (fovV/2)*(fovV/2));

  return angleDeg <= halfDiag;
}

// ── Основной расчёт ─────────────────────────────────────────
self.onmessage = function(e) {
  const { cam, shadowBoxes, wallBoxes } = e.data;

  const allBoxes = [...shadowBoxes, ...wallBoxes];
  const nShadow  = shadowBoxes.length;

  const STEP  = 0.25;
  const RANGE = 25;
  const HALF  = STEP / 2;

  const positions = [];
  const indices   = [];
  let   vi        = 0;

  const cx = cam.x, cy = cam.height, cz = cam.z;

  for (let x = -RANGE; x <= RANGE; x += STEP) {
    for (let z = -RANGE; z <= RANGE; z += STEP) {

      if (!isInCameraFOV(cam, x, z)) continue;

      const tx = x - cx, ty = 0.1 - cy, tz = z - cz;
      const dist = Math.sqrt(tx*tx + ty*ty + tz*tz);
      const dx = tx/dist, dy = ty/dist, dz = tz/dist;

      const hit = nearestHit(cx, cy, cz, dx, dy, dz, allBoxes);
      if (!hit) continue;
      if (hit.dist >= dist - 0.4) continue;

      // Первое препятствие — стена → не рисуем
      if (hit.idx >= nShadow) continue;

      // Обратный луч — стена между точкой и камерой?
      const bdx = -dx, bdy = -dy, bdz = -dz;
      const wh = nearestHit(x, 0.1, z, bdx, bdy, bdz, wallBoxes);
      if (wh && wh.dist < dist - 0.4) continue;

      // Квадрат в меш
      const x0 = x - HALF, x1 = x + HALF;
      const z0 = z - HALF, z1 = z + HALF;
      const y  = 0.05;

      positions.push(x0,y,z0, x1,y,z0, x1,y,z1, x0,y,z1);
      indices.push(vi, vi+1, vi+2, vi, vi+2, vi+3);
      vi += 4;
    }
  }

  // Передаём буферы через transferable — нулевое копирование
  const posArr = new Float32Array(positions);
  const idxArr = new Uint32Array(indices);
  self.postMessage({ posArr, idxArr }, [posArr.buffer, idxArr.buffer]);
};