/* ================================================================
   editor.js — 3D-редактор RetailVision v6 (полный рефактор)
   Стены — только объекты. Единая система объектов.
   ================================================================ */

const API = "http://127.0.0.1:8000";

// ================================================================
// RENDERER
// ================================================================
const canvas  = document.getElementById("scene");
const layout  = document.getElementById("editor-layout");
const PANEL_W = 248;

function getViewSize() {
  const lw = layout.clientWidth  || window.innerWidth;
  const lh = layout.clientHeight || (window.innerHeight - 76);
  return { w: Math.max(lw - PANEL_W * 2, 100), h: Math.max(lh, 100) };
}

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const { w: _iW, h: _iH } = getViewSize();
renderer.setSize(_iW, _iH);

const cam3d = new THREE.PerspectiveCamera(50, _iW / _iH, 0.1, 300);
cam3d.position.set(12, 20, 18);
cam3d.lookAt(0, 0, 0);

function onResize() {
  const { w, h } = getViewSize();
  renderer.setSize(w, h);
  cam3d.aspect = w / h;
  cam3d.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);
setTimeout(onResize, 200);

// ================================================================
// ORBIT
// ================================================================
let rmbMoved = false;
const orbit = (() => {
  let rmb=false, mmb=false, lx=0, ly=0;
  let theta=Math.PI/4, phi=Math.PI/3.2, r=28;
  const tgt = new THREE.Vector3();
  const upd = () => {
    cam3d.position.set(
      tgt.x + r*Math.sin(phi)*Math.sin(theta),
      tgt.y + r*Math.cos(phi),
      tgt.z + r*Math.sin(phi)*Math.cos(theta)
    );
    cam3d.lookAt(tgt);
  };
  upd();
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("mousedown", e => {
    if (e.button===2) { rmb=true; rmbMoved=false; }
    if (e.button===1) { mmb=true; e.preventDefault(); }
    lx=e.clientX; ly=e.clientY;
  });
  window.addEventListener("mouseup", e => {
    if (e.button===2) rmb=false;
    if (e.button===1) mmb=false;
  });
  window.addEventListener("mousemove", e => {
    const dx=e.clientX-lx, dy=e.clientY-ly;
    lx=e.clientX; ly=e.clientY;
    if (rmb && (Math.abs(dx)>1||Math.abs(dy)>1)) {
      rmbMoved=true;
      theta -= dx*0.005;
      phi = Math.max(0.05, Math.min(Math.PI/2.05, phi+dy*0.005));
      upd();
    }
    if (mmb) {
      const sp=r*0.001;
      const right=new THREE.Vector3().crossVectors(cam3d.getWorldDirection(new THREE.Vector3()),new THREE.Vector3(0,1,0)).normalize();
      tgt.addScaledVector(right,-dx*sp);
      tgt.y+=dy*sp;
      upd();
    }
  });
  canvas.addEventListener("wheel", e=>{ r=Math.max(2,Math.min(100,r+e.deltaY*0.06)); upd(); },{passive:true});
  return { reset(){ theta=Math.PI/4; phi=Math.PI/3.2; r=28; tgt.set(0,0,0); upd(); } };
})();

// ================================================================
// СЦЕНА
// ================================================================
const scene3d = new THREE.Scene();
scene3d.background = new THREE.Color(0x0d0f14);
scene3d.fog = new THREE.FogExp2(0x0d0f14, 0.01);
scene3d.add(new THREE.AmbientLight(0xffffff, 0.5));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(15,25,10); sun.castShadow=true;
sun.shadow.mapSize.setScalar(2048);
sun.shadow.camera.left=sun.shadow.camera.bottom=-40;
sun.shadow.camera.right=sun.shadow.camera.top=40;
sun.shadow.camera.far=120;
scene3d.add(sun);
scene3d.add(new THREE.GridHelper(80,80,0x252936,0x1a1d26));

const hitPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(200,200),
  new THREE.MeshBasicMaterial({visible:false,side:THREE.DoubleSide})
);
hitPlane.rotation.x = -Math.PI/2;
scene3d.add(hitPlane);

// ================================================================
// КОНСТАНТЫ
// ================================================================
const WALL_T = 0.15;   // толщина стены
const SNAP_R = 0.7;    // радиус snap

// ================================================================
// МАТЕРИАЛЫ
// ================================================================
const MAT = {
  wall:      () => new THREE.MeshStandardMaterial({color:0x3a4055,roughness:0.8}),
  wallSel:   () => new THREE.MeshStandardMaterial({color:0xff6644,roughness:0.8}),
  shelf:     () => new THREE.MeshStandardMaterial({color:0x1a3a6e,roughness:0.7,metalness:0.2}),
  counter:   () => new THREE.MeshStandardMaterial({color:0x5a2a1a,roughness:0.6}),
  door:      () => new THREE.MeshStandardMaterial({color:0xc8860a,roughness:0.4}),
  doorFrame: () => new THREE.MeshStandardMaterial({color:0x8b5e2a,roughness:0.5}),
  camBody:   () => new THREE.MeshStandardMaterial({color:0x0055cc,roughness:0.4,metalness:0.5}),
  pole:      () => new THREE.MeshStandardMaterial({color:0x334455}),
};

// ================================================================
// UNDO/REDO
// ================================================================
const undoStack = [], redoStack = [];
const MAX_UNDO  = 40;

function sceneSnapshot() {
  return JSON.stringify(objects.map(o => ({...o.data})));
}

function pushUndo() {
  undoStack.push(sceneSnapshot());
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  redoStack.length = 0;
}

function restoreSnapshot(snap) {
  const dataArr = JSON.parse(snap);
  removeAllObjects();
  dataArr.forEach(d => _createAndAdd(d.type, d));
  deselectAll();
}

function undo() { if (!undoStack.length) return; redoStack.push(sceneSnapshot()); restoreSnapshot(undoStack.pop()); }
function redo() { if (!redoStack.length) return; undoStack.push(sceneSnapshot()); restoreSnapshot(redoStack.pop()); }

// ================================================================
// ОБЪЕКТЫ — единый массив
// ================================================================
let objects    = [];    // [{mesh, type, data, frustum?, wallMeshIdx?}]
let objCounter = 0;
const meshToEntry = new WeakMap();

function registerMeshes(en) {
  en.mesh.traverse(c => { if(c.isMesh||c.isGroup) meshToEntry.set(c,en); });
  meshToEntry.set(en.mesh, en);
}

// Snap-точки — конечные точки стен
const snapPts = []; // [{x,z}]

function addSnapPt(x,z) {
  if (!snapPts.some(p=>Math.abs(p.x-x)<0.01&&Math.abs(p.z-z)<0.01))
    snapPts.push({x,z});
}

function snapNearest(x, z, excludeX1=null, excludeZ1=null, excludeX2=null, excludeZ2=null) {
  let best = null, bd = SNAP_R;
  for (const p of snapPts) {
    // Исключаем свои концы
    if (excludeX1 !== null && Math.abs(p.x - excludeX1) < 0.01 && Math.abs(p.z - excludeZ1) < 0.01) continue;
    if (excludeX2 !== null && Math.abs(p.x - excludeX2) < 0.01 && Math.abs(p.z - excludeZ2) < 0.01) continue;

    const d = Math.hypot(p.x - x, p.z - z);
    if (d < bd) {
      bd = d;
      best = p;
    }
  }
  return best ? {x: best.x, z: best.z, snapped: true} : {x, z, snapped: false};
}

// Snap-dot
let _snapDotMesh = null;
function getSnapDot() {
  if (!_snapDotMesh) {
    _snapDotMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.12,8,8),
      new THREE.MeshBasicMaterial({color:0x00e5a0})
    );
    _snapDotMesh.visible=false;
    scene3d.add(_snapDotMesh);
  }
  return _snapDotMesh;
}

// ================================================================
// СТЕНА — единственный способ добавить стену
// data: {type:"wall", id, label, x, z, length, rot, h}
// ================================================================
function wallEndpoints(d) {
  const half=(d.length||4)/2, rad=THREE.MathUtils.degToRad(d.rot||0);
  const cosR=Math.cos(rad), sinR=Math.sin(rad);
  return {
    ax: d.x-cosR*half, az: d.z+sinR*half,
    bx: d.x+cosR*half, bz: d.z-sinR*half,
  };
}

function buildWallMesh(d) {
  const len=d.length||4, h=d.h||3;
  const m=new THREE.Mesh(new THREE.BoxGeometry(len,h,WALL_T), MAT.wall());
  m.position.set(d.x,h/2,d.z);
  m.rotation.y=THREE.MathUtils.degToRad(d.rot||0);
  m.castShadow=true; m.receiveShadow=true;
  return m;
}

function addWall(overrideData={}) {
  pushUndo();
  const id=++objCounter;
  const data={type:"wall",id,label:`Стена ${id}`,x:0,z:0,length:4,rot:0,h:3,...overrideData};
  _createAndAdd("wall",data);
  selectEntry(objects.find(o=>o.data.id===data.id)||null);
}

// ================================================================
// ДВЕРЬ
// data: {type:"door", id, label, x, z, rot, w}
// ================================================================
function buildDoorMesh(d) {
  const grp=new THREE.Group();
  const fL=new THREE.Mesh(new THREE.BoxGeometry(0.1,2.3,0.22),MAT.doorFrame());
  fL.position.set(-d.w/2-0.05,1.15,0);
  const fR=fL.clone(); fR.position.x=d.w/2+0.05;
  const fT=new THREE.Mesh(new THREE.BoxGeometry(d.w+0.2,0.12,0.22),MAT.doorFrame());
  fT.position.set(0,2.3,0);
  const panel=new THREE.Mesh(new THREE.BoxGeometry(d.w,2.1,0.22),MAT.door());
  panel.position.set(0,1.05,0);
  const handle=new THREE.Mesh(new THREE.CylinderGeometry(0.03,0.03,0.18,6),new THREE.MeshStandardMaterial({color:0xdddddd,metalness:0.9}));
  handle.rotation.z=Math.PI/2; handle.position.set(d.w/2-0.12,1.05,0.08);
  grp.add(fL,fR,fT,panel,handle);
  grp.position.set(d.x,0,d.z);
  grp.rotation.y=THREE.MathUtils.degToRad(d.rot||0);
  return grp;
}

// Snap двери к ближайшей стене
function snapDoorToNearestWall(px,pz) {
  let bd=Infinity,bx=px,bz=pz,brot=0;
  for (const en of objects) {
    if (en.type!=="wall") continue;
    const d=en.data;
    const ep=wallEndpoints(d);
    const dx=ep.bx-ep.ax, dz=ep.bz-ep.az;
    const len2=dx*dx+dz*dz; if(len2<0.001) continue;
    const t=Math.max(0,Math.min(1,((px-ep.ax)*dx+(pz-ep.az)*dz)/len2));
    const cx=ep.ax+t*dx, cz=ep.az+t*dz;
    const dist=Math.hypot(px-cx,pz-cz);
    if (dist<bd){bd=dist;bx=cx;bz=cz;brot=-Math.atan2(dz,dx)*180/Math.PI;}
  }
  return {x:parseFloat(bx.toFixed(2)),z:parseFloat(bz.toFixed(2)),rot:brot};
}

// ================================================================
// СТЕЛЛАЖ
// ================================================================
function buildShelfMesh(d) {
  const m=new THREE.Mesh(new THREE.BoxGeometry(d.w,d.h,d.d),MAT.shelf());
  m.position.set(d.x,d.h/2,d.z);
  m.rotation.y=THREE.MathUtils.degToRad(d.rot||0);
  m.castShadow=true;
  return m;
}

// ================================================================
// ПРИЛАВОК
// ================================================================
function buildCounterMesh(d) {
  const m=new THREE.Mesh(new THREE.BoxGeometry(d.w,d.h,d.d),MAT.counter());
  m.position.set(d.x,d.h/2,d.z);
  m.rotation.y=THREE.MathUtils.degToRad(d.rot||0);
  m.castShadow=true;
  return m;
}

// ================================================================
// КАМЕРА
// ================================================================
function buildCameraMeshes(d) {
  const grp=new THREE.Group();
  const pivot=new THREE.Group();
  pivot.rotation.y=THREE.MathUtils.degToRad((d.yaw||0)+180);
  const tilt=new THREE.Group();
  tilt.rotation.x=THREE.MathUtils.degToRad(d.pitch||0);
  pivot.add(tilt);
  const body=new THREE.Mesh(new THREE.BoxGeometry(0.25,0.15,0.4),MAT.camBody());
  const lens=new THREE.Mesh(new THREE.CylinderGeometry(0.06,0.09,0.14,10),new THREE.MeshStandardMaterial({color:0x111111}));
  lens.rotation.x=Math.PI/2; lens.position.z=0.24; body.add(lens);
  tilt.add(body);
  const arrow=new THREE.Mesh(new THREE.ConeGeometry(0.07,0.5,8),new THREE.MeshBasicMaterial({color:0x00aaff}));
  arrow.rotation.x=Math.PI/2; arrow.position.z=0.65;
  tilt.add(arrow);
  grp.add(pivot);
  const pole=new THREE.Mesh(new THREE.CylinderGeometry(0.02,0.02,d.height,6),MAT.pole());
  pole.position.y=-d.height/2; grp.add(pole);
  grp.add(makeLabel(d.label));
  grp.position.set(d.x,d.height,d.z);
  return grp;
}

function buildFrustum(d) {
  const {x,z,height:h,yaw,pitch,fov,img_width:iw=1280,img_height:ih=720}=d;
  const fx=(iw/2)/Math.tan(THREE.MathUtils.degToRad(fov/2));
  const cx=iw/2,cy=ih/2;
  const yR=THREE.MathUtils.degToRad(yaw||0);
  const pR=THREE.MathUtils.degToRad(pitch||0);
  function pixelRay(u,v) {
    const ndx=(u-cx)/fx, ndy=-(v-cy)/fx;
    const cosP=Math.cos(pR),sinP=Math.sin(pR);
    const ry_c= cosP*ndy - sinP;
    const rz_c= sinP*ndy - cosP;
    const cosY=Math.cos(yR),sinY=Math.sin(yR);
    const rx_w= cosY*ndx + sinY*rz_c;
    const rz_w=-sinY*ndx + cosY*rz_c;
    return new THREE.Vector3(rx_w,ry_c,rz_w).normalize();
  }
  const pos=new THREE.Vector3(x,h,z), far=7;
  const fp=[[0,0],[iw,0],[iw,ih],[0,ih]].map(([u,v])=>pos.clone().addScaledVector(pixelRay(u,v),far));
  const pts=[];
  for(let i=0;i<4;i++) pts.push(pos.clone(),fp[i].clone());
  for(let i=0;i<4;i++) pts.push(fp[i].clone(),fp[(i+1)%4].clone());
  return new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x0099ff,transparent:true,opacity:0.55}));
}

function makeLabel(text,color="#0099ff") {
  const c=document.createElement("canvas"); c.width=128;c.height=40;
  const ctx=c.getContext("2d");
  ctx.clearRect(0,0,128,40);
  ctx.font="bold 14px monospace"; ctx.fillStyle=color;
  ctx.textAlign="center"; ctx.textBaseline="middle";
  ctx.fillText(text,64,20);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true,depthTest:false}));
  sp.scale.set(1.2,0.4,1);
  return sp;
}

// ================================================================
// ФАБРИКА: создать mesh по типу
// ================================================================
function buildMesh(type,data) {
  switch(type) {
    case "wall":    return buildWallMesh(data);
    case "door":    return buildDoorMesh(data);
    case "shelf":   return buildShelfMesh(data);
    case "counter": return buildCounterMesh(data);
    case "camera":  return buildCameraMeshes(data);
  }
  return null;
}

// Создать entry из data и добавить в сцену+массив
function _createAndAdd(type,data) {
  const mesh=buildMesh(type,data);
  if (!mesh) return null;
  scene3d.add(mesh);
  const en={mesh,type,data:{...data}};
  objects.push(en);
  registerMeshes(en);
  // Стена → регистрируем snap-точки
  if (type==="wall") {
    const ep=wallEndpoints(data);
    addSnapPt(ep.ax,ep.az);
    addSnapPt(ep.bx,ep.bz);
  }
  return en;
}

function removeAllObjects() {
  for (const en of objects) {
    scene3d.remove(en.mesh);
    en.mesh.traverse(o=>{if(o.geometry)o.geometry.dispose();});
    if (en.frustum) {scene3d.remove(en.frustum);en.frustum.geometry.dispose();}
  }
  objects=[];
  snapPts.length=0;
  clearBlindSpots();
}

// ================================================================
// SNAP ОБЪЕКТОВ К СТЕНАМ (shelf/counter)
// ================================================================
const WALL_SNAP_DIST = 0.7;

function snapObjToWall(px,pz,objD) {
  let bd=WALL_SNAP_DIST,best=null;
  for (const en of objects) {
    if (en.type!=="wall") continue;
    const d=en.data;
    const ep=wallEndpoints(d);
    const dx=ep.bx-ep.ax,dz=ep.bz-ep.az,len2=dx*dx+dz*dz;
    if(len2<0.001) continue;
    const t=Math.max(0,Math.min(1,((px-ep.ax)*dx+(pz-ep.az)*dz)/len2));
    const cx=ep.ax+t*dx,cz=ep.az+t*dz;
    const dist=Math.hypot(px-cx,pz-cz);
    if (dist<bd) {
      bd=dist;
      const rad=THREE.MathUtils.degToRad(d.rot||0);
      const cosR=Math.cos(rad),sinR=Math.sin(rad);
      const nx=-sinR,nz=cosR; // нормаль к стене
      const halfD=(objD||0.5)/2+WALL_T/2;
      const side=((px-cx)*nx+(pz-cz)*nz)>=0?1:-1;
      best={
        x:parseFloat((cx+nx*halfD*side).toFixed(2)),
        z:parseFloat((cz+nz*halfD*side).toFixed(2)),
        rot:-Math.atan2(dz,dx)*180/Math.PI,
      };
    }
  }
  return best;
}

// ================================================================
// GIZMO
// ================================================================
let gizmoGroup=null,gizmoTarget=null,dragGizmo=null;
let gizmoStartAngle=0;

function buildGizmo(noRot=false) {
  const grp=new THREE.Group();
  function arrow(axis) {
    const color=axis==="x"?0xff4444:0x4488ff;
    const g=new THREE.Group();
    const shaft=new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.04,1.2,8),new THREE.MeshBasicMaterial({color,depthTest:false}));
    shaft.position.y=0.6;
    const head=new THREE.Mesh(new THREE.ConeGeometry(0.12,0.35,8),new THREE.MeshBasicMaterial({color,depthTest:false}));
    head.position.y=1.4;
    const hit=new THREE.Mesh(new THREE.CylinderGeometry(0.22,0.22,1.8,8),new THREE.MeshBasicMaterial({visible:false,depthTest:false}));
    hit.position.y=0.9; hit.userData.axis=axis;
    g.add(shaft,head,hit);
    return g;
  }
  const ax=arrow("x"); ax.rotation.z=-Math.PI/2; grp.add(ax);
  const az=arrow("z"); az.rotation.x= Math.PI/2; grp.add(az);
  if (!noRot) {
    const ring=new THREE.Mesh(new THREE.TorusGeometry(0.5,0.03,8,32),new THREE.MeshBasicMaterial({color:0x00e5a0,depthTest:false}));
    ring.rotation.x=Math.PI/2;
    const ringHit=new THREE.Mesh(new THREE.TorusGeometry(0.5,0.14,8,32),new THREE.MeshBasicMaterial({visible:false,depthTest:false}));
    ringHit.rotation.x=Math.PI/2; ringHit.userData.axis="rot";
    grp.add(ring,ringHit);
  }
  grp.renderOrder=999;
  return grp;
}

function showGizmo(en) {
  if (gizmoGroup) {scene3d.remove(gizmoGroup);gizmoGroup=null;}
  gizmoTarget=en;
  if (!en||en.type==="door") return;
  gizmoGroup=buildGizmo(en.type==="wall");
  gizmoGroup.position.set(en.mesh.position.x,0.3,en.mesh.position.z);
  scene3d.add(gizmoGroup);
}

function updateGizmoPos() {
  if (!gizmoGroup||!gizmoTarget) return;
  gizmoGroup.position.set(gizmoTarget.mesh.position.x,0.3,gizmoTarget.mesh.position.z);
}

// ================================================================
// ВЫДЕЛЕНИЕ
// ================================================================
const _origCol=new WeakMap();

function highlight(en) {
  en.mesh.traverse(o=>{
    if(!o.isMesh||!o.material) return;
    if(!_origCol.has(o)) _origCol.set(o,o.material.color.getHex());
    o.material.color.setHex(0x00e5a0);
    if(o.material.emissive) o.material.emissive.setHex(0x003322);
  });
}
function unhighlight(en) {
  en.mesh.traverse(o=>{
    if(!o.isMesh||!o.material) return;
    const c=_origCol.get(o);
    if(c!==undefined) o.material.color.setHex(c);
    if(o.material.emissive) o.material.emissive.setHex(0x000000);
  });
}

let selected=null;

function selectEntry(en) {
  // Скрываем frustum/зоны предыдущей камеры
  if (selected&&selected.type==="camera") hideCameraViz(selected);
  if (selected) unhighlight(selected);

  selected=en;
  if (en) { highlight(en); showGizmo(en); }
  else showGizmo(null);

  // Показываем frustum/зоны для новой камеры
  if (en&&en.type==="camera") {
    showCameraViz(en);
    computeBlindSpots(en);
  }

  // Кнопка "Дверь" — активна только если выбрана стена
  const doorBtn=document.getElementById("add-door");
  if (doorBtn) {
    const wallSel=en&&en.type==="wall";
    doorBtn.disabled=!wallSel;
    doorBtn.classList.toggle("add-btn-disabled",!wallSel);
    doorBtn.title=wallSel?"Разместить дверь на выбранной стене":"Выберите стену для размещения двери";
  }

  showProps(en);
}

function deselectAll() { selectEntry(null); }

// ================================================================
// КАМЕРА — frustum и слепые зоны (только при выборе)
// ================================================================
function showCameraViz(en) {
  if (!en.frustum) {
    en.frustum=buildFrustum(en.data);
    scene3d.add(en.frustum);
  }
  computeBlindSpots(en);
}

function hideCameraViz(en) {
  if (en.frustum) {
    scene3d.remove(en.frustum);
    en.frustum.geometry.dispose();
    en.frustum=null;
  }
  clearBlindSpots();
}

// ================================================================
// СЛЕПЫЕ ЗОНЫ — ТОЛЬКО ЗА СТЕЛЛАЖАМИ И ПРИЛАВКАМИ (облегчённая версия)
// ================================================================
const blindMeshes = [];

function clearBlindSpots() {
  blindMeshes.forEach(m => {
    scene3d.remove(m);
    if (m.geometry) m.geometry.dispose();
  });
  blindMeshes.length = 0;
}

// ── Геометрия препятствий в виде OBB (учитывает поворот вокруг Y) ──
function toOBB(type, d) {
  if (type === "shelf" || type === "counter") {
    return {
      cx: d.x || 0,
      cy: (d.h || 1) / 2,
      cz: d.z || 0,
      hx: (d.w || 1) / 2,
      hy: (d.h || 1) / 2,
      hz: (d.d || 1) / 2,
      yaw: THREE.MathUtils.degToRad(d.rot || 0),
    };
  }
  if (type === "wall") {
    return {
      cx: d.x || 0,
      cy: (d.h || 3) / 2,
      cz: d.z || 0,
      hx: (d.length || 4) / 2,
      hy: (d.h || 3) / 2,
      hz: WALL_T / 2,
      yaw: THREE.MathUtils.degToRad(d.rot || 0),
    };
  }
  return null;
}

let _blindWorker  = null;
let _blindBusy    = false;
let _pendingBlind = null; // последний запрос пока Worker занят

// Код Worker-а инлайн через Blob — не зависит от пути файла
const _workerSrc = `
function rayOBB(ox,oy,oz,dx,dy,dz,b){
  const c=Math.cos(b.yaw), s=Math.sin(b.yaw);

  // Преобразуем луч в локальные координаты OBB (обратный поворот вокруг Y)
  const rox =  c*(ox-b.cx) - s*(oz-b.cz);
  const roy =  oy-b.cy;
  const roz =  s*(ox-b.cx) + c*(oz-b.cz);

  const rdx =  c*dx - s*dz;
  const rdy =  dy;
  const rdz =  s*dx + c*dz;

  let tmin = -Infinity, tmax = Infinity;

  function axis(o, d, h){
    if (Math.abs(d) < 1e-9) {
      if (o < -h || o > h) return false;
      return true;
    }
    const t1 = (-h - o) / d;
    const t2 = ( h - o) / d;
    const lo = Math.min(t1, t2);
    const hi = Math.max(t1, t2);
    tmin = Math.max(tmin, lo);
    tmax = Math.min(tmax, hi);
    return tmin <= tmax;
  }

  if (!axis(rox, rdx, b.hx)) return Infinity;
  if (!axis(roy, rdy, b.hy)) return Infinity;
  if (!axis(roz, rdz, b.hz)) return Infinity;
  if (tmax < 0) return Infinity;
  return tmin < 0 ? tmax : tmin;
}
function nearestHit(ox,oy,oz,dx,dy,dz,boxes){
  let best=Infinity,bi=-1;
  for(let i=0;i<boxes.length;i++){
    const t=rayOBB(ox,oy,oz,dx,dy,dz,boxes[i]);
    if(t<best){best=t;bi=i;}
  }
  return bi>=0?{dist:best,idx:bi}:null;
}
function inFOV(cam,px,pz){
  const tx=px-cam.x, ty=-cam.height, tz=pz-cam.z;
  const len=Math.sqrt(tx*tx+ty*ty+tz*tz); if(!len) return false;
  const nx=tx/len,ny=ty/len,nz=tz/len;
  const yR=cam.yaw*Math.PI/180, pR=cam.pitch*Math.PI/180;
  const cosP=Math.cos(pR),sinP=Math.sin(pR),cosY=Math.cos(yR),sinY=Math.sin(yR);
  const cdx=-sinY*cosP,cdy=-sinP,cdz=-cosY*cosP;
  const cl=Math.sqrt(cdx*cdx+cdy*cdy+cdz*cdz);
  const dot=(nx*cdx+ny*cdy+nz*cdz)/cl;
  const ang=Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI;
  const fovV=2*Math.atan(Math.tan(cam.fov/2*Math.PI/180)*(cam.ih/cam.iw))*180/Math.PI;
  return ang<=Math.sqrt((cam.fov/2)*(cam.fov/2)+(fovV/2)*(fovV/2));
}
self.onmessage=function(e){
  const {cam,shadowBoxes,wallBoxes}=e.data;
  const all=[...shadowBoxes,...wallBoxes], ns=shadowBoxes.length;
  const STEP=0.25,RANGE=25,HALF=STEP/2;
  const pos=[],idx=[];let vi=0;
  const cx=cam.x,cy=cam.height,cz=cam.z;
  for(let x=-RANGE;x<=RANGE;x+=STEP){
    for(let z=-RANGE;z<=RANGE;z+=STEP){
      if(!inFOV(cam,x,z)) continue;
      const tx=x-cx,ty=0.1-cy,tz=z-cz;
      const dist=Math.sqrt(tx*tx+ty*ty+tz*tz);
      const dx=tx/dist,dy=ty/dist,dz=tz/dist;
      const h=nearestHit(cx,cy,cz,dx,dy,dz,all);
      if(!h||h.dist>=dist-0.4) continue;
      if(h.idx>=ns) continue;
      const wh=nearestHit(x,0.1,z,-dx,-dy,-dz,wallBoxes);
      if(wh&&wh.dist<dist-0.4) continue;
      const x0=x-HALF,x1=x+HALF,z0=z-HALF,z1=z+HALF,y=0.05;
      pos.push(x0,y,z0,x1,y,z0,x1,y,z1,x0,y,z1);
      idx.push(vi,vi+1,vi+2,vi,vi+2,vi+3);
      vi+=4;
    }
  }
  const pa=new Float32Array(pos),ia=new Uint32Array(idx);
  self.postMessage({posArr:pa,idxArr:ia},[pa.buffer,ia.buffer]);
};
`;
const _workerBlob = new Blob([_workerSrc], { type: "application/javascript" });
const _workerURL  = URL.createObjectURL(_workerBlob);

function _buildBlindMesh(posArr, idxArr) {
  // Убираем старый меш и ставим новый атомарно
  clearBlindSpots();
  if (!posArr.length) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: 0xff2200, transparent: true, opacity: 0.40,
    side: THREE.DoubleSide, depthWrite: false,
  }));
  scene3d.add(mesh);
  blindMeshes.push(mesh);
}

function _ensureBlindWorker() {
  if (_blindWorker) return;
  _blindWorker = new Worker(_workerURL);
  _blindWorker.onmessage = function(e) {
    _blindBusy = false;
    _buildBlindMesh(e.data.posArr, e.data.idxArr);
    // Если пока считали пришёл новый запрос — сразу запускаем только его
    if (_pendingBlind) {
      const p = _pendingBlind;
      _pendingBlind = null;
      _postBlindJob(p.cam, p.shadowBoxes, p.wallBoxes);
    }
  };
  _blindWorker.onerror = function() {
    _blindBusy = false;
    _pendingBlind = null;
    if (_blindWorker) _blindWorker.terminate();
    _blindWorker = null;
  };
}

function _postBlindJob(cam, shadowBoxes, wallBoxes) {
  _ensureBlindWorker();
  if (!_blindWorker) return;
  _blindBusy = true;
  _blindWorker.postMessage({ cam, shadowBoxes, wallBoxes });
}

function computeBlindSpots(camEntry) {
  if (!camEntry || camEntry.type !== "camera") { clearBlindSpots(); return; }

  const d = camEntry.data;
  const shadowBoxes = [], wallBoxes = [];

  for (const en of objects) {
    if (en.type === "shelf" || en.type === "counter") {
      const obb = toOBB(en.type, en.data);
      if (obb) shadowBoxes.push(obb);
    } else if (en.type === "wall") {
      const obb = toOBB("wall", en.data);
      if (obb) wallBoxes.push(obb);
    }
  }
  if (!shadowBoxes.length) {
    clearBlindSpots();
    _pendingBlind = null;
    return;
  }

  const cam = {
    x: d.x, z: d.z, height: d.height || 3.0,
    yaw: d.yaw || 0, pitch: d.pitch || 0,
    fov: d.fov || 90,
    iw: d.img_width || 1280, ih: d.img_height || 720,
  };

  // Реалтайм-режим: если worker занят, просто переписываем последнюю задачу
  if (_blindBusy) {
    _pendingBlind = { cam, shadowBoxes, wallBoxes };
    return;
  }
  _postBlindJob(cam, shadowBoxes, wallBoxes);
}

// ================================================================
// LIVE-REBUILD (изменение параметров → мгновенный rebuild mesh)
// ================================================================
function liveRebuild(en) {
  const d = en.data;
  scene3d.remove(en.mesh);
  en.mesh.traverse(o => { if (o.geometry) o.geometry.dispose(); });

  if (en.type === "camera") {
    if (en.frustum) {
      scene3d.remove(en.frustum);
      en.frustum.geometry.dispose();
      en.frustum = null;
    }

    en.mesh = buildCameraMeshes(d);
    scene3d.add(en.mesh);
    registerMeshes(en);

    // === НОВОЕ: пересоздаём frustum + слепую зону ===
    en.frustum = buildFrustum(d);
    scene3d.add(en.frustum);

    if (selected === en) {
      showCameraViz(en);
      computeBlindSpots(en);        // ← ДОБАВИТЬ ЭТУ СТРОКУ
    }
    return;
  }

  // остальной код без изменений
  en.mesh = buildMesh(en.type, d);
  scene3d.add(en.mesh);
  registerMeshes(en);

  if (en.type === "wall") {
    const ep = wallEndpoints(d);
    addSnapPt(ep.ax, ep.az);
    addSnapPt(ep.bx, ep.bz);
  }

  if (selected === en) {
    highlight(en);
    showGizmo(en);
    updateGizmoPos();
  }
}

// ================================================================
// ПАНЕЛЬ СВОЙСТВ
// ================================================================
const PROP_PANELS=["props-empty","props-camera","props-shelf","props-counter","props-door","props-wall-obj"];

function showProps(en) {
  PROP_PANELS.forEach(id=>{const el=document.getElementById(id);if(el)el.style.display="none";});
  if (!en){document.getElementById("props-empty").style.display="block";return;}
  const d=en.data;
  if (en.type==="camera") {
    document.getElementById("props-camera").style.display="block";
    document.getElementById("p-cam-label").value=d.label||"";
    document.getElementById("p-cam-address").value=d.address||"";
    document.getElementById("p-cam-h").value=d.height??3;
    document.getElementById("p-cam-yaw").value=d.yaw??0;
    document.getElementById("p-cam-pitch").value=d.pitch??60;
    document.getElementById("p-cam-fov").value=d.fov??90;
  } else if (en.type==="shelf") {
    document.getElementById("props-shelf").style.display="block";
    document.getElementById("p-shelf-label").value=d.label||"";
    document.getElementById("p-shelf-w").value=d.w||1.5;
    document.getElementById("p-shelf-d").value=d.d||0.5;
    document.getElementById("p-shelf-h").value=d.h||2;
    document.getElementById("p-shelf-rot").value=d.rot||0;
  } else if (en.type==="counter") {
    document.getElementById("props-counter").style.display="block";
    document.getElementById("p-counter-label").value=d.label||"";
    document.getElementById("p-counter-w").value=d.w||2;
    document.getElementById("p-counter-d").value=d.d||0.8;
    document.getElementById("p-counter-h").value=d.h||1;
    document.getElementById("p-counter-rot").value=d.rot||0;
  } else if (en.type==="door") {
    document.getElementById("props-door").style.display="block";
    document.getElementById("p-door-label").value=d.label||"";
    document.getElementById("p-door-w").value=d.w||1;
  } else if (en.type==="wall") {
    document.getElementById("props-wall-obj").style.display="block";
    document.getElementById("p-wobj-label").value=d.label||"";
    document.getElementById("p-wobj-len").value=d.length||4;
    document.getElementById("p-wobj-rot").value=d.rot||0;
    document.getElementById("p-wobj-h").value=d.h||3;
  }
}

// Live-inputs: изменение → немедленный rebuild
function bindLive(id,field,parse) {
  const el=document.getElementById(id); if(!el) return;
  el.addEventListener("input",()=>{
    if(!selected) return;
    selected.data[field]=parse?parse(el.value):el.value;
    liveRebuild(selected);
  });
  el.addEventListener("change",()=>{if(selected) pushUndo();});
}

bindLive("p-cam-label","label"); bindLive("p-cam-address","address"); bindLive("p-cam-h","height",parseFloat);
bindLive("p-cam-yaw","yaw",parseFloat); bindLive("p-cam-pitch","pitch",parseFloat);
bindLive("p-cam-fov","fov",parseFloat);
bindLive("p-shelf-label","label"); bindLive("p-shelf-w","w",parseFloat);
bindLive("p-shelf-d","d",parseFloat); bindLive("p-shelf-h","h",parseFloat);
bindLive("p-shelf-rot","rot",parseFloat);
bindLive("p-counter-label","label"); bindLive("p-counter-w","w",parseFloat);
bindLive("p-counter-d","d",parseFloat); bindLive("p-counter-h","h",parseFloat);
bindLive("p-counter-rot","rot",parseFloat);
bindLive("p-door-label","label"); bindLive("p-door-w","w",parseFloat);
bindLive("p-wobj-label","label"); bindLive("p-wobj-len","length",parseFloat);
bindLive("p-wobj-rot","rot",parseFloat); bindLive("p-wobj-h","h",parseFloat);

// ================================================================
// КНОПКИ ПАНЕЛИ
// ================================================================
function deleteSelected() {
  if (!selected) return;
  pushUndo();
  scene3d.remove(selected.mesh);
  selected.mesh.traverse(o=>{if(o.geometry)o.geometry.dispose();});
  if (selected.frustum){scene3d.remove(selected.frustum);selected.frustum.geometry.dispose();}
  objects.splice(objects.indexOf(selected),1);
  selected=null;
  showGizmo(null);
  clearBlindSpots();
  showProps(null);
  // Перестраиваем snap-точки
  snapPts.length=0;
  for (const o of objects) if (o.type==="wall"){const ep=wallEndpoints(o.data);addSnapPt(ep.ax,ep.az);addSnapPt(ep.bx,ep.bz);}
}

function dupSelected(overX=1,overZ=1) {
  if (!selected) return;
  pushUndo();
  const d={...selected.data,id:++objCounter,label:`${selected.data.label} (2)`,x:selected.data.x+overX,z:selected.data.z+overZ};
  const en=_createAndAdd(selected.type,d);
  selectEntry(en);
}

document.getElementById("p-cam-apply")&&document.getElementById("p-cam-apply").addEventListener("click",()=>{});
document.getElementById("p-cam-del").addEventListener("click",deleteSelected);
document.getElementById("p-shelf-apply")&&document.getElementById("p-shelf-apply").addEventListener("click",()=>{});
document.getElementById("p-shelf-del").addEventListener("click",deleteSelected);
document.getElementById("p-shelf-dup").addEventListener("click",()=>dupSelected(1,1));
document.getElementById("p-counter-apply")&&document.getElementById("p-counter-apply").addEventListener("click",()=>{});
document.getElementById("p-counter-del").addEventListener("click",deleteSelected);
document.getElementById("p-counter-dup").addEventListener("click",()=>dupSelected(1,1));
document.getElementById("p-door-apply")&&document.getElementById("p-door-apply").addEventListener("click",()=>{});
document.getElementById("p-door-del").addEventListener("click",deleteSelected);
document.getElementById("p-wobj-del").addEventListener("click",deleteSelected);
document.getElementById("p-wobj-dup").addEventListener("click",()=>dupSelected(0.5,0.5));

// ================================================================
// КНОПКИ ДОБАВЛЕНИЯ
// ================================================================
function addObject(type,extra={}) {
  pushUndo();
  const id=++objCounter;
  const defaults={
    wall:    {length:4,rot:0,h:3,label:`Стена ${id}`},
    shelf:   {w:1.5,d:0.5,h:2,rot:0,label:`Стеллаж ${id}`},
    counter: {w:2,d:0.8,h:1,rot:0,label:`Прилавок ${id}`},
    camera:  {height:3,yaw:0,pitch:60,fov:90,img_width:1280,img_height:720,label:`CAM ${id}`},
  };
  const data={type,id,x:0,z:0,...defaults[type],...extra};
  const en=_createAndAdd(type,data);
  selectEntry(en);
}

function addDoor() {
  if (!selected||selected.type!=="wall") return;
  pushUndo();
  const id=++objCounter;
  const snap=snapDoorToNearestWall(selected.data.x,selected.data.z);
  const data={type:"door",id,x:snap.x,z:snap.z,rot:snap.rot,w:1,label:`Дверь ${id}`};
  const en=_createAndAdd("door",data);
  selectEntry(en);
}

document.getElementById("add-wall-obj").addEventListener("click",()=>addObject("wall"));
document.getElementById("add-door").addEventListener("click",addDoor);
document.getElementById("add-shelf").addEventListener("click",()=>addObject("shelf"));
document.getElementById("add-counter").addEventListener("click",()=>addObject("counter"));
document.getElementById("add-camera").addEventListener("click",()=>addObject("camera"));

// ================================================================
// RAYCAST
// ================================================================
const raycaster=new THREE.Raycaster();

function ndcFromEvent(e) {
  const r=canvas.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1,-((e.clientY-r.top)/r.height)*2+1);
}

function hitFloor(e) {
  raycaster.setFromCamera(ndcFromEvent(e),cam3d);
  const hits=raycaster.intersectObject(hitPlane);
  return hits.length?hits[0].point:null;
}

function pickObject(e) {
  raycaster.setFromCamera(ndcFromEvent(e),cam3d);

  function firstHit(types) {
    const tgts=[];
    for (const en of objects) {
      if(types&&!types.includes(en.type)) continue;
      en.mesh.traverse(c=>{if(c.isMesh)tgts.push(c);});
    }
    const hits=raycaster.intersectObjects(tgts,false);
    if(!hits.length) return null;
    for (const hit of hits) {
      let node=hit.object;
      while(node){if(meshToEntry.has(node))return meshToEntry.get(node);node=node.parent;}
    }
    return null;
  }

  // Приоритет: wall > shelf/counter/camera > door
  return firstHit(null);
}

function pickGizmoAxis(e) {
  if (!gizmoGroup) return null;
  raycaster.setFromCamera(ndcFromEvent(e),cam3d);
  const tgts=[];
  gizmoGroup.traverse(c=>{if(c.isMesh&&c.userData.axis)tgts.push(c);});
  const hits=raycaster.intersectObjects(tgts,false);
  return hits.length?hits[0].object.userData.axis:null;
}

// ================================================================
// МЫШЬ
// ================================================================
let dragEntry=null, mouseDownPt=null, hasDragged=false;

canvas.addEventListener("mousedown", e => {
  if (e.button !== 0) return;
  mouseDownPt = { x: e.clientX, y: e.clientY };
  hasDragged = false;

  const axis = pickGizmoAxis(e);
  if (axis && gizmoTarget) {
    dragGizmo = axis;
    const pt = hitFloor(e);
    if (axis === "rot" && pt) gizmoStartAngle = Math.atan2(pt.z - gizmoTarget.data.z, pt.x - gizmoTarget.data.x);
    return;
  }

  const picked = pickObject(e);
  dragEntry = (selected && picked === selected) ? selected : null;
});

window.addEventListener("mousemove",e=>{
  if (mouseDownPt&&(Math.abs(e.clientX-mouseDownPt.x)>8||Math.abs(e.clientY-mouseDownPt.y)>8))
    hasDragged=true;

  const pt=hitFloor(e);
  if (pt) document.getElementById("footer-coords").textContent=`${pt.x.toFixed(1)} × ${pt.z.toFixed(1)} м`;

  // Gizmo drag
  if (dragGizmo&&gizmoTarget&&hasDragged) {
    if (!pt) return;
    const d=gizmoTarget.data;
    if (dragGizmo==="x") {
      d.x=parseFloat(pt.x.toFixed(2));
      gizmoTarget.mesh.position.x=pt.x;
      if (gizmoTarget.type==="wall") { const ep=wallEndpoints(d); addSnapPt(ep.ax,ep.az); addSnapPt(ep.bx,ep.bz); }
    } else if (dragGizmo==="z") {
      d.z=parseFloat(pt.z.toFixed(2));
      gizmoTarget.mesh.position.z=pt.z;
      if (gizmoTarget.type==="wall") { const ep=wallEndpoints(d); addSnapPt(ep.ax,ep.az); addSnapPt(ep.bx,ep.bz); }
    } else if (dragGizmo==="rot") {
      const angle=Math.atan2(pt.z-d.z,pt.x-d.x);
      const delta=-(angle-gizmoStartAngle)*180/Math.PI;
      if (gizmoTarget.type==="camera") {
        d.yaw=(d.yaw||0)+delta;
        const pivot=gizmoTarget.mesh.children.find(c=>c.isGroup);
        if(pivot) pivot.rotation.y=THREE.MathUtils.degToRad(d.yaw+180);
        if(gizmoTarget.frustum){scene3d.remove(gizmoTarget.frustum);gizmoTarget.frustum.geometry.dispose();}
        gizmoTarget.frustum=buildFrustum(d); scene3d.add(gizmoTarget.frustum);
        computeBlindSpots(gizmoTarget);
      } else {
        d.rot=(d.rot||0)+delta;
        gizmoTarget.mesh.rotation.y=THREE.MathUtils.degToRad(d.rot);
        if (gizmoTarget.type==="wall") { const ep=wallEndpoints(d); addSnapPt(ep.ax,ep.az); addSnapPt(ep.bx,ep.bz); }
      }
      gizmoStartAngle=angle;
    }
    if (gizmoTarget.type==="camera"&&dragGizmo!=="rot") {
      gizmoTarget.mesh.position.y=d.height;
      if(gizmoTarget.frustum){scene3d.remove(gizmoTarget.frustum);gizmoTarget.frustum.geometry.dispose();}
      gizmoTarget.frustum=buildFrustum(d); scene3d.add(gizmoTarget.frustum);
    }
    updateGizmoPos();
    showProps(gizmoTarget);
    return;
  }

// Обычный drag объекта
  if (dragEntry && hasDragged) {
    if (!pt) return;

    let nx = pt.x;
    let nz = pt.z;
    let nrot = null;

    if (dragEntry.type === "door") {
      const s = snapDoorToNearestWall(pt.x, pt.z);
      nx = s.x; nz = s.z; nrot = s.rot;
    }
    else if (dragEntry.type === "shelf" || dragEntry.type === "counter") {
      const s = snapObjToWall(pt.x, pt.z, dragEntry.data.d);
      if (s) { nx = s.x; nz = s.z; nrot = s.rot; }
    }
    else if (dragEntry.type === "wall") {
      // FIX: плавное движение + snap только к чужим концам
      const d = dragEntry.data;
      const half = (d.length || 4) / 2;
      const rad = THREE.MathUtils.degToRad(d.rot || 0);
      const edx = Math.cos(rad) * half;
      const edz = -Math.sin(rad) * half;

      const ep = wallEndpoints(d);           // текущие концы этой стены

      // Проверяем snap для обоих концов
      for (const sign of [1, -1]) {
        const testX = pt.x + sign * edx;
        const testZ = pt.z + sign * edz;
        const s = snapNearest(testX, testZ, ep.ax, ep.az, ep.bx, ep.bz);
        if (s.snapped) {
          nx = s.x - sign * edx;
          nz = s.z - sign * edz;
          break;
        }
      }
    }

    // ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    // Применяем финальные координаты ОДИН раз
    dragEntry.data.x = parseFloat(nx.toFixed(2));
    dragEntry.data.z = parseFloat(nz.toFixed(2));
    dragEntry.mesh.position.x = nx;
    dragEntry.mesh.position.z = nz;

    if (nrot !== null) {
      dragEntry.data.rot = nrot;
      dragEntry.mesh.rotation.y = THREE.MathUtils.degToRad(nrot);
    }

    if (dragEntry.type === "camera") {
      dragEntry.mesh.position.y = dragEntry.data.height;
      if (dragEntry.frustum) {
        scene3d.remove(dragEntry.frustum);
        dragEntry.frustum.geometry.dispose();
      }
      dragEntry.frustum = buildFrustum(dragEntry.data);
      scene3d.add(dragEntry.frustum);
      computeBlindSpots(dragEntry);
    }

    // Snap-dot (визуальная подсказка)
    if (dragEntry.type === "wall") {
      const d = dragEntry.data;
      const half = (d.length || 4) / 2;
      const rad = THREE.MathUtils.degToRad(d.rot || 0);
      const edx = Math.cos(rad) * half;
      const edz = -Math.sin(rad) * half;

      const s = snapNearest(pt.x + edx, pt.z + edz,   // проверяем один конец
                           wallEndpoints(d).ax, wallEndpoints(d).az,
                           wallEndpoints(d).bx, wallEndpoints(d).bz);

      if (s.snapped) {
        getSnapDot().position.set(s.x, 0.1, s.z);
        getSnapDot().visible = true;
      } else {
        getSnapDot().visible = false;
      }
    }

    updateGizmoPos();
  }
});

window.addEventListener("mouseup", e => {
  if (e.button === 0) {
    if ((dragEntry || dragGizmo) && hasDragged) pushUndo();

    // FIX: пересобираем snap-точки ТОЛЬКО после завершения drag стены
    if (dragEntry && dragEntry.type === "wall") {
      snapPts.length = 0;
      for (const o of objects) {
        if (o.type === "wall") {
          const ep = wallEndpoints(o.data);
          addSnapPt(ep.ax, ep.az);
          addSnapPt(ep.bx, ep.bz);
        }
      }
    }

    dragEntry = null;
    dragGizmo = null;
    getSnapDot().visible = false;
  }
});

canvas.addEventListener("click",e=>{
  if (e.button!==0||hasDragged) return;
  if (dragGizmo) return;
  const en=pickObject(e);
  selectEntry(en||null);
});

// ПКМ без движения — снять выделение
canvas.addEventListener("mouseup",e=>{
  if (e.button===2&&!rmbMoved) selectEntry(null);
});

// ================================================================
// КЛАВИАТУРА
// ================================================================
document.addEventListener("keydown",e=>{
  const inInput=["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName);
  if (e.key==="Escape") { selectEntry(null); return; }
  if (!inInput) {
    if (e.key==="Delete"||e.key==="Backspace") { deleteSelected(); return; }
    if (e.ctrlKey&&(e.key==="z"||e.code==="KeyZ")) { e.preventDefault(); undo(); return; }
    if (e.ctrlKey&&(e.key==="y"||e.code==="KeyY")) { e.preventDefault(); redo(); return; }
  }
});

// ================================================================
// СОХРАНЕНИЕ / ЗАГРУЗКА
// ================================================================
document.getElementById("btn-save").addEventListener("click",async()=>{
  const ok = window.confirm(
    "Сохранить текущую сцену?\n\n" +
    "Это перезапишет ранее сохраненную конфигурацию камер и локальную сцену."
  );
  if (!ok) {
    document.getElementById("footer-info").textContent = "Сохранение отменено";
    return;
  }

  // Смещаем сохранение к центру сцены, чтобы координаты были в диапазоне [0..width/height]
  let sceneW = 20, sceneH = 20;
  try {
    const r = await fetch(`${API}/scene`);
    const s = await r.json();
    sceneW = Number(s.width) || 20;
    sceneH = Number(s.height) || 20;
  } catch (_) {}

  const raw = objects.map(o => ({ ...o.data }));
  let offsetX = sceneW / 2;
  let offsetZ = sceneH / 2;
  if (raw.length) {
    const xs = raw.map(o => Number(o.x) || 0);
    const zs = raw.map(o => Number(o.z) || 0);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    offsetX = sceneW / 2 - cx;
    offsetZ = sceneH / 2 - cz;
  }

  const savedScene = raw.map(o => ({
    ...o,
    x: Number((Number(o.x || 0) + offsetX).toFixed(3)),
    z: Number((Number(o.z || 0) + offsetZ).toFixed(3)),
  }));

  // Камеры на сервер
  const cams = savedScene.filter(o => o.type === "camera").map(c => ({
    x: c.x, y: c.z, height: c.height,
    yaw: c.yaw, pitch: -(90 + (c.pitch)),
    fov: c.fov, img_width: c.img_width || 1280,
    img_height: c.img_height || 720, label: c.label,
    address: c.address || "",
  }));
  try {
    await fetch(`${API}/cameras`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cams)});
    document.getElementById("footer-info").textContent=`Сохранено ${cams.length} камер · центрировано · ${new Date().toLocaleTimeString()}`;
  } catch { document.getElementById("footer-info").textContent="Ошибка сохранения"; }
  // Вся сцена в localStorage
  localStorage.setItem("rv_scene", JSON.stringify(savedScene));
});

document.getElementById("btn-load").addEventListener("click",()=>{
  const raw=localStorage.getItem("rv_scene"); if(!raw) return;
  try {
    pushUndo();
    removeAllObjects();
    JSON.parse(raw).forEach(d=>_createAndAdd(d.type,d));
    deselectAll();
    document.getElementById("footer-info").textContent="Сцена загружена";
  } catch(err) { document.getElementById("footer-info").textContent="Ошибка: "+err.message; }
});

document.getElementById("btn-reset-view").addEventListener("click",()=>orbit.reset());

// ================================================================
// ПРЕСЕТЫ
// ================================================================
const PRESETS_KEY = "rv_presets";

function _getPresets() {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY)) || {}; } catch { return {}; }
}
function _savePresets(p) { localStorage.setItem(PRESETS_KEY, JSON.stringify(p)); }

function _refreshPresetList() {
  const sel = document.getElementById("preset-select");
  const cur = sel.value;
  sel.innerHTML = '<option value="">-- выберите --</option>';
  const p = _getPresets();
  Object.keys(p).sort().forEach(name => {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
  });
  if (cur && p[cur]) sel.value = cur;
}

function _currentSceneData() {
  const raw = objects.map(o => ({ ...o.data }));
  return { version: 1, objects: raw };
}

function _applySceneData(data) {
  if (!data || !data.objects) return;
  pushUndo();
  removeAllObjects();
  data.objects.forEach(d => _createAndAdd(d.type, d));
  deselectAll();
}

_refreshPresetList();

document.getElementById("btn-preset-save").addEventListener("click", () => {
  const name = document.getElementById("preset-name").value.trim();
  if (!name) { document.getElementById("footer-info").textContent = "Введите имя пресета"; return; }
  const p = _getPresets();
  p[name] = _currentSceneData();
  _savePresets(p);
  _refreshPresetList();
  document.getElementById("preset-select").value = name;
  document.getElementById("footer-info").textContent = `Пресет "${name}" сохранён`;
});

document.getElementById("btn-preset-load").addEventListener("click", () => {
  const name = document.getElementById("preset-select").value;
  if (!name) { document.getElementById("footer-info").textContent = "Выберите пресет"; return; }
  const p = _getPresets();
  if (!p[name]) { document.getElementById("footer-info").textContent = "Пресет не найден"; return; }
  _applySceneData(p[name]);
  document.getElementById("footer-info").textContent = `Пресет "${name}" загружен`;
});

document.getElementById("btn-preset-del").addEventListener("click", () => {
  const name = document.getElementById("preset-select").value;
  if (!name) return;
  if (!window.confirm(`Удалить пресет "${name}"?`)) return;
  const p = _getPresets();
  delete p[name];
  _savePresets(p);
  _refreshPresetList();
  document.getElementById("footer-info").textContent = `Пресет "${name}" удалён`;
});

document.getElementById("btn-preset-export").addEventListener("click", () => {
  const name = document.getElementById("preset-select").value;
  const p = _getPresets();
  const data = name ? p[name] : _currentSceneData();
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `scene_${name || 'current'}.json`;
  a.click(); URL.revokeObjectURL(url);
  document.getElementById("footer-info").textContent = `Экспорт "${name || 'текущей'}"`;
});

document.getElementById("btn-preset-import").addEventListener("click", () => {
  document.getElementById("preset-import-file").click();
});

document.getElementById("preset-import-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const data = JSON.parse(ev.target.result);
      if (!data.objects) throw new Error("Неверный формат");
      const name = file.name.replace(/\.json$/i, "") || "imported";
      const p = _getPresets();
      p[name] = data;
      _savePresets(p);
      _refreshPresetList();
      document.getElementById("preset-select").value = name;
      document.getElementById("footer-info").textContent = `Импорт "${name}"`;
    } catch (err) {
      document.getElementById("footer-info").textContent = "Ошибка: " + err.message;
    }
  };
  reader.readAsText(file);
  e.target.value = "";
});

// ================================================================
// WS
// ================================================================
let wsRetry=0;
function connectWS() {
  const dot=document.getElementById("ws-dot"),lbl=document.getElementById("ws-label");
  dot.className="status-dot connecting"; lbl.textContent="Подключение...";
  const ws=new WebSocket("ws://127.0.0.1:8000/ws");
  ws.onopen=()=>{dot.className="status-dot connected";lbl.textContent="Подключено";wsRetry=0;};
  ws.onclose=()=>{dot.className="status-dot error";lbl.textContent="Нет соединения";setTimeout(connectWS,Math.min(5000,1000*++wsRetry));};
}
connectWS();

// ================================================================
// RENDER
// ================================================================
let fc=0,lastFPS=performance.now();
(function animate(){
  requestAnimationFrame(animate);
  renderer.render(scene3d,cam3d);
  if(++fc%60===0){
    document.getElementById("footer-fps").textContent=Math.round(60000/(performance.now()-lastFPS))+" FPS";
    lastFPS=performance.now();
  }
})();