"use strict";

const $ = (id) => document.getElementById(id);
const COORDINATE_LIMIT = 10;
const SIMULATION_TIME_LIMIT = 15;
const ASSET_VERSION = "20261030-greek-labels";
const PY_ROOT = "/non-traversable-reach-avoid/scripts";
const PY_FILES = [
  "barrier/geometry_common.py", "barrier/circle_barrier.py", "barrier/polygon_barrier.py",
  "strategy/otp_solver.py", "strategy/equilibrium_strategy.py",
  "strategy/barrier_classifier.py", "simulation/simulation.py", "api/web_api.py"
];

const refs = {
  geometry: $("geometryCanvas"), value: $("valueCanvas"), message: $("canvasMessage"),
  solverStatus: $("solverStatus"), barrierPanel: $("barrierPanel"),
  simulationPanel: $("simulationPanel"), valuePanel: $("valuePanel"),
  barrierInfo: $("barrierInfo"), simInfo: $("simInfo"),
  currentValue: $("currentValue"), currentTime: $("currentTime"),
  currentStatus: $("currentStatus"), playPause: $("playPauseSim"),
  joystickPanel: $("joystickPanel"), joystick: $("virtualJoystick"),
  joystickKnob: $("joystickKnob"), joystickPlayer: $("joystickPlayer"),
  joystickAngle: $("joystickAngle")
};

const app = {
  pyodide: null, mode: "barrier", barrier: null, sim: null, running: false,
  computingBarrier: false, barrierQueued: false, drag: null,
  lastStepWall: 0, stepping: false, transform: null,
  manualPlayer: null, manualHeading: null, manualSpeed: 0, joystickPointer: null,
  views: {barrier:null, simulation:null},
  displayPItems: [],
  polygon: [[-1.15,-.65],[1.15,-.65],[1.25,.55],[.25,1.15],[-1.05,.75]]
};

function number(id) { return Number($(id).value); }
function setNumber(id, value) { $(id).value = String(Number(Number(value).toFixed(9))); }
function clampCoordinate(value) { return Math.max(-COORDINATE_LIMIT, Math.min(COORDINATE_LIMIT, value)); }
function point(x, y) { return [clampCoordinate(number(x)), clampCoordinate(number(y))]; }
function finitePoint(p) { return p.length === 2 && p.every(Number.isFinite); }
function fmtPoint(p) { return p ? `(${p[0].toFixed(4)}, ${p[1].toFixed(4)})` : "—"; }

function setDefaults() {
  setNumber("bD1x", 0); setNumber("bD1y", 1.5); setNumber("bD2x", -2); setNumber("bD2y", 0);
  setNumber("bRadius", 1); setNumber("bAlpha", 2/3); setNumber("bBeta", 1);
  setNumber("sAx", -0.406422989); setNumber("sAy", -3.339212412);
  setNumber("sD1x", 0); setNumber("sD1y", 1.5);
  setNumber("sD2x", -2); setNumber("sD2y", 0); setNumber("sRadius", 1);
  setNumber("sVA", .2); setNumber("sVD1", .3); setNumber("sVD2", .3); setNumber("simDt", .05);
  syncPolygonText(); updateBarrierSpeedSummary(); updatePlaybackUI();
}

function updateBarrierSpeedSummary() {
  $("barrierSpeedSummary").textContent = `α=${number("bAlpha").toFixed(3)}\u00a0\u00a0β=${number("bBeta").toFixed(3)}`;
}

function updatePlaybackUI() {
  const running=app.running && !app.sim?.terminated;
  refs.playPause.querySelector("span:first-child").textContent=running ? "Ⅱ" : "▶";
  refs.playPause.querySelector(".button-label").textContent=running ? "Pause" : "Play";
  refs.playPause.setAttribute("aria-label",running ? "Pause simulation" : "Start simulation");
}

function updateJoystick() {
  const active=app.manualPlayer!==null;
  refs.joystickPanel.classList.toggle("hidden",!active);
  if(active)refs.joystickPlayer.textContent=app.manualPlayer;
  const angle=app.manualHeading,speed=app.manualSpeed;
  const radius=35;
  refs.joystickKnob.style.left=angle===null?"50%":`calc(50% + ${radius*speed*Math.cos(angle)}px)`;
  refs.joystickKnob.style.top=angle===null?"50%":`calc(50% - ${radius*speed*Math.sin(angle)}px)`;
  const text=angle===null?"Centered":`${Math.round(angle*180/Math.PI)}° · ${Math.round(speed*100)}%`;
  refs.joystickAngle.textContent=text;
  refs.joystick.setAttribute("aria-valuetext",text);
  refs.joystick.setAttribute("aria-valuenow",String(Math.round(speed*100)));
}

function setJoystickFromPointer(event) {
  const rect=refs.joystick.getBoundingClientRect();
  const dx=event.clientX-(rect.left+rect.width/2);
  const dy=event.clientY-(rect.top+rect.height/2);
  const radius=35,deadZone=.08;
  const rawSpeed=Math.min(1,Math.hypot(dx,dy)/radius);
  app.manualSpeed=Math.max(0,(rawSpeed-deadZone)/(1-deadZone));
  app.manualHeading=app.manualSpeed===0?null:Math.atan2(-dy,dx);
  updateJoystick();
}

function centerJoystick() {
  app.manualHeading=null;app.manualSpeed=0;updateJoystick();
}

function barrierConfig() {
  return {D1: point("bD1x","bD1y"), D2: point("bD2x","bD2y"), R:number("bRadius"),
    alpha:number("bAlpha"), beta:number("bBeta"), resolution:220,
    vertices:app.polygon, samples_per_edge:45, grid_size:150};
}

function simulationConfig() {
  return {A:point("sAx","sAy"), D1:point("sD1x","sD1y"), D2:point("sD2x","sD2y"),
    R:number("sRadius"), v_A:number("sVA"), v_D1:number("sVD1"), v_D2:number("sVD2"), t_max:SIMULATION_TIME_LIMIT,
    barrier_resolution:180};
}

function validConvex(vertices) {
  if (vertices.length < 3 || !vertices.every(finitePoint)) return false;
  let sign = 0;
  for (let i=0; i<vertices.length; i++) {
    const a=vertices[i], b=vertices[(i+1)%vertices.length], c=vertices[(i+2)%vertices.length];
    const cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);
    if (Math.abs(cross) < 1e-10) continue;
    const current=Math.sign(cross); if (sign && current !== sign) return false; sign=current;
  }
  return sign !== 0;
}

function syncPolygonText() { $("polygonText").value = app.polygon.map(p => `${p[0].toFixed(3)}, ${p[1].toFixed(3)}`).join("\n"); }
function parsePolygon() {
  return $("polygonText").value.trim().split(/\n+/).map(line => line.trim().split(/[\s,]+/).map(Number));
}

async function pythonCall(expression, input) {
  const call = expression.replace("_web_input", JSON.stringify(input));
  return JSON.parse(await app.pyodide.runPythonAsync(call));
}

let barrierTimer = null;
function scheduleBarrier(delay=80) {
  if (!app.pyodide || app.mode !== "barrier") return;
  clearTimeout(barrierTimer); barrierTimer = setTimeout(computeBarrier, delay);
}

async function computeBarrier() {
  if (app.computingBarrier) { app.barrierQueued = true; return; }
  const config = barrierConfig();
  const shape = $("goalShape").value;
  if (![config.D1,config.D2].every(finitePoint) || ![config.R,config.alpha,config.beta].every(Number.isFinite)) return;
  if (shape === "polygon" && !validConvex(app.polygon)) {
    app.barrier = null;app.displayPItems=[];showMessage("The target polygon must be convex.");
    refs.barrierInfo.textContent = "Invalid geometry"; return;
  }
  app.computingBarrier = true; showMessage("Computing barrier…");
  try {
    const fn = shape === "circle" ? "web_api.compute_circle_barrier(_web_input)" : "web_api.compute_polygon_barrier(_web_input)";
    app.barrier = await pythonCall(fn, JSON.stringify(config));
    app.displayPItems = displayPItems(app.barrier, config);
    if (!app.views.barrier) app.views.barrier=viewFromPoints(barrierViewPoints(config,app.barrier));
    hideMessage();
    const b=app.barrier;
    const pPositions=app.displayPItems.length
      ? app.displayPItems.map(item=>`${item.displayLabel.padEnd(8)} ${fmtPoint(item.point)}`).join("\n")
      : "p        —";
    refs.barrierInfo.textContent = `D1       ${fmtPoint(config.D1)}\nD2       ${fmtPoint(config.D2)}\n${pPositions}\nsolve    ${b.computation_time.toFixed(3)} s`;
  } catch (error) {
    app.barrier=null;app.displayPItems=[];showMessage(cleanError(error));refs.barrierInfo.textContent="Barrier computation failed";
  } finally {
    app.computingBarrier=false;
    if (app.barrierQueued) { app.barrierQueued=false; scheduleBarrier(20); }
  }
}

function cleanError(error) {
  const text=String(error?.message || error).trim();
  return text.split("\n").at(-1).replace(/^(PythonError:\s*)?(ValueError|RuntimeError):\s*/,"") || "Solver error";
}
function showMessage(text) { refs.message.textContent=text; refs.message.classList.add("show"); }
function hideMessage() { refs.message.classList.remove("show"); }

let simulationGeneration=0, simulationTimer=null;
function scheduleSimulationReset(delay=80) {
  clearTimeout(simulationTimer); simulationTimer=setTimeout(resetSimulation,delay);
}

async function resetSimulation() {
  const generation=++simulationGeneration;
  app.running=false; app.stepping=false;
  updatePlaybackUI();
  try {
    const result=await pythonCall("web_api.init_simulation(_web_input)", JSON.stringify(simulationConfig()));
    if(generation!==simulationGeneration)return;
    app.sim=result;
    if (!app.views.simulation) app.views.simulation=viewFromPoints(simulationViewPoints(simulationConfig(),app.sim));
    setNumber("sAx",app.sim.A[0]);setNumber("sAy",app.sim.A[1]);
    if(app.sim.attacker_constrained) showMessage("Attacker projected to the current D1/D2 barrier.");
    else hideMessage();
    updateSimInfo();
  } catch(error) { if(generation===simulationGeneration){app.sim=null;showMessage(cleanError(error));refs.simInfo.textContent="Initialization failed";refs.currentTime.textContent="—";refs.currentStatus.textContent="Initialization failed";} }
}

async function stepSimulation() {
  if (app.stepping || !app.sim || app.sim.terminated) { if (app.sim?.terminated) app.running=false; return; }
  app.stepping=true;
  try {
    const liveBarrier=$("liveSimulationBarrier").checked ? "True" : "False";
    const manualPlayer=app.manualPlayer===null?"None":JSON.stringify(app.manualPlayer);
    const manualHeading=app.manualHeading===null?"None":String(app.manualHeading);
    app.sim=JSON.parse(await app.pyodide.runPythonAsync(`web_api.simulation_step(${number("simDt")}, ${liveBarrier}, ${manualPlayer}, ${manualHeading}, ${app.manualSpeed})`));
    updateSimInfo(); if (app.sim.terminated) { app.running=false; updatePlaybackUI(); }
  } catch(error) { app.running=false; showMessage(cleanError(error)); }
  finally { app.stepping=false; }
}

function updateSimInfo() {
  if (!app.sim) return;
  const s=app.sim;
  const sigma=s.lift_pair ? `(${s.lift_pair.map(v=>v>0?"+1":"−1").join(", ")})` : "—";
  refs.simInfo.textContent = `t      ${s.time.toFixed(4)}\nA      ${fmtPoint(s.A)}\nD1     ${fmtPoint(s.D1)}\nD2     ${fmtPoint(s.D2)}\nσ      ${sigma}\nV      ${s.V == null ? "—" : s.V.toFixed(7)}`;
  refs.currentTime.textContent=s.time.toFixed(3);
  refs.currentStatus.textContent=s.result || (app.running ? "Running" : s.time ? "Paused" : "Ready");
  refs.currentValue.textContent=`V(t) = ${s.V == null ? "—" : s.V.toFixed(3)}`;
  updatePlaybackUI();
}

function switchTab(mode) {
  app.mode=mode; app.running=false;
  document.body.dataset.mode=mode;
  document.querySelectorAll(".tab").forEach(b=>{const active=b.dataset.tab===mode;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active));});
  refs.barrierPanel.classList.toggle("hidden",mode!=="barrier");
  refs.simulationPanel.classList.toggle("hidden",mode!=="simulation");
  refs.valuePanel.classList.toggle("hidden",mode!=="simulation");
  updatePlaybackUI(); hideMessage(); if (mode==="barrier") scheduleBarrier(0); else if (!app.sim) resetSimulation();
}

function resizeCanvas(canvas) {
  const dpr=Math.min(devicePixelRatio || 1,2), rect=canvas.getBoundingClientRect();
  const w=Math.max(1,Math.round(rect.width*dpr)), h=Math.max(1,Math.round(rect.height*dpr));
  if (canvas.width!==w || canvas.height!==h) { canvas.width=w; canvas.height=h; }
  return {ctx:canvas.getContext("2d"), width:w, height:h, dpr};
}

function viewFromPoints(points) {
  const valid=points.filter(finitePoint);
  if (!valid.length) return {cx:0,cy:0,span:8};
  const xs=valid.map(p=>p[0]),ys=valid.map(p=>p[1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const rawSpan=Math.max(maxX-minX,maxY-minY,2);
  return {cx:(minX+maxX)/2,cy:(minY+maxY)/2,span:1.26*rawSpan+.4};
}

function barrierViewPoints(config,barrier=app.barrier) {
  const points=[config.D1,config.D2];
  if($("goalShape").value==="circle")points.push([-config.R,-config.R],[config.R,config.R]);
  else points.push(...app.polygon);
  if(barrier){points.push(...barrier.barrier);barrier.p_items.forEach(item=>points.push(item.point));}
  return points;
}

function simulationViewPoints(config,simulation=app.sim) {
  const points=[[-config.R,-config.R],[config.R,config.R],config.A,config.D1,config.D2];
  if(simulation){points.push(...simulation.barrier);["A","D1","D2"].forEach(k=>points.push(...simulation.trajectory[k]));if(simulation.otp)points.push(simulation.otp);}
  return points;
}

function makeTransform(width,height,view) {
  const pad=42, usableW=width-2*pad, usableH=height-2*pad;
  const scale=Math.min(usableW,usableH)/view.span;
  const {cx,cy}=view;
  return {scale,cx,cy,width,height,
    toCanvas(p){return [width/2+(p[0]-cx)*scale,height/2-(p[1]-cy)*scale]},
    toWorld(p){return [cx+(p[0]-width/2)/scale,cy-(p[1]-height/2)/scale]}};
}
function worldToCanvas(x,y) { return app.transform.toCanvas([x,y]); }
function canvasToWorld(px,py) { return app.transform.toWorld([px,py]); }

function drawAxes(ctx,t) {
  ctx.save(); ctx.strokeStyle="#e6ebee"; ctx.lineWidth=1;
  for (let x=Math.ceil(t.cx-(t.width/2)/t.scale); x<=t.cx+(t.width/2)/t.scale; x++) line(ctx,t,[x,-1e3],[x,1e3]);
  for (let y=Math.ceil(t.cy-(t.height/2)/t.scale); y<=t.cy+(t.height/2)/t.scale; y++) line(ctx,t,[-1e3,y],[1e3,y]);
  ctx.strokeStyle="#aab6bd"; ctx.lineWidth=1.3; line(ctx,t,[0,-1e3],[0,1e3]); line(ctx,t,[-1e3,0],[1e3,0]); ctx.restore();
}
function line(ctx,t,a,b) { const p=t.toCanvas(a),q=t.toCanvas(b);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke(); }
function polyline(ctx,t,pts,style="#222",width=2,dash=[]) {
  if (!pts || pts.length<2) return; ctx.save();ctx.strokeStyle=style;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();
  pts.forEach((p,i)=>{const q=t.toCanvas(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.restore();
}
function circle(ctx,t,center,r,fill,stroke="#26343c",width=1.5) { const p=t.toCanvas(center);ctx.beginPath();ctx.arc(p[0],p[1],Math.abs(r*t.scale),0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke(); }
function marker(ctx,t,p,color,label,shape="circle") {
  const q=t.toCanvas(p),isP=label.startsWith("p_{");ctx.save();ctx.fillStyle=color;ctx.strokeStyle="white";ctx.lineWidth=isP?1.4:2;ctx.beginPath();
  if(shape==="star") { for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,r=i%2?4:9,x=q[0]+r*Math.cos(a),y=q[1]+r*Math.sin(a);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.closePath(); }
  else ctx.arc(q[0],q[1],isP?4:7,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle="#15242d";
  const subscript=label.match(/^p_\{(.+)\}$/);
  if(subscript){ctx.font="italic 600 12px system-ui";ctx.fillText("p",q[0]+10,q[1]-9);ctx.font="600 9px system-ui";ctx.fillText(subscript[1],q[0]+17,q[1]-5);}
  else {ctx.font="600 12px system-ui";ctx.fillText(label,q[0]+10,q[1]-9);}
  ctx.restore();
}

const sourceColors={D1:"#276fbf",D2:"#1597a5",p1:"#2b9a66",p2:"#805ad5",p3:"#a25d2f",p4:"#d05b83"};
function liftedAngle(defender,point,side) {
  const tau=2*Math.PI,target=Math.atan2(point[1],point[0]),angle=Math.atan2(defender[1],defender[0]);
  const lower=angle+tau*Math.floor((target-angle)/tau);
  return lower+(side>0?tau:0);
}

function displayPItems(barrier,config) {
  const activeSources=new Set(barrier.segments.map(segment=>segment.source));
  const active=barrier.p_items.filter(item=>activeSources.has(item.label)).map(item=>{
    const sides=item.mode
      ? item.mode.map(side=>side==="+"?1:-1)
      : item.lift_pair;
    const d1=liftedAngle(config.D1,item.point,sides[0]);
    const d2=liftedAngle(config.D2,item.point,sides[1]);
    return {...item,displayGroup:d1<=d2+1e-12?1:2};
  });
  for(const group of [1,2]) {
    const members=active.filter(item=>item.displayGroup===group);
    members.forEach((item,index)=>item.displayLabel=members.length>1?`p_{${group},${index+1}}`:`p_{${group}}`);
  }
  return active;
}

function barrierDetailSectors(barrier) {
  const sectors=[];
  const segments=barrier.segments,n=segments.length;
  for(let index=0;index<n;index++) {
    const segment=segments[index];
    const item=app.displayPItems.find(candidate=>candidate.label===segment.source);
    if(!item || segment.points.length<2)continue;
    const switches=[];
    if(["D1","D2"].includes(segments[(index-1+n)%n].source))switches.push(segment.points[0]);
    if(["D1","D2"].includes(segments[(index+1)%n].source))switches.push(segment.points.at(-1));
    sectors.push({item,arc:segment.points,switches});
  }
  return sectors;
}

function drawSector(ctx,t,center,arc,color) {
  const origin=t.toCanvas(center);ctx.save();ctx.fillStyle=color;ctx.globalAlpha=.07;ctx.beginPath();ctx.moveTo(...origin);
  arc.forEach(point=>ctx.lineTo(...t.toCanvas(point)));ctx.closePath();ctx.fill();ctx.restore();
}

function switchMarker(ctx,t,point,color) {
  const q=t.toCanvas(point);ctx.save();ctx.fillStyle="#fff";ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],4,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.restore();
}

function renderBarrier(ctx,w,h) {
  const c=barrierConfig(), shape=$("goalShape").value;
  const view=app.views.barrier||viewFromPoints(barrierViewPoints(c));
  app.transform=makeTransform(w,h,view); drawAxes(ctx,app.transform);
  if(shape==="circle") circle(ctx,app.transform,[0,0],c.R,"#e1e5e7");
  else { const closed=[...app.polygon,app.polygon[0]]; ctx.fillStyle="#e1e5e7";const p0=app.transform.toCanvas(closed[0]);ctx.beginPath();ctx.moveTo(...p0);closed.slice(1).forEach(p=>ctx.lineTo(...app.transform.toCanvas(p)));ctx.fill();polyline(ctx,app.transform,closed,"#26343c",1.5); }
  if(app.barrier) {
    const pColors=Object.fromEntries(app.displayPItems.map(item=>[item.label,item.displayGroup===1?"#2b9a66":"#805ad5"]));
    const showDetails=$("showAux").checked;
    const sectors=showDetails?barrierDetailSectors(app.barrier):[];
    if(showDetails) {
      Object.entries(app.barrier.individual).forEach(([name,p])=>polyline(ctx,app.transform,p,sourceColors[name],1,[5,4]));
      sectors.forEach(({item,arc})=>drawSector(ctx,app.transform,item.point,arc,pColors[item.label]));
    }
    app.barrier.segments.forEach(s=>polyline(ctx,app.transform,s.points,pColors[s.source]||sourceColors[s.source]||"#543d8b",3));
    if(showDetails)sectors.forEach(({item,switches})=>{
      const color=pColors[item.label];marker(ctx,app.transform,item.point,"#222",item.displayLabel);
      switches.forEach(point=>{polyline(ctx,app.transform,[item.point,point],color,1.2);switchMarker(ctx,app.transform,point,color);});
    });
  }
  marker(ctx,app.transform,c.D1,"#276fbf","D1"); marker(ctx,app.transform,c.D2,"#1597a5","D2");
  if(shape==="polygon") app.polygon.forEach((p,i)=>marker(ctx,app.transform,p,"#59666e",String(i+1)));
}

function renderSimulation(ctx,w,h) {
  const c=simulationConfig(), s=app.sim;
  const view=app.views.simulation||viewFromPoints(simulationViewPoints(c,s));
  app.transform=makeTransform(w,h,view);drawAxes(ctx,app.transform);circle(ctx,app.transform,[0,0],c.R,"#e1e5e7");
  if(s) {
    polyline(ctx,app.transform,s.barrier,s.barrier_is_live?"#c06b18":"#a66321",s.barrier_is_live?3.2:2.6,s.barrier_is_live?[]:[8,4]);
    polyline(ctx,app.transform,s.trajectory.A,"#c8473d",2.5);polyline(ctx,app.transform,s.trajectory.D1,"#276fbf",2.5);polyline(ctx,app.transform,s.trajectory.D2,"#1597a5",2.5);
    if(s.otp) marker(ctx,app.transform,s.otp,"#202b31","OTP","star");
    marker(ctx,app.transform,s.A,"#c8473d","A");
    marker(ctx,app.transform,app.drag==="sD1"?c.D1:s.D1,"#276fbf","D1");
    marker(ctx,app.transform,app.drag==="sD2"?c.D2:s.D2,"#1597a5","D2");
    if(s.headings) Object.entries(s.headings).forEach(([k,a])=>{if(a==null)return;const p=s[k],q=[p[0]+.3*Math.cos(a),p[1]+.3*Math.sin(a)];ctx.strokeStyle=k==="A"?"#c8473d":k==="D1"?"#276fbf":"#1597a5";line(ctx,app.transform,p,q)});
  }
}

let lastValueRender=0;
function renderValue(now) {
  if(app.mode!=="simulation" || refs.valuePanel.classList.contains("collapsed") || now-lastValueRender<40)return;
  lastValueRender=now;
  const {ctx,width:w,height:h,dpr}=resizeCanvas(refs.value), scale=dpr;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const ts=app.sim?.value_times||[],vs=app.sim?.value_history||[];
  const pad={l:48*scale,r:18*scale,t:14*scale,b:30*scale};
  const plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
  const tMax=Math.max(SIMULATION_TIME_LIMIT,ts.at(-1)||0,1);
  let vMin=vs.length?Math.min(0,...vs):-.05,vMax=vs.length?Math.max(0,...vs):.05;
  if(vMax-vMin<.05){const center=(vMax+vMin)/2;vMin=center-.025;vMax=center+.025;}
  const xy=(t,v)=>[pad.l+t/tMax*plotW,pad.t+(vMax-v)/(vMax-vMin)*plotH];

  ctx.strokeStyle="#b8bdb9";ctx.lineWidth=scale;ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  const zero=xy(0,0)[1];ctx.strokeStyle="#c9cdca";ctx.setLineDash([4*scale,4*scale]);ctx.beginPath();ctx.moveTo(pad.l,zero);ctx.lineTo(w-pad.r,zero);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#6b716d";ctx.font=`${11*scale}px system-ui`;ctx.textBaseline="middle";ctx.textAlign="left";
  for(const value of [vMin,(vMin+vMax)/2,vMax]){const y=xy(0,value)[1];ctx.fillText(value.toFixed(2),4*scale,y);}
  ctx.textBaseline="top";
  for(const value of [0,tMax/2,tMax]){const x=xy(value,vMin)[0];ctx.textAlign=value===0?"left":value===tMax?"right":"center";ctx.fillText(value.toFixed(value?1:0),x,h-pad.b+7*scale);}
  ctx.textAlign="left";ctx.fillText("V",pad.l+6*scale,3*scale);ctx.textAlign="right";ctx.fillText("t",w-4*scale,h-pad.b+7*scale);
  if(!ts.length)return;
  ctx.strokeStyle="#176b68";ctx.lineWidth=2*scale;ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();
  vs.forEach((value,index)=>{const p=xy(ts[index],value);index?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.stroke();
  const current=xy(ts.at(-1),vs.at(-1));ctx.fillStyle="#176b68";ctx.beginPath();ctx.arc(...current,4*scale,0,Math.PI*2);ctx.fill();
}

function render(now=performance.now()) {
  const {ctx,width,height}=resizeCanvas(refs.geometry);ctx.clearRect(0,0,width,height);ctx.fillStyle="#fff";ctx.fillRect(0,0,width,height);
  app.mode==="barrier"?renderBarrier(ctx,width,height):renderSimulation(ctx,width,height);renderValue(now);
  if(app.running && !app.stepping) { const interval=1000*number("simDt"); if(now-app.lastStepWall>=interval){app.lastStepWall=now;stepSimulation();} }
  requestAnimationFrame(render);
}

function pointerPosition(event) { const r=refs.geometry.getBoundingClientRect(), sx=refs.geometry.width/r.width, sy=refs.geometry.height/r.height;return [(event.clientX-r.left)*sx,(event.clientY-r.top)*sy]; }
refs.geometry.addEventListener("wheel",e=>{
  if(!app.transform)return;
  e.preventDefault();
  const canvasPoint=pointerPosition(e),anchor=app.transform.toWorld(canvasPoint);
  const current=app.views[app.mode]||{cx:app.transform.cx,cy:app.transform.cy,span:Math.min(app.transform.width-84,app.transform.height-84)/app.transform.scale};
  const span=Math.max(.2,Math.min(80,current.span*Math.exp(e.deltaY*.001)));
  const scale=Math.min(app.transform.width-84,app.transform.height-84)/span;
  app.views[app.mode]={
    cx:anchor[0]-(canvasPoint[0]-app.transform.width/2)/scale,
    cy:anchor[1]+(canvasPoint[1]-app.transform.height/2)/scale,
    span,
  };
},{passive:false});
function nearestDraggable(canvasPoint) {
  const candidates=[];
  if(app.mode==="barrier") { const c=barrierConfig();candidates.push(["bD1",c.D1],["bD2",c.D2]);if($("goalShape").value==="polygon")app.polygon.forEach((p,i)=>candidates.push([`vertex:${i}`,p])); }
  else if(!app.running && (!app.sim || app.sim.time===0)) { const c=simulationConfig();candidates.push(["sA",c.A],["sD1",c.D1],["sD2",c.D2]); }
  let best=null,distance=18;for(const [name,p] of candidates){const q=app.transform.toCanvas(p),d=Math.hypot(q[0]-canvasPoint[0],q[1]-canvasPoint[1]);if(d<distance){best=name;distance=d}}return best;
}
refs.geometry.addEventListener("pointerdown",e=>{if(!app.pyodide)return;const p=pointerPosition(e);app.drag=nearestDraggable(p);if(app.drag){if(app.drag==="sD1"||app.drag==="sD2")clearTimeout(simulationTimer);refs.geometry.setPointerCapture(e.pointerId);e.preventDefault();}});
refs.geometry.addEventListener("pointermove",e=>{if(!app.drag)return;const p=canvasToWorld(...pointerPosition(e));
  const bounded=p.map(clampCoordinate);
  if(app.drag.startsWith("vertex:")){app.polygon[Number(app.drag.split(":")[1])]=bounded;syncPolygonText();scheduleBarrier();}
  else { const map={bD1:["bD1x","bD1y"],bD2:["bD2x","bD2y"],sA:["sAx","sAy"],sD1:["sD1x","sD1y"],sD2:["sD2x","sD2y"]};setNumber(map[app.drag][0],bounded[0]);setNumber(map[app.drag][1],bounded[1]);if(app.mode==="barrier")scheduleBarrier();else if(app.drag==="sA")scheduleSimulationReset(); }
});
function finishDrag() { const dragged=app.drag;app.drag=null;if(dragged?.startsWith("s"))scheduleSimulationReset(0); }
refs.geometry.addEventListener("pointerup",finishDrag);refs.geometry.addEventListener("pointercancel",finishDrag);

function resetBarrierInputs() {
  if($("goalShape").value==="circle") { setNumber("bD1x",0);setNumber("bD1y",1.5);setNumber("bD2x",-2);setNumber("bD2y",0); }
  else { setNumber("bD1x",-2.35);setNumber("bD1y",2.05);setNumber("bD2x",2.45);setNumber("bD2y",1.95);app.polygon=[[-1.15,-.65],[1.15,-.65],[1.25,.55],[.25,1.15],[-1.05,.75]];syncPolygonText(); }
  setNumber("bRadius",1);setNumber("bAlpha",2/3);setNumber("bBeta",1);
  app.views.barrier=null;updateBarrierSpeedSummary();scheduleBarrier(0);
}

function bindUI() {
  const coordinateInputs=["bD1x","bD1y","bD2x","bD2y","sAx","sAy","sD1x","sD1y","sD2x","sD2y"];
  coordinateInputs.forEach(id=>$(id).addEventListener("input",()=>{const value=number(id);if(Number.isFinite(value)&&value!==clampCoordinate(value))setNumber(id,clampCoordinate(value));}));
  document.querySelectorAll(".tab").forEach(b=>b.addEventListener("click",()=>switchTab(b.dataset.tab)));
  $("goalShape").addEventListener("change",()=>{$("polygonControls").classList.toggle("hidden",$("goalShape").value!=="polygon");$("radiusLabel").classList.toggle("hidden",$("goalShape").value!=="circle");app.barrier=null;app.views.barrier=null;scheduleBarrier(0)});
  ["bD1x","bD1y","bD2x","bD2y","bRadius","bAlpha","bBeta"].forEach(id=>$(id).addEventListener("input",()=>{updateBarrierSpeedSummary();scheduleBarrier();}));
  $("showAux").addEventListener("change",()=>{});
  $("resetBarrier").addEventListener("click",resetBarrierInputs);
  $("applyPolygon").addEventListener("click",()=>{const p=parsePolygon();if(validConvex(p)){app.polygon=p;syncPolygonText();app.views.barrier=null;scheduleBarrier(0)}else{app.barrier=null;showMessage("The target polygon must be convex.")}});
  $("addVertex").addEventListener("click",()=>{const a=app.polygon.at(-1),b=app.polygon[0];app.polygon.push([(a[0]+b[0])/2,(a[1]+b[1])/2]);syncPolygonText();showMessage("Move the new vertex, then Apply vertices.")});
  $("removeVertex").addEventListener("click",()=>{if(app.polygon.length>3){app.polygon.pop();syncPolygonText();scheduleBarrier(0)}});
  ["sAx","sAy","sD1x","sD1y","sD2x","sD2y","sRadius","sVA","sVD1","sVD2"].forEach(id=>$(id).addEventListener("change",resetSimulation));
  document.querySelectorAll("[data-manual-player]").forEach(input=>input.addEventListener("change",()=>{
    document.querySelectorAll("[data-manual-player]").forEach(other=>{if(other!==input)other.checked=false});
    app.manualPlayer=input.checked?input.dataset.manualPlayer:null;
    centerJoystick();
  }));
  refs.joystick.addEventListener("pointerdown",event=>{
    refs.joystick.classList.add("is-active");refs.joystick.setPointerCapture(event.pointerId);app.joystickPointer=event.pointerId;setJoystickFromPointer(event);event.preventDefault();
  });
  refs.joystick.addEventListener("pointermove",event=>{if(event.pointerId===app.joystickPointer)setJoystickFromPointer(event)});
  const releaseJoystick=event=>{if(event.pointerId===app.joystickPointer){app.joystickPointer=null;refs.joystick.classList.remove("is-active");centerJoystick()}};
  refs.joystick.addEventListener("pointerup",releaseJoystick);
  refs.joystick.addEventListener("pointercancel",releaseJoystick);
  refs.joystick.addEventListener("keydown",event=>{
    const headings={ArrowRight:0,ArrowUp:Math.PI/2,ArrowLeft:Math.PI,ArrowDown:-Math.PI/2};
    if(event.key in headings){app.manualHeading=headings[event.key];app.manualSpeed=1;updateJoystick();event.preventDefault()}
    else if(event.key===" "||event.key==="Escape"){centerJoystick();event.preventDefault()}
  });
  refs.joystick.addEventListener("keyup",event=>{if(event.key.startsWith("Arrow")){centerJoystick();event.preventDefault()}});
  $("liveSimulationBarrier").addEventListener("change",updateSimInfo);
  refs.playPause.addEventListener("click",async()=>{
    if(app.running){app.running=false;updateSimInfo();return;}
    if(!app.sim||app.sim.terminated)await resetSimulation();
    if(!app.sim)return;
    app.running=true;app.lastStepWall=performance.now();updateSimInfo();
  });
  $("stepSim").addEventListener("click",async()=>{app.running=false;updatePlaybackUI();await stepSimulation()});
  $("resetSim").addEventListener("click",resetSimulation);
  $("valueToggle").addEventListener("click",()=>{
    const collapsed=refs.valuePanel.classList.toggle("collapsed");
    $("valueToggle").setAttribute("aria-expanded",String(!collapsed));
    if(!collapsed)lastValueRender=0;
  });
}

function setSolverStatus(text,state) {
  refs.solverStatus.className=`solver-status ${state}`;
  refs.solverStatus.querySelector("span:last-child").textContent=text;
}

async function initialize() {
  setDefaults();bindUI();requestAnimationFrame(render);
  try {
    app.pyodide=await loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"});
    await app.pyodide.loadPackage("numpy");app.pyodide.FS.mkdirTree(PY_ROOT);
    for(const directory of ["barrier","strategy","simulation","api"])app.pyodide.FS.mkdirTree(`${PY_ROOT}/${directory}`);
    await Promise.all(PY_FILES.map(async name=>{const url=new URL(`scripts/${name}`,document.baseURI);url.searchParams.set("v",ASSET_VERSION);const response=await fetch(url,{cache:"no-store"});if(!response.ok)throw new Error(`Could not load scripts/${name}`);app.pyodide.FS.writeFile(`${PY_ROOT}/${name}`,await response.text())}));
    await app.pyodide.runPythonAsync(`import inspect, sys\nfor name in ('barrier','strategy','simulation','api'):\n    sys.path.insert(0, '${PY_ROOT}/' + name)\nimport web_api\nif len(inspect.signature(web_api.simulation_step).parameters) < 5:\n    raise RuntimeError('incompatible cached web_api.py')`);
    document.querySelectorAll("[data-solver-control]").forEach(x=>x.disabled=false);
    setSolverStatus("Solver ready","ready");hideMessage();await computeBarrier();
  } catch(error) { setSolverStatus("Solver failed","error");showMessage(`Pyodide initialization failed: ${cleanError(error)}`); }
}

window.addEventListener("DOMContentLoaded",initialize);
