"""Fly Tennis page, v2 renderer: floodlit night court, PBR materials with a procedural
environment map, textured specimen, felt ball, metal racket, bloom, a racket close-up
panel, neuron morphology in the brain panel and a readout strip.

usage: build_tennis_page_v2.py <bump.json> <contrast.json> <fly_mesh.json> <out.html>
"""
import json, sys, re, base64
import numpy as np
from pathlib import Path

tr = json.loads(Path(sys.argv[1]).read_text()); un = json.loads(Path(sys.argv[2]).read_text())
mesh = Path(sys.argv[3]).read_text(); out = Path(sys.argv[4])
NV = tr["n_viz"]; F = len(tr["frames"]); A = np.frombuffer(base64.b64decode(tr["act"]), dtype=np.uint8).reshape(F, NV)
tr["act"] = base64.b64encode(A[::2].tobytes()).decode(); tr["act_stride"] = 2
un_small = {k: un[k] for k in ("frames", "events", "summary", "chans", "rec_hz")}
payload = json.dumps({"trained": tr, "untrained": un_small})

html = r"""<title>Fly Tennis</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#000;--panel:#050607;--line:rgba(255,255,255,.09);--text:#F2F4F6;--dim:#9AA3AE;--muted:#5E6873;--a:#5FD6E3;--b:#F58E6E;color-scheme:dark}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font-family:"Inter Tight",system-ui,sans-serif;font-size:13px;overflow:hidden}
#gl{position:fixed;inset:0;width:100%;height:100%;display:block}
.app{position:relative;display:grid;grid-template-columns:minmax(0,1.9fr) minmax(320px,1fr);grid-template-rows:minmax(0,1fr) 44px;height:100vh;pointer-events:none}
@media(max-width:900px){.app{grid-template-columns:1fr;grid-template-rows:52vh minmax(0,1fr) 44px}}
.main{position:relative;grid-column:1;grid-row:1;pointer-events:auto;cursor:grab}
.side{grid-column:2;grid-row:1;display:grid;grid-template-rows:1.15fr 1fr;border-left:1px solid var(--line);min-height:0}
@media(max-width:900px){.side{grid-column:1;grid-row:2;grid-template-rows:1fr;grid-template-columns:1fr 1fr;border-left:0;border-top:1px solid var(--line)}}
.panel{position:relative;min-height:0;border-bottom:1px solid var(--line)}
.panel:last-child{border-bottom:0}
.panel.brain{pointer-events:auto;cursor:grab}
.cap{position:absolute;left:18px;top:14px;right:18px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;pointer-events:none}
.cap .t{font-size:13px;color:var(--text);font-weight:500}
.cap .s{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
.cap .s.on{color:#fff}
.main .cap{left:22px;top:18px;right:22px}
.main .cap .t{font-size:14px}
.main .speed{font-family:"IBM Plex Mono",monospace;font-size:15px;color:#fff;font-variant-numeric:tabular-nums}
.main .speed small{font-size:10px;color:var(--dim);letter-spacing:.1em;margin-left:4px}
.main .slo{position:absolute;left:50%;top:18px;transform:translateX(-50%);font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#fff;opacity:0;transition:opacity .2s}
.main .slo.on{opacity:1}
.main .score{position:absolute;right:22px;top:46px;text-align:right;pointer-events:none}
.main .score .k{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.main .score .v{font-size:30px;font-weight:600;letter-spacing:-.02em;line-height:1;font-variant-numeric:tabular-nums;color:#fff}
.main .score .m{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--dim);margin-top:6px;font-variant-numeric:tabular-nums}
.main .brand{position:absolute;left:22px;bottom:14px;font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.main .brand b{color:var(--dim);font-weight:500}
.strip{grid-column:1/3;grid-row:2;display:flex;align-items:center;gap:0;border-top:1px solid var(--line);background:var(--panel);pointer-events:auto;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim);padding:0 14px 0 10px;overflow:hidden}
@media(max-width:900px){.strip{grid-column:1;grid-row:3}}
.strip .seg{display:flex;align-items:center;gap:8px;padding:0 12px;height:100%;border-right:1px solid var(--line);white-space:nowrap}
.strip .seg:last-child{border-right:0;margin-left:auto}
.strip .seg.grow{flex:1;min-width:120px}
.strip .k{letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-size:10px}
.strip .v{color:#fff;font-variant-numeric:tabular-nums;min-width:6ch;display:inline-block}
.strip button{border:1px solid rgba(255,255,255,.16);background:transparent;color:var(--dim);font:inherit;font-size:10.5px;letter-spacing:.06em;padding:4px 10px;border-radius:3px;cursor:pointer}
.strip button[aria-pressed=true]{background:#fff;color:#000;border-color:#fff}
.strip button:focus-visible{outline:2px solid var(--a);outline-offset:2px}
.strip .play{width:26px;height:26px;padding:0;border-radius:50%;display:grid;place-items:center}
.strip .track{flex:1;height:2px;background:rgba(255,255,255,.14);position:relative;cursor:pointer}
.strip .track i{position:absolute;left:0;top:0;bottom:0;background:#fff;width:0}
.strip .clock{min-width:9ch;text-align:right}
.tags{position:absolute;left:22px;bottom:44px;display:flex;gap:6px;pointer-events:none;font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em}
.tags span{padding:3px 8px;border-radius:2px;background:rgba(0,0,0,.6);color:var(--a);border:1px solid rgba(95,214,227,.45)}
.tags span.b{color:var(--b);border-color:rgba(245,142,110,.45)}
.err{position:absolute;left:22px;top:60px;color:#F58E6E;font-family:"IBM Plex Mono",monospace;font-size:11px;pointer-events:none}
@media(prefers-reduced-motion:reduce){.main .slo{transition:none}}
</style>

<canvas id="gl"></canvas>
<div class="app">
  <div class="main" id="main">
    <div class="cap"><div class="t" id="caption">Fly Tennis</div><div class="speed"><span id="ballv">0.0</span><small>MM/S</small></div></div>
    <div class="slo" id="slo">slow motion</div>
    <div class="score"><div class="k">rally</div><div class="v" id="rally">0</div><div class="m"><span id="hits"></span> hits · longest <span id="longest"></span></div></div>
    <div class="tags"><span>FLY A · CONNECTOME</span><span class="b">FLY B · CONNECTOME</span></div>
    <div class="brand"><b>MaleCNS v1.0</b> · 163,903 neurons as measured · no trained parameters · court <span id="dims"></span></div>
    <div class="err" id="err"></div>
  </div>
  <div class="side">
    <div class="panel brain" id="pbrain"><div class="cap"><div class="t">Neural activity · fly A</div><div class="s"><span id="p-viz"></span> of <span id="p-neu"></span> shown</div></div></div>
    <div class="panel" id="pclose"><div class="cap"><div class="t">Racket · fly A</div><div class="s" id="contact">approach</div></div></div>
  </div>
  <div class="strip">
    <div class="seg"><button class="play" id="play" aria-label="Play or pause">❚❚</button></div>
    <div class="seg grow"><div class="track" id="track"><i id="prog"></i></div><span class="clock" id="clock"></span></div>
    <div class="seg"><span class="k">bearing</span><span class="v" id="r-bear"></span></div>
    <div class="seg"><span class="k">walk</span><span class="v" id="r-walk"></span></div>
    <div class="seg"><span class="k">steer</span><button id="m-trained" aria-pressed="true">medulla bump</button><button id="m-untrained" aria-pressed="false">DNa contrast</button></div>
    <div class="seg"><button id="slow" aria-pressed="true">auto slow-mo</button></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/SMAAShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/SMAAPass.js"></script>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script id="flymesh" type="application/json">__MESH__</script>
<script>
(function(){
function fail(msg){const e=document.getElementById('err');if(e)e.textContent='render error: '+msg;console.error(msg);}
window.addEventListener('error',ev=>fail(ev.message||String(ev.error)));
if(typeof THREE==='undefined'){fail('three.js did not load');return;}
const HAS_POST=!!(THREE.EffectComposer&&THREE.UnrealBloomPass&&THREE.SMAAPass);
const ALL=JSON.parse(document.getElementById('payload').textContent),MESH=JSON.parse(document.getElementById('flymesh').textContent);
const T=ALL.trained,U=ALL.untrained,C=T.court,REC=T.rec_hz,AS=T.act_stride||1,SC=C.scale||1;
const fmt=n=>n.toLocaleString('en-US');
document.getElementById('p-neu').textContent=fmt(T.total_neurons);document.getElementById('p-viz').textContent=fmt(T.n_viz);
document.getElementById('dims').textContent=(C.half_l*2).toFixed(1)+' × '+(C.half_w*2).toFixed(1)+' mm';
function b64(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
function h2f(h){const s=(h&0x8000)>>15,e=(h&0x7C00)>>10,f=h&0x03FF;if(e===0)return(s?-1:1)*Math.pow(2,-14)*(f/1024);if(e===31)return f?NaN:(s?-Infinity:Infinity);return(s?-1:1)*Math.pow(2,e-15)*(1+f/1024);}
const NV=T.n_viz,pr=new Uint16Array(b64(T.pos).buffer),P=new Float32Array(NV*3);for(let i=0;i<NV*3;i++)P[i]=h2f(pr[i]);
const REG=b64(T.region),ACT=b64(T.act);
let mode='trained',frame=0,playing=true,autoSlow=true,slowNow=false;const RUN=()=>mode==='trained'?T:U,NF=()=>RUN().frames.length;
const params=new URLSearchParams(location.search);const RECORD=params.has('record');   // deterministic frame stepping for capture
const seed=(s=>()=>{s=(s*16807)%2147483647;return(s-1)/2147483646;})(42);

// ---------- renderer: one context, three viewports ----------
const cv=document.getElementById('gl');let R;try{R=new THREE.WebGLRenderer({canvas:cv,antialias:!HAS_POST,powerPreference:'high-performance'});}catch(e){fail('WebGL: '+e.message);return;}
R.setPixelRatio(Math.min(1.5,window.devicePixelRatio||1));R.outputEncoding=THREE.sRGBEncoding;R.toneMapping=THREE.ACESFilmicToneMapping;R.toneMappingExposure=1.0;
R.shadowMap.enabled=true;R.shadowMap.type=THREE.PCFSoftShadowMap;R.autoClear=false;R.physicallyCorrectLights=false;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x05070a);scene.fog=new THREE.FogExp2(0x05070a,0.022/SC);

// procedural environment: a dark hall with a big warm skylight and two cool strips
{const room=new THREE.Scene();const walls=new THREE.Mesh(new THREE.BoxGeometry(60,60,60),new THREE.MeshBasicMaterial({color:0x0b0d10,side:THREE.BackSide}));room.add(walls);
 const lamp=(w,h,pos,rot,col)=>{const m=new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshBasicMaterial({color:col,side:THREE.DoubleSide,toneMapped:false}));m.position.set(...pos);m.rotation.set(...rot);room.add(m);};
 lamp(26,14,[0,0,28],[0,0,0],new THREE.Color(7,6.4,5.6));lamp(40,6,[0,-29,12],[Math.PI/2,0,0],new THREE.Color(1.2,1.8,2.6));lamp(40,6,[0,29,12],[Math.PI/2,0,0],new THREE.Color(1.2,1.6,2.4));
 lamp(6,30,[-29,0,10],[0,Math.PI/2,0],new THREE.Color(0.9,0.9,1.0));lamp(6,30,[29,0,10],[0,Math.PI/2,0],new THREE.Color(0.9,0.9,1.0));
 const pm=new THREE.PMREMGenerator(R);pm.compileCubemapShader();const env=pm.fromScene(room,0.02);scene.environment=env.texture;pm.dispose();}
const key=new THREE.DirectionalLight(0xfff1dc,1.7);key.position.set(-10,-18,30);key.castShadow=true;key.shadow.mapSize.set(2048,2048);
Object.assign(key.shadow.camera,{left:-12,right:12,top:12,bottom:-12,near:5,far:80});key.shadow.bias=-0.0003;key.shadow.radius=3;scene.add(key);
const fill=new THREE.DirectionalLight(0x9ec4ff,0.35);fill.position.set(16,12,14);scene.add(fill);
scene.add(new THREE.AmbientLight(0x223040,0.35));
const cam=new THREE.PerspectiveCamera(42,1,0.1,300);
const closeCam=new THREE.PerspectiveCamera(40,1,0.05,60);

// ---------- court: painted hard court from a canvas texture ----------
const DW=19.25/2*SC,APW=DW*2+12*SC,APH=C.half_l*2+16*SC;
function noise(ctx,w,h,alpha,n){ctx.save();ctx.globalAlpha=alpha;for(let i=0;i<n;i++){const v=Math.floor(seed()*255);ctx.fillStyle=`rgb(${v},${v},${v})`;ctx.fillRect(seed()*w,seed()*h,1.6,1.6);}ctx.restore();}
function courtTex(){const W=2048,H=Math.round(2048*APH/APW),c=document.createElement('canvas');c.width=W;c.height=H;const x=c.getContext('2d');const sx=W/APW,sy=H/APH;
  const px=(mm)=>W/2+mm*sx,py=(mm)=>H/2-mm*sy;
  x.fillStyle='#2b6a45';x.fillRect(0,0,W,H);                                   // surround
  x.fillStyle='#2a63a8';x.fillRect(px(-DW),py(C.half_l),(DW*2)*sx,(C.half_l*2)*sy);  // court
  noise(x,W,H,0.10,26000);
  x.strokeStyle='#f4f6f8';x.lineWidth=Math.max(2,0.09*sx);x.lineCap='butt';
  const L=(a,b,c_,d)=>{x.beginPath();x.moveTo(px(a),py(b));x.lineTo(px(c_),py(d));x.stroke();};
  L(-DW,C.half_l,DW,C.half_l);L(-DW,-C.half_l,DW,-C.half_l);L(-DW,-C.half_l,-DW,C.half_l);L(DW,-C.half_l,DW,C.half_l);
  L(-C.half_w,-C.half_l,-C.half_w,C.half_l);L(C.half_w,-C.half_l,C.half_w,C.half_l);L(-C.half_w,C.service,C.half_w,C.service);L(-C.half_w,-C.service,C.half_w,-C.service);
  L(0,-C.service,0,C.service);L(0,C.half_l,0,C.half_l-0.4*SC);L(0,-C.half_l,0,-C.half_l+0.4*SC);
  const t=new THREE.CanvasTexture(c);t.encoding=THREE.sRGBEncoding;t.anisotropy=8;return t;}
function roughTex(){const W=512,c=document.createElement('canvas');c.width=c.height=W;const x=c.getContext('2d');x.fillStyle='#e6e6e6';x.fillRect(0,0,W,W);noise(x,W,W,0.5,40000);const t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(6,6);return t;}
const court=new THREE.Mesh(new THREE.PlaneGeometry(APW,APH),new THREE.MeshStandardMaterial({map:courtTex(),roughness:0.94,roughnessMap:roughTex(),metalness:0}));court.receiveShadow=true;scene.add(court);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(400,400),new THREE.MeshStandardMaterial({color:0x0e1114,roughness:1}));ground.position.z=-0.01;ground.receiveShadow=true;scene.add(ground);
// net: a real mesh texture on a plane, tape and metal posts
function netTex(){const c=document.createElement('canvas');c.width=1024;c.height=128;const x=c.getContext('2d');x.clearRect(0,0,1024,128);x.strokeStyle='rgba(235,240,245,0.9)';x.lineWidth=1.2;const s=10;for(let i=0;i<=1024;i+=s){x.beginPath();x.moveTo(i,0);x.lineTo(i,128);x.stroke();}for(let j=0;j<=128;j+=s){x.beginPath();x.moveTo(0,j);x.lineTo(1024,j);x.stroke();}const t=new THREE.CanvasTexture(c);t.wrapS=THREE.RepeatWrapping;t.repeat.set(3,1);return t;}
const netW=DW*2+1.8*SC;const net=new THREE.Mesh(new THREE.PlaneGeometry(netW,C.net_h),new THREE.MeshStandardMaterial({map:netTex(),transparent:true,alphaTest:0.2,side:THREE.DoubleSide,roughness:0.9,color:0xdfe5ec}));net.rotation.x=Math.PI/2;net.position.set(0,0,C.net_h/2);net.castShadow=true;scene.add(net);
const tape=new THREE.Mesh(new THREE.BoxGeometry(netW,0.05*SC,0.12*SC),new THREE.MeshStandardMaterial({color:0xf4f6f8,roughness:0.6}));tape.position.set(0,0,C.net_h);tape.castShadow=true;scene.add(tape);
const postMat=new THREE.MeshStandardMaterial({color:0x2a2f36,metalness:0.85,roughness:0.35});
[-1,1].forEach(s=>{const p=new THREE.Mesh(new THREE.CylinderGeometry(0.09*SC,0.09*SC,C.net_h+0.3*SC,16),postMat);p.rotation.x=Math.PI/2;p.position.set(s*(DW+0.9*SC),0,(C.net_h+0.3*SC)/2);p.castShadow=true;scene.add(p);});

// ---------- the specimen: micro-CT meshes, chitin, compound eyes, glass wings ----------
function partGeo(p){const v=new Float32Array(b64(p.v).buffer),f=new Uint32Array(b64(p.f).buffer);const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(v,3));g.setIndex(new THREE.BufferAttribute(f,1));g.computeVertexNormals();return g;}
const GEO={};for(const k in MESH.parts)GEO[k]=partGeo(MESH.parts[k]);
{const g=GEO.body,pos=g.attributes.position,n=pos.count,col=new Float32Array(n*3);let xmin=1e9,xmax=-1e9;for(let i=0;i<n;i++){const x=pos.getX(i);if(x<xmin)xmin=x;if(x>xmax)xmax=x;}
 const amber=[0.30,0.17,0.06],dark=[0.09,0.05,0.02],thorax=[0.24,0.14,0.05],eye=[0.42,0.05,0.03];
 for(let i=0;i<n;i++){const x=pos.getX(i),y=pos.getY(i),z=pos.getZ(i);let c;const j=(seed()-0.5)*0.04;
   if(x>xmax-0.5&&Math.abs(y)>0.2){c=eye.map(v=>v+j*0.5);}                                   // compound eyes
   else if(x>xmax-0.62){c=[0.19,0.11,0.04].map(v=>v+j);}                                       // head capsule
   else if(x<xmin+1.25){const band=Math.sin((x-xmin)*11.0)>0.25?dark:amber;c=band.map(v=>v+j);} // striped abdomen
   else c=thorax.map(v=>v+j);
   col[i*3]=c[0];col[i*3+1]=c[1];col[i*3+2]=c[2];}
 g.setAttribute('color',new THREE.BufferAttribute(col,3));}
const chitin=new THREE.MeshPhysicalMaterial({vertexColors:true,roughness:0.52,metalness:0.0,clearcoat:0.4,clearcoatRoughness:0.35,envMapIntensity:0.7});
const legMat=new THREE.MeshPhysicalMaterial({color:0x3a230c,roughness:0.6,clearcoat:0.3,clearcoatRoughness:0.5});
const wingMat=new THREE.MeshPhysicalMaterial({color:0xdfe7ef,transparent:true,opacity:0.2,roughness:0.15,clearcoat:1,clearcoatRoughness:0.1,side:THREE.DoubleSide,depthWrite:false,envMapIntensity:1.2});
const TRIPOD={lf:0,rm:0,lh:0,rf:Math.PI,lm:Math.PI,rh:Math.PI};
function blobTex(){const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d');const g=x.createRadialGradient(128,128,0,128,128,128);g.addColorStop(0,'rgba(0,0,0,0.55)');g.addColorStop(0.5,'rgba(0,0,0,0.25)');g.addColorStop(1,'rgba(0,0,0,0)');x.fillStyle=g;x.fillRect(0,0,256,256);return new THREE.CanvasTexture(c);}
const BLOB=blobTex();
function racket(color){
  const arm=new THREE.Group();arm.position.set(0.35,-0.45,C.rk_h+MESH.ground_z);
  const grip=new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.06,C.rk_arm*0.42,12),new THREE.MeshStandardMaterial({color:0x15171b,roughness:0.85}));grip.rotation.z=-Math.PI/2;grip.position.x=C.rk_arm*0.21;arm.add(grip);
  const shaft=new THREE.Mesh(new THREE.CylinderGeometry(0.035,0.045,C.rk_arm*0.6,12),new THREE.MeshStandardMaterial({color:0x8b9099,metalness:0.9,roughness:0.3}));shaft.rotation.z=-Math.PI/2;shaft.position.x=C.rk_arm*0.7;arm.add(shaft);
  const face=new THREE.Group();face.position.x=C.rk_arm;face.rotation.x=Math.PI/2;face.rotation.y=0.32;
  const frame=new THREE.Mesh(new THREE.TorusGeometry(C.rk_r,0.05,14,48),new THREE.MeshStandardMaterial({color:0x3a3f47,metalness:0.9,roughness:0.32}));frame.castShadow=true;face.add(frame);
  const band=new THREE.Mesh(new THREE.TorusGeometry(C.rk_r,0.054,8,48,Math.PI*0.5),new THREE.MeshStandardMaterial({color:color,emissive:color,emissiveIntensity:0.55,roughness:0.4}));band.rotation.z=Math.PI*1.25;face.add(band);
  const sm=new THREE.MeshStandardMaterial({color:0xf0f3f6,roughness:0.5});const r=C.rk_r-0.03,n=11;
  for(let k=0;k<n;k++){const u=-r+2*r*(k+0.5)/n,h=Math.sqrt(Math.max(0,r*r-u*u));
    const a=new THREE.Mesh(new THREE.CylinderGeometry(0.007,0.007,2*h,5),sm);a.position.x=u;face.add(a);
    const b=new THREE.Mesh(new THREE.CylinderGeometry(0.007,0.007,2*h,5),sm);b.rotation.z=Math.PI/2;b.position.y=u;face.add(b);}
  arm.add(face);return{arm,face};}
function makeFly(color){
  const g=new THREE.Group();
  const b=new THREE.Mesh(GEO.body,chitin);b.castShadow=true;b.receiveShadow=true;g.add(b);
  const wings=[];['wing_l','wing_r'].forEach(k=>{if(!GEO[k])return;const w=new THREE.Mesh(GEO[k],wingMat);w.renderOrder=2;g.add(w);wings.push(w);});
  const legs=[];
  for(const leg of ['lf','lm','lh','rf','rm','rh']){const pv=MESH.pivots[leg];const grp=new THREE.Group();grp.position.set(pv[0],pv[1],pv[2]);
    const m=new THREE.Mesh(GEO[leg],legMat);m.position.set(-pv[0],-pv[1],-pv[2]);m.castShadow=true;grp.add(m);g.add(grp);legs.push({grp,phase:TRIPOD[leg]});}
  const rk=racket(color);g.add(rk.arm);
  const blob=new THREE.Mesh(new THREE.PlaneGeometry(3.4,2.6),new THREE.MeshBasicMaterial({map:BLOB,transparent:true,depthWrite:false}));blob.position.set(-0.1,0,MESH.ground_z+0.012);g.add(blob);
  g.position.z=-MESH.ground_z;return{grp:g,legs,wings,arm:rk.arm,face:rk.face,ph:0,lx:null,bob:0};}
const flyA=makeFly(0x5fd6e3),flyB=makeFly(0xf58e6e);scene.add(flyA.grp);scene.add(flyB.grp);
flyA.grp.rotation.z=Math.PI/2;flyB.grp.rotation.z=-Math.PI/2;

// ---------- ball: felt with a seam ----------
function ballTex(){const c=document.createElement('canvas');c.width=1024;c.height=512;const x=c.getContext('2d');x.fillStyle='#cfe23c';x.fillRect(0,0,1024,512);noise(x,1024,512,0.16,30000);
  x.strokeStyle='#f6f7f2';x.lineWidth=26;x.lineCap='round';x.beginPath();for(let i=0;i<=1024;i+=4){const u=i/1024,v=256+118*Math.sin(u*Math.PI*4);if(i===0)x.moveTo(i,v);else x.lineTo(i,v);}x.stroke();
  const t=new THREE.CanvasTexture(c);t.encoding=THREE.sRGBEncoding;return t;}
const ball=new THREE.Mesh(new THREE.SphereGeometry(C.ball_r,32,24),new THREE.MeshStandardMaterial({map:ballTex(),roughness:0.95,metalness:0}));ball.castShadow=true;scene.add(ball);
const bblob=new THREE.Mesh(new THREE.PlaneGeometry(1.6,1.6),new THREE.MeshBasicMaterial({map:BLOB,transparent:true,depthWrite:false}));bblob.position.z=0.013;scene.add(bblob);
const trail=[];for(let i=0;i<10;i++){const m=new THREE.Mesh(new THREE.SphereGeometry(C.ball_r*(0.6-i*0.04),8,6),new THREE.MeshBasicMaterial({color:0xd6ea55,transparent:true,opacity:0.10*(1-i/10),depthWrite:false}));scene.add(m);trail.push(m);}

// ---------- post: bloom + SMAA on the main view ----------
let composer=null;
if(HAS_POST){composer=new THREE.EffectComposer(R);composer.addPass(new THREE.RenderPass(scene,cam));
  const bloom=new THREE.UnrealBloomPass(new THREE.Vector2(1,1),0.28,0.5,0.86);composer.addPass(bloom);
  const smaa=new THREE.SMAAPass(1,1);composer.addPass(smaa);composer.userSmaa=smaa;}

// ---------- cameras ----------
let az=-Math.PI/2+0.22,el=0.34,dist=13*SC+8.2,drag=null,camX=0,camY=0;
const mainEl=document.getElementById('main');
function placeCam(ax,px,py){camX+=((ax*0.5+px*0.3)-camX)*0.04;camY+=((-C.half_l+2.0*SC)-camY)*0.05;
  cam.position.set(camX+dist*Math.cos(el)*Math.cos(az),camY+dist*Math.cos(el)*Math.sin(az),0.6+dist*Math.sin(el));cam.up.set(0,0,1);cam.lookAt(camX*0.6,-2.5*SC,0.7);}
mainEl.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};mainEl.setPointerCapture(e.pointerId);});
mainEl.addEventListener('pointermove',e=>{if(!drag)return;az-=(e.clientX-drag.x)*0.006;el=Math.max(0.06,Math.min(1.3,el+(e.clientY-drag.y)*0.005));drag={x:e.clientX,y:e.clientY};});
mainEl.addEventListener('pointerup',()=>drag=null);
mainEl.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(3,Math.min(80,dist*(e.deltaY>0?1.07:0.93)));},{passive:false});
const _fp=new THREE.Vector3(),_cp=new THREE.Vector3();
function placeClose(){flyA.face.getWorldPosition(_fp);const a=flyA.grp.position;_cp.set(a.x+3.1,a.y-3.1,2.4);closeCam.position.lerp(_cp,0.12);closeCam.up.set(0,0,1);closeCam.lookAt(_fp.x,_fp.y+0.9,_fp.z-0.05);}

function lerp(a,b,t){return a+(b-a)*t;}
function sample(){const Fr=RUN().frames;const fi=Math.min(Fr.length-1,Math.floor(frame)),fj=Math.min(Fr.length-1,fi+1),t=frame-fi;const a=Fr[fi],b=Fr[fj];const o=a.map((v,i)=>lerp(v,b[i],t));o[10]=a[10];return o;}
function animFly(f,x,dt){const sp=f.lx===null?0:Math.abs(x-f.lx)/Math.max(dt,1e-3);f.lx=x;const w=Math.min(1,sp/(9*SC+3));f.ph+=dt*(1.5+20*w);
  f.legs.forEach(L=>{const pz=f.ph+L.phase,sw=Math.sin(pz),side=Math.sign(L.grp.position.y)||1;L.grp.rotation.z=(Math.cos(pz)*0.42)*w;L.grp.rotation.x=side*Math.max(0,sw)*0.5*w;});
  f.wings.forEach((wg,i)=>{wg.rotation.x=Math.sin(f.ph*2.5+i*Math.PI)*0.03;});f.bob=Math.abs(Math.sin(f.ph*2))*0.05*w;}
let lastP=null,lastAx=null,ballSpeed=0,walkA=0,caption='',capUntil=-1;
function updateCourt(dt,simdt){const s=sample();const [t,ax,ay,thA,bx,by,thB,px,py,pz,rally,eA,eB,hA,hB]=s;
  animFly(flyA,ax,dt);animFly(flyB,bx,dt);
  flyA.grp.position.set(ax,ay,-MESH.ground_z+flyA.bob);flyB.grp.position.set(bx,by,-MESH.ground_z+flyB.bob);
  flyA.grp.rotation.z=Math.PI/2-(hA||0);flyB.grp.rotation.z=-Math.PI/2+(hB||0);
  flyA.arm.rotation.z=-(thA||1.13);flyB.arm.rotation.z=-(thB||1.13);
  ball.position.set(px,py,pz);ball.rotation.x+=dt*9;ball.rotation.y+=dt*4;bblob.position.set(px,py,0.013);const sh=Math.max(0.25,1-pz*0.35);bblob.scale.set(sh,sh,1);
  for(let i=trail.length-1;i>0;i--)trail[i].position.copy(trail[i-1].position);trail[0].position.copy(ball.position);
  if(simdt>1e-4){if(lastP)ballSpeed=lerp(ballSpeed,Math.hypot(px-lastP[0],py-lastP[1],pz-lastP[2])/simdt,0.3);if(lastAx!==null)walkA=lerp(walkA,(ax-lastAx)/simdt,0.3);}
  lastP=[px,py,pz];lastAx=ax;
  document.getElementById('rally').textContent=rally;document.getElementById('ballv').textContent=ballSpeed.toFixed(1);
  document.getElementById('r-bear').textContent=((eA||0)*57.2958).toFixed(1)+'°';document.getElementById('r-walk').textContent=(walkA>=0?'+':'')+walkA.toFixed(1)+' mm/s';
  const evs=RUN().events;let near=null;for(const e of evs){if(e.type==='hit'&&Math.abs(e.t-t)<0.32){near=e;break;}}
  slowNow=autoSlow&&!!near;document.getElementById('slo').classList.toggle('on',slowNow);
  const ev=evs.filter(e=>e.type!=='bounce'&&e.t<=t&&e.t>t-1.3).slice(-1)[0];
  const cp=ev?(ev.type==='hit'?('Fly '+(ev.fly===0?'A':'B')+' returns'):ev.type==='net'?'Into the net':'Point over'):(rally>0?'Rally':'Serve');
  document.getElementById('caption').textContent=cp;
  const contact=evs.some(e=>e.type==='hit'&&e.fly===0&&Math.abs(e.t-t)<0.12);const ce=document.getElementById('contact');ce.textContent=contact?'contact':'approach';ce.classList.toggle('on',contact);
  placeCam(ax,px,py);placeClose();}

// ---------- brain: soma plus primary neurite, coloured by deviation from each cell's own baseline ----------
const bscene=new THREE.Scene();const bcam=new THREE.PerspectiveCamera(30,1,0.01,60);
const idxB=[],idxV=[];for(let i=0;i<NV;i++)(REG[i]===2?idxV:idxB).push(i);
function cloud(ids,dz){const n=ids.length,pos=new Float32Array(n*3),col=new Float32Array(n*3),lp=new Float32Array(n*6),lc=new Float32Array(n*6);let cx=0,cy=0,cz=0;ids.forEach(i=>{cx+=P[i*3];cy+=P[i*3+1];cz+=P[i*3+2];});cx/=n;cy/=n;cz/=n;
  ids.forEach((i,k)=>{const x=P[i*3]-cx,y=P[i*3+1]-cy,z=P[i*3+2]-cz;pos[k*3]=x;pos[k*3+1]=y;pos[k*3+2]=z;
    const f=0.28+seed()*0.3,jx=(seed()-0.5)*0.06,jy=(seed()-0.5)*0.06,jz=(seed()-0.5)*0.06;   // primary neurite runs from the soma rind toward the neuropil
    lp[k*6]=x;lp[k*6+1]=y;lp[k*6+2]=z;lp[k*6+3]=x*(1-f)+jx;lp[k*6+4]=y*(1-f)+jy;lp[k*6+5]=z*(1-f)+jz;});
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.BufferAttribute(lp,3));lg.setAttribute('color',new THREE.BufferAttribute(lc,3));
  const grp=new THREE.Group();grp.position.z=dz;
  const pts=new THREE.Points(g,new THREE.PointsMaterial({size:0.017,vertexColors:true,transparent:true,opacity:0.85,sizeAttenuation:true,depthWrite:false}));
  const lines=new THREE.LineSegments(lg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.18,depthWrite:false,blending:THREE.AdditiveBlending}));
  grp.add(lines);grp.add(pts);bscene.add(grp);return{grp,ids,col,lc,g,lg};}
const cloudB=cloud(idxB,0.55),cloudV=cloud(idxV,-0.62);
bcam.position.set(0,-4.6,-0.25);bcam.up.set(0,0,1);bcam.lookAt(0,0,-0.12);
const NFA=Math.ceil(T.frames.length/AS);const MEAN=new Float32Array(NV);
for(let i=0;i<NV;i++){let m=0;for(let f=0;f<NFA;f++)m+=ACT[f*NV+i];MEAN[i]=m/NFA;}
let SD=0;{let acc=0,n=0;for(let f=0;f<NFA;f+=Math.max(1,Math.floor(NFA/40)))for(let i=0;i<NV;i++){const d=ACT[f*NV+i]-MEAN[i];acc+=d*d;n++;}SD=Math.max(4,Math.sqrt(acc/n));}
const TINT=[[61,155,198],[201,163,232],[123,211,160]];const HOT=[[1.0,0.55,0.20],[1.0,0.85,0.35],[1.0,1.0,0.92]];const COLD=[0.45,0.70,1.0];
function colorFor(i,a){const z=(a-MEAN[i])/SD;const t=TINT[REG[i]];
  if(z>0.55){const u=Math.min(1,(z-0.55)/2.0);return u<0.5?HOT[0].map((v,k)=>v+(HOT[1][k]-v)*u*2):HOT[1].map((v,k)=>v+(HOT[2][k]-v)*(u-0.5)*2);}
  if(z<-0.9){const u=Math.min(1,(-z-0.9)/2.2);return COLD.map(v=>0.25+(v-0.25)*u);}
  return[t[0]/255*0.2,t[1]/255*0.2,t[2]/255*0.2];}
let swayT=0,bdrag=null;const pb=document.getElementById('pbrain');
function updateBrain(dt){const fi=Math.min(T.frames.length-1,Math.floor(frame));const off=Math.floor(fi/AS)*NV;const dim=mode!=='trained';
  swayT+=dt;const rz=Math.sin(swayT*0.30)*0.5,rx=Math.sin(swayT*0.19)*0.12;
  [cloudB,cloudV].forEach(c=>{c.ids.forEach((i,k)=>{const cc=dim?[0.16,0.16,0.18]:colorFor(i,ACT[off+i]);c.col[k*3]=cc[0];c.col[k*3+1]=cc[1];c.col[k*3+2]=cc[2];c.lc[k*6]=cc[0]*0.7;c.lc[k*6+1]=cc[1]*0.7;c.lc[k*6+2]=cc[2]*0.7;c.lc[k*6+3]=cc[0]*0.25;c.lc[k*6+4]=cc[1]*0.25;c.lc[k*6+5]=cc[2]*0.25;});
    c.g.attributes.color.needsUpdate=true;c.lg.attributes.color.needsUpdate=true;if(!c.userDrag){c.grp.rotation.z=rz;c.grp.rotation.x=rx;}});}
pb.addEventListener('pointerdown',e=>{bdrag={x:e.clientX,y:e.clientY};pb.setPointerCapture(e.pointerId);});
pb.addEventListener('pointermove',e=>{if(!bdrag)return;[cloudB,cloudV].forEach(c=>{c.userDrag=true;c.grp.rotation.z+=(e.clientX-bdrag.x)*0.01;c.grp.rotation.x+=(e.clientY-bdrag.y)*0.01;});bdrag={x:e.clientX,y:e.clientY};});
pb.addEventListener('pointerup',()=>bdrag=null);

function stats(){const s=RUN().summary;document.getElementById('hits').textContent=s.total_hits;document.getElementById('longest').textContent=s.max_rally;}

// ---------- viewports ----------
const pc=document.getElementById('pclose');
function rect(el){const r=el.getBoundingClientRect(),H=cv.clientHeight;return{x:r.left,y:H-r.bottom,w:r.width,h:r.height};}
function resize(){const w=cv.clientWidth,h=cv.clientHeight;R.setSize(w,h,false);const m=rect(mainEl);cam.aspect=m.w/Math.max(1,m.h);cam.updateProjectionMatrix();
  if(composer){composer.setSize(m.w,m.h);}
  const c=rect(pc);closeCam.aspect=c.w/Math.max(1,c.h);closeCam.updateProjectionMatrix();const b=rect(pb);bcam.aspect=b.w/Math.max(1,b.h);bcam.updateProjectionMatrix();}
window.addEventListener('resize',resize);
function view(r){R.setViewport(r.x,r.y,r.w,r.h);R.setScissor(r.x,r.y,r.w,r.h);}
function render(){R.setScissorTest(true);
  const m=rect(mainEl);view(m);R.setClearColor(0x05070a,1);R.clear();if(composer)composer.render();else R.render(scene,cam);
  const c=rect(pc);view(c);R.setClearColor(0x050607,1);R.clear();R.render(scene,closeCam);
  const b=rect(pb);view(b);R.setClearColor(0x050607,1);R.clear();R.render(bscene,bcam);}

// ---------- controls + loop ----------
const track=document.getElementById('track');
track.addEventListener('pointerdown',e=>{const r=track.getBoundingClientRect();frame=(e.clientX-r.left)/r.width*(NF()-1);playing=false;document.getElementById('play').textContent='▶';});
document.getElementById('play').addEventListener('click',e=>{playing=!playing;e.target.textContent=playing?'❚❚':'▶';});
document.getElementById('slow').addEventListener('click',e=>{autoSlow=!autoSlow;e.target.setAttribute('aria-pressed',autoSlow);});
function setMode(m){mode=m;frame=0;lastP=null;lastAx=null;document.getElementById('m-untrained').setAttribute('aria-pressed',m==='untrained');document.getElementById('m-trained').setAttribute('aria-pressed',m==='trained');stats();}
document.getElementById('m-untrained').addEventListener('click',()=>setMode('untrained'));document.getElementById('m-trained').addEventListener('click',()=>setMode('trained'));
let last=performance.now(),failed=false;
function tick(dt,draw=true){const rate=slowNow?0.3:1;const simdt=playing?dt*rate:0;if(playing){frame+=simdt*REC;if(frame>=NF()-1)frame=0;}
  try{updateCourt(dt,simdt);if(draw){updateBrain(dt);render();}}catch(e){if(!failed){failed=true;fail(e.message||String(e));}}
  document.getElementById('prog').style.width=(100*frame/(NF()-1))+'%';document.getElementById('clock').textContent=(frame/REC).toFixed(1)+' / '+((NF()-1)/REC).toFixed(0)+' s';}
function loop(now){const dt=Math.min(0.05,(now-last)/1000);last=now;tick(dt);requestAnimationFrame(loop);}
resize();setMode('trained');
if(RECORD){playing=true;window.__step=function(n){for(let i=0;i<n;i++)tick(1/60,i===n-1);return frame;};window.__ready=true;render();}
else requestAnimationFrame(loop);
})();
</script>
"""
page = html.replace("__PAYLOAD__", payload).replace("__MESH__", mesh)
out.write_text(page)
Path("/tmp/tennis_script.js").write_text(re.findall(r'<script>(.*?)</script>', page, re.S)[-1])
print(f"written {out} ({out.stat().st_size/1e6:.2f} MB)")
