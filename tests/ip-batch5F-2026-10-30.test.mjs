// Batch 5F: frozen 1956581 host behavior, numerical fixtures and exported Python.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import cv from '@techstark/opencv-js';
import { OPS, FILTERS, needsOf, isParamVisible, passFailVerdict } from '../src/renderer/filters/registry.js';
import { buildPythonCode, PF_VALUE_OPS } from '../src/renderer/codegen.js';
import { TOOL_SECTIONS, OP_TO_FILTER } from '../src/renderer/ui/palette-panel.js';
import { OP_ICON, OP_PRIMARY, OPTION_GROUPS } from '../src/renderer/filters/ui-meta.js';
import { activeContour5F, edgeTaper5F, laplace5F, fusion5F, blind5F } from '../src/renderer/filters/batch5f.js';
import { assertWrapperBalance } from './_leak-helpers.mjs';
while (typeof cv.Mat !== 'function') await new Promise(r => setTimeout(r, 20));
const hash = x => crypto.createHash('sha256').update(x).digest('hex');
const defaults = id => Object.fromEntries(OPS[id].params.map(q => [q.key, q.default]));
const run = (id, src, p = {}, ctx = {}, c = cv) => OPS[id].apply(c, src, { ...defaults(id), ...p }, ctx);
const vals = ctx => Object.fromEntries(ctx.values.fields.map(f => [f.key, f.value]));
function image(w, h, fn, ch = 3) {
  return cv.matFromArray(h, w, ch === 1 ? cv.CV_8UC1 : ch === 4 ? cv.CV_8UC4 : cv.CV_8UC3,
    Uint8Array.from(Array.from({ length: w * h }, (_, i) => fn(i % w, Math.floor(i / w))).flat()));
}
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id;
  return { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries({ ...defaults(id), ...p }).map(([k, v]) => [id + '__' + k, v])) } };
}
function same(a, b) {
  assert.deepEqual([a.cols, a.rows, a.channels()], [b.cols, b.rows, b.channels()]);
  assert.deepEqual(Buffer.from(a.data), Buffer.from(b.data));
}
const LEGACY = {"filter_denoise":["salt_pepper_noise","gaussian_noise","median_blur","rank_filter","gaussian_blur","box_blur","bilateral","wiener_filter","guided_filter","anisotropic_diffusion","nlm_denoise","deconv_wiener","deconv_lucy"],"segment_edge":["binary_threshold","otsu_threshold","adaptive_threshold","multi_otsu_threshold","canny","sobel","gradient_dir","laplacian_edge","scharr_edge","prewitt_edge","roberts_edge","log_edge","frangi_vesselness","watershed","watershed_marker","grabcut","region_growing"],"enhance_tools":["brightness_contrast","gamma","custom_lut","hist_equalize","clahe","contrast_stretch","flatfield_correction","quantize","unsharp","laplacian_sharpen","custom_kernel","dehaze","lowlight_enhance","local_contrast"],"image_quality":["psnr","ssim","delta_e","seg_score","bf_score","corr2","piqe"],"image_arithmetic":["imabsdiff","imfuse","mask","mask_overlay","reconstruct","imarith","logical_ops","inpaint","hist_match","roi_apply"]};
const LEGACY_PARAMS = {"salt_pepper_noise":[{"key":"amount","type":"number","default":5}],"gaussian_noise":[{"key":"sigma","type":"number","default":20},{"key":"seed","type":"number","default":1},{"key":"kind","type":"select","default":"additive","options":[{"value":"additive"},{"value":"speckle"}]}],"median_blur":[{"key":"ksize","type":"number","default":3}],"rank_filter":[{"key":"mode","type":"select","default":"percentile","options":[{"value":"percentile"},{"value":"min"},{"value":"max"},{"value":"range"},{"value":"mode"}]},{"key":"percentile","type":"number","default":50},{"key":"size","type":"number","default":3}],"gaussian_blur":[{"key":"ksize","type":"number","default":5}],"box_blur":[{"key":"ksize","type":"number","default":5}],"bilateral":[{"key":"d","type":"number","default":7},{"key":"sigmaColor","type":"number","default":75},{"key":"sigmaSpace","type":"number","default":75}],"wiener_filter":[{"key":"ksize","type":"number","default":5},{"key":"noiseVar","type":"number","default":0}],"guided_filter":[{"key":"radius","type":"number","default":2},{"key":"eps","type":"number","default":0.01}],"anisotropic_diffusion":[{"key":"iterations","type":"number","default":10},{"key":"kappa","type":"number","default":20},{"key":"conduction","type":"select","default":"exponential","options":[{"value":"exponential"},{"value":"quadratic"}]}],"nlm_denoise":[{"key":"h","type":"number","default":10},{"key":"patch","type":"select","default":"5","options":[{"value":"3"},{"value":"5"},{"value":"7"}]},{"key":"search","type":"select","default":"11","options":[{"value":"7"},{"value":"11"},{"value":"15"}]}],"deconv_wiener":[{"key":"psf","type":"select","default":"motion","options":[{"value":"motion"},{"value":"disk"}]},{"key":"psfSize","type":"number","default":9},{"key":"psfAngle","type":"number","default":0},{"key":"nsr","type":"number","default":0.01}],"deconv_lucy":[{"key":"psf","type":"select","default":"motion","options":[{"value":"motion"},{"value":"disk"}]},{"key":"psfSize","type":"number","default":9},{"key":"psfAngle","type":"number","default":0},{"key":"iterations","type":"number","default":10}],"binary_threshold":[{"key":"thresh","type":"number","default":127},{"key":"invert","type":"boolean","default":false}],"otsu_threshold":[{"key":"invert","type":"boolean","default":false},{"key":"method","type":"select","default":"otsu","options":[{"value":"otsu"},{"value":"triangle"}]}],"adaptive_threshold":[{"key":"blockSize","type":"number","default":11},{"key":"C","type":"number","default":2}],"multi_otsu_threshold":[],"canny":[{"key":"t1","type":"number","default":50},{"key":"t2","type":"number","default":150}],"sobel":[{"key":"dir","type":"select","default":"both","options":[{"value":"x"},{"value":"y"},{"value":"both"}]},{"key":"ksize","type":"number","default":3}],"gradient_dir":[{"key":"method","type":"select","default":"sobel","options":[{"value":"sobel"},{"value":"prewitt"},{"value":"central"},{"value":"intermediate"},{"value":"roberts"}]},{"key":"view","type":"select","default":"magnitude","options":[{"value":"magnitude"},{"value":"direction"},{"value":"direction_gray"}]}],"laplacian_edge":[{"key":"ksize","type":"number","default":3}],"scharr_edge":[{"key":"dir","type":"select","default":"both","options":[{"value":"x"},{"value":"y"},{"value":"both"}]}],"prewitt_edge":[{"key":"dir","type":"select","default":"both","options":[{"value":"x"},{"value":"y"},{"value":"both"}]}],"roberts_edge":[{"key":"dir","type":"select","default":"both","options":[{"value":"d1"},{"value":"d2"},{"value":"both"}]}],"log_edge":[{"key":"sigma","type":"number","default":2},{"key":"zeroThresh","type":"number","default":8},{"key":"view","type":"select","default":"zerocross","options":[{"value":"zerocross"},{"value":"response"}]}],"frangi_vesselness":[{"key":"polarity","type":"select","default":"bright","options":[{"value":"bright"},{"value":"dark"}]},{"key":"sigmaMin","type":"number","default":1},{"key":"sigmaMax","type":"number","default":3},{"key":"beta","type":"number","default":0.5}],"watershed":[{"key":"ratio","type":"number","default":0.5}],"watershed_marker":[{"key":"polarity","type":"select","default":"dark","options":[{"value":"dark"},{"value":"bright"}]},{"key":"hHeight","type":"number","default":2},{"key":"view","type":"select","default":"boundaries","options":[{"value":"boundaries"},{"value":"labels"}]}],"grabcut":[{"key":"margin","type":"number","default":12},{"key":"iter","type":"number","default":2},{"key":"output","type":"select","default":"extract","options":[{"value":"extract"},{"value":"mask"}]}],"region_growing":[{"key":"seedX","type":"number","default":50},{"key":"seedY","type":"number","default":50},{"key":"tolerance","type":"number","default":12},{"key":"rangeMode","type":"select","default":"fixed","options":[{"value":"fixed"},{"value":"floating"}]},{"key":"view","type":"select","default":"mask","options":[{"value":"mask"},{"value":"overlay"}]}],"brightness_contrast":[{"key":"alpha","type":"number","default":1.2},{"key":"beta","type":"number","default":0}],"gamma":[{"key":"g","type":"number","default":1.5},{"key":"curve","type":"select","default":"power","options":[{"value":"power"},{"value":"log"}]}],"custom_lut":[{"key":"out0","type":"number","default":0},{"key":"in1","type":"number","default":64},{"key":"out1","type":"number","default":64},{"key":"in2","type":"number","default":128},{"key":"out2","type":"number","default":128},{"key":"in3","type":"number","default":192},{"key":"out3","type":"number","default":192},{"key":"out255","type":"number","default":255},{"key":"view","type":"select","default":"result","options":[{"value":"result"},{"value":"curve"}]}],"hist_equalize":[],"clahe":[{"key":"clip","type":"number","default":2},{"key":"tile","type":"number","default":8}],"contrast_stretch":[{"key":"lowPct","type":"number","default":1},{"key":"highPct","type":"number","default":1}],"flatfield_correction":[{"key":"blurSize","type":"number","default":121}],"quantize":[{"key":"levels","type":"number","default":4}],"unsharp":[{"key":"amount","type":"number","default":1},{"key":"radius","type":"number","default":5}],"laplacian_sharpen":[{"key":"strength","type":"number","default":1}],"custom_kernel":[{"key":"preset","type":"select","default":"sharpen","options":[{"value":"identity"},{"value":"sharpen"},{"value":"edge"},{"value":"emboss"},{"value":"outline"},{"value":"box3"},{"value":"box5"},{"value":"motion"},{"value":"disk"}]},{"key":"amount","type":"number","default":1},{"key":"psfSize","type":"number","default":9},{"key":"psfAngle","type":"number","default":0}],"dehaze":[{"key":"amount","type":"number","default":0.95},{"key":"patch","type":"number","default":15},{"key":"refine","type":"boolean","default":true},{"key":"view","type":"select","default":"result","options":[{"value":"result"},{"value":"transmission"},{"value":"dark"}]}],"lowlight_enhance":[{"key":"amount","type":"number","default":1},{"key":"patch","type":"number","default":15},{"key":"refine","type":"boolean","default":true},{"key":"alphaBlend","type":"boolean","default":false}],"local_contrast":[{"key":"edgeThreshold","type":"number","default":0.3},{"key":"amount","type":"number","default":0.5}],"psnr":[],"ssim":[{"key":"speed","type":"select","default":"full","options":[{"value":"full"},{"value":"fast"}]}],"delta_e":[{"key":"formula","type":"select","default":"cie76","options":[{"value":"cie76"},{"value":"ciede2000"},{"value":"cie94"},{"value":"angle"}]},{"key":"compare","type":"select","default":"pixel","options":[{"value":"pixel"},{"value":"mean_b"}]},{"key":"tol","type":"number","default":2.3},{"key":"tol","type":"number","default":2.3},{"key":"scaleMax","type":"number","default":20},{"key":"scaleMax","type":"number","default":20},{"key":"view","type":"select","default":"heatmap","options":[{"value":"heatmap"},{"value":"over"}]}],"seg_score":[],"bf_score":[{"key":"tolPct","type":"number","default":0.75}],"corr2":[],"piqe":[],"imabsdiff":[],"imfuse":[{"key":"mode","type":"select","default":"blend","options":[{"value":"blend"},{"value":"checkerboard"}]},{"key":"alpha","type":"number","default":50},{"key":"tileSize","type":"number","default":32}],"mask":[{"key":"invert","type":"boolean","default":false}],"mask_overlay":[{"key":"style","type":"select","default":"fill","options":[{"value":"fill"},{"value":"labels"},{"value":"outline"}]},{"key":"color","type":"select","default":"red","options":[{"value":"red"},{"value":"green"},{"value":"blue"},{"value":"yellow"},{"value":"cyan"},{"value":"magenta"}]},{"key":"opacity","type":"number","default":50},{"key":"thickness","type":"number","default":2}],"reconstruct":[],"imarith":[{"key":"operation","type":"select","default":"add","options":[{"value":"add"},{"value":"subtract"},{"value":"multiply"},{"value":"divide"}]}],"logical_ops":[{"key":"operation","type":"select","default":"and","options":[{"value":"and"},{"value":"or"},{"value":"xor"},{"value":"a_and_not_b"}]}],"inpaint":[{"key":"method","type":"select","default":"telea","options":[{"value":"telea"},{"value":"ns"}]},{"key":"radius","type":"number","default":3}],"hist_match":[],"roi_apply":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"}]},{"key":"x","type":"number","default":25},{"key":"y","type":"number","default":25},{"key":"w","type":"number","default":50},{"key":"h","type":"number","default":50},{"key":"invert","type":"boolean","default":false},{"key":"feather","type":"number","default":0},{"key":"view","type":"select","default":"result","options":[{"value":"result"},{"value":"outline"}]}]};
test('ip5f-G: 650 clean-tree host cases preserve pixels, info, values and Python bytes', () => {
  const result = {};
  for (const [kind, w, h] of [['rgb',37,29],['gray',37,29],['rgba',37,29],['rgb',1,1],['empty',0,0]]) {
    const d = Uint8Array.from({ length: w*h*3 }, (_,i) => (Math.imul(i+11,73) ^ (i>>4)*17)&255);
    const c = w ? cv.matFromArray(h,w,cv.CV_8UC3,d) : new cv.Mat(), src = new cv.Mat();
    if(kind==='gray') cv.cvtColor(c,src,cv.COLOR_RGB2GRAY);
    else if(kind==='rgba') cv.cvtColor(c,src,cv.COLOR_RGB2RGBA); else c.copyTo(src);
    try { for(const [fid,ids] of Object.entries(LEGACY)) for(const id of ids) {
      const qs = LEGACY_PARAMS[id], defs = Object.fromEntries(qs.map(q=>[q.key,q.default]));
      for(const extra of [{},...qs.filter(q=>q.type==='select').flatMap(q=>q.options.filter(o=>o.value!==q.default).map(o=>({[q.key]:o.value})))]) {
        const local={...defs,...extra},p={mode:id,...Object.fromEntries(Object.entries(local).map(([k,v])=>[id+'__'+k,v]))}, st={id:fid,params:p};
        const key=JSON.stringify([kind,w,h,fid,id,extra]),ctx={src2:c};let g=null;
        try {
          if(w&&needsOf(FILTERS[fid],p)==='gray'&&src.channels()>1){g=new cv.Mat();cv.cvtColor(src,g,src.channels()===4?cv.COLOR_RGBA2GRAY:cv.COLOR_RGB2GRAY);}
          let seed=12345;const random=Math.random;let out;
          try{Math.random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);out=FILTERS[fid].apply(cv,g||src,p,ctx);}finally{Math.random=random;}
          try{result[key]={image:hash(Buffer.from(out.data)),rows:out.rows,cols:out.cols,ch:out.channels(),info:ctx.info,values:ctx.values,python:hash(buildPythonCode([st]))};}finally{out.delete();}
        } catch(e) {result[key]={error:typeof e==='number'?'OpenCV native exception':String(e),python:hash(buildPythonCode([st]))};} finally{g?.delete();}
      }
    }}finally{src.delete();c.delete();}
  }
  assert.equal(Object.keys(result).length,650);
  assert.equal(hash(JSON.stringify(result)),'a4bf43c9e7a43b6e6135c900e98e61f9270f9b01931d2b96f7af3c499b880b7d');
});
const NEW = ['active_contour','deconv_blind','deconv_reg','local_laplacian','multissim','exposure_fusion'];
test('ip5f-schema: six operations are reachable in order and MS-SSIM feeds Pass/Fail',()=>{
  for(const [id,after] of [['active_contour','region_growing'],['deconv_blind','deconv_lucy'],['deconv_reg','deconv_blind'],['local_laplacian','local_contrast'],['multissim','ssim'],['exposure_fusion','reconstruct']]){
    assert.ok(OPS[id],id);const s=TOOL_SECTIONS.find(s=>s.ops.includes(id)),host=OP_TO_FILTER[id];
    assert.equal(s.ops[s.ops.indexOf(after)+1],id);
    assert.deepEqual(FILTERS[host].params[0].options.map(o=>o.value),s.ops);
    assert.ok(OP_ICON[id]);assert.equal(Object.values(OP_ICON).filter(x=>x===OP_ICON[id]).length,1);
    assert.match(OP_PRIMARY[id](defaults(id)),/[ก-๙]/);
    if(OPTION_GROUPS[host+'.mode'])assert.ok(OPTION_GROUPS[host+'.mode'].some(g=>g[1].includes(id)));
  }
  for(const id of ['deconv_wiener','deconv_lucy','deconv_blind','deconv_reg'])assert.equal(OPS[id].params.at(-1).key,'taper');
  assert.equal(OPS.inpaint.params[0].options.at(-1).value,'laplace');
  assert.equal(isParamVisible(OPS.inpaint.params.find(q=>q.key==='radius'),{method:'laplace'},OPS.inpaint.params),false);
  assert.ok(PF_VALUE_OPS.has('multissim'));assert.ok(PF_VALUE_OPS.has('active_contour'));
});
test('ip5f-active: Otsu segments a bright square, counts final area and draws an inner red boundary',()=>{
  const src=image(21,19,(x,y)=>x>=6&&x<=14&&y>=5&&y<=13?220:30,1);
  try{const c={},out=run('active_contour',src,{init:'otsu',smooth:0,view:'mask'},c);
    try{assert.deepEqual(Array.from(out.data),Array.from(src.data,v=>v===220?255:0));assert.equal(vals(c).area,81);assert.match(c.info,/c_in 220\.0 · c_out 30\.0/);}finally{out.delete();}
    const overCtx={},over=run('active_contour',src,{init:'otsu',smooth:0},overCtx);try{assert.deepEqual(Array.from(over.data.slice((5*21+10)*3,(5*21+10)*3+3)),[255,0,0]);assert.deepEqual(Array.from(over.data.slice((9*21+10)*3,(9*21+10)*3+3)),[220,220,220]);assert.match(overCtx.info,/Chan-Vese · 1 รอบ \(หยุดเมื่อนิ่ง\)/);}finally{over.delete();}
  }finally{src.delete();}
});
test('ip5f-active: the region means, 5-of-9 majority threshold and bias sign stay oriented',()=>{
  const seed=Uint8Array.from([1,1,1,1,1,0,0,0,0]);
  const tied=activeContour5F(new Uint8Array(9),3,3,seed,1,1,0);
  assert.equal(tied.mask[4],1,'a neighborhood with exactly five foreground samples remains foreground');
  const src=image(12,10,(x,y)=>x>=4&&x<=7&&y>=3&&y<=6?130:120,1);
  try{
    const inside=Uint8Array.from(src.data,v=>v===130?1:0);
    const neutral=activeContour5F(src.data,12,10,inside,1,0,0);
    const contracting=activeContour5F(src.data,12,10,inside,1,0,.2);
    const expanding=activeContour5F(src.data,12,10,inside,1,0,-.2);
    assert.equal(neutral.cin,130);assert.equal(neutral.cout,120);
    assert.ok(contracting.area<neutral.area);assert.ok(expanding.area>neutral.area);
  }finally{src.delete();}
});
test('ip5f-laplace: harmonic ramp, edge mask, constant color and unchanged exterior',()=>{
  const src=image(15,13,(x,y)=>[x*10+y*2,60,180]),mask=image(15,13,(x,y)=>(x>=5&&x<=8&&y>=4&&y<=7)?255:0,1);
  try{const out=run('inpaint',src,{method:'laplace'},{src2:mask});try{same(out,src);}finally{out.delete();}
    const edge=image(15,13,(x,y)=>x===0&&y===0?255:0,1);try{const out=run('inpaint',src,{method:'laplace'},{src2:edge});try{assert.deepEqual(Array.from(out.data.slice(0,3)),[6,60,180]);assert.deepEqual(Array.from(out.data.slice(3)),Array.from(src.data.slice(3)));}finally{out.delete();}}finally{edge.delete();}
  }finally{src.delete();mask.delete();}
});
test('ip5f-laplace: one red-black update starts from the measured border mean',()=>{
  const data=Uint8Array.from([0,10,20,30,40,50,60,70,80]),mask=Uint8Array.from([0,0,0,0,1,0,0,0,0]);
  const q=laplace5F(data,3,3,1,mask,1);
  assert.equal(q.data[4],40,'the four border values average to 40 before the first update');
});
test('ip5f-laplace: coarse initialization converges to a harmonic ramp before the budget ends',()=>{
  const src=image(60,50,(x,y)=>2*x+2*y,1),mask=image(60,50,(x,y)=>x>=20&&x<40&&y>=15&&y<35?255:0,1),ctx={src2:mask};
  try{
    const out=run('inpaint',src,{method:'laplace'},ctx);
    try{
      assert.match(ctx.info,/หยุดเมื่อนิ่ง/);
      for(let y=15;y<35;y++)for(let x=20;x<40;x++)assert.ok(Math.abs(out.data[y*60+x]-src.data[y*60+x])<=1,`harmonic pixel ${x},${y}`);
    }finally{out.delete();}
  }finally{src.delete();mask.delete();}
});
test('ip5f-laplace: 80x80 and budgeted 240x180 holes converge with the coarse solve',()=>{
  for(const [w,h,x0,x1,y0,y1] of [[80,80,20,60,20,55],[240,180,70,170,40,140]]){
    const src=image(w,h,x=>x,1),mask=image(w,h,(x,y)=>x>=x0&&x<x1&&y>=y0&&y<y1?255:0,1),ctx={src2:mask};
    try{const out=run('inpaint',src,{method:'laplace'},ctx);try{
      assert.match(ctx.info,/หยุดเมื่อนิ่ง/);
      if(w===240)assert.match(ctx.info,/ภาพใหญ่ จึงลดเหลือ/);
      for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++)assert.ok(Math.abs(out.data[y*w+x]-x)<=1,`${w}x${h} harmonic pixel ${x},${y}`);
    }finally{out.delete();}}finally{src.delete();mask.delete();}
  }
});
test('ip5f-local-laplacian: identity short circuit and beta changes large contrast',()=>{
  const src=image(35,29,(x,y)=>x<17?[20,30,40]:[220,200,180]);try{
    const sameOut=run('local_laplacian',src,{alpha:1,beta:1});try{same(sameOut,src);}finally{sameOut.delete();}
    const a=run('local_laplacian',src,{alpha:.5,beta:1}),b=run('local_laplacian',src,{alpha:.5,beta:.2}),k8=run('local_laplacian',src,{alpha:.5,beta:.2,levels:'8'}),k32=run('local_laplacian',src,{alpha:.5,beta:.2,levels:'32'});
    try{assert.notEqual(hash(a.data),hash(b.data));assert.notDeepEqual(Array.from(k8.data),Array.from(k32.data),'the number of intensity levels must affect remapping');}finally{a.delete();b.delete();k8.delete();k32.delete();}
  }finally{src.delete();}
});
test('ip5f-multissim: identical, flat luminance difference, adaptive scales and missing B',()=>{
  const a=image(45,45,()=>[100,100,100]),b=image(45,45,()=>[150,150,150]);try{
    const c={src2:a},out=run('multissim',a,{},c);try{assert.ok(Math.abs(vals(c).msssim-1)<1e-7);assert.match(c.info,/\(3 scale\)/);assert.equal(passFailVerdict({metric:'msssim',lo:.99,hi:1},{op:'multissim',values:c.values}).result,'OK');}finally{out.delete();}
    const c2={src2:b},o2=run('multissim',a,{},c2);try{assert.ok(vals(c2).msssim>0&&vals(c2).msssim<1);}finally{o2.delete();}
    const no={},pas=run('multissim',a,{},no);try{same(pas,a);assert.equal(no.values.fields.length,0);}finally{pas.delete();}
  }finally{a.delete();b.delete();}
});
test('ip5f-fusion: identical exposures, zero weights give their mean, B is optional',()=>{
  const a=image(33,27,(x,y)=>[x*6,y*8,80]),b=image(33,27,(x,y)=>[x*6+20,y*8+20,120]);try{
    const out=run('exposure_fusion',a,{}, {src2:a});try{same(out,a);}finally{out.delete();}
    const avg=run('exposure_fusion',a,{wContrast:0,wSaturation:0,wExposure:0},{src2:b});try{for(let i=0;i<a.data.length;i++)assert.ok(Math.abs(avg.data[i]-Math.floor((a.data[i]+b.data[i])/2+.5))<=1);}finally{avg.delete();}
    const no=run('exposure_fusion',a);try{same(no,a);}finally{no.delete();}
  }finally{a.delete();b.delete();}
});
test('ip5f-fusion: saturation uses channel standard deviation',()=>{
  const a=image(17,17,()=>[255,0,0]),b=image(17,17,()=>[128,128,0]);
  const ga=new cv.Mat(),gb=new cv.Mat();
  try{
    cv.cvtColor(a,ga,cv.COLOR_RGB2GRAY);cv.cvtColor(b,gb,cv.COLOR_RGB2GRAY);
    const q=fusion5F(cv,a,b,ga,gb,[0,1,0]);
    assert.deepEqual(Array.from(q.data.slice(0,3)),[213,43,0]);
  }finally{ga.delete();gb.delete();a.delete();b.delete();}
});
test('ip5f-fusion: exact half values round upward consistently',()=>{
  const a=image(17,17,()=>[10,10,10]),b=image(17,17,()=>[11,11,11]),ga=new cv.Mat(),gb=new cv.Mat();
  try{cv.cvtColor(a,ga,cv.COLOR_RGB2GRAY);cv.cvtColor(b,gb,cv.COLOR_RGB2GRAY);const q=fusion5F(cv,a,b,ga,gb,[0,0,0]);assert.deepEqual(Array.from(q.data.slice(0,3)),[11,11,11]);}
  finally{ga.delete();gb.delete();a.delete();b.delete();}
});
test('ip5f-edgetaper: both edges use nearest-edge distance and blur weight',()=>{
  const src=image(7,7,(x)=>[x*x*5,x*x*5,x*x*5]);
  try{
    const q=edgeTaper5F(cv,src,{n:3,k:Array(9).fill(1/9)});
    try{
      assert.equal(q.data[(3*7)*3],62,'left edge is fully blurred across the wrapped frame');
      assert.equal(q.data[(3*7+6)*3],102,'right edge is also fully blurred, rather than using left-edge distance');
      assert.equal(q.data[(3*7+1)*3],7,'the interior blend uses alpha*input + (1-alpha)*blur');
    }finally{q.delete();}
  }finally{src.delete();}
});
test('ip5f-blind/reg: constant signal, normalized nonnegative PSF visualization and taper compatibility',()=>{
  const src=image(33,27,()=>[80,100,140]);try{
    for(const id of ['deconv_blind','deconv_reg']){const out=run(id,src,{psfSize:3,iterations:2});try{for(let i=0;i<out.data.length;i++)assert.ok(Math.abs(out.data[i]-src.data[i])<=1);}finally{out.delete();}}
    const psf=run('deconv_blind',src,{psfSize:3,iterations:2,view:'psf'});try{assert.equal(psf.rows,27);assert.equal(psf.cols,27);assert.equal(Math.max(...psf.data),255);}finally{psf.delete();}
    for(const id of ['deconv_wiener','deconv_lucy']){
      const c1={},c2={},one=run(id,src,{},c1),two=run(id,src,{taper:false},c2);try{same(one,two);assert.deepEqual(c1,c2);assert.equal(buildPythonCode([step(id)]),buildPythonCode([step(id,{taper:false})]));}finally{one.delete();two.delete();}
    }
    for(const id of ['deconv_wiener','deconv_lucy','deconv_blind','deconv_reg']){
      const c={},out=run(id,src,{psfSize:3,iterations:2,taper:true},c);
      try{assert.deepEqual([out.cols,out.rows],[src.cols,src.rows]);assert.match(c.info,/edgetaper/);for(let i=0;i<out.data.length;i++)assert.ok(Math.abs(out.data[i]-src.data[i])<=1,`${id} constant signal`);}
      finally{out.delete();}
    }
  }finally{src.delete();}
});
test('ip5f-deconv-reg: regularization keeps its Laplacian penalty at large lambda',()=>{
  const src=image(23,19,(x,y)=>[(x*19+y*7)%256,(x*5+y*23)%256,(x*13+y*3)%256]);
  try{
    for(const lambda of [.6,1]){
      const reg=run('deconv_reg',src,{lambda}),wnr=run('deconv_wiener',src,{nsr:lambda});
      try{assert.notDeepEqual(Array.from(reg.data),Array.from(wnr.data),`lambda ${lambda} must retain |L|²`);}
      finally{reg.delete();wnr.delete();}
    }
  }finally{src.delete();}
});
test('ip5f-blind: PSF refinement retains its multiplicative prior',()=>{
  const src=image(7,6,(x,y)=>((y*7+x)*37+x*19)%256,1);
  try{
    const q=blind5F(cv,src,src,3,2);
    assert.deepEqual(q.psf.map(v=>Math.round(v*1e5)/1e5),[.11274,.14618,.08855,.10781,.14106,.08450,.10287,.13587,.08041]);
  }finally{src.delete();}
});
for(const [w,h,ch] of [[0,0,1],[1,1,1],[9,7,1],[9,7,4]])for(const [id,p] of [...NEW.map(id=>[id,{}]),['inpaint',{method:'laplace'}],['deconv_wiener',{taper:true}],['deconv_lucy',{taper:true}]])test(`ip5f-edges/leaks ${id} ${w}x${h} ch${ch}`,()=>{
  const src=w?image(w,h,(x,y)=>ch===1?(x+y)*15:[x*17,y*21,80,255],ch):new cv.Mat();
  try{assertWrapperBalance(cv,id,c=>run(id,src,p,{src2:src},c).delete());}finally{src.delete();}
});
let python=false,parity=false;
try{execFileSync('python',['-c','pass'],{stdio:'pipe'});python=true;if(!process.env.IP5F_NO_PY){execFileSync('python',['-c','import cv2,numpy'],{stdio:'pipe'});parity=true;}}catch{}
const JOBS=[...NEW.map(id=>[id,{}]),['active_contour',{init:'otsu',view:'mask',smooth:2,bias:.2}],['deconv_blind',{view:'psf',psfSize:3,iterations:2}],['deconv_blind',{psfSize:3,iterations:2,taper:true}],['deconv_reg',{psf:'disk',psfSize:2,taper:true}],['deconv_reg',{lambda:.6}],['deconv_reg',{lambda:1}],['local_laplacian',{alpha:1,beta:.5,levels:'8'}],['inpaint',{method:'laplace'}],['deconv_wiener',{taper:true}],['deconv_lucy',{taper:true}]];
test('ip5f-Python: all exports and composed Pass/Fail pipelines compile',{skip:!python},()=>{
  const codes=JOBS.map(([id,p])=>{const st=step(id,p);if(['multissim','exposure_fusion','inpaint'].includes(id))st.src2={pipeline:[]};return buildPythonCode([st]);});
  const ms=step('multissim');ms.src2={pipeline:[]};codes.push(buildPythonCode([ms,step('pass_fail',{metric:'msssim',lo:.8,hi:1})]));
  execFileSync('python',['-c',"import sys,json\nfor s in json.loads(sys.stdin.buffer.read()): compile(s,'export','exec')"],{input:JSON.stringify(codes),stdio:'pipe'});
});
test('ip5f-parity: every 5F path uses distinct files, bytes-safe stdin and documented sources',{skip:!parity},()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ip5f-parity-'));
  const rgb=image(45,37,(x,y)=>[(x*13+y*3)%256,(x*3+y*17)%256,(x*7+y*11)%256]);
  const other=image(23,19,(x,y)=>[(x*11+y*5+40)%256,(x*7+y*13+20)%256,(x*3+y*19+80)%256]);
  const gray=new cv.Mat(),rgba=new cv.Mat();cv.cvtColor(rgb,gray,cv.COLOR_RGB2GRAY);cv.cvtColor(rgb,rgba,cv.COLOR_RGB2RGBA);
  const tinyH=image(1,9,(x,y)=>[y*23,y*11,255-y*19]),tinyW=image(9,1,(x,y)=>[x*23,255-x*17,x*11]);
  const mask=image(45,37,(x,y)=>{const v=x>=17&&x<=22&&y>=13&&y<=18?255:0;return[v,v,v];});
  const large=image(640,480,(x,y)=>[(x*13+y*3)%256,(x*3+y*17)%256,(x*7+y*11)%256]);
  const cases=[
    {id:'active_contour',src:gray,p:{init:'otsu',view:'mask',smooth:2,bias:.2}},
    {id:'deconv_blind',src:rgba,p:{psfSize:3,iterations:2,taper:true}},
    {id:'deconv_reg',src:rgb,p:{lambda:.6}},
    {id:'deconv_reg',src:rgb,p:{lambda:1}},
    {id:'local_laplacian',src:rgba,p:{alpha:.5,beta:.5,levels:'32'}},
    {id:'multissim',src:gray,b:other,p:{}},
    {id:'exposure_fusion',src:rgb,b:other,p:{}},
    {id:'exposure_fusion',src:rgb,b:other,p:{wContrast:0,wSaturation:0,wExposure:0}},
    {id:'inpaint',src:rgba,b:mask,p:{method:'laplace'}},
    {id:'deconv_wiener',src:tinyH,p:{taper:true}},
    {id:'deconv_lucy',src:tinyW,p:{iterations:2,taper:true}},
    {id:'active_contour',src:large,p:{init:'otsu',view:'mask',smooth:1,iterations:1000}},
  ];
  const apps=[];
  function bytes(m){
    const rgb8=new cv.Mat(),bgr8=new cv.Mat();
    try{if(m.channels()===1)cv.cvtColor(m,rgb8,cv.COLOR_GRAY2RGB);else if(m.channels()===4)cv.cvtColor(m,rgb8,cv.COLOR_RGBA2RGB);else m.copyTo(rgb8);cv.cvtColor(rgb8,bgr8,cv.COLOR_RGB2BGR);return Buffer.from(bgr8.data);}
    finally{rgb8.delete();bgr8.delete();}
  }
  try{
    const manifest=cases.map((j,i)=>{
      const st=step(j.id,j.p),two=!!j.b;if(two)st.src2={pipeline:[]};
      const pipe=j.src.channels()===1?[step('grayscale'),st]:[st];
      const built=buildPythonCode(pipe);let code=built.slice(0,built.indexOf('if __name__ == "__main__":'));
      const inputs=[j.src,...(j.b?[j.b]:[])].map((m,k)=>{
        const file=path.join(dir,`i${i}_${k}.bin`),arg=two?(k?'path_b':'path_a'):'path';fs.writeFileSync(file,bytes(m));
        const before=code;code=code.replace(`cv2.imread(${arg})`,`np.fromfile(${arg}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`);assert.notEqual(code,before,`expected Python read for ${arg}`);return file;
      });
      const file=path.join(dir,`c${i}.py`),out=path.join(dir,`o${i}.bin`);fs.writeFileSync(file,code,'utf8');
      const ctx={...(j.b?{src2:j.b}:{})},app=run(j.id,j.src,j.p,ctx);apps.push(app);
      return {file,inputs,out,app,ctx,id:j.id,p:j.p};
    });
    const manifestFile=path.join(dir,'manifest.json'),resultsFile=path.join(dir,'results.json');
    fs.writeFileSync(manifestFile,JSON.stringify(manifest.map(({file,inputs,out})=>({file,inputs,out}))));
    fs.writeFileSync(path.join(dir,'driver.py'),`import contextlib,io,json,sys\nresults=[]\nfor c in json.load(open(sys.argv[1])):\n ns={'__name__':'case'}\n log=io.StringIO()\n with contextlib.redirect_stdout(log):\n  exec(compile(open(c['file'],encoding='utf8').read(),c['file'],'exec'),ns)\n  result=ns['process'](*c['inputs'])\n result.tofile(c['out'])\n results.append({'shape':result.shape,'stdout':log.getvalue()})\njson.dump(results,open(sys.argv[2],'w'))\n`);
    execFileSync('python',[path.join(dir,'driver.py'),manifestFile,resultsFile],{stdio:'pipe',maxBuffer:1<<25,env:{...process.env,PYTHONIOENCODING:'utf8'}});
    const results=JSON.parse(fs.readFileSync(resultsFile,'utf8'));
    for(let i=0;i<manifest.length;i++){
      const j=manifest[i],py=results[i],data=fs.readFileSync(j.out);
      if(py.shape.length===3)for(let k=0;k<data.length;k+=3)[data[k],data[k+2]]=[data[k+2],data[k]];
      const actual=Array.from(j.app.data),diff=actual.map((v,k)=>Math.abs(v-data[k]));
      assert.deepEqual(py.shape,[j.app.rows,j.app.cols,...(j.app.channels()>1?[j.app.channels()]:[])],j.id+' shape');
      const max=j.id==='active_contour'?0:j.id==='deconv_wiener'?2:1;
      const maxDiff=diff.reduce((m,v)=>Math.max(m,v),0);
      assert.ok(maxDiff<=max,`${j.id} max difference ${maxDiff}`);
      const diffLimit=j.id==='active_contour'?0:.005;
      assert.ok(diff.filter(Boolean).length/diff.length<=diffLimit,`${j.id} differing pixel fraction`);
      if(j.id==='multissim'||j.id==='exposure_fusion'){
        const suffix=' (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)';
        assert.ok(j.ctx.info.endsWith(suffix),j.ctx.info);
        assert.ok(py.stdout.trimEnd().endsWith(suffix),py.stdout);
      }
      if(j.id==='multissim')assert.ok(py.stdout.includes(vals(j.ctx).msssim.toFixed(4).slice(0,-1)),py.stdout);
    }
  }finally{
    for(const app of apps)app.delete();
    rgb.delete();other.delete();gray.delete();rgba.delete();tinyH.delete();tinyW.delete();mask.delete();large.delete();fs.rmSync(dir,{recursive:true,force:true});
  }
});
