// IP batch 5E: behavioral fixtures, frozen clean-tree goldens and Python parity.
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, MORPH_5A_LUT, defaultParams, needsOf, isParamVisible, passFailVerdict, deltaE94, simplifyClosedContour5D } from "../src/renderer/filters/registry.js";
import { TOOL_SECTIONS, OP_TO_FILTER } from "../src/renderer/ui/palette-panel.js";
import { OP_ICON, GROUP_ICON, OP_PRIMARY, OPTION_GROUPS } from "../src/renderer/filters/ui-meta.js";
import { buildPythonCode, PF_VALUE_OPS } from "../src/renderer/codegen.js";
import { jacobi5EPy, color5EPy } from "../src/renderer/codegen-batch5e.js";
import { jacobi5E, inverse3E, decorr5E, pca5E } from "../src/renderer/filters/batch5e.js";
import { assertLeakFree, assertWrapperBalance, makeThrowingCv } from "./_leak-helpers.mjs";
while (typeof cv.Mat !== "function")
    await new Promise((r) => setTimeout(r, 20));
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
const hash = (x) => crypto.createHash("sha256").update(x).digest("hex");
const vals = (ctx) => Object.fromEntries(ctx.values.fields.map((f) => [f.key, f.value]));
function image(w, h, fn, ch = 1) {
    const data = new Uint8Array(w * h * ch);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const v = fn(x, y);
            if (ch === 1)
                data[y * w + x] = v;
            else
                data.set(v, (y * w + x) * ch);
        }
    return cv.matFromArray(h, w, ch === 1 ? cv.CV_8UC1 : ch === 3 ? cv.CV_8UC3 : cv.CV_8UC4, data);
}
function run(id, src, p = {}, ctx = {}, c = cv) { return OPS[id].apply(c, src, { ...defaults(id), ...p }, ctx); }
function step(id, p = {}) {
    const fid = OP_TO_FILTER[id] || id, local = { ...defaults(id), ...p };
    return fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
}
function same(a, b) {
    assert.deepEqual([a.cols, a.rows, a.channels()], [b.cols, b.rows, b.channels()]);
    assert.deepEqual(Buffer.from(a.data), Buffer.from(b.data));
}

const LEGACY = {"morph_erode":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_dilate":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_open":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_close":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_gradient":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_tophat":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_blackhat":{"fid":"morph_tools","qs":[{"key":"shape","type":"select","default":"rect","options":[{"value":"rect"},{"value":"ellipse"},{"value":"cross"},{"value":"diamond"},{"value":"octagon"},{"value":"line"}]},{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1},{"key":"lineAngle","type":"number","default":0}]},"morph_fill_holes":{"fid":"morph_tools","qs":[]},"morph_clear_border":{"fid":"morph_tools","qs":[]},"morph_remove_small":{"fid":"morph_tools","qs":[{"key":"minArea","type":"number","default":150}]},"morph_clean":{"fid":"morph_tools","qs":[]},"morph_majority":{"fid":"morph_tools","qs":[{"key":"iterations","type":"number","default":1}]},"morph_skeleton":{"fid":"morph_tools","qs":[]},"morph_spur":{"fid":"morph_tools","qs":[{"key":"iterations","type":"number","default":1}]},"morph_thin":{"fid":"morph_tools","qs":[{"key":"iterations","type":"number","default":0}]},"morph_shrink":{"fid":"morph_tools","qs":[{"key":"iterations","type":"number","default":0}]},"morph_ult_erode":{"fid":"morph_tools","qs":[]},"morph_reconstruct_open":{"fid":"morph_tools","qs":[{"key":"ksize","type":"number","default":5},{"key":"iterations","type":"number","default":1}]},"morph_convex_hull":{"fid":"morph_tools","qs":[]},"morph_endpoints_branchpoints":{"fid":"morph_tools","qs":[]},"morph_hit_or_miss":{"fid":"morph_tools","qs":[{"key":"hmPattern","type":"select","default":"isolated","options":[{"value":"isolated"},{"value":"hole"},{"value":"corner"},{"value":"line_end"}]},{"key":"hmRotate","type":"boolean","default":true}]},"morph_regional_max":{"fid":"morph_tools","qs":[]},"morph_regional_min":{"fid":"morph_tools","qs":[]},"morph_h_maxima":{"fid":"morph_tools","qs":[{"key":"hHeight","type":"number","default":10}]},"morph_h_minima":{"fid":"morph_tools","qs":[{"key":"hHeight","type":"number","default":10}]},"morph_extended_max":{"fid":"morph_tools","qs":[{"key":"hHeight","type":"number","default":10}]},"morph_extended_min":{"fid":"morph_tools","qs":[{"key":"hHeight","type":"number","default":10}]},"colorspace":{"fid":"color_tools","qs":[{"key":"space","type":"select","default":"hsv","options":[{"value":"hsv"},{"value":"hls"},{"value":"lab"},{"value":"ycrcb"},{"value":"xyz"},{"value":"luv"},{"value":"yuv"}]}]},"colorspace_inverse":{"fid":"color_tools","qs":[{"key":"space","type":"select","default":"hsv","options":[{"value":"hsv"},{"value":"hls"},{"value":"lab"},{"value":"ycrcb"},{"value":"xyz"},{"value":"luv"},{"value":"yuv"}]}]},"white_balance":{"fid":"color_tools","qs":[{"key":"method","type":"select","default":"gray_world","options":[{"value":"gray_world"},{"value":"white_patch"},{"value":"ref_point"}]},{"key":"refX","type":"number","default":50},{"key":"refY","type":"number","default":50},{"key":"refSize","type":"number","default":5}]},"quantize":{"fid":"enhance_tools","qs":[{"key":"levels","type":"number","default":4}]},"hough_circles":{"fid":"analysis_detect","qs":[{"key":"param2","type":"number","default":40},{"key":"minDist","type":"number","default":30},{"key":"minR","type":"number","default":0},{"key":"maxR","type":"number","default":0},{"key":"dp","type":"number","default":1}]},"shape_filter":{"fid":"analysis_detect","qs":[{"key":"prop","type":"select","default":"circularity","options":[{"value":"circularity"},{"value":"solidity"},{"value":"eccentricity"},{"value":"extent"},{"value":"none"}]},{"key":"lo","type":"number","default":0.8},{"key":"hi","type":"number","default":1},{"key":"minArea","type":"number","default":50},{"key":"maxArea","type":"number","default":0},{"key":"view","type":"select","default":"overlay","options":[{"value":"overlay"},{"value":"mask"}]},{"key":"keepN","type":"number","default":0},{"key":"keepRank","type":"select","default":"largest","options":[{"value":"largest"},{"value":"smallest"}]}]},"delta_e":{"fid":"image_quality","qs":[{"key":"formula","type":"select","default":"cie76","options":[{"value":"cie76"},{"value":"ciede2000"},{"value":"cie94"}]},{"key":"compare","type":"select","default":"pixel","options":[{"value":"pixel"},{"value":"mean_b"}]},{"key":"tol","type":"number","default":2.3},{"key":"scaleMax","type":"number","default":20},{"key":"view","type":"select","default":"heatmap","options":[{"value":"heatmap"},{"value":"over"}]}]},"mask":{"fid":"image_arithmetic","qs":[{"key":"invert","type":"boolean","default":false}]},"mask_overlay":{"fid":"image_arithmetic","qs":[{"key":"style","type":"select","default":"fill","options":[{"value":"fill"},{"value":"labels"},{"value":"outline"}]},{"key":"color","type":"select","default":"red","options":[{"value":"red"},{"value":"green"},{"value":"blue"},{"value":"yellow"},{"value":"cyan"},{"value":"magenta"}]},{"key":"opacity","type":"number","default":50},{"key":"thickness","type":"number","default":2}]}};
const OLD_HOSTS = {"morph_tools": ["morph_erode", "morph_dilate", "morph_open", "morph_close", "morph_gradient", "morph_tophat", "morph_blackhat", "morph_fill_holes", "morph_clear_border", "morph_remove_small", "morph_clean", "morph_majority", "morph_skeleton", "morph_spur", "morph_thin", "morph_shrink", "morph_ult_erode", "morph_reconstruct_open", "morph_convex_hull", "morph_endpoints_branchpoints", "morph_hit_or_miss", "morph_regional_max", "morph_regional_min", "morph_h_maxima", "morph_h_minima", "morph_extended_max", "morph_extended_min"], "color_tools": ["colorspace", "colorspace_inverse", "white_balance"], "enhance_tools": ["quantize"], "analysis_detect": ["hough_circles", "shape_filter"], "image_quality": ["delta_e"], "image_arithmetic": ["mask", "mask_overlay"]};
test("ip5e-G. 525 clean-tree cases freeze pixels, info, values and Python bytes", () => {
    const result = {};
    for (const [kind, w, h] of [["rgb", 37, 29], ["gray", 37, 29], ["rgba", 37, 29], ["rgb", 1, 1], ["empty", 0, 0]]) {
        const data = Uint8Array.from({ length: w * h * 3 }, (_, i) => (Math.imul(i + 11, 73) ^ (i >> 4) * 17) & 255);
        const c = w ? cv.matFromArray(h, w, cv.CV_8UC3, data) : new cv.Mat(), src = new cv.Mat();
        if (kind === "gray")
            cv.cvtColor(c, src, cv.COLOR_RGB2GRAY);
        else if (kind === "rgba")
            cv.cvtColor(c, src, cv.COLOR_RGB2RGBA);
        else
            c.copyTo(src);
        try {
            for (const [fid, ids] of Object.entries(OLD_HOSTS))
                for (const id of ids) {
                    const qs = LEGACY[id].qs, defs = Object.fromEntries(qs.map(q => [q.key, q.default]));
                    const variants = [{}, ...qs.filter(q => q.type === "select").flatMap(q => q.options.filter(o => o.value !== q.default && !(id === "pass_fail" && ["mean", "std", "entropy", "corr2"].includes(o.value))).map(o => ({ [q.key]: o.value })))];
                    for (const extra of variants) {
                        const local = { ...defs, ...extra }, params = { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) }, st = { id: fid, params };
                        const key = JSON.stringify([kind, w, h, fid, id, extra]), ctx = { src2: c };
                        let g = null;
                        try {
                            if (w && needsOf(FILTERS[fid], params) === "gray" && src.channels() > 1) {
                                g = new cv.Mat();
                                cv.cvtColor(src, g, src.channels() === 4 ? cv.COLOR_RGBA2GRAY : cv.COLOR_RGB2GRAY);
                            }
                            // Salt/pepper uses Math.random. Seed only that op; original 85eafc5 source
                            // was replayed from OS temp to freeze these five stochastic pixel cases.
                            const random = Math.random;
                            let seed = 12345, out;
                            try {
                                if (id === "salt_pepper_noise")
                                    Math.random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
                                out = FILTERS[fid].apply(cv, g || src, params, ctx);
                            }
                            finally {
                                Math.random = random;
                            }
                            try {
                                result[key] = { image: hash(Buffer.from(out.data)), rows: out.rows, cols: out.cols, ch: out.channels(), info: ctx.info, values: ctx.values, python: hash(buildPythonCode([st])) };
                            }
                            finally {
                                out.delete();
                            }
                        }
                        catch (e) {
                            result[key] = { error: typeof e === "number" ? "OpenCV native exception" : String(e), python: hash(buildPythonCode([st])) };
                        }
                        finally {
                            g?.delete();
                        }
                    }
                }
        }
        finally {
            src.delete();
            c.delete();
        }
    }
    assert.equal(Object.keys(result).length, 525);
    assert.equal(hash(JSON.stringify(result)), "960003dba59607200d894bcd0bf72c387f5dd17da374af96ab67522a45b54122");
});
const NEW = ['morph_perimeter','morph_thicken','morph_keep_border','reconstruct','select_object','srgb_linear','decorrstretch','color_matrix'];
const EXT = [['hough_circles',{view:'mask'}],['colorspace',{space:'yiq'}],['colorspace_inverse',{space:'yiq'}],['white_balance',{method:'pca'}],['delta_e',{formula:'angle'}]];
test('ip5e-schema. placement, metadata, ranges and scalar wiring',()=>{
    for(const [id,after] of [['morph_perimeter','morph_fill_holes'],['morph_thicken','morph_shrink'],['morph_keep_border','morph_clear_border'],['reconstruct','mask_overlay'],['select_object','shape_filter'],['srgb_linear','colorspace_inverse'],['decorrstretch','white_balance'],['color_matrix','rgb_adjust']]){
        assert.ok(OPS[id],id);const section=TOOL_SECTIONS.find(s=>s.ops.includes(id)),host=OP_TO_FILTER[id];
        assert.equal(section.ops[section.ops.indexOf(after)+1],id);
        assert.deepEqual(FILTERS[host].params[0].options.map(o=>o.value),section.ops);
        assert.ok(OPS[id].desc&&OPS[id].modeHint);assert.match(OP_PRIMARY[id](defaults(id)),/[ก-๙]/);
        assert.equal(Object.values(OP_ICON).filter(x=>x===OP_ICON[id]).length,1);
        assert.ok(!Object.values(GROUP_ICON).includes(OP_ICON[id]));
        for(const q of OPS[id].params)assert.ok(q.label&&(q.desc||q.hint),id+q.key);
        if(OPTION_GROUPS[host+'.mode'])assert.ok(OPTION_GROUPS[host+'.mode'].some(g=>g[1].includes(id)),id);
    }
    for(const [id,p] of EXT)for(const [key,val] of Object.entries(p)){const q=OPS[id].params.find(q=>q.key===key);assert.ok(q);if(q.options)assert.equal(q.options.at(-1).value,val);}
    assert.equal(defaults('hough_circles').view,'overlay');
    assert.equal(defaults('morph_perimeter').perimConn,'4');assert.equal(defaults('morph_thicken').iterations,0);assert.ok(PF_VALUE_OPS.has('select_object'));
    for(const [id,key,range] of [['select_object','x',[0,100,.5]],['decorrstretch','tolPct',[0,10,.5]],['white_balance','pcaPercent',[.5,20,.5]],['color_matrix','m11',[-3,3,.01]],['color_matrix','o1',[-255,255,1]]]){const q=OPS[id].params.find(q=>q.key===key);assert.deepEqual([q.min,q.max,q.step],range);}
});
test('ip5e-perimeter. four/eight neighbors and exterior background',()=>{
    const src=image(5,5,(x,y)=>x===0&&y===0?0:255);
    try{for(const perimConn of ['4','8']){const ctx={},out=run('morph_perimeter',src,{perimConn},ctx);try{
        const expected=[];for(let y=0;y<5;y++)for(let x=0;x<5;x++){let edge=false;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if((dx||dy)&&(perimConn==='8'||!dx||!dy)){const xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=5||yy>=5||!src.data[yy*5+xx])edge=true;}expected.push(src.data[y*5+x]&&edge?255:0);}
        assert.deepEqual(Array.from(out.data),expected);assert.equal(ctx.info,`เส้นรอบ ${expected.filter(Boolean).length} px`);
    }finally{out.delete();}}}finally{src.delete();}
});
// Independent scalar thin oracle: visit only image pixels, leaving its two-pixel
// complement pad fixed in both parallel substeps. No deletions start at the frame.
function fixedPadThin5E(src,w,h,iterations){
    const a=Uint8Array.from(src,v=>v>127?1:0),offsets=[1,1-w,-w,-1-w,-1,w-1,w,w+1];
    for(let round=0;round<(iterations||w*h);round++){
        let changed=false;
        for(let step=0;step<2;step++){
            const remove=[];
            for(let y=2;y<h-2;y++)for(let x=2;x<w-2;x++){
                const i=y*w+x;if(!a[i])continue;let mask=0;
                for(let bit=0;bit<8;bit++)mask|=a[i+offsets[bit]]<<bit;
                if(MORPH_5A_LUT.thin[step][mask])remove.push(i);
            }
            for(const i of remove)a[i]=0;changed||=remove.length>0;
        }
        if(!changed)break;
    }
    return Uint8Array.from(a,v=>v*255);
}
test('ip5e-thicken. growth, component preservation, fixed-pad duality and guarded convergence',()=>{
    const src=image(25,17,(x,y)=>((x>=4&&x<=6)||(x>=18&&x<=20))&&y>=6&&y<=9?255:0);
    try{for(const iterations of [1,2,3,0]){const out=run('morph_thicken',src,{iterations}),pad=image(29,21,(x,y)=>x<2||y<2||x>=27||y>=19?255:255-src.data[(y-2)*25+x-2]),thin=fixedPadThin5E(pad.data,29,21,iterations),labels=new cv.Mat();try{
        assert.equal(cv.connectedComponents(out,labels,8)-1,2);assert.ok(cv.countNonZero(out)>cv.countNonZero(src));
        // Fixed padding also removes the old three-round border-hole exception.
        for(let y=0;y<17;y++)for(let x=0;x<25;x++)assert.equal(out.data[y*25+x],255-thin[(y+2)*29+x+2]);
        for(let i=0;i<src.data.length;i++)assert.ok(out.data[i]>=src.data[i]);
        if(!iterations){const stable=run('morph_thicken',out,{iterations:0});try{same(stable,out);}finally{stable.delete();}}
    }finally{out.delete();pad.delete();labels.delete();}}}finally{src.delete();}
});

test('ip5e-thicken-topology. boundary growth preserves components and holes',()=>{
    // Independent flood counts: foreground uses 8 neighbors; padded background uses 4.
    function topology(m){
        const w=m.cols+2,h=m.rows+2,a=new Uint8Array(w*h);
        for(let y=0;y<m.rows;y++)for(let x=0;x<m.cols;x++)a[(y+1)*w+x+1]=m.data[y*m.cols+x]>127?1:0;
        function count(value,diagonal){let n=0;const seen=new Uint8Array(a.length);
            for(let i=0;i<a.length;i++)if(!seen[i]&&a[i]===value){n++;seen[i]=1;const q=[i];
                for(let k=0;k<q.length;k++){const x=q[k]%w,y=Math.floor(q[k]/w);
                    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if((dx||dy)&&(diagonal||!dx||!dy)){
                        const xx=x+dx,yy=y+dy,j=yy*w+xx;
                        if(xx>=0&&yy>=0&&xx<w&&yy<h&&!seen[j]&&a[j]===value){seen[j]=1;q.push(j);}
                    }
                }
            }return n;
        }
        return [count(1,true),count(0,false)-1];
    }
    const twoBlobs=Array.from({length:17},(_,y)=>Array.from({length:25},(_,x)=>((x>=4&&x<=6)||(x>=18&&x<=20))&&y>=6&&y<=9?'#':'.').join(''));
    for(const rows of [['.###','#...','#...','....'],['#...','....','....','....'],['##.#','....','....','....'],['#.#.','#...','....','....'],['.....','.###.','.#.#.','.###.','.....'],twoBlobs]){
        const src=image(rows[0].length,rows.length,(x,y)=>rows[y][x]==='#'?255:0);
        try{const expected=topology(src);for(const iterations of [1,2,3,0]){const out=run('morph_thicken',src,{iterations});try{assert.deepEqual(topology(out),expected,JSON.stringify({rows,iterations}));}finally{out.delete();}}}finally{src.delete();}
    }
});
test('ip5e-border/select. complementary border mask, point selection, background and connectivity',()=>{
    const src=image(9,7,(x,y)=>(x<2&&y<2)||(x>=4&&x<=6&&y>=3&&y<=5)?255:0);
    try{const ctx={},keep=run('morph_keep_border',src,{},ctx),clear=run('morph_clear_border',src);try{assert.equal(ctx.info,'เก็บ 1 ชิ้นที่แตะขอบ');for(let i=0;i<src.data.length;i++)assert.equal(keep.data[i]+clear.data[i],src.data[i]);}finally{keep.delete();clear.delete();}
        for(const [x,y,area] of [[50,50,9],[100,100,0]]){const c={},out=run('select_object',src,{x,y},c);try{assert.equal(cv.countNonZero(out),area);assert.equal(vals(c).area,area);if(!area)assert.equal(c.info,'จุดที่เลือกไม่อยู่บนวัตถุ');}finally{out.delete();}}
        const over=run('select_object',src,{view:'overlay'});try{assert.equal(over.channels(),3);assert.deepEqual(Array.from(over.data.slice(0,3)),[90,90,90]);}finally{over.delete();}
    }finally{src.delete();}
    const diag=image(5,5,(x,y)=>x===y?255:0);try{for(const conn of ['4','8']){const c={},out=run('select_object',diag,{x:0,y:0,conn},c);try{assert.equal(vals(c).area,conn==='4'?1:5);}finally{out.delete();}}}finally{diag.delete();}
});
test('ip5e-reconstruct. grayscale propagation, clipping, nearest resize and missing B',()=>{
    const mask=image(7,3,x=>x===3?20:100),marker=image(7,3,(x,y)=>x===1&&y===1?150:0);
    try{const c={src2:mask},out=run('reconstruct',marker,{},c);try{assert.deepEqual(Array.from(out.data.slice(0,7)),[100,100,100,20,20,20,20]);assert.match(c.info,/ตัด marker/);}finally{out.delete();}
        const missing={},pass=run('reconstruct',marker,{},missing);try{same(pass,marker);assert.match(missing.info,/B/);}finally{pass.delete();}
        const small=image(1,1,()=>75);try{const c={src2:small},out=run('reconstruct',marker,{},c);try{assert.ok(Array.from(out.data).every(v=>v===75));assert.match(c.info,/ขนาด/);}finally{out.delete();}}finally{small.delete();}
    }finally{mask.delete();marker.delete();}
});
test('ip5e-YIQ. forward formula and inverse round-trip within two bytes',()=>{
    const src=image(20,11,(x,y)=>[(x*13+y*3)%256,(y*23+x*7)%256,(x*5+y*19)%256],3);
    try{const enc=run('colorspace',src,{space:'yiq'}),out=run('colorspace_inverse',enc,{space:'yiq'});try{
        assert.ok(Array.from(src.data).every((v,i)=>Math.abs(v-out.data[i])<=2));
        const [r,g,b]=Array.from(src.data.slice(3,6),v=>v/255);assert.deepEqual(Array.from(enc.data.slice(3,6)),[(.299*r+.587*g+.114*b)*255,(.596*r-.274*g-.322*b)*255/(2*.596)+127.5,(.211*r-.523*g+.312*b)*255/(2*.523)+127.5].map(v=>Math.max(0,Math.min(255,Math.floor(v+.5)))));
    }finally{enc.delete();out.delete();}}finally{src.delete();}
});
test('ip5e-LUT/matrix. every byte, identity, signed offsets and channel order',()=>{
    const src=image(256,1,x=>[x,255-x,x*7%256],3);
    try{for(const direction of ['to_linear','to_srgb']){const out=run('srgb_linear',src,{direction});try{assert.deepEqual(Array.from(out.data),Array.from(src.data,v=>{const c=v/255;return Math.floor(255*(direction==='to_linear'?(c<=.04045?c/12.92:((c+.055)/1.055)**2.4):(c<=.0031308?12.92*c:1.055*c**(1/2.4)-.055))+.5);}));}finally{out.delete();}}
        const out=run('color_matrix',src);try{same(out,src);}finally{out.delete();}
        const swap=run('color_matrix',src,{m11:0,m13:1,m33:0,m31:1,o2:-30});try{for(let i=0;i<src.data.length;i+=3)assert.deepEqual(Array.from(swap.data.slice(i,i+3)),[src.data[i+2],Math.max(0,src.data[i+1]-30),src.data[i]]);}finally{swap.delete();}
    }finally{src.delete();}
});
test('ip5e-decorrelation/PCA. constant singular data, gray and gain correction',()=>{
    const src=image(8,5,()=>[100,50,25],3);try{const c={},out=run('decorrstretch',src,{},c);try{same(out,src);assert.match(c.info,/เสื่อม|ซ้ำ|singular/);}finally{out.delete();}
        const wb={},balanced=run('white_balance',src,{method:'pca'},wb);try{assert.deepEqual(Array.from(balanced.data.slice(0,3)),[58,58,58]);assert.match(wb.info,/PCA/);}finally{balanced.delete();}
    }finally{src.delete();}
});
test('ip5e-circles. mask uses exactly the overlay detector centers/radii and filled LINE_8',()=>{
    const src=image(96,96,()=>0),expected=new cv.Mat(96,96,cv.CV_8UC1,new cv.Scalar(0)),circles=[];
    cv.circle(src,new cv.Point(48,48),22,new cv.Scalar(255),3);
    const proxy=new Proxy(cv,{get(t,k){if(k==='circle')return (dst,center,r,color,...rest)=>{if(color[1]===255&&color[0]===0)circles.push([center.x,center.y,r]);return t.circle(dst,center,r,color,...rest);};return t[k];}});
    try{const p={param2:10,minR:15,maxR:30,minDist:30};run('hough_circles',src,p,{},proxy).delete();assert.ok(circles.length);
        for(const [x,y,r] of circles)cv.circle(expected,new cv.Point(x,y),r,new cv.Scalar(255),-1,cv.LINE_8);
        const out=run('hough_circles',src,{...p,view:'mask'});try{same(out,expected);}finally{out.delete();}
    }finally{src.delete();expected.delete();}
});
test('ip5e-errors. injected failures and throwing context setters release every Mat',()=>{
    const src=image(17,13,(x,y)=>[x*11,y*13,90,255],4),b=image(5,3,()=>80),gray=image(17,13,()=>0);
    try{for(const [id,p,method,input] of [['reconstruct',{},'resize',src],['select_object',{view:'overlay'},'line',gray],['morph_thicken',{},'connectedComponents',gray],['morph_keep_border',{},'connectedComponentsWithStats',gray],['hough_circles',{view:'mask'},'findContours',gray]])assertWrapperBalance(cv,id+method,c=>run(id,input,p,{src2:b},makeThrowingCv(c,method)),{expectThrow:/injected test failure/});
        for(const [id,p] of [...NEW.map(id=>[id,{}]),...EXT].filter(([id])=>id!=='colorspace'))assertWrapperBalance(cv,id+' info setter',c=>{const ctx={src2:b};Object.defineProperty(ctx,'info',{set(){throw Error('context failure');}});workerRun(id,src,p,ctx,c);},{expectThrow:/context failure/,minCreated:0});
    }finally{src.delete();b.delete();gray.delete();}
});
test('ip5e-native-leaks. normal color/morphology paths preserve native heap census',()=>{
    const src=image(17,13,(x,y)=>[x*11,y*13,90,255],4),b=image(5,3,()=>80);
    try{for(const [id,p] of [...NEW.map(id=>[id,{}]),...EXT])assertLeakFree(cv,id,c=>workerRun(id,src,p,{src2:b},c).delete(),{iters:20,warmup:5});}finally{src.delete();b.delete();}
});
test('ip5e-defaults. missing and explicit new defaults preserve exported Python and pixels',()=>{
    const src=image(11,9,(x,y)=>(x*13+y*23)%256);
    try{for(const [id,key,value] of [['hough_circles','view','overlay']]){
        const old=defaults(id);delete old[key];const a={},b={},x=OPS[id].apply(cv,src,old,a),y=run(id,src,{[key]:value},b);
        try{same(x,y);assert.deepEqual(a,b);const st=step(id);delete st.params[id+'__'+key];assert.equal(buildPythonCode([st]),buildPythonCode([step(id,{[key]:value})]));}finally{x.delete();y.delete();}
    }}finally{src.delete();}
});
test('ip5e-select-PF. selected area supports OK, NG and missing values',()=>{
    assert.equal(OP_PRIMARY.pass_fail({metric:'area',cond:'ge',lo:9}),'พื้นที่วัตถุที่เลือก ผ่านเมื่อ ≥ 9');
    const src=image(5,5,(x,y)=>x>=1&&x<=3&&y>=1&&y<=3?255:0);
    try{const c={},out=run('select_object',src,{},c);out.delete();const upstream={op:'select_object',values:c.values};assert.equal(passFailVerdict({metric:'area',cond:'ge',lo:9},upstream).result,'OK');assert.equal(passFailVerdict({metric:'area',cond:'ge',lo:10},upstream).result,'NG');assert.equal(passFailVerdict({metric:'area'},null).result,'NG');}finally{src.delete();}
});

test('ip5e-empty-B-export. a following grayscale step handles color passthrough',()=>{
    for(const [id,p] of [['reconstruct',{}],['delta_e',{formula:'angle'}]]){
        const first=step(id,p);first.src2={pipeline:[]};
        const code=buildPythonCode([first,step('grayscale')]);
        assert.match(code.slice(code.indexOf('(grayscale)')),/img = cv2.cvtColor\(img, cv2.COLOR_BGR2GRAY\)/,id);
        // Also compile the same transition inside a nested B chain.
        const outer=step('reconstruct');outer.src2={pipeline:[first,step('grayscale')]};
        const nested=buildPythonCode([outer]);
        assert.match(nested,/img_b = cv2.cvtColor\(img_b, cv2.COLOR_BGR2GRAY\)/,id+' nested B');
    }
});
test('ip5e-decorrelation. independent channels stay unchanged, singular channels remain finite',()=>{
    const src=image(8,1,x=>[30+(x&1)*30,80+((x>>1)&1)*30,140+((x>>2)&1)*30],3),singular=image(11,7,(x,y)=>[x*11+y*3,x*11+y*3,90],3);
    try{const identity=run('decorrstretch',src);try{same(identity,src);}finally{identity.delete();}for(const tolPct of [0,2,10]){const c={},out=run('decorrstretch',singular,{tolPct},c);try{assert.match(c.info,/เสื่อม/);assert.ok(Array.from(out.data).some(v=>v>0));}finally{out.delete();}}}finally{src.delete();singular.delete();}
});
// R2: an analytic inverse square root of [[1,.8,0],[.8,1,0],[0,0,1]].
function unequalBands5E() {
    return image(8,1,i=>{const x=(i&1)?1:-1,y=(i&2)?1:-1,z=(i&4)?1:-1;return [100+20*x,100+40*x+30*y,120+10*z];},3);
}
function correlationReference5E() {
    const a=(1/Math.sqrt(1.8)+1/Math.sqrt(.2))/2,b=(1/Math.sqrt(1.8)-1/Math.sqrt(.2))/2;
    return Buffer.from(Array.from({length:8},(_,i)=>{const x=(i&1)?1:-1,y=(i&2)?1:-1,z=(i&4)?1:-1;
        return [100+a*20*x+b*(20/50)*(40*x+30*y),100+b*(50/20)*20*x+a*(40*x+30*y),120+10*z].map(v=>Math.max(0,Math.min(255,Math.floor(v+.5))));
    }).flat());
}
test('ip5e-R2-correlation. MATLAB default matches analytic unequal-variance reference',()=>{
    const q=OPS.decorrstretch.params.find(q=>q.key==='mode');assert.equal(q?.default,'correlation');assert.deepEqual(q.options.map(o=>o.value),['correlation','covariance']);
    const src=unequalBands5E();try{const a=run('decorrstretch',src),b=run('decorrstretch',src,{mode:'correlation'}),c=run('decorrstretch',src,{mode:'covariance'});
        try{assert.deepEqual(Buffer.from(a.data),correlationReference5E());same(a,b);assert.notDeepEqual(Buffer.from(a.data),Buffer.from(c.data));}finally{a.delete();b.delete();c.delete();}
        const zero=image(8,1,i=>[src.data[i*3],src.data[i*3+1],90],3);try{const ctx={},out=run('decorrstretch',zero,{},ctx);try{assert.match(ctx.info,/เสื่อม/);for(let i=2;i<out.data.length;i+=3)assert.equal(out.data[i],90);}finally{out.delete();}}finally{zero.delete();}
    }finally{src.delete();}
});
test('ip5e-R2-covariance/PCA. frozen pre-R2 output and projection ties stay exact',()=>{
    const src=image(17,13,(x,y)=>[(x*17+y*9)%255,(x*7+y*23)%255,(x*29+y*3)%255],3);
    try{const hashes=['7290615d35d165a03d7a7aba98e2e5257fbd6ee9c98bce1f15101207ad9cdba8','0ab48031531ff396afd4689cbfadd071b81c28371ad3eebacbe4a76a94fbc01b','76f1e0784c21321af17ab032cb143220caf358503c905710baf3af54a57b8370'];
        for(const [i,tolPct] of [0,2,10].entries()){const out=run('decorrstretch',src,{mode:'covariance',tolPct});try{assert.equal(hash(out.data),hashes[i]);}finally{out.delete();}}
    }finally{src.delete();}
    const ties=Uint8Array.from([[30,60,90],[30,90,60],[60,90,30],[90,30,60],[90,60,30],[60,30,90],[255,0,0],[0,255,0],[0,0,255]].flat());
    for(const percent of [.5,3.5,20])assert.deepEqual(pca5E(ties,percent),{light:[0.408248290463863,0.408248290463863,0.8164965809277261],gains:[1.3333333333333333,1.3333333333333333,0.6666666666666666]});
});
test('ip5e-R2-thicken-frame. a centered 3x3 blob fills 15x11 without border dots',()=>{
    const src=image(15,11,(x,y)=>x>=6&&x<=8&&y>=4&&y<=6?255:0);try{const out=run('morph_thicken',src,{iterations:0});
        try{assert.deepEqual(Array.from(out.data),Array(165).fill(255));}finally{out.delete();}
        const before=['695e020747ba6733523b19db1d02af189aec89fa0cfb7be2a92c9140dabc0557','d9bfd8fddc7494c4d96f29d91e405afea5ec51d4e8f94732ab9915b3e7709511'];
        for(const iterations of [1,2]){const finite=run('morph_thicken',src,{iterations});try{assert.equal(hash(finite.data),before[iterations-1]);}finally{finite.delete();}}
    }finally{src.delete();}
});
test('ip5e-R2-angle-labels. exactly one control per key uses the formula unit in every host',()=>{
    for(const [fid,f] of Object.entries(FILTERS)){
        const formula=f.params.find(q=>q.key==='formula'||q.key==='delta_e__formula');if(!formula||!formula.options.some(o=>o.value==='angle'))continue;
        const prefix=formula.key.slice(0,-'formula'.length);
        for(const value of ['angle','cie76','cie94','ciede2000']){
            const p={...defaultParams(fid),[formula.key]:value,...(prefix?{mode:'delta_e'}:{})};
            for(const key of ['tol','scaleMax']){const qs=f.params.filter(q=>q.key===prefix+key&&isParamVisible(q,p,f.params));assert.equal(qs.length,1,fid+key);
                assert.match(qs[0].label,value==='angle'?/องศา/:/ΔE/);if(value==='angle')assert.doesNotMatch(qs[0].hint||'',/CIE76|ΔE/);
            }
        }
    }
});
test('ip5e-R2-helper-count. uncertain channels do not duplicate later kernels or Jacobi',()=>{
    for(const [id,p] of [['reconstruct',{}],['delta_e',{formula:'angle'}]]){
        const first=step(id,p);first.src2={pipeline:[]};
        const pipe=[first,step('decorrstretch'),step('srgb_linear'),step('color_matrix')];
        for(const stages of [pipe,[...pipe,step('white_balance',{method:'pca'}),step('decorrstretch')]]){
            const code=buildPythonCode(stages);assert.equal((code.match(/def jacobi5e\(/g)||[]).length,1);assert.equal((code.match(/lut5e = \[\]/g)||[]).length,1);assert.equal((code.match(/m5e, o5e = /g)||[]).length,1);
            const outer=step('reconstruct');outer.src2={pipeline:stages};const nested=buildPythonCode([outer,...stages]);assert.equal((nested.match(/def jacobi5e\(/g)||[]).length,1);
        }
    }
});
test('ip5e-angle. orthogonal/zero RGB vectors, degrees, mean reference and scalars',()=>{
    const a=image(3,1,x=>[[255,0,0],[0,0,0],[100,0,0]][x],3),b=image(3,1,()=>[0,100,0],3);try{for(const compare of ['pixel','mean_b']){const c={src2:b},out=run('delta_e',a,{formula:'angle',compare,scaleMax:100},c);try{assert.deepEqual(Array.from(out.data),[230,0,230]);assert.equal(vals(c).de_mean,60);assert.match(c.info,/องศา/);assert.match(c.info,/60\.000/);assert.match(c.values.fields[0].label,/องศา/);assert.match(c.values.fields[3].label,/R/);}finally{out.delete();}}}finally{a.delete();b.delete();}
});
for(const [w,h,ch] of [[0,0,1],[1,1,1],[9,7,1],[9,7,4]])for(const [id,p] of [...NEW.map(id=>[id,{}]),...EXT])test(`ip5e-edges/leaks ${id} ${w}x${h} ch${ch}`,()=>{
    const src=w?image(w,h,(x,y)=>ch===1?(x+y)*21:[x*17,y*29,80,255],ch):new cv.Mat();
    try{assertWrapperBalance(cv,id,c=>{const st=step(id,p);let g=null;try{if(w&&needsOf(FILTERS[st.id],st.params)==='gray'&&ch>1){g=new c.Mat();c.cvtColor(src,g,c.COLOR_RGBA2GRAY);}run(id,g||src,p,{src2:src},c).delete();}finally{g?.delete();}});}finally{src.delete();}
});
let syntaxPython=false,pyAvailable=false;try{execFileSync('python',['-c','pass'],{stdio:'pipe'});syntaxPython=true;if(!process.env.IP5E_NO_PY){execFileSync('python',['-c','import cv2,numpy'],{stdio:'pipe'});pyAvailable=true;}}catch{}
test('ip5e-Jacobi. standard-library Python and JS have matching eigenpairs without numpy',{skip:!syntaxPython},()=>{
    const matrices=[[[2,1,0],[1,3,1],[0,1,5]],[[1,1,1],[1,1,1],[1,1,1]],[[0,0,0],[0,0,0],[0,0,0]],[[3421.25,-18.1,91.25],[-18.1,523.5,-16.7],[91.25,-16.7,931.3]]];
    const code=jacobi5EPy().join('\n')+"\nimport sys,json\njson.dump([jacobi5e(m) for m in json.loads(sys.stdin.buffer.read())],sys.stdout)";
    const results=JSON.parse(execFileSync('python',['-c',code],{input:JSON.stringify(matrices),stdio:'pipe'}));
    for(let i=0;i<matrices.length;i++){const js=jacobi5E(matrices[i]);assert.deepEqual(results[i],[js.values,js.vectors]);}
    const a=[[.299,.587,.114],[.596,-.274,-.322],[.211,-.523,.312]],inv=inverse3E(a);for(let r=0;r<3;r++)for(let c=0;c<3;c++)assert.ok(Math.abs(a[r].reduce((s,x,k)=>s+x*inv[k][c],0)-(r===c?1:0))<1e-14);
});
test('ip5e-covariance-HD. identical float64 multiplication before division at the singular threshold',{skip:!syntaxPython},()=>{
    const groups=[[[167,137,55],238847],[[166,159,11],898737],[[103,36,103],936016]],n5e=1920*1080,sums5e=[0,0,0],cross5e=[[0,0,0],[0,0,0],[0,0,0]],data=new Uint8Array(n5e*3);let index=0;
    for(const [rgb,count] of groups){for(let r=0;r<3;r++){sums5e[r]+=rgb[r]*count;for(let c=0;c<3;c++)cross5e[r][c]+=rgb[r]*rgb[c]*count;}for(let j=0;j<count;j++)data.set(rgb,(index++)*3);}
    const cov=cross5e.map((row,r)=>row.map((s,c)=>(s-sums5e[r]*sums5e[c]/n5e)/(n5e-1))),js=jacobi5E(cov);
    const expression=color5EPy('decorrstretch',{},true).find(s=>s.trimStart().startsWith('cov5e =')).trim();
    const code=jacobi5EPy().join('\n')+`\nimport sys,json\nn5e,sums5e,cross5e=json.loads(sys.stdin.buffer.read())\n${expression}\njson.dump(jacobi5e(cov5e),sys.stdout)`;
    const py=JSON.parse(execFileSync('python',['-c',code],{input:JSON.stringify([n5e,sums5e,cross5e]),stdio:'pipe'}));
    assert.deepEqual(py,[js.values,js.vectors]);assert.equal(decorr5E(data,0,'covariance').singular,py[0].some(e=>e<=1e-12));assert.ok(py[0].some(e=>e<=1e-12));
});
test('ip5e-R2-arithmetic. exported correlation/covariance arithmetic matches JS without numpy',{skip:!syntaxPython},()=>{
    const src=unequalBands5E(),fixtures=[Array.from(src.data),Array(24).fill(90),Array.from(src.data,(_,i)=>i%3===2?90:src.data[i])];src.delete();
    for(const mode of ['correlation','covariance']){
        const lines=color5EPy('decorrstretch',{mode},true),start=lines.findIndex(s=>s.trimStart().startsWith('data5e =')),end=lines.findIndex(s=>s.trimStart().startsWith('x5e ='));
        const arithmetic=lines.slice(start+1,end).map(s=>s.slice(4)).join('\n');
        const code=jacobi5EPy().join('\n')+`\nimport sys,json\nflat=json.loads(sys.stdin.buffer.read())\ndata5e=[flat[i:i+3] for i in range(0,len(flat),3)]\n${arithmetic}\nout=[]\nfor x in data5e:\n    for c in range(3):\n        raw=(x[0]-mean5e[0])*t5e[0][c]+(x[1]-mean5e[1])*t5e[1][c]+(x[2]-mean5e[2])*t5e[2][c]+mean5e[c]\n        out.append(max(0,min(255,math.floor(raw+.5))))\njson.dump(out,sys.stdout)`;
        for(const data of fixtures){const py=JSON.parse(execFileSync('python',['-c',code],{input:JSON.stringify(data),stdio:'pipe'}));assert.deepEqual(Array.from(decorr5E(Uint8Array.from(data),0,mode).data),py);}
    }
});
test('ip5e-R2-export-math-scope. a later sRGB import cannot shadow earlier color arithmetic',{skip:!syntaxPython},()=>{
    // Execute the actual exported process up to its first sqrt. Tiny input stubs
    // avoid requiring cv2/numpy; the arithmetic itself is checked independently.
    const harness=`import sys,json,types,math
class NumericReached(Exception): pass
def reached(value): raise NumericReached()
math.sqrt=reached
class Pixels:
    rows=[[10,20,30],[30,50,80]]
    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)
    def __ne__(self,other): return [True,True]
    def __getitem__(self,key): return self.rows[key[0]][key[1]] if isinstance(key,tuple) else self.rows[key]
class Img:
    ndim,size,shape=3,6,(1,2,3)
    def __getitem__(self,key): return self
    def reshape(self,*shape): return Pixels()
cv=types.ModuleType('cv2'); cv.imread=lambda path:Img()
np=types.ModuleType('numpy'); np.ascontiguousarray=lambda x:x
np.all=lambda x,axis:x; np.flatnonzero=lambda x:[i for i,v in enumerate(x) if v]
sys.modules['cv2']=cv; sys.modules['numpy']=np
for code in json.loads(sys.stdin.buffer.read()):
    ns={'__name__':'scope_test'}; exec(code,ns)
    try: ns['process']('input')
    except NumericReached: pass
    else: raise AssertionError('did not reach color arithmetic')
`;
    const pipes=[['decorrstretch',{}],['white_balance',{method:'pca'}]].map(([id,p])=>buildPythonCode([step(id,p),step('srgb_linear'),step('color_matrix')]));
    execFileSync('python',['-c',harness],{input:JSON.stringify(pipes),stdio:'pipe'});
});
test('ip5e-R2-parity-manifest. composed two-input jobs prepare even without Python cv2',()=>{
    const src=image(3,2,()=>[10,20,30],3),empty=new cv.Mat(),mask=image(3,2,()=>[40,50,60],3);
    try{for(const [id,p] of [['reconstruct',{}],['delta_e',{formula:'angle'}]])for(const b of [empty,mask]){
        const first=step(id,p);first.src2={pipeline:[]};const pipe=[first,step('decorrstretch'),step('srgb_linear'),step('color_matrix')];
        const codes=pythonCases([{id,p,src,b,pipe}],{prepareOnly:true});assert.equal(codes.length,1);assert.match(codes[0],/def process\(path_a, path_b\)/);
    }}finally{src.delete();empty.delete();mask.delete();}
});
test('ip5e-P-syntax. every new branch, invalid parameters and channel transitions compile',{skip:!syntaxPython},t=>{
    const pipes=[...NEW.map(id=>[step(id)]),...EXT.map(([id,p])=>[step(id,p)]),[step('colorspace',{space:'yiq'}),step('colorspace_inverse',{space:'yiq'}),step('rgb_adjust')],[step('select_object'),step('pass_fail',{metric:'area'})]];
    for(const id of NEW)pipes.push([step(id,Object.fromEntries(OPS[id].params.map(q=>[q.key,'bad'])))]);
    for(const id of [...NEW,'white_balance','delta_e'])for(const p of parityVariants(id)){
        const st=step(id,p);if(['reconstruct','delta_e'].includes(id))st.src2={pipeline:[]};pipes.push([st]);
    }
    pipes.push([step('colorspace',{space:'yiq'}),step('white_balance',{method:'pca'}),step('rgb_adjust')]);
    for(const [id,p] of [['reconstruct',{}],['delta_e',{formula:'angle'}]]){
        for(const next of [step('grayscale'),step('morph_keep_border'),step('rgb_adjust'),step('white_balance'),step('decorrstretch',{tolPct:2}),step('colorspace_inverse',{space:'yiq'}),step('srgb_linear'),step('quantize')]){
            const first=step(id,p);first.src2={pipeline:[]};pipes.push([first,next]);
            const outer=step('reconstruct');outer.src2={pipeline:[first,next]};pipes.push([outer]);
        }
        const first=step(id,p);first.src2={pipeline:[]};const stages=[first,step('decorrstretch'),step('srgb_linear'),step('color_matrix'),step('white_balance',{method:'pca'}),step('decorrstretch')];pipes.push(stages);
        const outer=step('reconstruct');outer.src2={pipeline:stages};pipes.push([outer,...stages]);
    }
    execFileSync('python',['-c',"import sys,json; codes=json.loads(sys.stdin.buffer.read()); [compile(c,'export','exec') for c in codes]"],{input:JSON.stringify(pipes.map(p=>buildPythonCode(p))),stdio:'pipe'});
    t.diagnostic(`${pipes.length} exported Python scripts compiled, including connected two-input branches`);
});
for(const [id,p] of [...NEW.map(id=>[id,{}]),...EXT])test('ip5e-P-'+id+JSON.stringify(p),{skip:!pyAvailable&&'python/cv2 unavailable (or IP5E_NO_PY=1)'},()=>{
    const src=image(29,23,(x,y)=>[(x*17+y*13)%255,(x*7+y*23)%255,(x*29+y*3)%255],3),b=image(29,23,(x,y)=>[x*5,y*7,200],3);
    try{const c={src2:b},js=workerRun(id,src,p,c),[py]=pythonCases([{id,p,src,appOrder:id==='colorspace',b:['reconstruct','delta_e'].includes(id)?b:undefined}]);try{assert.deepEqual(py.shape,[js.rows,js.cols,...(js.channels()===1?[]:[js.channels()])]);let max=0;for(let i=0;i<js.data.length;i++)max=Math.max(max,Math.abs(js.data[i]-py.data[i]));assert.equal(max,0,`${id} error ${max}`);}finally{js.delete();}}finally{src.delete();b.delete();}
});
function parityVariants(id){
    if(id==='morph_perimeter')return ['4','8'].map(perimConn=>({perimConn}));
    if(id==='morph_thicken')return [0,1,2,3].map(iterations=>({iterations}));
    if(id==='select_object')return ['4','8'].flatMap(conn=>['mask','overlay'].flatMap(view=>[0,50,100].map(x=>({conn,view,x,y:x}))));
    if(id==='srgb_linear')return ['to_linear','to_srgb'].map(direction=>({direction}));
    if(id==='decorrstretch')return ['correlation','covariance'].flatMap(mode=>[0,2,10].map(tolPct=>({mode,tolPct})));
    if(id==='color_matrix')return [{},{m11:0,m13:1,m31:1,m33:0,o2:-20},{m11:.393,m12:.769,m13:.189,m21:.349,m22:.686,m23:.168,m31:.272,m32:.534,m33:.131},{m11:-2,o1:120.5}];
    if(id==='white_balance')return [.5,3.5,20].map(pcaPercent=>({method:'pca',pcaPercent}));
    if(id==='delta_e')return ['pixel','mean_b'].flatMap(compare=>['heatmap','over'].map(view=>({formula:'angle',compare,view})));
    return [{}];
}
for(const id of [...NEW,'white_balance','delta_e'].filter(id=>!['hough_circles'].includes(id)))test(`ip5e-P-variants ${id}`,{skip:!pyAvailable&&'python/cv2 unavailable (or IP5E_NO_PY=1)'},()=>{
    const rgb=image(17,13,(x,y)=>[(x*17+y*9)%255,(x*7+y*23)%255,(x*29+y*3)%255],3),gray=image(17,13,(x,y)=>((x<4||x>10)&&y>2&&y<10)?255:0),b=image(9,7,(x,y)=>[x*17,y*23,190],3),owned=[rgb,gray,b];
    try{const jobs=[rgb,gray].flatMap(src=>parityVariants(id).map(p=>({id,src,p,b:['reconstruct','delta_e'].includes(id)?b:undefined}))),results=pythonCases(jobs);
        for(let i=0;i<jobs.length;i++){const j=jobs[i],c={src2:j.b},out=workerRun(id,j.src,j.p,c);try{const py=results[i];assert.deepEqual(py.shape,[out.rows,out.cols,...(out.channels()===1?[]:[out.channels()])]);let max=0;for(let k=0;k<out.data.length;k++)max=Math.max(max,Math.abs(out.data[k]-py.data[k]));assert.equal(max,0,`${id} ${JSON.stringify(j.p)} diff ${max}`);
            if(id==='select_object')assert.ok(py.stdout.includes(`area ${vals(c).area} px`)||vals(c).area===0);
            if(id==='delta_e'){const v=vals(c),fixed=n=>(Math.floor(n*1000+.5)/1000).toFixed(3);assert.ok(py.stdout.includes(`mean ${fixed(v.de_mean)} max ${fixed(v.de_max)}`),py.stdout);}
        }finally{out.delete();}}
    }finally{owned.forEach(m=>m.delete());}
});
test('ip5e-P-circles. real detected circles at native and downscaled sizes',{skip:!pyAvailable&&'python/cv2 unavailable'},()=>{
    for(const w of [96,720]){const src=image(w,96,()=>[0,0,0],3);try{cv.circle(src,new cv.Point(Math.floor(w/2),48),22,new cv.Scalar(255,255,255),3);const p={view:'mask',param2:10,minR:15,maxR:30,minDist:30},out=workerRun('hough_circles',src,p,{});try{assert.ok(cv.countNonZero(out));const [py]=pythonCases([{id:'hough_circles',src,p}]);assert.deepEqual(Buffer.from(out.data),py.data);}finally{out.delete();}}finally{src.delete();}}
});
test('ip5e-P-sequences. YIQ inverse, matrix, PCA and scalar Pass/Fail retain channel order',{skip:!pyAvailable&&'python/cv2 unavailable'},()=>{
    const src=image(13,11,(x,y)=>[x*17,y*19,(x*7+y*11)%255],3);
    try{for(const stages of [[['colorspace',{space:'yiq'}],['colorspace_inverse',{space:'yiq'}],['color_matrix',{m11:0,m13:1,m31:1,m33:0}]],[['colorspace',{space:'lab'}],['white_balance',{method:'pca'}],['color_matrix',{}]]]){
        let current=src.clone();try{for(const [id,p] of stages){const out=workerRun(id,current,p,{});current.delete();current=out;}const [py]=pythonCases([{src,pipe:stages.map(([id,p])=>step(id,p))}]);assert.deepEqual(Buffer.from(current.data),py.data);}finally{current.delete();}
    }
    const pipe=[step('select_object'),step('pass_fail',{metric:'area',cond:'ge',lo:0})],[py]=pythonCases([{src,pipe}]);assert.match(py.stdout,/RESULT: OK/);
    }finally{src.delete();}
});
test('ip5e-P-empty-B-transitions. runtime gray/color state reaches subsequent consumers',{skip:!pyAvailable&&'python/cv2 unavailable'},()=>{
    const src=image(13,11,(x,y)=>[x*17,y*19,(x*7+y*11)%255],3),empty=new cv.Mat(),mask=image(13,11,()=>[140,170,200],3);
    const consumers=[['grayscale',{}],['morph_keep_border',{}],['rgb_adjust',{}],['white_balance',{}],['decorrstretch',{tolPct:2}],['colorspace_inverse',{space:'yiq'}],['srgb_linear',{}],['quantize',{}]];
    try{for(const [id,p] of [['reconstruct',{}],['delta_e',{formula:'angle'}]])for(const b of [empty,mask])for(const appOrder of [false,true])for(const [next,q] of consumers){
        const first=step(id,p);first.src2={pipeline:[]};
        const prefix=appOrder?[step('colorspace',{space:'lab'})]:[],pipe=[...prefix,first,step(next,q)];
        let current=src.clone();
        try{
            if(appOrder){const out=workerRun('colorspace',current,{space:'lab'},{});current.delete();current=out;}
            for(const [op,params,ctx] of [[id,p,{src2:b}],[next,q,{}]]){const out=workerRun(op,current,params,ctx);current.delete();current=out;}
            const [py]=pythonCases([{id,p,src,b,pipe,appOrder:appOrder&&['rgb_adjust','white_balance','srgb_linear','quantize'].includes(next)}]);
            assert.deepEqual(py.shape,[current.rows,current.cols,...(current.channels()===1?[]:[current.channels()])]);
            let max=0;for(let i=0;i<current.data.length;i++)max=Math.max(max,Math.abs(current.data[i]-py.data[i]));
            assert.equal(max,0,`${id}, B empty=${b===empty}, app order=${appOrder}, ${next}: ${max}`);
        }finally{current.delete();}
    }}finally{src.delete();empty.delete();mask.delete();}
});
test('ip5e-P-thicken-topology. cropped boundary cases match every Python byte',{skip:!pyAvailable&&'python/cv2 unavailable'},()=>{
    for(const rows of [['.###','#...','#...','....'],['#...','....','....','....'],['##.#','....','....','....'],['#.#.','#...','....','....']]){
        const src=image(4,4,(x,y)=>Array(3).fill(rows[y][x]==='#'?255:0),3);
        try{for(const iterations of [1,2,3,0]){const out=workerRun('morph_thicken',src,{iterations},{});try{const [py]=pythonCases([{id:'morph_thicken',p:{iterations},src}]);assert.deepEqual(Buffer.from(out.data),py.data);}finally{out.delete();}}}finally{src.delete();}
    }
});
test('ip5e-P-R2-reference/frame. analytic correlation and fixed-pad shape match Python exactly',{skip:!pyAvailable&&'python/cv2 unavailable'},()=>{
    const src=unequalBands5E(),blob=image(15,11,(x,y)=>Array(3).fill(x>=6&&x<=8&&y>=4&&y<=6?255:0),3);
    try{
        for(const mode of ['correlation','covariance']){const out=run('decorrstretch',src,{mode});try{const [py]=pythonCases([{id:'decorrstretch',src,p:{mode}}]);assert.deepEqual(Buffer.from(out.data),py.data);if(mode==='correlation')assert.deepEqual(py.data,correlationReference5E());}finally{out.delete();}}
        const [py]=pythonCases([{id:'morph_thicken',src:blob,p:{iterations:0}}]);assert.deepEqual(py.shape,[11,15]);assert.deepEqual(py.data,Buffer.alloc(165,255));
    }finally{src.delete();blob.delete();}
});
test('ip5e-P-R2-pipeline. reconstruct/angle through color kernels retain exact pixels',{skip:!pyAvailable&&'python/cv2 unavailable'},()=>{
    const src=image(13,11,(x,y)=>[x*17,y*19,(x*7+y*11)%255],3),empty=new cv.Mat(),mask=image(13,11,()=>[140,170,200],3);
    try{for(const [id,p] of [['reconstruct',{}],['delta_e',{formula:'angle'}]])for(const b of [empty,mask])for(const appOrder of [false,true]){
        const first=step(id,p);first.src2={pipeline:[]};const stages=[['decorrstretch',{}],['srgb_linear',{}],['color_matrix',{}],['white_balance',{method:'pca'}],['decorrstretch',{mode:'covariance'}]];
        const prefix=appOrder?[step('colorspace',{space:'lab'})]:[],pipe=[...prefix,first,...stages.map(([op,q])=>step(op,q))];let current=src.clone();
        try{for(const [op,q,ctx] of [...(appOrder?[['colorspace',{space:'lab'},{}]]:[]),[id,p,{src2:b}],...stages.map(([op,q])=>[op,q,{}])]){const out=workerRun(op,current,q,ctx);current.delete();current=out;}
            const [py]=pythonCases([{id,p,src,b,pipe}]);assert.deepEqual(py.shape,[current.rows,current.cols,current.channels()]);assert.deepEqual(py.data,Buffer.from(current.data));
        }finally{current.delete();}
    }}finally{src.delete();empty.delete();mask.delete();}
});
function pythonCases(jobs, {prepareOnly=false} = {}) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ip5e-parity-"));
    function bytes(m) {
        if (m.rows * m.cols === 0)
            return Buffer.alloc(0);
        const rgb = new cv.Mat(), bgr = new cv.Mat();
        try {
            // imread always supplies three channels, including for gray source fixtures.
            if (m.channels() === 1)
                cv.cvtColor(m, rgb, cv.COLOR_GRAY2RGB);
            else if (m.channels() === 4)
                cv.cvtColor(m, rgb, cv.COLOR_RGBA2RGB);
            else
                m.copyTo(rgb);
            cv.cvtColor(rgb, bgr, cv.COLOR_RGB2BGR);
            return Buffer.from(bgr.data);
        }
        finally {
            rgb.delete();
            bgr.delete();
        }
    }
    try {
        const manifest = jobs.map((j, i) => {
            let pipe = j.pipe || [step(j.id, j.p)];
            if (j.b)
                pipe.find((st) => st.id === OP_TO_FILTER[j.id]).src2 = { pipeline: [] };
            if (j.src.channels() === 1)
                pipe = [step("grayscale"), ...pipe];
            let code = buildPythonCode(pipe);
            code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
            const inputs = [j.src, ...(j.b ? [j.b] : [])].map((m, k) => {
                const input = path.join(dir, `i${i}_${k}.bin`);
                fs.writeFileSync(input, bytes(m));
                const arg = j.b ? k ? "path_b" : "path_a" : "path";
                code = code.replace(`cv2.imread(${arg})`, `np.fromfile(${arg}, dtype=np.uint8).reshape((${m.rows}, ${m.cols}, 3))`);
                return input;
            });
            if (j.tail)
                code = code.replace("\n    return img\n", "\n    " + j.tail + "\n    return img\n");
            const file = path.join(dir, `c${i}.py`), out = path.join(dir, `o${i}.bin`);
            fs.writeFileSync(file, code);
            return { file, inputs, out };
        });
        if(prepareOnly)return manifest.map(c=>fs.readFileSync(c.file,'utf8'));
        fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(manifest));
        fs.writeFileSync(path.join(dir, "driver.py"), `import json,io,contextlib,sys,time
results=[]
for c in json.load(open(sys.argv[1])):
    ns={'__name__':'case'}
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(c['file'],encoding='utf8').read(),c['file'],'exec'),ns)
        start=time.perf_counter()
        result=ns['process'](*c['inputs'])
        elapsed_ms=(time.perf_counter()-start)*1000
    result.tofile(c['out'])
    results.append({'shape':result.shape,'stdout':buf.getvalue(),'elapsed_ms':elapsed_ms})
json.dump(results,open(sys.argv[2],'w'))
`);
        execFileSync("python", [path.join(dir, "driver.py"), path.join(dir, "manifest.json"), path.join(dir, "results.json")], { stdio: "pipe", env: { ...process.env, PYTHONIOENCODING: "utf8" }, maxBuffer: 1 << 25 });
        return JSON.parse(fs.readFileSync(path.join(dir, "results.json"), "utf8")).map((r, i) => {
            const data = fs.readFileSync(manifest[i].out);
            if (r.shape.length === 3 && !jobs[i].appOrder)
                for (let k = 0; k < data.length; k += 3)
                    [data[k], data[k + 2]] = [data[k + 2], data[k]];
            return { ...r, data };
        });
    }
    finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}
function workerRun(id, src, p, ctx, c = cv) {
    const st = step(id, p);
    let gray = null;
    try {
        if (needsOf(FILTERS[st.id], st.params) === "gray" && src.channels() > 1) {
            gray = new c.Mat();
            c.cvtColor(src, gray, src.channels() === 4 ? c.COLOR_RGBA2GRAY : c.COLOR_RGB2GRAY);
        }
        return run(id, gray || src, p, ctx, c);
    }
    finally {
        gray?.delete();
    }
}
