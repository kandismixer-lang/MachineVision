import { finiteOr } from "./filters/registry.js";
const number = (v,d,lo,hi) => Math.min(hi,Math.max(lo,finiteOr(v,d)));
const lines = s => s.split("\n");
export const jacobi5EPy = () => lines(`import math
def jacobi5e(matrix):
    a = [list(row) for row in matrix]
    v = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    for sweep in range(50):
        if abs(a[0][1]) < 1e-14 and abs(a[0][2]) < 1e-14 and abs(a[1][2]) < 1e-14:
            break
        for p, q in [(0,1),(0,2),(1,2)]:
            if abs(a[p][q]) < 1e-14:
                continue
            tau = (a[q][q] - a[p][p]) / (2 * a[p][q])
            t = (1 if tau >= 0 else -1) / (abs(tau) + math.sqrt(1 + tau * tau))
            c = 1 / math.sqrt(1 + t * t)
            s, apq = t * c, a[p][q]
            a[p][p] -= t * apq
            a[q][q] += t * apq
            a[p][q] = a[q][p] = 0.
            for k in range(3):
                if k != p and k != q:
                    kp, kq = a[k][p], a[k][q]
                    a[k][p] = a[p][k] = c * kp - s * kq
                    a[k][q] = a[q][k] = s * kp + c * kq
                vp, vq = v[k][p], v[k][q]
                v[k][p], v[k][q] = c * vp - s * vq, s * vp + c * vq
    return [a[0][0],a[1][1],a[2][2]], v`);

// All color kernels read explicit app-order RGB and return BGR for the exporter.
export function color5EPy(id,p,mirrored) {
  const begin = [`gray5e = img.ndim == 2`, `rgb5e = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if gray5e else np.ascontiguousarray(img[:, :, :3]${mirrored?"[:, :, ::-1]":""})`];
  const finish = ["img = np.ascontiguousarray(out5e[:, :, ::-1])"];
  if(id==="yiq"||id==="yiq_inverse"){
    const inverse=id==="yiq_inverse";
    const body=lines(`a5e = [[.299,.587,.114],[.596,-.274,-.322],[.211,-.523,.312]]
if ${inverse?"True":"False"}:
    co = [[0.,0.,0.] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            co[r][c] = a5e[(r+1)%3][(c+1)%3]*a5e[(r+2)%3][(c+2)%3]-a5e[(r+1)%3][(c+2)%3]*a5e[(r+2)%3][(c+1)%3]
    det = a5e[0][0]*co[0][0]+a5e[0][1]*co[0][1]+a5e[0][2]*co[0][2]
    a5e = [[co[c][r]/det for c in range(3)] for r in range(3)]
x5e = rgb5e.astype(np.float64)
${inverse?`x5e[:, :, 0] /= 255
x5e[:, :, 1] = (x5e[:, :, 1]-127.5)*(2*.596)/255
x5e[:, :, 2] = (x5e[:, :, 2]-127.5)*(2*.523)/255`:`x5e /= 255`}
out5e = np.empty_like(rgb5e)
for c in range(3):
    v5e = a5e[c][0]*x5e[:, :, 0]+a5e[c][1]*x5e[:, :, 1]+a5e[c][2]*x5e[:, :, 2]
    v5e = v5e*255 if ${inverse?"True":"False"} or c == 0 else v5e*255/(2*(.596 if c == 1 else .523))+127.5
    out5e[:, :, c] = np.clip(np.floor(v5e+.5),0,255).astype(np.uint8)`);
    // Color Space Convert exports app channel order; inverse exports ordinary BGR.
    return [...begin,...(inverse?["if gray5e:","    out5e = rgb5e", "else:",...body.map(s=>"    "+s)]:body),...(inverse?finish:["img = out5e"]),`print('YIQ ${inverse?"-> RGB":"(NTSC)"}')`];
  }
  if(id==="srgb_linear")return [
    "import math", "lut5e = []", "for v5e in range(256):", "    c5e = v5e / 255",
    p.direction==="to_srgb"?"    y5e = 12.92*c5e if c5e <= .0031308 else 1.055*c5e**(1/2.4)-.055":"    y5e = c5e/12.92 if c5e <= .04045 else ((c5e+.055)/1.055)**2.4",
    "    lut5e.append(max(0,min(255,math.floor(y5e*255+.5))))", "img = cv2.LUT(img, np.array(lut5e,np.uint8))",`print('${p.direction==="to_srgb"?"Linear -> sRGB":"sRGB -> Linear"}')`];
  if(id==="color_matrix"){
    const m=Array.from({length:3},(_,r)=>Array.from({length:3},(_,c)=>number(p[`m${r+1}${c+1}`],r===c?1:0,-3,3))),o=[1,2,3].map(c=>number(p[`o${c}`],0,-255,255));
    return [...begin,`m5e, o5e = ${JSON.stringify(m)}, ${JSON.stringify(o)}`,"x5e = rgb5e.astype(np.float64)","out5e = np.empty_like(rgb5e)","for c in range(3):","    v5e = m5e[c][0]*x5e[:, :, 0]+m5e[c][1]*x5e[:, :, 1]+m5e[c][2]*x5e[:, :, 2]+o5e[c]","    out5e[:, :, c] = np.clip(np.floor(v5e+.5),0,255).astype(np.uint8)",...finish,"print('Color Matrix: 3x3 + offset')"];
  }
  if(id==="decorrstretch")return [...begin,"if gray5e:","    print('gray image: no colors to stretch')","else:",...[
    "import math",...lines(`data5e = rgb5e.reshape(-1,3)
n5e = len(data5e)
sums5e = [0,0,0]
cross5e = [[0,0,0] for _ in range(3)]
for pixel5e in data5e:
    x = [int(v) for v in pixel5e]
    for r in range(3):
        sums5e[r] += x[r]
        for c in range(3):
            cross5e[r][c] += x[r]*x[c]
mean5e = [s/n5e for s in sums5e]
cov5e = [[(cross5e[r][c]-float(sums5e[r])*float(sums5e[c])/n5e)/(n5e-1) if n5e > 1 else 0. for c in range(3)] for r in range(3)]
sigma5e = [math.sqrt(max(0,cov5e[c][c])) for c in range(3)]
${p.mode==='covariance'?'matrix5e = cov5e':`inv_sigma5e = [1/s if s > 0 else 0. for s in sigma5e]
matrix5e = [[inv_sigma5e[r]*cov5e[r][c]*inv_sigma5e[c] for c in range(3)] for r in range(3)]`}
e5e, v5e = jacobi5e(matrix5e)
inv5e = [1/math.sqrt(e) if e > 1e-12 else 0. for e in e5e]
t5e = [[0.,0.,0.] for _ in range(3)]
for r in range(3):
    for c in range(3):
        s = 0.
        for k in range(3):
            s += v5e[r][k]*inv5e[k]*v5e[c][k]
        t5e[r][c] = ${p.mode==='covariance'?'s*sigma5e[c]':'inv_sigma5e[r]*s*sigma5e[c]'}
x5e = data5e.astype(np.float64)
raw5e = np.empty_like(x5e)
for c in range(3):
    raw5e[:, c] = (x5e[:, 0]-mean5e[0])*t5e[0][c]+(x5e[:, 1]-mean5e[1])*t5e[1][c]+(x5e[:, 2]-mean5e[2])*t5e[2][c]+mean5e[c]
tol5e = ${number(p.tolPct,0,0,10)}
if tol5e > 0:
    for c in range(3):
        ordered5e = np.sort(raw5e[:, c],kind='stable')
        def quantile5e(p):
            pos = (n5e-1)*p
            j = math.floor(pos)
            return ordered5e[j]+(ordered5e[min(n5e-1,j+1)]-ordered5e[j])*(pos-j)
        lo5e, hi5e = quantile5e(tol5e/100), quantile5e(1-tol5e/100)
        if hi5e > lo5e:
            raw5e[:, c] = (raw5e[:, c]-lo5e)*255/(hi5e-lo5e)
out5e = np.clip(np.floor(raw5e+.5),0,255).astype(np.uint8).reshape(rgb5e.shape)
img = np.ascontiguousarray(out5e[:, :, ::-1])
print('Decorrelation stretch' + (' (singular color axis)' if any(e <= 1e-12 for e in e5e) else ''))`)
  ].map(s=>"    "+s)];
  if(id==="pca")return [...begin,"if gray5e:","    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)","    print('white balance: gray image, nothing to adjust')","else:",...[
    "import math",...lines(`data5e = rgb5e.reshape(-1,3)
valid5e = np.all(data5e != 255,axis=1)
pixels5e = np.flatnonzero(valid5e)
n5e = len(pixels5e)
sums5e = [0,0,0]
for i in pixels5e:
    for c in range(3):
        sums5e[c] += int(data5e[i,c])
mean5e = [s/255/n5e if n5e else 0. for s in sums5e]
norm5e = math.sqrt(mean5e[0]*mean5e[0]+mean5e[1]*mean5e[1]+mean5e[2]*mean5e[2])
out5e = rgb5e.copy()
if not n5e or not norm5e:
    print('PCA: no unsaturated nonzero pixels to estimate illuminant')
else:
    m5e = [v/norm5e for v in mean5e]
    x5e = data5e.astype(np.float64)/255
    proj5e = x5e[pixels5e,0]*m5e[0]+x5e[pixels5e,1]*m5e[1]+x5e[pixels5e,2]*m5e[2]
    pixels5e = pixels5e[np.argsort(proj5e,kind='stable')]
    k5e = max(1,math.floor(${number(p.pcaPercent,3.5,.5,20)}/100*n5e))
    chosen5e = list(pixels5e[:k5e])+list(pixels5e[n5e-k5e:])
    a5e = [[0.,0.,0.] for _ in range(3)]
    for i in chosen5e:
        for r in range(3):
            for c in range(3):
                a5e[r][c] += float(x5e[i,r])*float(x5e[i,c])
    e5e, v5e = jacobi5e(a5e)
    largest5e = 0
    for c in range(1,3):
        if e5e[c] > e5e[largest5e]:
            largest5e = c
    light5e = [abs(row[largest5e]) for row in v5e]
    target5e = (light5e[0]+light5e[1]+light5e[2])/3
    gains5e = [target5e/v if v > 1e-3 else 1. for v in light5e]
    for c in range(3):
        out5e[:, :, c] = np.clip(np.floor(rgb5e[:, :, c].astype(np.float64)*gains5e[c]+.5),0,255).astype(np.uint8)
    print('PCA illuminant R/G/B ' + '/'.join(f'{math.floor(v*1000+.5)/1000:.3f}' for v in light5e))
img = np.ascontiguousarray(out5e${mirrored?"[:, :, ::-1]":""})`)
  ].map(s=>"    "+s)];
  throw new Error("Unknown batch5E color kernel: "+id);
}
