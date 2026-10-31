import { finiteOr, boolOr, LUCY_WORK_BUDGET } from './filters/registry.js';
const lines = s => s.split('\n');
const num = (v,d,lo,hi) => Math.min(hi,Math.max(lo,finiteOr(v,d)));
const integer = (v,d,lo,hi) => Math.round(num(v,d,lo,hi));
export function taper5FPy() { return lines(`def edgetaper5f(src, psf):
    n = psf.shape[0]
    r = (n - 1) // 2
    h, w = src.shape[:2]
    px, py = [0.] * n, [0.] * n
    for y in range(n):
        for x in range(n):
            px[x] += float(psf[y,x])
            py[y] += float(psf[y,x])
    def axis(projection, length):
        ac = [0.] * len(projection)
        for d in range(len(ac)):
            for k in range(len(ac)-d):
                ac[d] += projection[k]*projection[k+d]
        return np.array([1-ac[min(x,length-1-x)]/ac[0] if min(x,length-1-x)<len(ac) and ac[0]>0 else 1. for x in range(length)],np.float64)
    alpha = axis(py,h)[:,None]*axis(px,w)[None,:]
    gray = src.ndim == 2
    source = src[:,:,None].astype(np.float64) if gray else src.astype(np.float64)
    active5f = np.flatnonzero(alpha.reshape(-1) != 1)
    image5f = source[:,:,0] if gray else source
    padded5f = cv2.copyMakeBorder(image5f,r,r,r,r,cv2.BORDER_WRAP)
    kernel5f = np.ascontiguousarray(np.flip(psf.astype(np.float64),(0,1)))
    blur5f = cv2.filter2D(padded5f,-1,kernel5f,borderType=cv2.BORDER_CONSTANT)[r:r+h,r:r+w]
    if gray:
        blur5f = blur5f[:,:,None]
    flat5f = source.reshape((-1,source.shape[2]))
    aa5f = alpha.reshape(-1)[active5f,None]
    flatBlur5f = blur5f.reshape((-1,source.shape[2]))
    flat5f[active5f] = np.clip(np.floor(aa5f*flat5f[active5f]+(1-aa5f)*flatBlur5f[active5f]+.5),0,255)
    out5f = flat5f.reshape(source.shape).astype(np.uint8)
    return out5f[:,:,0] if gray else out5f`); }
export function withTaper5FPy(base) {
  const out=[...taper5FPy()];
  for(const line of base){out.push(line);if(line.startsWith('psf = psf_kernel('))out.push('img = edgetaper5f(img, psf)');}
  return out;
}
export function regularized5FPy(base,p) {
  const out=[];
  for(const line of (boolOr(p.taper,false)?withTaper5FPy(base):base)){
    if(line.trim().startsWith('print("wiener deconvolution:')){out.push(`print('Regularized deconvolution · λ ${num(p.lambda,.01,.0001,1)}')`);continue;}
    out.push(line.replace('den = hr * hr + hi * hi + nsr','den = hr * hr + hi * hi + nsr * (lr * lr + li * li)'));
    if(line==='otf = cv2.dft(ph, flags=cv2.DFT_COMPLEX_OUTPUT)')out.push(...lines(`lap5f = np.zeros((hp,wp),np.float32)
lap5f[0,0] = -4
lap5f[0,1] = lap5f[0,-1] = lap5f[1,0] = lap5f[-1,0] = 1
lotf5f = cv2.dft(lap5f,flags=cv2.DFT_COMPLEX_OUTPUT)
lr, li = lotf5f[:,:,0].astype(np.float64), lotf5f[:,:,1].astype(np.float64)`));
  }return out;
}
export function active5FPy(p) {
  const asked=integer(p.iterations,100,1,1000),smooth=integer(p.smooth,1,0,5),margin=num(p.margin,10,1,45),bias=num(p.bias,0,-1,1);
  return lines(`h5f, w5f = img.shape[:2]
data5f = img.astype(np.float64)
asked5f, smooth5f = ${asked}, ${smooth}
iters5f = min(asked5f,max(1,${LUCY_WORK_BUDGET}//(w5f*h5f*(1+smooth5f))))
${p.init==='otsu'?`mask5f = cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1] > 0`:`mx5f, my5f = int(w5f*${margin}/100+.5), int(h5f*${margin}/100+.5)
yy5f, xx5f = np.indices((h5f,w5f))
mask5f = (xx5f>=mx5f)&(xx5f<w5f-mx5f)&(yy5f>=my5f)&(yy5f<h5f-my5f)`}
def means5f(mask):
    ni = int(np.count_nonzero(mask))
    no = mask.size-ni
    si = int(np.sum(img[mask],dtype=np.int64))
    so = int(np.sum(img[~mask],dtype=np.int64))
    return si/ni if ni else so/max(1,no), so/no if no else si/max(1,ni), ni
stable5f = False
for used5f in range(1,iters5f+1):
    cin5f, cout5f, _ = means5f(mask5f)
    f5f = (data5f-cout5f)**2-(data5f-cin5f)**2-${bias}*65025*.01
    next5f = np.where(f5f>0,True,np.where(f5f<0,False,mask5f))
    for _ in range(smooth5f):
        pad5f = np.pad(next5f.astype(np.uint8),1,mode='edge')
        count5f = np.zeros_like(next5f,dtype=np.int16)
        for dy5f in range(3):
            for dx5f in range(3):
                count5f += pad5f[dy5f:dy5f+h5f,dx5f:dx5f+w5f]
        next5f = count5f >= 5
    stable5f = bool(np.array_equal(next5f,mask5f))
    mask5f = next5f
    if stable5f:
        break
cin5f, cout5f, area5f = means5f(mask5f)
note5f = f' · ภาพใหญ่ จึงลดเหลือ {iters5f} รอบ (ขอ {asked5f} รอบ)' if iters5f < asked5f else ''
print(f"Chan-Vese · {used5f} รอบ ({'หยุดเมื่อนิ่ง' if stable5f else 'ครบ'}) · c_in {np.floor(cin5f*10+.5)/10:.1f} · c_out {np.floor(cout5f*10+.5)/10:.1f}{note5f}")
${p.view==='mask'?`img = mask5f.astype(np.uint8)*255`:`pad5f = np.pad(mask5f,1,mode='constant')
inner5f = mask5f & ~(pad5f[:-2,1:-1]&pad5f[2:,1:-1]&pad5f[1:-1,:-2]&pad5f[1:-1,2:])
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
img[inner5f] = (0,0,255)`}`);
}
export function laplace5FPy(binary) {
  return [...binary,...lines(`nz5f = int(cv2.countNonZero(b_bin))
total5f = img.shape[0]*img.shape[1]
if nz5f == 0:
    print('mask ว่าง ไม่มีบริเวณให้ซ่อม')
elif nz5f*100 > 25*total5f:
    print('ไม่ซ่อม: mask ครอบเกิน 25%')
else:
    gray5f = img.ndim == 2
    src5f = img[:,:,None] if gray5f else img[:,:,:3]
    h5f,w5f,ch5f = src5f.shape
    mask5f = b_bin > 0
    def state5f(data,mask):
        h,w,ch = data.shape
        border = np.zeros((h,w),bool)
        border[:-1,:] |= mask[1:,:]
        border[1:,:] |= mask[:-1,:]
        border[:,:-1] |= mask[:,1:]
        border[:,1:] |= mask[:,:-1]
        border &= ~mask
        count = int(np.count_nonzero(border))
        mean = np.sum(data[border],axis=0,dtype=np.float64)/count if count else np.zeros(ch,np.float64)
        values = data.astype(np.float64).copy()
        values[mask] = mean
        yy5f,xx5f = np.nonzero(mask)
        parity5f = (yy5f+xx5f)&1
        red = (yy5f[parity5f==0],xx5f[parity5f==0])
        black = (yy5f[parity5f==1],xx5f[parity5f==1])
        redn = (red[0]>0).astype(np.int32)+(red[0]<h-1)+(red[1]>0)+(red[1]<w-1)
        blackn = (black[0]>0).astype(np.int32)+(black[0]<h-1)+(black[1]>0)+(black[1]<w-1)
        def neighbors5f(coords):
            y,x = coords
            near = []
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                yn,xn = y+dy,x+dx
                ok = (yn>=0)&(yn<h)&(xn>=0)&(xn<w)
                near.append((np.clip(yn,0,h-1)*w+np.clip(xn,0,w-1),ok))
            return tuple(near)
        rednb5f,blacknb5f = neighbors5f(red),neighbors5f(black)
        span = max(1,int(np.max(xx5f)-np.min(xx5f)+1) if len(xx5f) else 1,int(np.max(yy5f)-np.min(yy5f)+1) if len(yy5f) else 1)
        return {'values':values,'mask':mask,'h':h,'w':w,'ch':ch,'span5f':span,'red5f':red,'black5f':black,'redn5f':redn,'blackn5f':blackn,'rednb5f':rednb5f,'blacknb5f':blacknb5f}
    def down5f(level):
        data,mask = level['values'],level['mask']
        h,w,ch = data.shape
        nh,nw = (h+1)//2,(w+1)//2
        out = np.zeros((nh,nw,ch),np.float64)
        outmask = np.zeros((nh,nw),bool)
        count = np.zeros((nh,nw),np.float64)
        for dy in range(2):
            for dx in range(2):
                sh,sw = data[dy::2,dx::2].shape[:2]
                if sh and sw:
                    out[:sh,:sw] += data[dy::2,dx::2]
                    outmask[:sh,:sw] |= mask[dy::2,dx::2]
                    count[:sh,:sw] += 1
        out /= count[:,:,None]
        return state5f(out,outmask)
    def solve5f(level,limit):
        values,mask = level['values'],level['mask']
        h,w = mask.shape
        flat = values.reshape((-1,level['ch']))
        omega = 2/(1+np.sin(np.pi/(level['span5f']+1)))
        stable = False
        for used in range(1,limit+1):
            change = 0.
            for coords,n,near in ((level['red5f'],level['redn5f'],level['rednb5f']),(level['black5f'],level['blackn5f'],level['blacknb5f'])):
                y,x = coords
                indices = y*w+x
                sums = np.zeros((len(y),level['ch']),np.float64)
                for neighbor,valid in near:
                    sums += flat[neighbor]*valid[:,None]
                old = flat[indices].copy()
                updated = old+omega*(sums/np.maximum(1,n)[:,None]-old)
                if len(y):
                    change = max(change,float(np.max(np.abs(updated-old))))
                    flat[indices] = updated
            if change < .01:
                stable = True
                break
        return (used if limit else 0),stable
    hierarchy5f = [state5f(src5f.astype(np.float64),mask5f)]
    budget5f = min(5000,max(1,${LUCY_WORK_BUDGET}//nz5f))
    while max(hierarchy5f[-1]['w'],hierarchy5f[-1]['h']) > 32 and len(hierarchy5f)<budget5f:
        hierarchy5f.append(down5f(hierarchy5f[-1]))
    max_coarse5f = (budget5f-1)//(len(hierarchy5f)-1) if len(hierarchy5f)>1 else 0
    coarse_share5f = min(12,max_coarse5f,max(4,budget5f//(4*len(hierarchy5f)))) if len(hierarchy5f)>1 else 0
    fine_share5f = budget5f-coarse_share5f*(len(hierarchy5f)-1)
    used5f,stable5f,fine_used5f = 0,False,0
    for level_index5f in range(len(hierarchy5f)-1,-1,-1):
        level5f = hierarchy5f[level_index5f]
        if level_index5f<len(hierarchy5f)-1:
            coarse5f = hierarchy5f[level_index5f+1]
            y5f,x5f = np.nonzero(level5f['mask'])
            cx5f,cy5f = x5f//2,y5f//2
            dx5f,dy5f = (x5f%2)/2,(y5f%2)/2
            cx15f,cy15f = np.minimum(coarse5f['w']-1,cx5f+1),np.minimum(coarse5f['h']-1,cy5f+1)
            a5f=coarse5f['values'][cy5f,cx5f]
            b5f=coarse5f['values'][cy5f,cx15f]
            c5f=coarse5f['values'][cy15f,cx5f]
            d5f=coarse5f['values'][cy15f,cx15f]
            level5f['values'][y5f,x5f] = a5f*(1-dx5f[:,None])*(1-dy5f[:,None])+b5f*dx5f[:,None]*(1-dy5f[:,None])+c5f*(1-dx5f[:,None])*dy5f[:,None]+d5f*dx5f[:,None]*dy5f[:,None]
        rounds5f,stable5f = solve5f(level5f,fine_share5f if level_index5f==0 else coarse_share5f)
        used5f += rounds5f
        if level_index5f==0:
            fine_used5f = rounds5f
    img = np.clip(np.floor(hierarchy5f[0]['values']+.5),0,255).astype(np.uint8)
    if gray5f:
        img = img[:,:,0]
    note5f = f' · ภาพใหญ่ จึงลดเหลือ {budget5f} รอบ (ขอ 5000 รอบ)' if budget5f < 5000 and not stable5f and fine_used5f>=fine_share5f else ''
    print(f"Laplace (regionfill) · {nz5f} px · {used5f} รอบ · {'หยุดเมื่อนิ่ง' if stable5f else 'ครบ'}{note5f}")`)];
}
export function blind5FPy(p,grayCode) {
  const n=Math.min(31,integer(p.psfSize,9,3,31)|1),asked=integer(p.iterations,10,1,50);
  return [...(boolOr(p.taper,false)?taper5FPy():[]),...lines(`n5f, asked5f = ${n}, ${asked}
r5f = (n5f-1)//2
psf5f = np.full((n5f,n5f),1/(n5f*n5f),np.float64)
${boolOr(p.taper,false)?'img = edgetaper5f(img,psf5f)':''}
h5f,w5f = img.shape[:2]
ch5f = 1 if img.ndim == 2 else 3
work5f = ch5f+1+(n5f*n5f+8)//9
iters5f = min(asked5f,max(1,${LUCY_WORK_BUDGET}//(w5f*h5f*work5f)))
g5f = (img if img.ndim==2 else cv2.cvtColor(img,${grayCode})).astype(np.float32)
est5f = g5f.copy()
planes5f = [(c.astype(np.float32),c.astype(np.float32)) for c in cv2.split(img)] if ch5f>1 else []
def ratio5f(g,est,kf):
    c = cv2.filter2D(est,-1,kf,borderType=cv2.BORDER_REFLECT_101)
    ok = c>1e-6
    return np.where(ok,g/np.where(ok,c,np.float32(1)),np.float32(0)).astype(np.float32)
def step5f(g,est,kk,kf):
    ratio = ratio5f(g,est,kf)
    corr = cv2.filter2D(ratio,-1,kk,borderType=cv2.BORDER_REFLECT_101)
    return (est*corr).astype(np.float32)
for it5f in range(iters5f):
    kk5f = psf5f.astype(np.float32)
    kf5f = np.ascontiguousarray(kk5f[::-1,::-1])
    est5f = step5f(g5f,est5f,kk5f,kf5f)
    planes5f = [(g,step5f(g,e,kk5f,kf5f)) for g,e in planes5f]
    rr5f = ratio5f(g5f,est5f,kf5f)
    padded5f = cv2.copyMakeBorder(est5f,r5f,r5f,r5f,r5f,cv2.BORDER_REFLECT_101)
    corr5f = cv2.matchTemplate(padded5f,rr5f,cv2.TM_CCORR)[::-1,::-1].astype(np.float64)
    next5f = np.maximum(0.,psf5f*corr5f)
    total5f = float(np.cumsum(next5f.reshape(-1),dtype=np.float64)[-1])
    psf5f = next5f/total5f if total5f>0 else np.full((n5f,n5f),1/(n5f*n5f),np.float64)
${p.view==='psf'?`side5f = min(h5f,w5f)
index5f = np.floor(np.arange(side5f)*n5f/side5f).astype(np.int32)
out5f = psf5f[index5f[:,None],index5f[None,:]]/float(np.max(psf5f))*255`:`out5f = cv2.merge([e for g,e in planes5f]) if planes5f else est5f
out5f = out5f.astype(np.float64)`}
img = np.clip(np.floor(out5f+.5),0,255).astype(np.uint8)
note5f = f' · ภาพใหญ่ จึงลดเหลือ {iters5f} รอบ (ขอ {asked5f} รอบ)' if iters5f<asked5f else ''
print(f'Blind deconvolution {iters5f} รอบ · PSF {n5f}×{n5f}{note5f}${boolOr(p.taper,false)?' · edgetaper':''}')`)];
}
export function localLaplacian5FPy(p,grayCode) {
  const sigma=num(p.sigma,.2,.01,1),alpha=num(p.alpha,.5,.01,5),beta=num(p.beta,1,0,5),K=[8,16,32].includes(Number(p.levels))?Number(p.levels):16;
  return lines(`sigma5f, alpha5f, beta5f, k5f = ${sigma}, ${alpha}, ${beta}, ${K}
nlev5f = max(1,min(8,int(np.floor(np.log2(min(img.shape[:2]))))-3))
if not (alpha5f==1 and beta5f==1) and min(img.shape[:2])>=16:
    gray5f = img if img.ndim==2 else cv2.cvtColor(img,${grayCode})
    light5f = gray5f.astype(np.float64)/255
    gaussian5f = [light5f.astype(np.float32)]
    for level5f in range(nlev5f):
        gaussian5f.append(cv2.pyrDown(gaussian5f[-1]))
    acc5f = [np.zeros(g.shape,np.float64) for g in gaussian5f[:-1]]
    noise5f = .01**(alpha5f-1)
    for index5f in range(k5f):
        gamma5f = index5f/(k5f-1)
        d5f = light5f-gamma5f
        t5f = np.abs(d5f)
        u5f = t5f/sigma5f
        f5f = np.where(u5f>=.01,np.power(u5f,alpha5f),u5f*noise5f)
        r5f = np.where(t5f<=sigma5f,gamma5f+np.sign(d5f)*sigma5f*f5f,gamma5f+np.sign(d5f)*(sigma5f+beta5f*(t5f-sigma5f)))
        remapped5f = [r5f.astype(np.float32)]
        for level5f in range(nlev5f):
            remapped5f.append(cv2.pyrDown(remapped5f[-1]))
        for level5f in range(nlev5f):
            g5f = gaussian5f[level5f].astype(np.float64)
            up5f = cv2.pyrUp(remapped5f[level5f+1],dstsize=(g5f.shape[1],g5f.shape[0])).astype(np.float64)
            weight5f = np.maximum(0,1-np.abs(g5f-gamma5f)*(k5f-1))
            acc5f[level5f] += weight5f*(remapped5f[level5f].astype(np.float64)-up5f)
    result5f = gaussian5f[-1].astype(np.float64)
    for level5f in range(nlev5f-1,-1,-1):
        target5f = gaussian5f[level5f]
        up5f = cv2.pyrUp(result5f.astype(np.float32),dstsize=(target5f.shape[1],target5f.shape[0])).astype(np.float64)
        result5f = up5f+acc5f[level5f]
    result5f = np.clip(result5f,0,1)
    if img.ndim==2:
        out5f = result5f*255
    else:
        gain5f = result5f/(np.where(gray5f>0,gray5f,1).astype(np.float64)/255)
        out5f = np.where((gray5f>0)[:,:,None],img.astype(np.float64)*gain5f[:,:,None],result5f[:,:,None]*255)
    img = np.clip(np.floor(out5f+.5),0,255).astype(np.uint8)
print(f'Local Laplacian · σ {sigma5f} · α {alpha5f} · β {beta5f} · {k5f} ระดับ · {nlev5f} ชั้น')`);
}
export function multissim5FPy(bgray) {
  return lines(`if img_b.ndim==3:
    img_b = cv2.cvtColor(img_b,${bgray})
resized5f = img.shape[:2]!=img_b.shape[:2]
if resized5f:
    img_b = cv2.resize(img_b,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
h5f,w5f = img.shape[:2]
scales5f = 1
while scales5f<5 and min(w5f,h5f)>=11*2**scales5f:
    scales5f += 1
weights5f = [.0448,.2856,.3001,.2363,.1333][:scales5f]
sum5f = sum(weights5f)
weights5f = [v/sum5f for v in weights5f]
a5f,b5f = img.astype(np.float64),img_b.astype(np.float64)
msssim5f = 1.
for scale5f,weight5f in enumerate(weights5f):
    af5f,bf5f = a5f.astype(np.float32),b5f.astype(np.float32)
    ma5f = cv2.GaussianBlur(af5f,(11,11),1.5)
    mb5f = cv2.GaussianBlur(bf5f,(11,11),1.5)
    aa5f,bb5f,ab5f = ma5f*ma5f,mb5f*mb5f,ma5f*mb5f
    va5f = cv2.GaussianBlur(af5f*af5f,(11,11),1.5)-aa5f
    vb5f = cv2.GaussianBlur(bf5f*bf5f,(11,11),1.5)-bb5f
    vab5f = cv2.GaussianBlur(af5f*bf5f,(11,11),1.5)-ab5f
    t15f = (ab5f.astype(np.float64)*2+float(np.float32((.01*255)**2))).astype(np.float32)
    t25f = (vab5f.astype(np.float64)*2+float(np.float32((.03*255)**2))).astype(np.float32)
    t35f = (aa5f+bb5f)+np.float32((.01*255)**2)
    t45f = (va5f+vb5f)+np.float32((.03*255)**2)
    lum5f = float(np.cumsum((t15f/t35f).reshape(-1),dtype=np.float64)[-1])/a5f.size
    cs5f = float(np.cumsum((t25f/t45f).reshape(-1),dtype=np.float64)[-1])/a5f.size
    msssim5f *= max(0.,cs5f)**weight5f
    if scale5f==scales5f-1:
        msssim5f *= max(0.,lum5f)**weight5f
    if scale5f==0:
        map5f = (t15f*t25f)/(t35f*t45f)
        norm5f = cv2.normalize(map5f,None,0,255,cv2.NORM_MINMAX)
        first5f = np.clip(np.rint(norm5f),0,255).astype(np.uint8)
    if scale5f<scales5f-1:
        def down5f(a):
            h,w = a.shape
            h,w = h//2*2,w//2*2
            return (a[:h:2,:w:2]+a[:h:2,1:w:2]+a[1:h:2,:w:2]+a[1:h:2,1:w:2])/4
        a5f,b5f = down5f(a5f),down5f(b5f)
img = first5f
print(f'MS-SSIM {msssim5f:.4f} ({scales5f} scale)' + (' (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)' if resized5f else ''))`);
}
export function fusion5FPy(p,mirrored,alignB) {
  return [...alignB,...lines(`a5f = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) if img.ndim==2 else np.ascontiguousarray(img[:,:,:3]${mirrored?'[:,:,::-1]':''})
b5f = cv2.cvtColor(img_b,cv2.COLOR_GRAY2RGB) if img_b.ndim==2 else np.ascontiguousarray(img_b[:,:,:3]${mirrored?'[:,:,::-1]':''})
resized5f = a5f.shape[:2]!=b5f.shape[:2]
if resized5f:
    b5f = cv2.resize(b5f,(a5f.shape[1],a5f.shape[0]),interpolation=cv2.INTER_LINEAR)
nlev5f = max(1,int(np.floor(np.log2(min(a5f.shape[:2]))))-1)
wc5f,ws5f,we5f = ${num(p.wContrast,1,0,3)},${num(p.wSaturation,1,0,3)},${num(p.wExposure,1,0,3)}
kern5f = [1.,4.,6.,4.,1.]
def reflect5f(i,n):
    if n==1:
        return np.zeros_like(i)
    period = 2*n-2
    k = i%period
    return np.where(k<n,k,period-k)
def down5f(src):
    h,w = src.shape[:2]
    ow,oh = (w+1)//2,(h+1)//2
    xidx = np.arange(ow)*2
    tmp = np.zeros((h,ow)+src.shape[2:],np.float64)
    for k in range(5):
        tmp += np.take(src,reflect5f(xidx+k-2,w),axis=1).astype(np.float64)*kern5f[k]
    tmp /= 16
    yidx = np.arange(oh)*2
    out = np.zeros((oh,ow)+src.shape[2:],np.float64)
    for k in range(5):
        out += np.take(tmp,reflect5f(yidx+k-2,h),axis=0)*kern5f[k]
    return out/16
def up5f(src,shape):
    h,w = src.shape[:2]
    oh,ow = shape
    x = np.arange(ow)
    tmp = np.zeros((h,ow)+src.shape[2:],np.float64)
    for k in range(5):
        d = k-2
        valid = ((x-d)%2)==0
        index = reflect5f(np.floor_divide(x-d,2),w)
        term = np.take(src,index,axis=1).astype(np.float64)*kern5f[k]
        tmp += term*valid.reshape((1,ow)+(1,)*(src.ndim-2))
    tmp /= 8
    y = np.arange(oh)
    out = np.zeros((oh,ow)+src.shape[2:],np.float64)
    for k in range(5):
        d = k-2
        valid = ((y-d)%2)==0
        index = reflect5f(np.floor_divide(y-d,2),h)
        term = np.take(tmp,index,axis=0)*kern5f[k]
        out += term*valid.reshape((oh,1)+(1,)*(src.ndim-2))
    return out/8
def weights5f(src):
    gray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY).astype(np.int32)
    h,w = gray.shape
    yy,xx = np.indices((h,w))
    lap = gray[reflect5f(yy-1,h),xx]+gray[reflect5f(yy+1,h),xx]+gray[yy,reflect5f(xx-1,w)]+gray[yy,reflect5f(xx+1,w)]-4*gray
    contrast = np.abs(lap.astype(np.float64)/255)
    rgb = src.astype(np.float64)/255
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    mean = (r+g+b)/3
    sat = np.sqrt(((r-mean)**2+(g-mean)**2+(b-mean)**2)/3)
    exposed = np.exp(-((r-.5)**2)/.08)*np.exp(-((g-.5)**2)/.08)*np.exp(-((b-.5)**2)/.08)
    contrastw = contrast if wc5f==1 else np.ones_like(contrast) if wc5f==0 else np.power(contrast,wc5f)
    satw = sat if ws5f==1 else np.ones_like(sat) if ws5f==0 else np.power(sat,ws5f)
    exposedw = exposed if we5f==1 else np.ones_like(exposed) if we5f==0 else np.power(exposed,we5f)
    return contrastw*satw*exposedw+1e-12
wa5f,wb5f = weights5f(a5f),weights5f(b5f)
total5f = wa5f+wb5f
wa5f,wb5f = wa5f/total5f,wb5f/total5f
bands5f = []
sizes5f = [a5f.shape[:2]]
curA5f,curB5f = a5f.astype(np.float64)/255,b5f.astype(np.float64)/255
curWa5f,curWb5f = wa5f,wb5f
for level5f in range(nlev5f):
    last5f = level5f==nlev5f-1
    if last5f:
        aa5f,bb5f = curA5f,curB5f
    else:
        nextA5f,nextB5f = down5f(curA5f),down5f(curB5f)
        nextWa5f,nextWb5f = down5f(curWa5f),down5f(curWb5f)
        shape5f = curA5f.shape[:2]
        aa5f = curA5f-up5f(nextA5f,shape5f)
        bb5f = curB5f-up5f(nextB5f,shape5f)
        sizes5f.append(nextA5f.shape[:2])
    bands5f.append(curWa5f[:,:,None]*aa5f+curWb5f[:,:,None]*bb5f)
    if not last5f:
        curA5f,curB5f,curWa5f,curWb5f = nextA5f,nextB5f,nextWa5f,nextWb5f
result5f = bands5f[-1]
for level5f in range(nlev5f-2,-1,-1):
    result5f = up5f(result5f,sizes5f[level5f])+bands5f[level5f]
scaled5f = np.clip(result5f*255,0,255)
lower5f = np.floor(scaled5f)
rounded5f = lower5f+((scaled5f-lower5f)>=.5-1e-10)
img = np.ascontiguousarray(rounded5f.astype(np.uint8)[:,:,::-1])
print(f'Exposure fusion · {nlev5f} ชั้น' + (' (ปรับขนาด B ให้เท่า A ก่อนคำนวณ เพราะสองภาพขนาดไม่ตรงกัน)' if resized5f else ''))`)];
}
