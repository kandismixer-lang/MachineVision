// Batch 5F numerical kernels. Python mirrors the precision and operation order.
export const byte5F = v => Number.isFinite(v) ? Math.max(0, Math.min(255, Math.floor(v + .5))) : 0;
const roundFusion5F = v => {const x=Math.max(0,Math.min(255,v)),lo=Math.floor(x);return lo+(x-lo>=.5-1e-10?1:0);};
const reflect101 = (x, n) => { if(n===1)return 0; const p=2*n-2,k=((x%p)+p)%p; return k<n?k:p-k; };
// Projection autocorrelation at distance from the nearer frame edge. Overlapping
// taper bands on tiny images use the nearer edge; singleton axes are fully blurred.
export function taperAxis5F(projection, length) {
  const ac = new Float64Array(projection.length);
  for(let d=0;d<ac.length;d++)for(let k=0;k<ac.length-d;k++)ac[d]+=projection[k]*projection[k+d];
  return Float64Array.from({length},(_,x)=>{const d=Math.min(x,length-1-x);return d<ac.length&&ac[0]>0?1-ac[d]/ac[0]:1;});
}
export function edgeTaper5F(cv, src, psf) {
  const owned=[],mat=()=>{const m=new cv.Mat();owned.push(m);return m;};
  try {
    const n=psf.n,r=(n-1)/2,w=src.cols,h=src.rows,ch=src.channels();
    const px=new Float64Array(n),py=new Float64Array(n);
    for(let y=0;y<n;y++)for(let x=0;x<n;x++){const k=psf.k[y*n+x];px[x]+=k;py[y]+=k;}
    const ax=taperAxis5F(px,w),ay=taperAxis5F(py,h);
    const kernel=cv.matFromArray(n,n,cv.CV_32F,psf.k);owned.push(kernel);
    const flip=mat(),f=mat(),pad=mat(),blur=mat();
    cv.flip(kernel,flip,-1);src.convertTo(f,cv.CV_32F);
    cv.copyMakeBorder(f,pad,r,r,r,r,cv.BORDER_WRAP);
    cv.filter2D(pad,blur,-1,flip,new cv.Point(-1,-1),0,cv.BORDER_CONSTANT);
    const input=src.data,bd=blur.data32F,out=new Uint8Array(w*h*ch),pw=w+2*r;
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)for(let c=0;c<ch;c++){
      const i=(y*w+x)*ch+c,a=ay[y]*ax[x];out[i]=byte5F(a*input[i]+(1-a)*bd[((y+r)*pw+x+r)*ch+c]);
    }
    return cv.matFromArray(h,w,src.type(),out);
  } finally {for(const m of owned)m.delete();}
}
export function means5F(data,mask) {
  let si=0,so=0,ni=0,no=0;
  for(let i=0;i<data.length;i++)if(mask[i]){si+=data[i];ni++;}else{so+=data[i];no++;}
  return {cin:ni?si/ni:so/Math.max(1,no),cout:no?so/no:si/Math.max(1,ni),area:ni};
}
export function activeContour5F(data,w,h,initial,iters,smooth,bias) {
  let mask=Uint8Array.from(initial),used=0,stable=false;
  for(;used<iters;){
    const old=mask,{cin,cout}=means5F(data,old);let next=new Uint8Array(old.length);
    for(let i=0;i<next.length;i++){const v=data[i],f=(v-cout)**2-(v-cin)**2-bias*65025*.01;next[i]=f>0?1:f<0?0:old[i];}
    for(let s=0;s<smooth;s++){
      const tmp=new Uint8Array(next.length);
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){
        let count=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)count+=next[Math.max(0,Math.min(h-1,y+dy))*w+Math.max(0,Math.min(w-1,x+dx))];
        tmp[y*w+x]=count>=5?1:0;
      }next=tmp;
    }
    used++;stable=next.every((v,i)=>v===old[i]);mask=next;if(stable)break;
  }
  return {mask,used,stable,...means5F(data,mask)};
}
function laplaceState5F(data,w,h,ch,mask) {
  const border=new Uint8Array(w*h),red=[],black=[];
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x]){
    ((x+y)&1?black:red).push(y*w+x);
    for(const [xx,yy]of [[x,y-1],[x,y+1],[x-1,y],[x+1,y]])if(xx>=0&&yy>=0&&xx<w&&yy<h&&!mask[yy*w+xx])border[yy*w+xx]=1;
  }
  const sums=new Float64Array(ch);let count=0;
  for(let i=0;i<border.length;i++)if(border[i]){count++;for(let c=0;c<ch;c++)sums[c]+=data[i*ch+c];}
  const mean=Float64Array.from(sums,v=>count?v/count:0),values=Float64Array.from(data);
  for(const ids of [red,black])for(const i of ids)for(let c=0;c<ch;c++)values[i*ch+c]=mean[c];
  return {w,h,ch,mask,red,black,values};
}
function downLaplace5F(level) {
  const {w,h,ch,mask,values}=level,nw=Math.ceil(w/2),nh=Math.ceil(h/2),data=new Float64Array(nw*nh*ch),nextMask=new Uint8Array(nw*nh);
  for(let y=0;y<nh;y++)for(let x=0;x<nw;x++){
    const i=y*nw+x;let n=0;
    for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){const xx=x*2+dx,yy=y*2+dy;if(xx<w&&yy<h){const j=yy*w+xx;if(mask[j])nextMask[i]=1;n++;for(let c=0;c<ch;c++)data[i*ch+c]+=values[j*ch+c];}}
    for(let c=0;c<ch;c++)data[i*ch+c]/=n;
  }
  return laplaceState5F(data,nw,nh,ch,nextMask);
}
function solveLaplace5F(level,budget) {
  const {w,h,ch,red,black,values:a}=level,omega=2/(1+Math.sin(Math.PI/Math.max(w,h)));let used=0,stable=false;
  for(;used<budget;){
    let max=0;
    for(const ids of [red,black])for(const i of ids){
      const y=Math.floor(i/w),x=i%w,n=(y>0)+(y<h-1)+(x>0)+(x<w-1);
      for(let c=0;c<ch;c++){
        const j=i*ch+c;let s=0;
        if(y>0)s+=a[j-w*ch];if(y<h-1)s+=a[j+w*ch];if(x>0)s+=a[j-ch];if(x<w-1)s+=a[j+ch];
        const v=n?a[j]+omega*(s/n-a[j]):a[j];max=Math.max(max,Math.abs(v-a[j]));a[j]=v;
      }
    }
    used++;if(max<.01){stable=true;break;}
  }
  return {used,stable};
}
export function laplace5F(data,w,h,ch,mask,budget) {
  let level=laplaceState5F(data,w,h,ch,mask),factor=1;const hierarchy=[level];
  while(Math.max(level.w,level.h)>32){level=downLaplace5F(level);factor*=2;hierarchy.push(level);}
  let used=0;
  if(hierarchy.length>1&&budget>=3){
    const coarse=hierarchy.at(-1),q=solveLaplace5F(coarse,Math.min(500,Math.max(1,Math.floor(budget/3))));used+=q.used;
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x;if(!mask[i])continue;
      const cx=Math.floor(x/factor),cy=Math.floor(y/factor),dx=(x%factor)/factor,dy=(y%factor)/factor,cx1=Math.min(coarse.w-1,cx+1),cy1=Math.min(coarse.h-1,cy+1);
      for(let c=0;c<ch;c++){
        const a=coarse.values[(cy*coarse.w+cx)*ch+c],b=coarse.values[(cy*coarse.w+cx1)*ch+c],d=coarse.values[(cy1*coarse.w+cx)*ch+c],e=coarse.values[(cy1*coarse.w+cx1)*ch+c];
        hierarchy[0].values[i*ch+c]=a*(1-dx)*(1-dy)+b*dx*(1-dy)+d*(1-dx)*dy+e*dx*dy;
      }
    }
  }
  const fine=hierarchy[0],q=solveLaplace5F(fine,budget-used);used+=q.used;
  return {data:Uint8Array.from(fine.values,byte5F),used,stable:q.stable};
}
// The shared Lucy update uses exactly the original float32 filter2D path.
export function lucyStep5F(cv,g,est,K,Kf,cc,ratio,corr) {
  const n=g.rows*g.cols,anchor=new cv.Point(-1,-1);
  cv.filter2D(est,cc,-1,Kf,anchor,0,cv.BORDER_REFLECT_101);
  {const gd=g.data32F,cd=cc.data32F,rd=ratio.data32F;for(let i=0;i<n;i++)rd[i]=cd[i]>1e-6?gd[i]/cd[i]:0;}
  cv.filter2D(ratio,corr,-1,K,anchor,0,cv.BORDER_REFLECT_101);
  {const ed=est.data32F,cr=corr.data32F;for(let i=0;i<n;i++)ed[i]=ed[i]*cr[i];}
}
export function blind5F(cv,src,gray,n,iters) {
  const owned=[],mat=()=>{const m=new cv.Mat();owned.push(m);return m;};
  try {
    const w=src.cols,h=src.rows,ch=src.channels(),r=(n-1)/2;
    let psf=Array(n*n).fill(1/(n*n));
    const K=mat(),Kf=mat(),cc=mat(),corr=mat(),ratio=mat(),padded=mat(),psfCorr=mat();ratio.create(h,w,cv.CV_32F);
    const g=mat();gray.convertTo(g,cv.CV_32F);const est=g.clone();owned.push(est);
    const planes=[];
    if(ch>1)for(let c=0;c<ch;c++){
      const p=cv.matFromArray(h,w,cv.CV_32F,Float32Array.from({length:w*h},(_,i)=>src.data[i*ch+c]));owned.push(p);
      const e=p.clone();owned.push(e);planes.push([p,e]);
    }
    for(let it=0;it<iters;it++){
      K.create(n,n,cv.CV_32F);K.data32F.set(psf);cv.flip(K,Kf,-1);
      lucyStep5F(cv,g,est,K,Kf,cc,ratio,corr);
      for(const [p,e]of planes)lucyStep5F(cv,p,e,K,Kf,cc,ratio,corr);
      cv.filter2D(est,cc,-1,Kf,new cv.Point(-1,-1),0,cv.BORDER_REFLECT_101);
      {const gd=g.data32F,cd=cc.data32F,rd=ratio.data32F;for(let i=0;i<rd.length;i++)rd[i]=cd[i]>1e-6?gd[i]/cd[i]:0;}
      cv.copyMakeBorder(est,padded,r,r,r,r,cv.BORDER_REFLECT_101);cv.matchTemplate(padded,ratio,psfCorr,cv.TM_CCORR);
      const cd=psfCorr.data32F,next=new Float64Array(n*n);let total=0;
      for(let ky=0;ky<n;ky++)for(let kx=0;kx<n;kx++){
        const i=ky*n+kx,sum=cd[(n-1-ky)*n+n-1-kx];next[i]=Math.max(0,psf[i]*sum);total+=next[i];
      }
      psf=total>0?Array.from(next,v=>v/total):Array(n*n).fill(1/(n*n));
    }
    const out=new Uint8Array(w*h*ch);
    if(ch===1)out.set(Uint8Array.from(est.data32F,byte5F));
    else for(let c=0;c<ch;c++){const d=planes[c][1].data32F;for(let i=0;i<d.length;i++)out[i*ch+c]=byte5F(d[i]);}
    return {data:out,psf};
  } finally {for(const m of owned)m.delete();}
}
function ssimScale5F(cv,a,b,w,h,keepMap) {
  const owned=[],mat=()=>{const m=new cv.Mat();owned.push(m);return m;};
  try {
    const A=cv.matFromArray(h,w,cv.CV_32F,a),B=cv.matFromArray(h,w,cv.CV_32F,b);owned.push(A,B);
    const ma=mat(),mb=mat(),aa=mat(),bb=mat(),ab=mat(),va=mat(),vb=mat(),vab=mat(),map=keepMap?mat():null;
    const win=new cv.Size(11,11);cv.GaussianBlur(A,ma,win,1.5);cv.GaussianBlur(B,mb,win,1.5);
    cv.multiply(A,A,aa);cv.multiply(B,B,bb);cv.multiply(A,B,ab);
    cv.GaussianBlur(aa,va,win,1.5);cv.GaussianBlur(bb,vb,win,1.5);cv.GaussianBlur(ab,vab,win,1.5);
    if(map)map.create(h,w,cv.CV_32F);
    const f=Math.fround,C1=f((.01*255)**2),C2=f((.03*255)**2),mad=ma.data32F,mbd=mb.data32F;
    const ad=va.data32F,bd=vb.data32F,cd=vab.data32F,md=map?map.data32F:null;let l=0,cs=0;
    for(let i=0;i<a.length;i++){
      const a2=f(mad[i]*mad[i]),b2=f(mbd[i]*mbd[i]),ab2=f(mad[i]*mbd[i]);
      const t1=f(2*ab2+C1),t2=f(2*f(cd[i]-ab2)+C2),t3=f(f(a2+b2)+C1),t4=f(f(f(ad[i]-a2)+f(bd[i]-b2))+C2);
      l+=f(t1/t3);cs+=f(t2/t4);if(map)md[i]=f(f(t1*t2)/f(t3*t4));
    }
    let mapBytes=null;
    if(map){const norm=mat(),bytes=mat();cv.normalize(map,norm,0,255,cv.NORM_MINMAX);norm.convertTo(bytes,cv.CV_8U);mapBytes=Uint8Array.from(bytes.data);}
    return {l:l/a.length,cs:cs/a.length,map:mapBytes};
  }finally{for(const m of owned)m.delete();}
}
export function multissim5F(cv,a,b,w,h) {
  const weights=[.0448,.2856,.3001,.2363,.1333];let k=1;
  while(k<5&&Math.min(w,h)>=11*2**k)k++;
  const total=weights.slice(0,k).reduce((s,v)=>s+v,0);let score=1,first;
  let aa=Float64Array.from(a),bb=Float64Array.from(b);
  for(let s=0;s<k;s++){
    const q=ssimScale5F(cv,aa,bb,w,h,!s);if(!s)first=q.map;
    const weight=weights[s]/total;score*=Math.max(0,q.cs)**weight;if(s===k-1)score*=Math.max(0,q.l)**weight;
    if(s<k-1){const nw=Math.floor(w/2),nh=Math.floor(h/2),down=d=>Float64Array.from({length:nw*nh},(_,i)=>{const y=Math.floor(i/nw)*2,x=(i%nw)*2,j=y*w+x;return(d[j]+d[j+1]+d[j+w]+d[j+w+1])/4;});aa=down(aa);bb=down(bb);w=nw;h=nh;}
  }
  return {score,scales:k,map:first};
}
const PYR5F=[1,4,6,4,1];
function downPyramid5F(src,w,h,ch) {
  const ow=Math.ceil(w/2),oh=Math.ceil(h/2),horizontal=new Float64Array(h*ow*ch),out=new Float64Array(oh*ow*ch);
  for(let y=0;y<h;y++)for(let x=0;x<ow;x++)for(let c=0;c<ch;c++){
    let sum=0;for(let k=0;k<5;k++)sum+=PYR5F[k]*src[(y*w+reflect101(2*x+k-2,w))*ch+c];
    horizontal[(y*ow+x)*ch+c]=sum/16;
  }
  for(let y=0;y<oh;y++)for(let x=0;x<ow;x++)for(let c=0;c<ch;c++){
    let sum=0;for(let k=0;k<5;k++)sum+=PYR5F[k]*horizontal[(reflect101(2*y+k-2,h)*ow+x)*ch+c];
    out[(y*ow+x)*ch+c]=sum/16;
  }
  return out;
}
function upPyramid5F(src,w,h,ch,ow,oh) {
  const horizontal=new Float64Array(h*ow*ch),out=new Float64Array(oh*ow*ch);
  for(let y=0;y<h;y++)for(let x=0;x<ow;x++)for(let c=0;c<ch;c++){
    let sum=0;for(let d=-2;d<=2;d++)if(((x-d)&1)===0)sum+=PYR5F[d+2]*src[(y*w+reflect101((x-d)/2,w))*ch+c];
    horizontal[(y*ow+x)*ch+c]=sum/8;
  }
  for(let y=0;y<oh;y++)for(let x=0;x<ow;x++)for(let c=0;c<ch;c++){
    let sum=0;for(let d=-2;d<=2;d++)if(((y-d)&1)===0)sum+=PYR5F[d+2]*horizontal[(reflect101((y-d)/2,h)*ow+x)*ch+c];
    out[(y*ow+x)*ch+c]=sum/8;
  }
  return out;
}
export function fusion5F(cv,a,b,grayA,grayB,weights) {
  const w=a.cols,h=a.rows,n=w*h,nlev=Math.max(1,Math.floor(Math.log2(Math.min(w,h)))-1);
  const source=(src,gray)=>{
    const pixels=src.data,grayBytes=gray.data,contr=new Float64Array(n),out=new Float64Array(n*3);
    for(let i=0;i<n;i++){
      const x=i%w,y=Math.floor(i/w),v=grayBytes[i];
      const up=grayBytes[reflect101(y-1,h)*w+x],dn=grayBytes[reflect101(y+1,h)*w+x],lf=grayBytes[y*w+reflect101(x-1,w)],rt=grayBytes[y*w+reflect101(x+1,w)];
      contr[i]=Math.abs((up+dn+lf+rt-4*v)/255);
      const r=pixels[i*3]/255,g=pixels[i*3+1]/255,bl=pixels[i*3+2]/255,m=(r+g+bl)/3;
      const sat=Math.sqrt(((r-m)**2+(g-m)**2+(bl-m)**2)/3),exp=Math.exp(-((r-.5)**2)/.08)*Math.exp(-((g-.5)**2)/.08)*Math.exp(-((bl-.5)**2)/.08);
      out[i*3]=r;out[i*3+1]=g;out[i*3+2]=bl;
      contr[i]=Math.pow(contr[i],weights[0])*Math.pow(sat,weights[1])*Math.pow(exp,weights[2])+1e-12;
    }
    return {image:out,weight:contr};
  };
  const A=source(a,grayA),B=source(b,grayB),wa=A.weight,wb=B.weight;
  for(let i=0;i<n;i++){const sum=wa[i]+wb[i];wa[i]/=sum;wb[i]/=sum;}
  const ga=[A.image],gb=[B.image],pa=[wa],pb=[wb],sizes=[[w,h]];
  for(let l=1;l<nlev;l++){
    const [pw,ph]=sizes[l-1],nw=Math.ceil(pw/2),nh=Math.ceil(ph/2);
    ga.push(downPyramid5F(ga[l-1],pw,ph,3));gb.push(downPyramid5F(gb[l-1],pw,ph,3));
    pa.push(downPyramid5F(pa[l-1],pw,ph,1));pb.push(downPyramid5F(pb[l-1],pw,ph,1));sizes.push([nw,nh]);
  }
  const bands=[];
  for(let l=0;l<nlev;l++){
    const [lw,lh]=sizes[l],last=l===nlev-1,ua=last?null:upPyramid5F(ga[l+1],sizes[l+1][0],sizes[l+1][1],3,lw,lh),ub=last?null:upPyramid5F(gb[l+1],sizes[l+1][0],sizes[l+1][1],3,lw,lh);
    const band=new Float64Array(lw*lh*3);
    for(let i=0;i<lw*lh;i++)for(let c=0;c<3;c++){
      const j=i*3+c,av=last?ga[l][j]:ga[l][j]-ua[j],bv=last?gb[l][j]:gb[l][j]-ub[j];band[j]=pa[l][i]*av+pb[l][i]*bv;
    }
    bands.push(band);
  }
  let result=bands[nlev-1];
  for(let l=nlev-2;l>=0;l--){const [tw,th]=sizes[l],up=upPyramid5F(result,sizes[l+1][0],sizes[l+1][1],3,tw,th);result=Float64Array.from(bands[l],(v,i)=>up[i]+v);}
  return {data:Uint8Array.from(result,v=>roundFusion5F(v*255)),levels:nlev};
}
