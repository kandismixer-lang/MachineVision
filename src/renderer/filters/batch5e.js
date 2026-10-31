// Float64 kernels for batch 5E. Keep the operation order in codegen-batch5e.js.
export const byte5E = v => Math.max(0, Math.min(255, Math.floor(v + 0.5)));
export function jacobi5E(matrix) {
  const a = matrix.map(r => [...r]), v = [[1,0,0],[0,1,0],[0,0,1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    if (Math.abs(a[0][1]) < 1e-14 && Math.abs(a[0][2]) < 1e-14 && Math.abs(a[1][2]) < 1e-14) break;
    for (const [p,q] of [[0,1],[0,2],[1,2]]) {
      if (Math.abs(a[p][q]) < 1e-14) continue;
      const tau = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
      const c = 1 / Math.sqrt(1 + t * t), s = t * c, apq = a[p][q];
      a[p][p] -= t * apq; a[q][q] += t * apq; a[p][q] = a[q][p] = 0;
      for (let k = 0; k < 3; k++) {
        if (k !== p && k !== q) {
          const kp = a[k][p], kq = a[k][q];
          a[k][p] = a[p][k] = c * kp - s * kq;
          a[k][q] = a[q][k] = s * kp + c * kq;
        }
        const vp = v[k][p], vq = v[k][q];
        v[k][p] = c * vp - s * vq; v[k][q] = s * vp + c * vq;
      }
    }
  }
  return { values: [a[0][0],a[1][1],a[2][2]], vectors: v };
}
const YIQ = [[.299,.587,.114],[.596,-.274,-.322],[.211,-.523,.312]];
export function inverse3E(a) {
  const c = Array.from({length:3},()=>[0,0,0]);
  for(let r=0;r<3;r++)for(let k=0;k<3;k++)c[r][k]=a[(r+1)%3][(k+1)%3]*a[(r+2)%3][(k+2)%3]-a[(r+1)%3][(k+2)%3]*a[(r+2)%3][(k+1)%3];
  const det=a[0][0]*c[0][0]+a[0][1]*c[0][1]+a[0][2]*c[0][2];
  return c.map((_,r)=>c.map(row=>row[r]/det));
}
const INV_YIQ = inverse3E(YIQ);
export function yiq5E(data, inverse = false) {
  const out = new Uint8Array(data.length), a = inverse ? INV_YIQ : YIQ;
  for(let i=0;i<data.length;i+=3){
    const x = inverse ? [data[i]/255,(data[i+1]-127.5)*(2*.596)/255,(data[i+2]-127.5)*(2*.523)/255] : [data[i]/255,data[i+1]/255,data[i+2]/255];
    for(let c=0;c<3;c++){
      let v=a[c][0]*x[0]+a[c][1]*x[1]+a[c][2]*x[2];
      v=inverse||c===0?v*255:v*255/(2*(c===1?.596:.523))+127.5;
      out[i+c]=byte5E(v);
    }
  }
  return out;
}
export function srgbLut5E(inverse) {
  return Uint8Array.from({length:256},(_,i)=>{const c=i/255;return byte5E(255*(inverse?(c<=.0031308?12.92*c:1.055*c**(1/2.4)-.055):(c<=.04045?c/12.92:((c+.055)/1.055)**2.4)));});
}
export function decorr5E(data,tol,mode='correlation') {
  const n=data.length/3,sums=[0,0,0],cross=[[0,0,0],[0,0,0],[0,0,0]];
  for(let i=0;i<data.length;i+=3)for(let r=0;r<3;r++){sums[r]+=data[i+r];for(let c=0;c<3;c++)cross[r][c]+=data[i+r]*data[i+c];}
  const mean=sums.map(s=>s/n),cov=cross.map((row,r)=>row.map((s,c)=>n>1?(s-sums[r]*sums[c]/n)/(n-1):0));
  const sigma=cov.map((row,c)=>Math.sqrt(Math.max(0,row[c]))),invSigma=sigma.map(s=>s>0?1/s:0),correlation=mode!=='covariance';
  // A constant band has a zero row/column rather than an undefined correlation.
  const matrix=correlation?cov.map((row,r)=>row.map((s,c)=>invSigma[r]*s*invSigma[c])):cov;
  const {values,vectors:v}=jacobi5E(matrix),inv=values.map(x=>x>1e-12?1/Math.sqrt(x):0);
  const transform=Array.from({length:3},(_,r)=>Array.from({length:3},(_,c)=>{let s=0;for(let k=0;k<3;k++)s+=v[r][k]*inv[k]*v[c][k];return correlation?invSigma[r]*s*sigma[c]:s*sigma[c];}));
  const out=new Float64Array(data.length);
  for(let i=0;i<data.length;i+=3)for(let c=0;c<3;c++)out[i+c]=(data[i]-mean[0])*transform[0][c]+(data[i+1]-mean[1])*transform[1][c]+(data[i+2]-mean[2])*transform[2][c]+mean[c];
  if(tol>0)for(let c=0;c<3;c++){
    const sorted=Array.from({length:n},(_,i)=>out[3*i+c]).sort((a,b)=>a-b);
    const quantile=p=>{const pos=(n-1)*p,j=Math.floor(pos);return sorted[j]+(sorted[Math.min(n-1,j+1)]-sorted[j])*(pos-j);};
    const lo=quantile(tol/100),hi=quantile(1-tol/100);
    if(hi>lo)for(let i=c;i<out.length;i+=3)out[i]=(out[i]-lo)*255/(hi-lo);
  }
  return { data: Uint8Array.from(out,byte5E), singular: values.some(x=>x<=1e-12) };
}
export function pca5E(data,percent) {
  const pixels=[],sums=[0,0,0];
  for(let i=0;i<data.length;i+=3)if(data[i]!==255&&data[i+1]!==255&&data[i+2]!==255){pixels.push(i);for(let c=0;c<3;c++)sums[c]+=data[i+c];}
  const n=pixels.length;
  if(!n)return null;
  const mean=sums.map(s=>s/255/n),norm=Math.sqrt(mean[0]*mean[0]+mean[1]*mean[1]+mean[2]*mean[2]);
  if(!norm)return null;
  const m=mean.map(x=>x/norm),projections=new Float64Array(data.length/3);
  for(const i of pixels)projections[i/3]=(data[i]/255)*m[0]+(data[i+1]/255)*m[1]+(data[i+2]/255)*m[2];
  pixels.sort((i,j)=>projections[i/3]-projections[j/3]||i-j);
  const k=Math.max(1,Math.floor(percent/100*n)),selected=[...pixels.slice(0,k),...pixels.slice(n-k)],a=[[0,0,0],[0,0,0],[0,0,0]];
  for(const i of selected)for(let r=0;r<3;r++)for(let c=0;c<3;c++)a[r][c]+=(data[i+r]/255)*(data[i+c]/255);
  const {values,vectors}=jacobi5E(a);let largest=0;for(let c=1;c<3;c++)if(values[c]>values[largest])largest=c;
  const light=vectors.map(row=>Math.abs(row[largest])),target=(light[0]+light[1]+light[2])/3;
  return {light,gains:light.map(x=>x>1e-3?target/x:1)};
}
export function angle5E(r,g,b,R,G,B) {
  const norm=Math.sqrt(r*r+g*g+b*b)*Math.sqrt(R*R+G*G+B*B);
  return norm===0?0:Math.acos(Math.max(-1,Math.min(1,(r*R+g*G+b*B)/norm)))*180/Math.PI;
}
// Grow only from existing components; never add an 8-neighbor bridge between labels.
// Seeds and neighbor visits are row-major, matching the Python queue exactly.
export function guardThicken5E(candidate,owner,w,h,simple) {
  const ring=[[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1]];
  const queue=new Int32Array(w*h),blocked=new Uint8Array(w*h);let tail=0;
  for(let i=0;i<owner.length;i++)if(owner[i])queue[tail++]=i;
  const initial=tail;
  for(let head=0;head<tail;head++){
    const at=queue[head],x=at%w,y=Math.floor(at/w);
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
      const xx=x+dx,yy=y+dy,i=yy*w+xx;
      if(xx<0||yy<0||xx>=w||yy>=h||owner[i]||blocked[i]||!candidate[i])continue;
      let label=0,conflict=false;
      for(let oy=-1;oy<=1;oy++)for(let ox=-1;ox<=1;ox++){
        const nx=xx+ox,ny=yy+oy;
        if(nx<0||ny<0||nx>=w||ny>=h)continue;
        const other=owner[ny*w+nx];if(other){if(label&&other!==label)conflict=true;label=other;}
      }
      if(conflict)blocked[i]=1;
      else if(label){
        let mask=0;
        for(let bit=0;bit<8;bit++){
          const nx=xx+ring[bit][0],ny=yy+ring[bit][1];
          if(nx>=0&&ny>=0&&nx<w&&ny<h&&owner[ny*w+nx])mask|=1<<bit;
        }
        // An addition is simple under foreground-8/background-4 connectivity.
        // Unlike the complement's thin LUT alone, this also preserves holes.
        if(simple[mask]){owner[i]=label;queue[tail++]=i;}
      }
    }
  }
  return tail!==initial;
}
