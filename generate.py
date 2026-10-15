import argparse, hashlib, math, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms
P=[[(12,20,45),(62,72,134),(203,128,181),(255,225,195)],[(14,36,49),(29,128,136),(151,213,195),(250,246,220)],[(32,19,48),(104,46,126),(231,84,142),(255,198,114)],[(241,221,233),(195,207,245),(189,232,222),(255,243,204)],[(12,12,18),(57,51,68),(136,109,95),(235,208,161)]]
def render(i, dest, w=6000, h=4000):
    if not 0<=i<10000: raise ValueError('index outside campaign')
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('gradient-v3-'+str(i)).encode()).digest()[:8],'big'))
    pal=np.array(P[(i//17)%len(P)],dtype=np.float32)
    mode=i%5; angle=float(rng.uniform(-3.14,3.14)); cx=float(rng.uniform(.2,.8)); cy=float(rng.uniform(.2,.8))
    x=np.linspace(0,1,w,dtype=np.float32)[None,:]
    image=Image.new('RGB',(w,h))
    for top in range(0,h,96):
        n=min(96,h-top); y=np.arange(top,top+n,dtype=np.float32)[:,None]/h
        dx=x-cx; dy=y-cy
        if mode==0: t=(dx*math.cos(angle)+dy*math.sin(angle)+.8)/1.6
        elif mode==1: t=np.sqrt(dx*dx+dy*dy)*1.2
        elif mode==2: t=(np.arctan2(dy,dx)+math.pi)/(2*math.pi)
        elif mode==3: t=x*.7+y*.3+.07*np.sin(8*x+7*y)
        else: t=np.sqrt((dx*1.4)**2+(dy*.7)**2)+.06*np.sin(10*x-6*y)
        t=np.broadcast_to(np.clip(t,0,1),(n,w)); z=t*3; k=np.minimum(z.astype(np.int8),2)
        f=(z-k)[...,None]; f=f*f*(3-2*f)
        rgb=pal[k]*(1-f)+pal[k+1]*f
        image.paste(Image.fromarray(np.uint8(np.clip(rgb+.5,0,255)),'RGB'),(0,top))
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    image.save(dest,'JPEG',quality=94,subsampling=0,icc_profile=icc,optimize=True)
    return {'index':i,'filename':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=2);p.add_argument('--width',type=int,default=6000);p.add_argument('--height',type=int,default=4000);p.add_argument('--out',default='output')
    a=p.parse_args()
    if a.start<0 or not 1<=a.count<=50 or a.start+a.count>10000: p.error('invalid campaign range')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    m=[render(i,out/f'gradient_{i+1:05d}.jpg',a.width,a.height) for i in range(a.start,a.start+a.count)]
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__': main()
