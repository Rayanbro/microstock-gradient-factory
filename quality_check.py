"""Technical JPEG checks; failures stop the workflow before publishing artifacts."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
def check(folder, width=6000, height=4000):
    root=Path(folder)
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest: raise ValueError('empty manifest')
    failures=[]; thumbs=[]
    for entry in manifest:
        p=root/entry['filename']
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.format!='JPEG' or im.mode!='RGB' or im.size!=(width,height): raise ValueError('wrong JPEG format or dimensions')
                if not im.info.get('icc_profile'): raise ValueError('missing ICC color profile')
                thumb=np.asarray(im.convert('RGB').resize((48,32)),dtype=np.float32)
                if float(thumb.std(axis=(0,1)).mean())<5: raise ValueError('nearly flat image')
                thumbs.append((entry['filename'],thumb))
        except Exception as exc: failures.append(f"{p.name}: {exc}")
    # Similarity gate is conservative: flag essentially identical low-res images.
    for a in range(len(thumbs)):
        for b in range(a):
            if float(np.mean(np.abs(thumbs[a][1]-thumbs[b][1])))<0.6:
                failures.append(f"visually near-identical: {thumbs[a][0]} / {thumbs[b][0]}")
    (root/'quality-report.json').write_text(json.dumps({'checked':len(manifest),'passed':not failures,'issues':failures},indent=2))
    if failures: raise ValueError('; '.join(failures[:8]))
    return len(manifest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='output');p.add_argument('--width',type=int,default=6000);p.add_argument('--height',type=int,default=4000);a=p.parse_args()
    print('Passed:',check(a.out,a.width,a.height))
