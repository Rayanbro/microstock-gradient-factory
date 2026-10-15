from pathlib import Path
from PIL import Image
from generate import render
def test_render(tmp_path):
    a=tmp_path/'a.jpg';b=tmp_path/'b.jpg'
    x=render(0,a,120,80);y=render(0,b,120,80)
    assert x['sha256']==y['sha256']
    with Image.open(a) as im:
        assert im.size==(120,80) and im.mode=='RGB' and im.info.get('icc_profile')
def test_distinct(tmp_path):
    hashes=[render(i,tmp_path/f'{i}.jpg',120,80)['sha256'] for i in range(5)]
    assert len(set(hashes))==5
