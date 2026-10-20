"""Download the official 45-scene O-HAZE archive with CRC checks and resume support."""
from pathlib import Path
import json, urllib.request, struct, zlib, hashlib, concurrent.futures, time, argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'datasets/o_haze')
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1];dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
u=url='https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip'
with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=40) as r:
    total_size=int(r.headers['Content-Length'])
def get(a,b):
 r=urllib.request.urlopen(urllib.request.Request(u,headers={'Range':f'bytes={a}-{b}'}),timeout=40)
 assert r.status==206,(r.status,'Range not supported');return r.read()
t=get(total_size-65557,total_size-1);i=t.rfind(b'PK\x05\x06');v=struct.unpack_from('<4s4H2LH',t,i);cd=get(v[6],v[6]+v[5]-1);pos=0;rows=[]
while pos<len(cd):
 assert cd[pos:pos+4]==b'PK\x01\x02';method=struct.unpack_from('<H',cd,pos+10)[0];crc,comp,size=struct.unpack_from('<LLL',cd,pos+16);nl,el,cl=struct.unpack_from('<HHH',cd,pos+28);offset=struct.unpack_from('<L',cd,pos+42)[0];name=cd[pos+46:pos+46+nl].decode('utf-8');rows.append({'name':name,'compressed':comp,'size':size,'method':method,'crc':crc,'offset':offset});pos+=46+nl+el+cl

rows=[r for r in rows if r['name'].lower().endswith('.jpg')]
def download(r):
 name=Path(r['name']).name;folder='clear' if '/GT/' in r['name'] else 'hazy';p=dest/folder/name;p.parent.mkdir(exist_ok=True)
 if p.exists() and p.stat().st_size==r['size']:data=p.read_bytes()
 else:
  # Local file names are flattened; never extract archive paths to the filesystem.
  offset=r['offset'];end=offset+30+len(r['name'].encode())+65535+r['compressed']
  for attempt in range(3):
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{min(end,total_size-1)}'})
    with urllib.request.urlopen(req,timeout=100) as response:
     assert response.status==206;raw=response.read()
    assert raw[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',raw,26);data=raw[30+nl+el:30+nl+el+r['compressed']]
    if r['method']==8:data=zlib.decompress(data,-15)
    assert len(data)==r['size'];assert (zlib.crc32(data)&0xffffffff)==r['crc'];p.write_bytes(data);break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 assert (zlib.crc32(data)&0xffffffff)==r['crc']
 return {'file':str(p.relative_to(repo)).replace('\\','/'),'source_member':r['name'],'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
manifest=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for f in concurrent.futures.as_completed([pool.submit(download,r) for r in rows]):
  manifest.append(f.result())
  if len(manifest)%10==0:print(f'Downloaded and checksum-verified {len(manifest)}/90 images',flush=True)
(dest/'source_manifest.json').write_text(json.dumps({'source_url':url,'dataset':'O-HAZE','images':sorted(manifest,key=lambda r:r['file'])},indent=2),encoding='utf-8')
print('Official dataset complete: 45 paired scenes.',flush=True)