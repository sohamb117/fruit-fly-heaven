"""Measured BANC geometry and EM overview for the existing anatomical viewer.

Only display geometry and microscopy are sampled. No simulation cells or edges
are selected or omitted here. Coordinates stay in BANC micrometers.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import io
import json
import subprocess
import numpy as np
import pyarrow.feather as feather
from cloudvolume import CloudVolume

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/prepared/banc888'
OUT=DATA/'anatomy'
CACHE=ROOT/'data/raw/banc888/anatomy'
BASE='https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome'
SWC=BASE+'/compiled_data/banc_888/banc_banc_space_swc'
EM='https://storage.googleapis.com/seunglab_lee_fly_cns_001_alignment/aligned/v0'


def download(url,path):
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.part')
        subprocess.run(['curl','-fsSL','--retry','2',url,'-o',str(temporary)],check=True,capture_output=True)
        temporary.replace(path)
    return path.read_bytes()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(planes):
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    ids=np.fromfile(DATA/'ids.bin','<u8')
    table={r['banc_888_id']:r for r in feather.read_table(ROOT/'data/raw/banc888/meta.feather').to_pylist()}
    rows=[table.get(str(i),{}) for i in ids]
    pos=[];owners=[];invalid_positions=[];fallback_positions=[]
    for i,r in enumerate(rows):
        for field,scale in [('root_position_nm',np.array([.001,.001,.001])),('position',np.array([.004,.004,.045]))]:
            text=r.get(field)
            if not text:continue
            p=np.fromstring(text.strip('[]()'),sep=',')
            if len(p)!=3 or not np.isfinite(p).all():continue
            p=p*scale
            if (p<0).any() or (p>np.array([1048.576,1179.648,315.45])).any():
                invalid_positions.append({'index':i,'field':field,'position_um':p.tolist()});continue
            pos.append(p);owners.append(i)
            if field=='position' and r.get('root_position_nm'):fallback_positions.append(i)
            break
    np.array(pos,'<f4').tofile(OUT/'positions.bin');np.array(owners,'<u4').tofile(OUT/'neuron-indices.bin')
    labels=[r.get('cell_type') or r.get('cell_class') or r.get('super_class') or 'Unclassified' for r in rows]
    (OUT/'neurons.json').write_text(json.dumps({'ids':[str(i) for i in ids],'labels':labels},separators=(',',':')))
    groups=json.loads((DATA/'console/groups.json').read_text())
    chosen=set(sum((v for k,v in groups.items() if k not in ('odor_left','odor_right','sweet')),[]))
    # Include representative cell types from brain, VNC and every sensory class.
    by_type={}
    for i,r in enumerate(rows):by_type.setdefault((r.get('region'),r.get('cell_class')),[]).append(i)
    rng=np.random.default_rng(888)
    for values in by_type.values():chosen.update(rng.choice(values,min(3,len(values)),replace=False).tolist())

    def skeleton(i):
        path=CACHE/f'{ids[i]}_l2.swc'
        try:raw=download(f'{SWC}/{ids[i]}_l2.swc',path)
        except subprocess.CalledProcessError:
            path=CACHE/f'{ids[i]}_pcg.swc'
            raw=download(f'{BASE}/neuron_skeletons/swcs-from-pcg-skel/{ids[i]}.swc',path)
        a=np.loadtxt(io.BytesIO(raw),comments='#',ndmin=2)
        if a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('Malformed SWC')
        lookup={int(v):k for k,v in enumerate(a[:,0])}
        edges=np.array([[lookup[int(parent)],k] for k,parent in enumerate(a[:,6]) if parent>=0 and int(parent) in lookup],'<u4').reshape(-1,2)
        return i,a[:,2:5].astype('<f4'),edges,digest(path)
    items=[];failures=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(skeleton,i):i for i in sorted(chosen)}
        for done,future in enumerate(as_completed(jobs),1):
            try:items.append(future.result())
            except Exception as error:failures.append({'index':jobs[future],'message':str(error)})
            if done%100==0:print(f'Skeletons {done}/{len(chosen)}',flush=True)
    vertices=[];branch_owners=[];edges=[];ranges=[];offset=edge_offset=0
    for i,v,e,h in sorted(items):
        vertices.append(v);branch_owners.append(np.full(len(v),i,'<u4'));edges.append(e+offset)
        ranges.append({'index':i,'id':str(ids[i]),'vertexOffset':offset,'vertexCount':len(v),'edgeOffset':edge_offset,'edgeCount':len(e),'sha256':h})
        offset+=len(v);edge_offset+=len(e)
    if not items:raise RuntimeError('No BANC skeletons downloaded')
    np.concatenate(vertices).tofile(OUT/'skeleton-positions.bin');np.concatenate(branch_owners).tofile(OUT/'skeleton-neurons.bin');np.concatenate(edges).astype('<u4').tofile(OUT/'skeleton-edges.bin')
    # Measured neuropil surfaces, including both brain and ventral nerve cord.
    surface=CloudVolume('precomputed://'+BASE+'/region_outlines',progress=False,use_https=True)
    mesh=surface.mesh.get([3,4],fuse=True)
    (mesh.vertices/1000).astype('<f4').tofile(OUT/'surface-positions.bin');mesh.faces.astype('<u4').tofile(OUT/'surface-triangles.bin')
    info=json.loads(download(EM+'/info',CACHE/'em-info.json'))['scales'][6]
    step=max(1,(info['size'][2]-1)//(planes-1));zs=list(range(0,info['size'][2],step))
    volume_path=OUT/'em-volume.bin'
    dims=[info['size'][0]//4,info['size'][1]//4,len(zs)]
    if not volume_path.exists() or volume_path.stat().st_size!=int(np.prod(dims)):
        def section(z):
            path=CACHE/f'em-z{z}.bin'
            if not path.exists():
                volume=CloudVolume('precomputed://'+EM,mip=6,progress=False,parallel=1,fill_missing=False,use_https=True)
                plane=np.asarray(volume[:,:,z:z+1])[:,:,0,0]
                # Area average in x/y. Selected z planes are measured, not
                # synthesized; the viewer interpolates the coarse overview.
                reduced=plane.reshape(dims[0],4,dims[1],4).mean(axis=(1,3)).astype('u1')
                reduced.ravel(order='F').tofile(path)
            return np.fromfile(path,'u1').reshape(dims[:2],order='F')
        result=np.empty(dims,dtype='u1')
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs={pool.submit(section,z):i for i,z in enumerate(zs)}
            for done,future in enumerate(as_completed(jobs),1):
                result[:,:,jobs[future]]=future.result()
                if done%8==0:print(f'EM planes {done}/{len(zs)}',flush=True)
        result.ravel(order='F').tofile(volume_path)
    manifest={'schema':'fruit-fly-anatomy-v1','dataset':'BANC v888','units':'µm','neuronCount':len(ids),'mappedNeurons':len(pos),
              'missingIndices':sorted(set(range(len(ids)))-set(owners)),'positionMeaning':'Annotated representative point; not necessarily soma',
              'invalidPositions':invalid_positions,'fallbackPositionIndices':fallback_positions,
              'bounds':[np.min(pos,axis=0).tolist(),np.max(pos,axis=0).tolist()],
              'skeletonCount':len(items),'skeletonVertices':offset,'skeletonEdges':edge_offset,'skeletonRanges':ranges,'skeletonFailures':failures,
              'surfaceVertices':len(mesh.vertices),'surfaceTriangles':len(mesh.faces),
              'volume':{'dimensions':dims,'spacing':[2.048,2.048,.045*step],'origin':[0,0,0],'order':'x-fastest'},
              'neuroglancerSource':BASE+'/neuron_meshes',
              'sources':{'annotations':'banc_888_meta.feather','skeletons':[SWC,BASE+'/neuron_skeletons/swcs-from-pcg-skel'],'surface':BASE+'/region_outlines','em':EM},
              'notes':['All simulated neurons and connections are retained. Display skeletons are sampled.',
                       'The EM overview averages 4x4 xy pixels at mip 6 and samples z planes; it does not resolve synapses.'],
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in OUT.iterdir() if p.suffix in ('.bin','.json') and p.name!='metadata.json'}}
    (OUT/'metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('mappedNeurons','skeletonCount','skeletonVertices','volume')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--em-planes',type=int,default=64);args=parser.parse_args()
    main(args.em_planes)
