"""Download and prepare the article's public real-data cases (no manuscript I/O).

python download_real_data.py --out ../study_output/real --cache ../source_cache
Use --offline with a cache previously filled by this script. Source SHA-256
checks are mandatory, including in offline mode. No data live in the scripts
directory. UCI data: Ben Williams, DOI 10.24432/C58G7V, CC BY 4.0.
Coastline: Natural Earth contributors, public domain. See SOURCES.json output.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile
import numpy as np
from scipy.io import loadmat
import shapefile
from matplotlib.path import Path as PolygonPath

SOURCES = {
    'handwriting.zip': {
        'url': 'https://archive.ics.uci.edu/static/public/175/character%2Btrajectories.zip',
        'sha256': '5d2db017ef0d8cf0e65ed060c9e90399f78eb9f1e3cb63e22ca8c3ef4ba67d52',
        'citation': 'Williams, B. (2006). Character Trajectories. UCI Machine Learning Repository. DOI: 10.24432/C58G7V.',
        'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'landing_page': 'https://archive.ics.uci.edu/dataset/175/character+trajectories',
    },
    'coastline.zip': {
        'url': 'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip',
        'sha256': 'bfa04cdbcbef07ef90dfca1dabb48062eca29900a113df0f389303e255484017',
        'citation': 'Made with Natural Earth. 1:10 million coastline. Natural Earth contributors.',
        'license': 'Public domain',
        'license_url': 'https://www.naturalearthdata.com/about/terms-of-use/',
        'landing_page': 'https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/',
        'archive_version': '5.0.0-pre9',
    },
}
ISLANDS = {'Sri_Lanka': (80.7, 7.8), 'Madagascar': (46.5, -19.),
           'Cuba': (-79.5, 22.), 'Iceland': (-19., 65.),
           'Taiwan': (121., 23.7), 'Tasmania': (146.7, -42.)}


def external(path):
    p = Path(path).resolve(); source = Path(__file__).parent.resolve()
    if p == source or source in p.parents:
        raise ValueError('Data/output/cache directories must be outside the scripts directory.')
    return p


def fetch(cache, offline=False):
    cache = external(cache); cache.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        target = cache/name
        if not target.exists():
            if offline: raise FileNotFoundError(f'Offline cache missing {target}')
            temp = target.with_suffix('.part')
            print(f'Downloading {name} from its original public provider', flush=True)
            try:
                with urlopen(spec['url'], timeout=120) as src, temp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                if hashlib.sha256(temp.read_bytes()).hexdigest() != spec['sha256']:
                    raise ValueError(f'{name}: source changed or download incomplete; SHA-256 mismatch.')
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'{target}: SHA-256 mismatch; do not silently substitute another release.')
    return cache


def clean(points):
    """Remove consecutive points closer than 1e-12 times the bbox diagonal."""
    points = np.asarray(points, dtype=float)
    scale = np.linalg.norm(np.ptp(points, axis=0))
    if not np.isfinite(points).all() or scale <= 0: raise ValueError('Invalid source curve')
    kept = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i]-points[kept[-1]]) > 1e-12*scale: kept.append(i)
    return points[kept], np.array(kept)


def project(lonlat, centre):
    """Spherical Lambert azimuthal equal-area coordinates, radius 6371 km.

    The map is a documented working geometry, not geodesic interpolation.
    Fixed centres are recorded for every island.
    """
    lon, lat = np.deg2rad(np.asarray(lonlat)).T
    lon0, lat0 = np.deg2rad(centre)
    d = lon-lon0
    k = np.sqrt(2/(1+np.sin(lat0)*np.sin(lat)+np.cos(lat0)*np.cos(lat)*np.cos(d)))
    return 6371*np.column_stack((k*np.cos(lat)*np.sin(d),
        k*(np.cos(lat0)*np.sin(lat)-np.sin(lat0)*np.cos(lat)*np.cos(d))))


def select_indices(points, n, regime, closed):
    """Actual retained source vertices; no interpolated pseudo-observations.

    Index mode rounds equally spaced source indices. Arc mode greedily takes
    nearest feasible source vertices to equally spaced polygonal arc targets,
    reserving one source vertex for each subsequent target. Endpoints retained.
    """
    total = len(points); count = n+1 if closed else n
    if count >= total: raise ValueError('Need withheld source vertices.')
    if regime == 'index': return np.rint(np.linspace(0,total-1,count)).astype(int)
    if regime != 'arclength': raise ValueError('Unknown sampling regime')
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    indices = [0]
    for j, goal in enumerate(np.linspace(0,arc[-1],count)[1:-1],start=1):
        lo = indices[-1]+1; hi = total-count+j
        insertion = int(np.searchsorted(arc,goal))
        choices = np.unique(np.clip([insertion-1,insertion],lo,hi))
        indices.append(int(choices[np.argmin(abs(arc[choices]-goal))]))
    return np.array(indices+[total-1])


def prepare(out, cache, offline=False):
    out=external(out); out.mkdir(parents=True,exist_ok=True)
    cache=fetch(cache,offline); (out/'curves').mkdir(exist_ok=True)
    cases=[]
    with zipfile.ZipFile(cache/'handwriting.zip') as z:
        member=next(n for n in z.namelist() if n.endswith('.mat'))
        mat=loadmat(BytesIO(z.read(member)),simplify_cells=True)
    consts=mat['consts']; labels=np.asarray(consts['charlabels']).ravel()
    key=np.asarray(consts['key']).ravel()
    factors=np.asarray(consts['datanorm']).ravel()[:2]
    counts={int(k):0 for k in np.unique(labels)}
    rejected=[]
    for i,(v,label) in enumerate(zip(mat['mixout'],labels)):
        label=int(label)
        if counts[label]>=2: continue
        # Undo the supplied per-channel scaling, integrate the first two
        # velocity channels, and translate to start at the origin. The common
        # time-step factor cancels in scale-normalized shape comparisons.
        positions=np.cumsum(np.asarray(v)[:2].T*factors,axis=0)
        positions-=positions[0]
        points,indices=clean(positions)
        if len(points)<65:
            rejected.append({'source_index_1based':i+1,'reason':'fewer than 65 cleaned vertices'})
            continue
        counts[label]+=1
        char=str(key[label-1])
        ident=f'pen_{label:02d}_{counts[label]:02d}'
        np.savez_compressed(out/'curves'/f'{ident}.npz',points=points,source_indices=indices)
        cases.append({'id':ident,'domain':'handwriting','closed':False,'label':char,
                      'source_index_1based':i+1,'source_vertices':len(positions),
                      'vertices':len(points),'removed_vertices':len(positions)-len(points)})
    if len(cases)!=40: raise ValueError(f'Expected 40 selected handwriting traces; got {len(cases)}')
    with zipfile.ZipFile(cache/'coastline.zip') as z:
        reader=shapefile.Reader(**{ext:BytesIO(z.read(f'ne_10m_coastline.{ext}')) for ext in ('shp','shx','dbf')})
        shapes=reader.shapes()
        version=z.read('ne_10m_coastline.VERSION.txt').decode().strip()
        if version!=SOURCES['coastline.zip']['archive_version']: raise ValueError('Unexpected coastline release')
        for name,centre in ISLANDS.items():
            matches=[]
            for i,s in enumerate(shapes):
                x=np.asarray(s.points)
                if len(s.parts)==1 and np.array_equal(x[0],x[-1]) and PolygonPath(x).contains_point(centre):
                    matches.append((i,x))
            if len(matches)!=1: raise ValueError(f'Expected one enclosing closed ring for {name}')
            feature,lonlat=matches[0]; points,indices=clean(project(lonlat,centre))
            points[-1]=points[0]
            ident='coast_'+name
            np.savez_compressed(out/'curves'/f'{ident}.npz',points=points,source_indices=indices,lonlat=lonlat[indices])
            cases.append({'id':ident,'domain':'coastline','closed':True,'label':name.replace('_',' '),
                          'feature_index_0based':feature,'projection_centre_lonlat':centre,
                          'source_vertices':len(lonlat),'vertices':len(points),
                          'removed_vertices':len(lonlat)-len(points)})
    sources={'retrieval_date':'2026-09-25','sources':SOURCES,
             'handwriting_xy_denormalization_factors':factors.tolist(),
             'selection':'First two eligible traces per source label in original file order; six fixed named islands.',
             'rejected_before_selection':rejected,
             'modifications':'Velocity integration and per-channel denormalization for handwriting; spherical equal-area projection for coastlines; consecutive near-duplicate removal; no extra smoothing.',
             'scope':'One handwriting author; generalized 1:10 million cartographic contours. References are source polylines, not physical ground truth.'}
    (out/'SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    return cases


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--offline',action='store_true')
    a=p.parse_args();print(json.dumps(prepare(a.out,a.cache,a.offline),indent=2))
