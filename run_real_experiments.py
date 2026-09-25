"""Real-source reconstruction and seam experiments.

python run_real_experiments.py --out ../study_output/real --cache ../source_cache
--offline requires the two original archives in the cache. Every fit uses
retained source vertices; withheld points are used only for evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.spatial import cKDTree
from download_real_data import prepare,select_indices,external
from spline_core import coupled,averaged,natural,bending_report,seam_metrics,endpoint_approximation
from run_experiments import save_fit,csv_write,json_write


class PolylineDistance:
    """Nearest distance to a finite union of line segments.

    A nearest-vertex distance is an upper bound U. Any better segment has
    midpoint distance <= U + half its length. Query all midpoints within
    U + the longest half-length, then compute exact segment projections.
    This bound avoids an unverified nearest-k candidate shortcut.
    """
    def __init__(self,points):
        self.points=np.asarray(points);self.a=self.points[:-1];self.v=np.diff(self.points,axis=0)
        self.l2=np.einsum('ij,ij->i',self.v,self.v)
        if np.min(self.l2)<=0:raise ValueError('Polyline has zero-length segments')
        self.tree=cKDTree(self.a+self.v/2);self.vertices=cKDTree(self.points)
        self.halfmax=float(np.sqrt(self.l2.max())/2)

    def __call__(self,queries):
        q=np.asarray(queries);upper=self.vertices.query(q)[0]
        candidates=self.tree.query_ball_point(q,(upper+self.halfmax)*(1+1e-12)+1e-14)
        lengths=np.array([len(c) for c in candidates]);owners=np.repeat(np.arange(len(q)),lengths)
        ids=np.concatenate(candidates).astype(int)
        delta=q[owners]-self.a[ids]
        t=np.clip(np.einsum('ij,ij->i',delta,self.v[ids])/self.l2[ids],0,1)
        residual=delta-t[:,None]*self.v[ids]
        d2=np.einsum('ij,ij->i',residual,residual)
        result=np.full(len(q),np.inf);np.minimum.at(result,owners,d2)
        return np.sqrt(result)


def controlled_polyline(f,scale,tolerance=2e-6):
    """Cubic chord approximation with a second-derivative error bound.

    On each cubic span, max ||C''|| is attained at an endpoint because C''
    is affine. Chord error <= max||C''|| * delta_t**2 / 8, including for
    vector curves. Return vertices and the maximum theoretical chord bound/S.
    This controls source-point distances, not a two-sided Hausdorff estimate.
    """
    knots=np.unique(f.knots[(f.knots>=0)&(f.knots<=1)])
    grids=[];bounds=[]
    for a,b in zip(knots[:-1],knots[1:]):
        acc=float(np.linalg.norm(f(np.array([a,b]),2),axis=1).max())
        count=max(1,int(np.ceil((b-a)*np.sqrt(acc/(8*tolerance*scale)))))
        if count>200000:raise ValueError('Excessive subdivision; inspect geometry')
        grids.append(np.linspace(a,b,count+1)[:-1]);bounds.append(acc*((b-a)/count)**2/(8*scale))
    t=np.r_[np.concatenate(grids),1.]
    return t,f(t),float(max(bounds))


def reconstruction(polyline,source,indices,closed,source_distance=None):
    scale=np.linalg.norm(np.ptp(source,axis=0));unique_count=len(source)-int(closed)
    mask=np.ones(unique_count,dtype=bool);mask[indices[indices<unique_count]]=False
    if not mask.any():raise ValueError('No held-out source vertices')
    distances=PolylineDistance(polyline)(source[:unique_count])/scale
    held=distances[mask]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(source,axis=0),axis=1))]
    fraction=arc[:unique_count]/arc[-1]
    boundary=mask&((fraction<=.1)|(fraction>=.9))
    reverse=(source_distance or PolylineDistance(source))(polyline)/scale
    return {'heldout_n':int(mask.sum()),'heldout_p95':float(np.quantile(held,.95)),
            'heldout_rms':float(np.sqrt(np.mean(held**2))),'heldout_max':float(held.max()),
            'endpoint_n':int(boundary.sum()) if not closed else 0,
            'endpoint_p95':float(np.quantile(distances[boundary],.95)) if not closed and boundary.any() else '',
            'source_vertex_symmetric':float(max(distances.max(),reverse.max()))}


def run(out,cache,offline=False):
    out=external(out);start=time.perf_counter();cases=prepare(out,cache,offline)
    (out/'fits').mkdir(exist_ok=True);(out/'anchors').mkdir(exist_ok=True)
    rows=[];audit=[]
    for i,c in enumerate(cases):
        source=np.load(out/'curves'/f"{c['id']}.npz")['points'];scale=np.linalg.norm(np.ptp(source,axis=0))
        source_distance=PolylineDistance(source)
        for n in (8,16,32):
            for regime in ('index','arclength'):
                indices=select_indices(source,n,regime,c['closed']);d=source[indices]
                case_id=f"{c['id']}_n{n}_{regime}"
                np.savez_compressed(out/'anchors'/f'{case_id}.npz',points=d,indices=indices)
                modes=[('periodic',a) for a in (0.,.5,1.)] if c['closed'] else [(m,a) for a in (0.,.5,1.) for m in ('averaged','repeated','natural')]
                modes.append(('linear',1.))
                for method,alpha in modes:
                    row={'id':case_id,'curve_id':c['id'],'domain':c['domain'],'label':c['label'],
                         'n':n,'regime':regime,'method':method,'alpha':alpha,'status':'ok'}
                    try:
                        if method=='linear':
                            poly=d;row.update(chord_bound=0.,residual=0.,bending='',integrator='not_applicable',quadrature_relative='',min_speed='')
                        else:
                            if method=='averaged':f=averaged(d,alpha)
                            elif method=='natural':f=natural(d,alpha)
                            else:f=coupled(d,alpha,c['closed'])
                            _,poly,bound=controlled_polyline(f,scale)
                            e,qerr,gdiff,speed,integrator=bending_report(f,scale)
                            row.update(chord_bound=bound,residual=float(np.linalg.norm(f(f.t)-d,axis=1).max()/scale),
                                       bending=e if np.isfinite(e) else '',quadrature_relative=qerr,
                                       gauss_disagreement=gdiff,min_speed=speed,integrator=integrator)
                            if not np.isfinite(e):row['status']='nonregular_energy'
                            elif integrator=='adaptive_warning':row['status']='energy_warning'
                            save_fit(out/'fits'/f'{case_id}_{method}_a{alpha:g}.npz',f)
                        row.update(reconstruction(poly,source,indices,c['closed'],source_distance))
                        row['distance_polyline_vertices']=len(poly)
                        # Fixed audit subset: first trace of each class, all six
                        # islands, arclength anchors at n=16, coupled alpha=.5.
                        if n==16 and regime=='arclength' and alpha==.5 and method in ('repeated','periodic') and (c['closed'] or c['id'].endswith('_01')):
                            _,fine,fine_bound=controlled_polyline(f,scale,5e-7)
                            check=reconstruction(fine,source,indices,c['closed'],source_distance)
                            audit.append({'id':case_id,'coarse_p95':row['heldout_p95'],'fine_p95':check['heldout_p95'],
                                          'p95_change':abs(row['heldout_p95']-check['heldout_p95']),
                                          'max_change':abs(row['heldout_max']-check['heldout_max']),
                                          'coarse_bound':row['chord_bound'],'fine_bound':fine_bound})
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        row.update(status='failed',message=str(exc))
                    rows.append(row)
        print(f"Real-data benchmark {i+1}/{len(cases)}: {c['id']}",flush=True)
    csv_write(out/'benchmark.csv',rows);csv_write(out/'resolution_audit.csv',audit)
    seams=[]
    for c in cases:
        if not c['closed']:continue
        source=np.load(out/'curves'/f"{c['id']}.npz")['points'];scale=np.linalg.norm(np.ptp(source,axis=0))
        indices=select_indices(source,32,'arclength',True);d=source[indices][:-1]
        origin={};sites=None;grid=np.linspace(0,1,2049)
        for shift in (0,8,16,24):
            x=np.roll(d,-shift,axis=0);x=np.vstack([x,x[0]])
            for mode in ('direct','geometric','speed_matched','periodic'):
                f=coupled(x,.5,True) if mode=='periodic' else endpoint_approximation(x,.5,None if mode=='direct' else mode)
                if shift==0:origin[mode]=f;sites=f.t
                aligned=origin[mode]((grid+sites[shift])%1)
                row={'curve_id':c['id'],'label':c['label'],'shift':shift,'method':mode,**seam_metrics(f,scale),
                     'aligned_origin_change':float(np.linalg.norm(f(grid)-aligned,axis=1).max()/scale),
                     'retained_vertex_residual':float(np.linalg.norm(f(f.t)-x,axis=1).max()/scale)}
                seams.append(row);save_fit(out/'fits'/f"seam_{c['id']}_{shift}_{mode}.npz",f)
    csv_write(out/'seams.csv',seams)
    meta={'study_date':'2026-09-25','curve_count':len(cases),'handwriting_traces':40,'islands':6,
          'anchor_sets':len(cases)*6,'benchmark_rows':len(rows),'cubic_fits':sum(r['method']!='linear' for r in rows),
          'linear_fits':sum(r['method']=='linear' for r in rows),'seam_experiments':len(seams),'resolution_audits':len(audit),
          'selection':'First two eligible original-order traces per character class; six specified islands; no fit-outcome selection.',
          'distance_tolerance':2e-6,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'runtime_seconds':time.perf_counter()-start,
          'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
          'scope':'Exploratory reconstruction of processed source geometry. No between-writer, physical shoreline, denoising, classification, or topology guarantees.'}
    json_write(out/'provenance.json',meta)
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--offline',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.out,a.cache,a.offline),indent=2))
