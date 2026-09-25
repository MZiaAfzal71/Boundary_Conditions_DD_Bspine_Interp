"""Run the article benchmark. Usage: python run_experiments.py --out OUTPUT.

Use --seeds 10 for the delivered study. Uniform sampling is run once per
family/size. Every failure is retained in results; no successful-case filtering.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from spline_core import (coupled,averaged,natural,metrics,endpoint_approximation,
                         seam_metrics,refine,Fit,energy,sampled_curve)
from generate_data import generate,truth_curve,case


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_fit(path,f):
    np.savez_compressed(path,knots=f.knots,controls=f.controls,t=f.t,
                        matrix=f.matrix,metadata=json.dumps(f.metadata))


def load_fit(path):
    from scipy.interpolate import BSpline
    with np.load(path,allow_pickle=False) as z:
        return Fit(BSpline(z['knots'],z['controls'],3,extrapolate=False),z['t'],z['knots'],
                   z['matrix'],z['controls'],json.loads(str(z['metadata'])))


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def run(out,seeds=10,pilot=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'fits').mkdir(exist_ok=True)
    start=time.perf_counter();cases=generate(out/'data',seeds)
    rows=[]
    for idx,c in enumerate(cases):
        d=np.array(c['points']);truth=truth_curve(c['family'],np.linspace(0,1,4097))
        for alpha in (0.,.5,1.):
            methods=('periodic',) if c['closed'] else ('averaged','repeated','natural')
            for name in methods:
                row={k:c[k] for k in ('id','family','n','regime','seed','closed')}
                row.update(method=name,alpha=alpha,status='ok')
                try:
                    tick=time.perf_counter()
                    if name=='averaged':f=averaged(d,alpha)
                    elif name=='natural':f=natural(d,alpha)
                    else:f=coupled(d,alpha,c['closed'])
                    row['fit_seconds']=time.perf_counter()-tick
                    row.update(metrics(f,d,truth))
                    if not np.isfinite(row['bending']):
                        row['bending']='';row['status']='nonregular_energy'
                    save_fit(out/'fits'/f"{c['id']}_{name}_a{alpha:g}.npz",f)
                except (ValueError,np.linalg.LinAlgError) as exc:
                    row.update(status='failed',message=str(exc))
                rows.append(row)
        if (idx+1)%30==0:print(f'Benchmark {idx+1}/{len(cases)} cases',flush=True)
    csv_write(out/'benchmark.csv',rows)
    # Full derivative diagnostics across boundary multipliers.
    boundary=[]
    for family in ('sine','spiral'):
        c=case(family,16,'clustered',0);d=np.array(c['points'])
        truth=truth_curve(family,np.linspace(0,1,4097))
        for factor in (.25,.5,1.,2.,4.,16.,64.):
            f=coupled(d,exterior_scale=factor)
            row={'family':family,'exterior_scale':factor,**metrics(f,d,truth)}
            boundary.append(row);save_fit(out/'fits'/f'boundary_{family}_{factor:g}.npz',f)
    csv_write(out/'boundary.csv',boundary)
    # All 12 seam origins; same elliptic polygon, not independent datasets.
    seams=[];origin_fits={};origin_sites=None
    d=np.array(case('ellipse',12,'jittered',7)['points'])[:-1]
    for shift in range(len(d)):
        x=np.roll(d,-shift,axis=0);x=np.vstack([x,x[0]])
        scale=np.linalg.norm(np.ptp(x,axis=0))
        for mode in ('direct','geometric','speed_matched','periodic'):
            if mode=='periodic':f=coupled(x,closed=True)
            else:f=endpoint_approximation(x,support=None if mode=='direct' else mode)
            if shift==0:origin_fits[mode]=f;origin_sites=f.t
            grid=np.linspace(0,1,1001)
            aligned=origin_fits[mode]((grid+origin_sites[shift])%1)
            row={'shift':shift,'method':mode,**seam_metrics(f,scale),'bending':energy(f,scale,96)}
            row['aligned_origin_change']=float(np.max(np.linalg.norm(f(grid)-aligned,axis=1))/scale)
            seams.append(row);save_fit(out/'fits'/f'seam_{shift}_{mode}.npz',f)
    csv_write(out/'seams.csv',seams)
    # Pilot: the fixed 24 seed-zero cases at n=8,16. No tuning on reference error.
    pilot_rows=[]
    if pilot:
        selected=[c for c in cases if c['seed']==0 and c['n'] in (8,16)]
        for i,c in enumerate(selected):
            d=np.array(c['points']);truth=truth_curve(c['family'],np.linspace(0,1,4097))
            base=coupled(d,closed=c['closed']);tick=time.perf_counter()
            f,info=refine(d,closed=c['closed'])
            mb,mf=metrics(base,d,truth),metrics(f,d,truth)
            row={'id':c['id'],'closed':c['closed'],'family':c['family'],
                 'seconds':time.perf_counter()-tick,**info}
            row['displacement']=json.dumps(info.get('displacement',[]))
            for k in ('bending','hausdorff_sampled','condition','residual','quadrature_relative'):
                row['base_'+k],row['refined_'+k]=mb[k],mf[k]
            pilot_rows.append(row);save_fit(out/'fits'/f"pilot_{c['id']}.npz",f)
            print(f'Refinement pilot {i+1}/{len(selected)}',flush=True)
        csv_write(out/'refinement.csv',pilot_rows)
    # Resolution audit covers every family/regime at two sizes, seed zero.
    audit=[]
    for c in cases:
        if c['seed']!=0 or c['n'] not in (8,32):continue
        d=np.array(c['points']);f=coupled(d,closed=c['closed'])
        coarse=metrics(f,d,truth_curve(c['family'],np.linspace(0,1,4097)))
        fine=metrics(f,d,truth_curve(c['family'],np.linspace(0,1,8193)),4097)
        audit.append({'id':c['id'],'coarse':coarse['hausdorff_sampled'],
                      'fine':fine['hausdorff_sampled'],
                      'absolute_change':abs(fine['hausdorff_sampled']-coarse['hausdorff_sampled'])})
    csv_write(out/'resolution_audit.csv',audit)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    metadata={'study_date':'2026-09-25','case_count':len(cases),'fit_count':len(rows),
              'seeds':seeds,'sizes':[8,16,32],'pilot_count':len(pilot_rows),
              'alphas':[0,.5,1],'runtime_seconds':time.perf_counter()-start,
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'platform':platform.platform(),'machine':platform.machine(),'script_sha256':hashes,
              'interpretation':'Exploratory synthetic boundary study. Not preregistered; not a real-data validation.',
              'distance_metric':'Hausdorff distance between finite parameter-sampled point sets; not certified continuous distance.',
              'license_note':'Synthetic formulas and provisional legacy coordinates are generated locally; confirm author rights before public release.'}
    json_write(out/'provenance.json',metadata)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,default=10);p.add_argument('--skip-refinement',action='store_true')
    args=p.parse_args();print(json.dumps(run(args.out,args.seeds,not args.skip_refinement),indent=2))
