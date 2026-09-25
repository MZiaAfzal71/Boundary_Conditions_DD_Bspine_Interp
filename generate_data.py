"""Generate all study input data. Usage: python generate_data.py --out OUTPUT.

No downloads, images, or LaTeX inputs are required. Coordinates transcribed
from the supplied manuscript are explicitly provisional historical examples.
"""
import argparse
import json
from pathlib import Path
import numpy as np

FAMILIES=('sine','spiral','ellipse','flower')
CLOSED={'sine':False,'spiral':False,'ellipse':True,'flower':True}


def truth_curve(name,t):
    t=np.asarray(t)
    if name=='sine':return np.c_[t,.25*np.sin(2*np.pi*t)+.08*np.sin(6*np.pi*t)]
    if name=='spiral':
        a=1.6*np.pi*t;r=.3+.7*t
        return np.c_[r*np.cos(a),r*np.sin(a)]
    a=2*np.pi*t
    if name=='ellipse':return np.c_[1.5*np.cos(a),np.sin(a)]
    if name=='flower':
        r=1+.25*np.cos(5*a)
        return np.c_[r*np.cos(a),r*np.sin(a)]
    raise ValueError(name)


def case(name,n,regime,seed):
    closed=CLOSED[name];q=n if closed else n-1
    rng=np.random.default_rng(seed)
    if regime=='uniform':h=np.ones(q)
    elif regime=='jittered':h=.25+np.exp(.9*rng.normal(size=q))
    elif regime=='clustered':
        phase=rng.uniform(0,2*np.pi)
        h=.15+np.exp(1.5*np.cos(2*np.pi*np.arange(q)/q+phase)+.25*rng.normal(size=q))
    else:raise ValueError(regime)
    tau=np.r_[0,np.cumsum(h)/h.sum()]
    tau[-1]=1.0
    d=truth_curve(name,tau)
    if closed:d[-1]=d[0]
    return {'id':f'{name}_n{n}_{regime}_s{seed}','family':name,'n':n,
            'regime':regime,'seed':seed,'closed':closed,'tau':tau.tolist(),'points':d.tolist()}


def legacy_cases():
    closed=[[3.2667,3.4956],[3.4315,5.2091],[-1.1818,5.6046],[-2.0057,2.6718],
            [.0703,-3.2597],[-3.192,-3.4903],[-3.2249,-5.138],[1.4214,-5.3027],
            [2.3441,-2.8972],[.0703,3.4956],[3.2667,3.4956]]
    a=[[0,0],[.43,1.06],[1.22,-1.25],[2.64,1.85],[3.53,1.82],[4.35,-2.57],
       [5.34,-2.53],[7.22,3.66],[8.21,3.69],[9.63,-4.12],[11.08,-3.98]]
    b=[[0,0],[.43,1.06],[1.35,3.27],[3.25,3.27],[6,4],[6.67,.57],
       [8.54,2.01],[9.58,2.76],[11.2,3.48],[11.96,1.18],[14.58,.07]]
    c=b[:7]+[[9.73,3.23],[11.2,3.48],[14,2],[14.51,.35]]
    return [{'id':name,'closed':cl,'points':d,'status':'provisional_pdf_transcription',
             'source':'supplied main.pdf; printed coordinates, not digitized curve traces',
             'limitations':'original exterior knots, solver and manual displacements unavailable'}
            for name,cl,d in [('legacy_closed',True,closed),('legacy_open_a',False,a),
                              ('legacy_open_b',False,b),('legacy_open_c',False,c)]]


def generate(out,seeds=10,sizes=(8,16,32)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=[case(f,n,r,s) for f in FAMILIES for n in sizes
          for r in ('uniform','jittered','clustered') for s in range(1 if r=='uniform' else seeds)]
    (out/'cases.json').write_text(json.dumps(data,indent=2)+'\n')
    (out/'legacy.json').write_text(json.dumps(legacy_cases(),indent=2)+'\n')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=10)
    a=p.parse_args();print(f'Generated {len(generate(a.out,a.seeds))} synthetic cases.')
