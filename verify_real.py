"""Independent checks for source selection and real-data distance calculations."""
import json
import numpy as np
from download_real_data import clean,project,select_indices,external
from run_real_experiments import PolylineDistance,controlled_polyline,reconstruction
from spline_core import coupled


def run_checks():
    passed=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        passed.append(name)
    rng=np.random.default_rng(7401)
    vertices=rng.normal(size=(43,2));queries=rng.normal(size=(101,2))
    # Independent exhaustive point/segment matrix.
    delta=queries[:,None,:]-vertices[None,:-1,:];v=np.diff(vertices,axis=0)
    t=np.clip(np.sum(delta*v,axis=2)/np.sum(v*v,axis=1),0,1)
    brute=np.linalg.norm(delta-t[:,:,None]*v,axis=2).min(axis=1)
    check('segment distances equal exhaustive search',np.max(abs(brute-PolylineDistance(vertices)(queries)))<1e-13)
    check('long segments remain candidates',PolylineDistance(np.array([[-100.,0.],[100.,0.],[0.,1.],[.01,1.]]))(np.array([[0.,.02]]))[0]<.02000000001)
    line=np.column_stack((np.linspace(0,1,81)**3,np.linspace(0,1,81)**2))
    for regime in ('index','arclength'):
        ids=select_indices(line,32,regime,False)
        check(f'{regime} distinct retained endpoints',len(ids)==32 and ids[0]==0 and ids[-1]==80 and np.all(np.diff(ids)>0))
    closed=np.column_stack((np.cos(np.linspace(0,2*np.pi,121)),np.sin(np.linspace(0,2*np.pi,121))));closed[-1]=closed[0]
    ids=select_indices(closed,32,'arclength',True)
    check('closed selection keeps 32 unique anchors and one closure',len(ids)==33 and ids[0]==0 and ids[-1]==120)
    result=reconstruction(closed,closed,ids,True)
    check('closure duplicate excluded from held-out count',result['heldout_n']==88 and result['heldout_max']<1e-14)
    points=np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.],[2.,0.]])
    p,idx=clean(points)
    check('cleaning preserves original source indices',np.array_equal(idx,[0,2,4]) and len(p)==3)
    check('projection maps its centre to zero',np.max(abs(project(np.array([[80.7,7.8]]),(80.7,7.8))))<1e-10)
    f=coupled(line[select_indices(line,16,'index',False)])
    scale=np.linalg.norm(np.ptp(line,axis=0));t,p,bound=controlled_polyline(f,scale)
    # Evaluate dense interior fractions against same-parameter chord points;
    # the point-to-polyline distance can only be smaller.
    s=np.linspace(0,1,11)
    fine=(t[:-1,None]+np.diff(t)[:,None]*s).ravel()
    chords=(p[:-1,None,:]*(1-s[None,:,None])+p[1:,None,:]*s[None,:,None]).reshape(-1,2)
    actual=float(np.linalg.norm(f(fine)-chords,axis=1).max()/scale)
    check('cubic chord error respects derivative bound',actual<=bound*(1+1e-8) and bound<=2e-6)
    check('distance metric similarity invariance',np.max(abs(PolylineDistance(vertices*7+8)(queries*7+8)/7-brute))<1e-13)
    return {'count':len(passed),'checks':passed,'maximum_distance_error':float(np.max(abs(brute-PolylineDistance(vertices)(queries)))),
            'chord_actual_error':actual,'chord_error_bound':bound}


if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
