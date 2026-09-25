"""Independent numerical checks. Usage: python verify.py --out CHECKS.json."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, BSpline
from spline_core import (coupled,averaged,natural,parameters,robin_reference,
                         endpoint_approximation,mapping,energy,refine,validate)
from generate_data import case,truth_curve


def run_checks():
    results={}
    def record(name,value,tol):
        value=float(value)
        results[name]={'value':value,'tolerance':tol,'passed':bool(value<=tol)}
        if value>tol:raise AssertionError(f'{name}: {value} > {tol}')
    rng=np.random.default_rng(20260925);grid=np.linspace(0,1,1001)
    periodic_errors=[];robin_errors=[];boundary_errors=[];residuals=[];basis_errors=[]
    for n in (5,8,16):
        for alpha in (0,.5,1):
            for seed in range(4):
                c=case('flower',n,'jittered',seed);d=np.array(c['points'])
                f=coupled(d,alpha,True)
                g=CubicSpline(f.t,d,bc_type='periodic')
                periodic_errors.append(np.max(np.abs(f(grid)-g(grid))))
                residuals.append(np.max(np.abs(f(f.t)-d)))
                for order in (0,1,2):
                    denom=max(1,np.max(np.abs(f(grid,order))))
                    boundary_errors.append(np.max(np.abs(f(0,order)-f(1,order)))/denom)
                c=case('sine',n,'jittered',seed);d=np.array(c['points'])
                f=coupled(d,alpha)
                r=2/(f.t[0]-f.knots[2]);s=2/(f.knots[-3]-f.t[-1])
                g=robin_reference(d,f.t,r,s)
                robin_errors.append(np.max(np.abs(f(grid)-g(grid))))
                for gfit in (f,averaged(d,alpha),natural(d,alpha)):
                    residuals.append(np.max(np.abs(gfit(gfit.t)-d)))
                b=BSpline(f.knots,np.eye(len(f.controls)),3)
                basis_errors.append(np.max(np.abs(b(grid).sum(axis=1)-1)))
    record('periodic_vs_independent_power_basis',max(periodic_errors),2e-10)
    record('repeated_vs_independent_robin_system',max(robin_errors),2e-10)
    record('interpolation_residual',max(residuals),1e-10)
    record('periodic_derivatives_0_1_2',max(boundary_errors),1e-9)
    record('partition_of_unity',max(basis_errors),2e-14)
    # Endpoint identities and natural reference use distinct equations.
    d=np.array(case('sine',16,'clustered',4)['points']);f=coupled(d)
    err=[]
    for t,r in [(0,2/(0-f.knots[2])),(1,-2/(f.knots[-3]-1))]:
        err.append(np.linalg.norm(f(t,2)-r*f(t,1))/max(1,np.linalg.norm(f(t,2))))
    record('endpoint_robin_identity',max(err),1e-10)
    g=natural(d);record('natural_vs_scipy',np.max(np.abs(g(grid)-CubicSpline(g.t,d,bc_type='natural')(grid))),1e-10)
    # Change the outer four knots while keeping the nearest exterior knots.
    u=f.knots.copy();u[0]-=.3;u[1]-=.05;u[-2]+=.09;u[-1]+=.4
    altered=coupled(d,sites=f.t,frozen_exterior=u)
    record('irrelevance_of_four_outer_exterior_knots',np.max(np.abs(f(grid)-altered(grid))),1e-10)
    reflected=coupled(d,exterior='reflected')
    record('constant_vs_reflected_geometry',np.max(np.abs(f(grid)-reflected(grid))),1e-10)
    # Symmetries of the full algorithm, rather than just fixed basis functions.
    theta=.713;q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    shifted=2.7*d@q.T+np.array([3,-4]);g=coupled(shifted)
    record('similarity_equivariance',np.max(np.abs(g(grid)-(2.7*f(grid)@q.T+[3,-4]))),1e-10)
    g=coupled(d[::-1]);record('open_reversal',np.max(np.abs(g(grid)-f(1-grid))),1e-10)
    dclosed=np.array(case('flower',16,'jittered',1)['points'])
    fclosed=coupled(dclosed,closed=True)
    shift=5;rolled=np.roll(dclosed[:-1],-shift,axis=0);rolled=np.vstack([rolled,rolled[0]])
    greindexed=coupled(rolled,closed=True)
    record('periodic_cyclic_reindexing',np.max(np.abs(greindexed(grid)-fclosed((grid+fclosed.t[shift])%1))),1e-10)
    # Endpoint-support closure: geometric G2 does not imply parametric C2.
    d=np.array(case('ellipse',12,'jittered',7)['points'])
    for support in ('geometric','speed_matched'):
        g=endpoint_approximation(d,support=support)
        v=g([0,1],1);a=g([0,1],2)
        cross=v[:,0]*a[:,1]-v[:,1]*a[:,0]
        record(f'{support}_endpoint_curvature',np.max(np.abs(cross/np.linalg.norm(v,axis=1)**3)),1e-9)
        record(f'{support}_positive_tangent_alignment',1-np.dot(v[0],v[1])/np.prod(np.linalg.norm(v,axis=1)),1e-12)
        if support=='speed_matched':record('speed_matched_C1',np.linalg.norm(v[0]-v[1]),1e-10)
    # Finite differences only away from breakpoints.
    x=np.array([.113,.319,.731]);eps=1e-6
    record('analytic_derivative_vs_difference',np.max(np.abs((f(x+eps)-f(x-eps))/(2*eps)-f(x,1))),1e-6)
    bad=0
    for data,closed in [(np.zeros((4,2)),False),(np.array([[0,0],[1,1],[1,1],[2,2]]),False),
                        (np.array([[0,0],[1,0],[1,1],[0,1]]),True)]:
        try:validate(data,closed)
        except ValueError:bad+=1
    record('invalid_inputs_not_rejected',3-bad,0)
    # The strict displacement bound is a mathematical invariant for arbitrary signs.
    h=np.diff(f.t);bounds=.25*np.minimum(h[:-1],h[1:]);margin=1
    for _ in range(100):
        z=f.t.copy();z[1:-1]+=rng.uniform(-1,1,len(bounds))*bounds
        margin=min(margin,np.min(np.diff(z)/h))
    record('minimum_relative_gap_shortfall',max(0,.5-margin),1e-14)
    result={'checks':results,'count':len(results),'all_passed':True,'python':platform.python_version()}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out')
    args=p.parse_args();result=run_checks()
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
