"""Render all new article figures from numerical outputs, as PDF/SVG/PNG.

Usage: python generate_figures.py --results OUTPUT --out ARTICLE/figures
Historical figures remain archival material; they are not recreated as claims.
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from spline_core import coupled,averaged,natural,Fit,parameters,energy
from generate_data import legacy_cases,case,truth_curve
from run_experiments import load_fit

COLORS=['#1768AC','#E1812C','#158568','#A74D8D']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180,
                     'axes.labelsize':9,'legend.fontsize':8})


def rows(path):
    with open(path) as f:return list(csv.DictReader(f))


def save(fig,out,name):
    for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)


def draw_curve(ax,f,label,color,style='-'):
    x=f(np.linspace(0,1,1501));ax.plot(x[:,0],x[:,1],style,color=color,lw=1.4,label=label)


def generate(results,out):
    results=Path(results);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grid=np.linspace(0,1,1001)
    bench=rows(results/'benchmark.csv')
    # Representative historical point sets under the now-explicit rules.
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
    legacy=legacy_cases()
    for panel,idx in zip(ax,[0,1]):
        d=np.array(legacy[idx]['points']);panel.plot(d[:,0],d[:,1],'o:',color='.65',ms=3,lw=.7,label='Data polygon')
        panel.set_aspect('equal',adjustable='datalim');panel.set_xlabel('x');panel.set_ylabel('y')
    for alpha,color in zip((.5,1.),COLORS):draw_curve(ax[0],coupled(legacy[0]['points'],alpha,True),f'Periodic, alpha={alpha:g}',color)
    for method,label,color in [(averaged,'Averaged knots',COLORS[0]),(coupled,'Repeated controls',COLORS[1]),(natural,'Natural boundary',COLORS[2])]:
        draw_curve(ax[1],method(legacy[1]['points'],.5),label,color)
    ax[0].set_title('(a) Closed historical data');ax[1].set_title('(b) Open historical data, alpha=0.5')
    for a in ax:a.legend(loc='best',frameon=False)
    save(fig,out,'fig01_legacy')
    # Boundary parameter sweep and its endpoint-speed effect.
    fig,ax=plt.subplots(1,2,figsize=(7.1,3),layout='constrained')
    d=np.array(case('spiral',16,'clustered',0)['points'])
    truth=truth_curve('spiral',grid);ax[0].plot(*truth.T,color='.7',lw=3,label='Reference')
    for factor,color in zip((.25,1,16),COLORS):
        f=load_fit(results/'fits'/f'boundary_spiral_{factor:g}.npz')
        draw_curve(ax[0],f,f'Exterior multiplier {factor:g}',color)
        ax[1].plot(grid,np.linalg.norm(f(grid,1),axis=1),color=color,lw=1.3,label=f'{factor:g}')
    ax[0].plot(*d.T,'k.',ms=3);ax[0].set_aspect('equal',adjustable='datalim')
    ax[0].set_title('(a) Same points and parameter values');ax[0].legend(frameon=False,fontsize=7)
    ax[1].set_title('(b) Speed near the first endpoint');ax[1].set_xlim(0,.18)
    ax[1].set_xlabel('Parameter t');ax[1].set_ylabel('Speed');ax[1].legend(title='Multiplier',frameon=False)
    save(fig,out,'fig02_boundary')
    # Distribution summaries, with all cases visible as fliers.
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    groups=[[float(r['hausdorff_sampled']) for r in bench if r['method']==m and float(r['alpha'])==.5]
            for m in ('averaged','repeated','natural')]
    boxes=ax[0].boxplot(groups,patch_artist=True,tick_labels=['Averaged','Repeated','Natural'],showfliers=True)
    for b,c in zip(boxes['boxes'],COLORS):b.set_facecolor(c);b.set_alpha(.55)
    groups=[[float(r['hausdorff_sampled']) for r in bench if r['method']=='periodic' and float(r['alpha'])==a] for a in (0,.5,1)]
    boxes=ax[1].boxplot(groups,patch_artist=True,tick_labels=['Uniform','Centripetal','Chord'],showfliers=True)
    for b,c in zip(boxes['boxes'],COLORS):b.set_facecolor(c);b.set_alpha(.55)
    for a in ax:a.set_yscale('log');a.set_ylabel('Sampled Hausdorff distance / S');a.grid(axis='y',alpha=.15)
    ax[0].set_title('(a) Open curves, alpha=0.5');ax[1].set_title('(b) Periodic curves')
    save(fig,out,'fig03_benchmark')
    # Seam geometry and curvature, directly from saved fits.
    fig,ax=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
    d=np.array(case('ellipse',12,'jittered',7)['points']);scale=np.linalg.norm(np.ptp(d,axis=0))
    for mode,label,color in [('geometric','Collinear supports',COLORS[0]),('speed_matched','Speed matched',COLORS[1]),('periodic','Periodic interpolation',COLORS[2])]:
        f=load_fit(results/'fits'/f'seam_0_{mode}.npz');draw_curve(ax[0],f,label,color)
        v,a=f(grid,1),f(grid,2);speed=np.linalg.norm(v,axis=1)
        curvature=scale*(v[:,0]*a[:,1]-v[:,1]*a[:,0])/speed**3
        ax[1].plot(grid,curvature,color=color,lw=1.25,label=label)
    ax[0].plot(*d.T,'k.',ms=4);ax[0].plot(*d[0],'ko',ms=5,mfc='white');ax[0].set_aspect('equal',adjustable='datalim')
    ax[0].set_title('(a) Seam marked by an open circle');ax[0].legend(frameon=False,fontsize=7)
    ax[1].set_title('(b) Curvature exposes seam flattening');ax[1].set_xlabel('Parameter t');ax[1].set_ylabel('Signed curvature x S')
    save(fig,out,'fig04_seam')
    # Refinement trade-off, reference geometry never used by the search.
    pilot=rows(results/'refinement.csv');fig,ax=plt.subplots(figsize=(5.8,3.5),layout='constrained')
    for family,color in zip(('sine','spiral','ellipse','flower'),COLORS):
        data=[r for r in pilot if r['family']==family]
        x=[float(r['refined_bending'])/float(r['base_bending']) for r in data]
        y=[float(r['refined_hausdorff_sampled'])/float(r['base_hausdorff_sampled']) for r in data]
        ax.scatter(x,y,s=36,c=color,label=family,edgecolor='white',lw=.5)
    ax.axhline(1,color='.4',ls='--',lw=1);ax.axvline(1,color='.4',ls=':',lw=1)
    ax.set_xlabel('Bending energy after / before');ax.set_ylabel('Sampled distance after / before')
    ax.set_title('Knot refinement: fairness gains do not ensure fidelity gains');ax.legend(frameon=False,ncol=2)
    save(fig,out,'fig05_refinement')
    # One knot, two meanings of locality.
    d=np.array(case('flower',16,'jittered',0)['points']);base=coupled(d,closed=True)
    z=base.t.copy();j=8;z[j]+=.2*min(z[j]-z[j-1],z[j+1]-z[j])
    moved=coupled(d,closed=True,sites=base.t,breakpoints=z)
    fixed=BSpline(moved.knots,base.controls,3,extrapolate=False)
    fig,ax=plt.subplots(figsize=(6.6,2.6),layout='constrained')
    for value,label,color in [(fixed(grid),'Controls fixed',COLORS[0]),(moved(grid),'Controls re-solved',COLORS[1])]:
        err=np.linalg.norm(value-base(grid),axis=1)
        ax.semilogy(grid,np.maximum(err,1e-16),label=label,color=color,lw=1.4)
    ax.axvline(base.t[j],ls=':',color='.4');ax.set_xlabel('Parameter t');ax.set_ylabel('Pointwise change')
    ax.set_title('Local basis support and global interpolation response');ax.legend(frameon=False)
    save(fig,out,'fig06_sensitivity')
    # Quadrature diagnostic.
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    disagreements=np.array([float(r['gauss_disagreement']) for r in bench])
    ax[0].hist(np.log10(np.maximum(disagreements,1e-16)),bins=32,color=COLORS[0])
    ax[0].axvline(-3,color=COLORS[1],ls='--');ax[0].set_xlabel('log10 relative difference: 48 vs 96 nodes')
    ax[0].set_ylabel('Number of fits');ax[0].set_title('(a) Fixed quadrature can disagree')
    f=coupled(case('flower',8,'jittered',3)['points'],alpha=0,closed=True)
    dense=np.linspace(0,1,10001);sp=np.linalg.norm(f(dense,1),axis=1)
    ax[1].semilogy(dense,sp,color=COLORS[2]);ax[1].set_xlabel('Parameter t');ax[1].set_ylabel('Speed')
    ax[1].set_title('(b) A difficult eight-vertex flower')
    save(fig,out,'fig07_quadrature')
    return sorted(p.name for p in out.glob('*.pdf'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print('\n'.join(generate(a.results,a.out)))
