"""Generate real-data figures, tables, and scalar macros from recorded results.

python report_real_results.py --results ../study_output/real --figures ../study_output/figures --tables ../study_output/tables
No manuscript is edited or compiled.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_figures import save,COLORS
from generate_tables import table,num
from run_experiments import load_fit

LABELS={'averaged':'Averaged knots','repeated':'Repeated controls','natural':'Natural boundary',
        'periodic':'Periodic cubic','linear':'Piecewise linear'}


def generate(results,figures,tables):
    root=Path(results);figures=Path(figures);tables=Path(tables)
    figures.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    with open(root/'benchmark.csv') as f: rows=list(csv.DictReader(f))
    with open(root/'seams.csv') as f: seams=list(csv.DictReader(f))
    with open(root/'resolution_audit.csv') as f: audit=list(csv.DictReader(f))
    cases=json.loads((root/'cases.json').read_text());meta=json.loads((root/'provenance.json').read_text())
    if any(r['status']=='failed' for r in rows):raise ValueError('Report failed real-data fits explicitly before aggregation')
    def med(data,key):
        values=[float(r[key]) for r in data if r.get(key,'')!='']
        return float(np.median(values)) if values else float('nan')
    def selected(domain,method,alpha):
        return [r for r in rows if r['domain']==domain and r['method']==method and float(r['alpha'])==alpha]
    open_table=[];closed_table=[];summary={}
    for domain in ('handwriting','coastline'):
        methods=('averaged','repeated','natural') if domain=='handwriting' else ('periodic',)
        modes=[(m,a) for a in (0.,.5,1.) for m in methods]+[('linear',1.)]
        stats=[]
        for method,alpha in modes:
            group=selected(domain,method,alpha)
            e95=med(group,'heldout_p95');edge=med(group,'endpoint_p95') if domain=='handwriting' else med(group,'heldout_max')
            energy_warnings=sum(r.get('integrator')=='adaptive_warning' or r.get('bending','')=='' for r in group) if method!='linear' else 0
            bending=med(group,'bending') if method!='linear' and not energy_warnings else None
            stats.append({'method':method,'alpha':alpha,'count':len(group),'heldout_p95':e95,
                          'endpoint_p95' if domain=='handwriting' else 'heldout_max':edge,
                          'bending':bending,'unresolved_energy_count':energy_warnings})
            row=[LABELS[method],f'{alpha:g}' if method!='linear' else '--',num(e95),num(edge)]
            if domain=='handwriting':row.append(r'--$^{\dagger}$' if energy_warnings else ('--' if bending is None else num(bending)))
            (open_table if domain=='handwriting' else closed_table).append(row)
        summary[domain]=stats
    table(tables/'real_open.tex','lrrrr',['Method',r'$\alpha$',r'Median $E_{95}$',r'Median $E^{\partial}_{95}$',r'Median $\widehat E_b$'],open_table)
    table(tables/'real_closed.tex','lrrr',['Method',r'$\alpha$',r'Median $E_{95}$',r'Median $E_{\max}$'],closed_table)
    # Each trace is the aggregation unit: six paired settings -> one median.
    contrasts=[];summary['paired']=[]
    for baseline in ('averaged','natural','linear'):
        a={r['id']:r for r in selected('handwriting','repeated',.5)}
        b={r['id']:r for r in selected('handwriting',baseline,1. if baseline=='linear' else .5)}
        for metric,label in [('heldout_p95',r'$E_{95}$'),('endpoint_p95',r'$E^{\partial}_{95}$')]:
            per_curve={}
            for ident,r in a.items():
                if r[metric]=='' or b[ident][metric]=='':raise ValueError('Missing paired endpoint metric')
                per_curve.setdefault(r['curve_id'],[]).append(float(r[metric])-float(b[ident][metric]))
            values=np.array([np.median(v) for v in per_curve.values()])
            # A difference within the combined two-curve chord error budget
            # is not called a win. Linear has exact segment distances.
            threshold=2e-6 if baseline=='linear' else 4e-6
            entry={'baseline':baseline,'metric':metric,'median_difference':float(np.median(values)),
                   'wins':int(np.sum(values < -threshold)),'losses':int(np.sum(values > threshold)),
                   'within_tolerance':int(np.sum(abs(values)<=threshold)),'threshold':threshold}
            summary['paired'].append(entry)
            contrasts.append([LABELS[baseline],label,num(entry['median_difference']),
                              f"{entry['wins']}/{entry['within_tolerance']}/{entry['losses']}"])
    table(tables/'real_paired.tex','llrr',['Reference', 'Metric', 'Median difference', 'Win/tie/loss'],contrasts)
    seam_table=[];summary['seams']=[]
    for mode,label in [('direct','Endpoint only'),('geometric','Collinear supports'),('speed_matched','Speed matched'),('periodic','Periodic cubic')]:
        group=[r for r in seams if r['method']==mode]
        entry={'method':mode,'median_J1':med(group,'seam_d1'),'median_J2':med(group,'seam_d2'),
               'max_origin_change':max(float(r['aligned_origin_change']) for r in group),
               'max_endpoint_curvature':max(max(abs(float(r['curvature_left'])),abs(float(r['curvature_right']))) for r in group)}
        summary['seams'].append(entry)
        seam_table.append([label,num(entry['median_J1']),num(entry['median_J2']),num(entry['max_origin_change'])])
    table(tables/'real_seams.tex','lrrr',['Construction',r'Median $J_1$',r'Median $J_2$','Max. origin change'],seam_table)
    dataset_table=[]
    pens=[c for c in cases if c['domain']=='handwriting']
    dataset_table.append(['Pen trajectories','40',f"{min(c['vertices'] for c in pens)}--{max(c['vertices'] for c in pens)}",'CC BY 4.0'])
    for c in cases:
        if c['closed']:dataset_table.append([c['label'],'1',str(c['vertices']-1),'Public domain'])
    table(tables/'real_datasets.tex','lrrl',['Source curves','Count','Vertices per curve','Reuse terms'],dataset_table)
    summary.update(meta)
    summary.update(max_residual=max(float(r.get('residual',0)) for r in rows),
                   max_p95_resolution_change=max(float(r['p95_change']) for r in audit),
                   max_distance_resolution_change=max(float(r['max_change']) for r in audit),
                   nonregular_energy_count=sum(r['status']=='nonregular_energy' for r in rows),
                   quadrature_warning_count=sum(r.get('integrator')=='adaptive_warning' for r in rows),
                   max_reported_chord_bound=max(float(r['chord_bound']) for r in rows))
    (tables/'real_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    checks=json.loads((root/'verification.json').read_text())
    values={'RealCurves':len(cases),'RealFits':meta['cubic_fits'],'RealLinearFits':meta['linear_fits'],
            'RealAnchorSets':meta['anchor_sets'],'RealSeamFits':len(seams),'RealChecks':checks['count'],
            'RealResidual':num(summary['max_residual']),'RealDistanceChange':num(summary['max_p95_resolution_change']),
            'RealAudits':len(audit),'RealQuadratureWarnings':summary['quadrature_warning_count']}
    (tables/'real_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in values.items())+'\n')
    # Fixed representatives: first selected trajectory in each of first four
    # enumerated source classes. Selection is independent of fitting error.
    reps=[c for c in pens if c['id'].endswith('_01')][:4]
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained')
    for ax,c in zip(axes.ravel(),reps):
        source=np.load(root/'curves'/f"{c['id']}.npz")['points'];s=np.linalg.norm(np.ptp(source,axis=0))
        ident=c['id']+'_n16_arclength';d=np.load(root/'anchors'/f'{ident}.npz')['points']
        ax.plot(*(source/s).T,color='.75',lw=3,label='Source trace')
        for m,color,style in zip(('averaged','repeated','natural'),COLORS,('-','--',':')):
            f=load_fit(root/'fits'/f'{ident}_{m}_a0.5.npz');x=f(np.linspace(0,1,2001))/s
            ax.plot(*x.T,color=color,ls=style,lw=1.2,label=LABELS[m])
        ax.plot(*(d/s).T,'k.',ms=3,label='Retained vertices');ax.set_aspect('equal',adjustable='datalim')
        ax.set_title(f"Character {c['label']}, source #{c['source_index_1based']}")
        ax.set_xlabel('x / S');ax.set_ylabel('y / S')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    save(fig,figures,'fig08_real_handwriting')
    fig,axes=plt.subplots(2,3,figsize=(7.1,5.5),layout='constrained')
    for ax,c in zip(axes.ravel(),[c for c in cases if c['closed']]):
        source=np.load(root/'curves'/f"{c['id']}.npz")['points'];s=np.linalg.norm(np.ptp(source,axis=0));origin=source.mean(axis=0)
        ident=c['id']+'_n32_arclength';d=np.load(root/'anchors'/f'{ident}.npz')['points']
        f=load_fit(root/'fits'/f'{ident}_periodic_a0.5.npz');x=f(np.linspace(0,1,3001))
        ax.plot(*((source-origin)/s).T,color='.7',lw=2,label='Source coastline')
        ax.plot(*((d-origin)/s).T,color=COLORS[1],ls=':',lw=1,label='Piecewise linear')
        ax.plot(*((x-origin)/s).T,color=COLORS[0],lw=1,label='Periodic cubic')
        ax.plot(*((d-origin)/s).T,'k.',ms=2,label='Retained vertices');ax.set_aspect('equal',adjustable='datalim')
        ax.set_title(c['label']);ax.set_xlabel('x / S');ax.set_ylabel('y / S')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
    save(fig,figures,'fig09_real_coastlines')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
    summary['density']=[]
    for ax,domain in zip(axes,('handwriting','coastline')):
        methods=('averaged','repeated','natural','linear') if domain=='handwriting' else ('periodic','linear')
        for method,color in zip(methods,COLORS):
            group=selected(domain,method,1. if method=='linear' else .5)
            vals=[med([r for r in group if int(r['n'])==n],'heldout_p95') for n in (8,16,32)]
            ax.plot([8,16,32],vals,'o-',color=color,label=LABELS[method])
            summary['density'].append({'domain':domain,'method':method,'n':[8,16,32],'median_p95':vals})
        ax.set_yscale('log');ax.set_xticks([8,16,32]);ax.set_xlabel('Retained vertices N')
        ax.set_ylabel('Median held-out distance E95');ax.set_title('Pen trajectories' if domain=='handwriting' else 'Island coastlines')
        ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.15)
    save(fig,figures,'fig10_real_density')
    (tables/'real_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--figures',required=True);p.add_argument('--tables',required=True)
    a=p.parse_args();print(json.dumps(generate(a.results,a.figures,a.tables),indent=2))
