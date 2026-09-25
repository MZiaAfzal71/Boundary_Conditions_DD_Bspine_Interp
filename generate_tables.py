"""Generate article tables and scalar macros from CSV results.

Usage: python generate_tables.py --results OUTPUT --out ARTICLE/tables
This writes computed table fragments only; it never compiles or edits a paper.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_rows(path):
    with open(path) as f:return list(csv.DictReader(f))


def median(rows,key):return float(np.median([float(r[key]) for r in rows]))


def num(v):
    if v==0:return '0'
    if abs(v)<.001 or abs(v)>=10000:
        a,b=f'{v:.2e}'.split('e')
        return rf'${a}\times 10^{{{int(b)}}}$'
    return f'{v:.4g}'


def table(path,columns,head,rows):
    text=[r'\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(head)+r' \\',r'\midrule']
    text+=[' & '.join(str(x) for x in row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    Path(path).write_text('\n'.join(text)+'\n')


def paired_interval(rows,alpha,reps=2000):
    a={r['id']:r for r in rows if r['alpha']==str(float(alpha)) and r['method']=='averaged'}
    b={r['id']:r for r in rows if r['alpha']==str(float(alpha)) and r['method']=='repeated'}
    strata={}
    for key in sorted(a):
        group=(a[key]['family'],a[key]['n'],a[key]['regime'])
        strata.setdefault(group,[]).append(float(b[key]['hausdorff_sampled'])-float(a[key]['hausdorff_sampled']))
    values=list(strata.values());rng=np.random.default_rng(625+int(100*alpha))
    draws=[np.median(np.concatenate([rng.choice(v,len(v),replace=True) for v in values])) for _ in range(reps)]
    return float(np.median(np.concatenate(values))),np.percentile(draws,[2.5,97.5]).tolist()


def generate(results,out):
    results=Path(results);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bench=read_rows(results/'benchmark.csv');seams=read_rows(results/'seams.csv')
    pilot=read_rows(results/'refinement.csv');audit=read_rows(results/'resolution_audit.csv')
    check=json.loads((results/'verification.json').read_text())
    labels={'averaged':'Averaged knots','repeated':'Repeated controls','natural':'Natural boundary','periodic':'Periodic cubic'}
    open_rows=[];closed_rows=[]
    for alpha in (0.,.5,1.):
        for method in ('averaged','repeated','natural','periodic'):
            rows=[r for r in bench if r['method']==method and float(r['alpha'])==alpha]
            # Do not silently remove failed fits from aggregate results.
            if any(r['status']!='ok' for r in rows):raise ValueError('Resolve/report benchmark failures before summarizing.')
            item=[labels[method],f'{alpha:g}',num(median(rows,'hausdorff_sampled')),num(median(rows,'bending'))]
            (closed_rows if method=='periodic' else open_rows).append(item)
    table(out/'open_results.tex','lrrr',['Method',r'$\alpha$',r'Median $H_s/S$',r'Median $S E_b$'],open_rows)
    table(out/'closed_results.tex','lrrr',['Method',r'$\alpha$',r'Median $H_s/S$',r'Median $S E_b$'],closed_rows)
    paired=[];pair_values={}
    for alpha in (0.,.5,1.):
        mid,ci=paired_interval(bench,alpha);pair_values[str(alpha)]={'median':mid,'ci':ci}
        paired.append([f'{alpha:g}',num(mid),num(ci[0]),num(ci[1])])
    table(out/'paired_results.tex','rrrr',[r'$\alpha$',r'Median $\Delta H_s/S$',r'2.5\%',r'97.5\%'],paired)
    seam_rows=[]
    for method,label in [('direct','Direct'),('geometric','Collinear supports'),('speed_matched','Speed matched'),('periodic','Periodic interpolation')]:
        rows=[r for r in seams if r['method']==method]
        seam_rows.append([label,num(median(rows,'seam_angle_deg')),num(median(rows,'seam_d1')),
                          num(median(rows,'seam_d2')),num(max(float(r['aligned_origin_change']) for r in rows))])
    table(out/'seam_results.tex','lrrrr',['Construction',r'Angle ($^\circ$)',r'$J_1$',r'$J_2$',r'Max. $I$'],seam_rows)
    reductions=np.array([1-float(r['refined_bending'])/float(r['base_bending']) for r in pilot])
    hchange=np.array([float(r['refined_hausdorff_sampled'])/float(r['base_hausdorff_sampled'])-1 for r in pilot])
    summary={'case_count':len({r['id'] for r in bench}),'fit_count':len(bench),'failures':sum(r['status']!='ok' for r in bench),
             'maximum_residual':max(float(r['residual']) for r in bench),
             'adaptive_count':sum(r['integrator'].startswith('adaptive') for r in bench),
             'integration_warnings':sum(r['integrator']=='adaptive_warning' for r in bench),
             'large_gauss_disagreement_count':sum(float(r['gauss_disagreement'])>.001 for r in bench),
             'maximum_final_quadrature_error':max(float(r['quadrature_relative']) for r in bench),
             'minimum_speed':min(float(r['min_speed']) for r in bench),
             'pilot_count':len(pilot),'pilot_accepted':sum(r['accepted']=='True' for r in pilot),
             'pilot_median_bending_reduction_percent':100*float(np.median(reductions)),
             'pilot_median_distance_change_percent':100*float(np.median(hchange)),
             'pilot_distance_worse_count':int(np.sum(hchange>.001)),
             'sampling_max_absolute_change':max(float(r['absolute_change']) for r in audit),
             'paired_stratified_bootstrap':pair_values,'test_count':check['count'],
             'periodic_equivalence_error':check['checks']['periodic_vs_independent_power_basis']['value'],
             'robin_equivalence_error':check['checks']['repeated_vs_independent_robin_system']['value']}
    macros={'StudyCases':str(summary['case_count']),'StudyFits':str(summary['fit_count']),
            'StudyChecks':str(summary['test_count']),'StudyFailures':str(summary['failures']),
            'MaxResidual':num(summary['maximum_residual']),'AdaptiveCount':str(summary['adaptive_count']),
            'GaussMismatchCount':str(summary['large_gauss_disagreement_count']),
            'PilotCases':str(summary['pilot_count']),'PilotAccepted':str(summary['pilot_accepted']),
            'PilotReduction':f"{summary['pilot_median_bending_reduction_percent']:.2f}",
            'PilotDistanceWorse':str(summary['pilot_distance_worse_count']),
            'SamplingChange':num(summary['sampling_max_absolute_change']),
            'PeriodicError':num(summary['periodic_equivalence_error']),
            'RobinError':num(summary['robin_equivalence_error'])}
    (out/'numbers.tex').write_text('% Generated by generate_tables.py; do not edit numbers manually.\n'+
                                 '\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(generate(a.results,a.out),indent=2))
