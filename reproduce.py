"""Generate the study's numerical data, figures, and computed tables.

Quick start (Python >=3.11):
  python -m pip install -r requirements.txt
  python reproduce.py --output ../study_output

The full study downloads two public archives (about 11 MB) on first use.
Use --offline after the source cache is populated; --cache selects an existing
external cache. --synthetic-only skips downloads; --real-only runs just the
real-data extension. The default command regenerates the complete article.
The output directory must be OUTSIDE this scripts directory. Nothing compiles
LaTeX or creates a PDF article. Copy output/figures and output/tables into the
separate article directory. Figures are generated as PDF, SVG, and PNG.
Run python verify.py for checks alone. The default 10 seeds match the delivered
article. More seeds change the results and require updating the article text.
"""
import argparse
import hashlib
import json
from pathlib import Path
from generate_data import generate
from verify import run_checks
from run_experiments import run
from generate_figures import generate as figures
from generate_tables import generate as tables


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,default=10)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--synthetic-only',action='store_true')
    modes.add_argument('--real-only',action='store_true')
    p.add_argument('--cache');p.add_argument('--offline',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();source=Path(__file__).parent.resolve()
    if args.seeds<1:p.error('--seeds must be positive')
    out.mkdir(parents=True,exist_ok=True)
    checks=run_checks();(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(f"Passed {checks['count']} checks.",flush=True)
    report={'output':str(out),'checks':checks['count']}
    if not args.real_only:
        meta=run(out,args.seeds)
        figs=figures(out,out/'figures');summary=tables(out,out/'tables')
        report.update(figures=len(figs),cases=meta['case_count'],fits=meta['fit_count'],summary=summary)
    if not args.synthetic_only:
        from verify_real import run_checks as real_checks
        from run_real_experiments import run as real_run
        from report_real_results import generate as real_report
        realout=out/'real';realout.mkdir(exist_ok=True)
        rc=real_checks();(realout/'verification.json').write_text(json.dumps(rc,indent=2)+'\n')
        print(f"Passed {rc['count']} real-data checks.",flush=True)
        rm=real_run(realout,args.cache or out/'source_cache',args.offline)
        real_report(realout,out/'figures',out/'tables')
        report.update(real=rm,real_checks=rc['count'])
    manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(out.rglob('*')) if f.is_file() and f.name!='checksums.json'}
    (out/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
