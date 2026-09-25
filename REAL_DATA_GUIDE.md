# Real-world geometry extension

The revised article now tests the boundary analysis on recorded handwriting and mapped island coastlines. These are reconstruction studies using independently sourced data. Withheld vertices belong to the same processed source curves as the retained vertices; they are not independently measured physical ground truth.

## Public sources and attribution

| Source | Original download | Attribution and reuse |
|---|---|---|
| Character Trajectories | https://archive.ics.uci.edu/static/public/175/character%2Btrajectories.zip | Williams, B. (2006), UCI Machine Learning Repository, DOI [10.24432/C58G7V](https://doi.org/10.24432/C58G7V). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Adaptations are documented below. |
| Natural Earth 1:10 million coastline | https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip | Made with Natural Earth. Natural Earth contributors. [Public domain](https://www.naturalearthdata.com/about/terms-of-use/). Projected and subsampled for this study. |

Downloaded on 25 September 2026. The original archives are preserved under `supplement/results/source_cache/`. The numerical source metadata are in `supplement/results/real/SOURCES.json`.

SHA-256 checksums:

```text
handwriting.zip
5d2db017ef0d8cf0e65ed060c9e90399f78eb9f1e3cb63e22ca8c3ef4ba67d52

coastline.zip
bfa04cdbcbef07ef90dfca1dabb48062eca29900a113df0f389303e255484017
```

The coastline archive's internal version is `5.0.0-pre9`; the current landing-page label is different. The script pins the actual archive hash and internal version. If a provider replaces the download, the script stops on a checksum mismatch. Use the preserved snapshot rather than silently accepting a different release.

Preserve this attribution, license links, and modification description when redistributing the adapted data. The download archives belong in an external cache or article supplement, not in the scripts repository.

## Exact selection and transformations

Handwriting: select the first two original-order eligible recordings in each of the 20 source character classes. Eligibility means at least 65 vertices after consecutive near-duplicate removal. The delivered selection rejected no preceding candidate and uses 40 traces from one writer. The source IDs are recorded explicitly in `real/cases.json`.

The source channels are processed velocities. Multiply the first two channels by their corresponding `consts.datanorm` values, `[48.866640026065134, 74.1867275749229]`, cumulatively sum, and translate the first point to zero. The common time-step scale is immaterial to the normalized shape measurements. This reconstructs processed paths, not raw tablet coordinates. The provider's original smoothing and alignment remain part of the data; no additional smoothing is applied.

Coastlines: select the unique single-part closed ring containing each fixed interior point, then use that point as the centre of a spherical Lambert azimuthal equal-area projection with radius 6,371 km.

| Outline | Longitude | Latitude | Zero-based source feature |
|---|---:|---:|---:|
| Sri Lanka | 80.7 | 7.8 | 145 |
| Madagascar | 46.5 | -19.0 | 18 |
| Cuba | -79.5 | 22.0 | 2942 |
| Iceland | -19.0 | 65.0 | 410 |
| Taiwan | 121.0 | 23.7 | 204 |
| Tasmania | 146.7 | -42.0 | 1515 |

Consecutive near-duplicates are removed using a threshold of `1e-12` times the full source bounding-box diagonal. Original indices are preserved in each processed NPZ. Closed curves retain exactly one repeated endpoint. Geographic projection is a defined working geometry; no claim of geodesic or surveying accuracy is made.

## Experimental design

- Retain 8, 16, or 32 actual source vertices under index-based and arc-length-based policies, preserving endpoints or closure. Arc-length selection uses nearest feasible source vertices and enforces increasing indices.
- Use the remaining distinct source vertices only for evaluation. The source curve defines the offline subsampling task; it is not a prediction setting where all withheld geometry is unknown at acquisition time.
- Test uniform, centripetal, and chord-length parameterizations. Open curves compare averaged clamped knots, repeated endpoint controls, and natural boundaries. Closed curves use periodic interpolation.
- Add a piecewise-linear baseline with identical retained vertices. Its classical curvature-squared energy is not comparable, so report only distances for it.
- Evaluate 276 retained sets, 2,268 cubic reconstructions, and 276 linear reconstructions. Retain all outcomes, including warning diagnostics.
- Test four seam constructions at four origins on each of the six 32-vertex arc-length coastline selections: 96 seam experiments.

`E95` is the 95th percentile of withheld source-point distances divided by the full source diagonal. `Emax` is their maximum. The open-curve endpoint measure restricts evaluation to the first and last 10% of source polygonal arc length. These metrics weight source vertices; they are not uniform-arc-length integrals.

Cubic curves are subdivided using the second-derivative chord-error bound. The nominal distance tolerance is `2e-6` times the source diagonal. Point-to-polyline queries use exact segment projection with a complete candidate-search bound, checked against exhaustive search. Twenty-six fixed cases are audited at the tighter tolerance `5e-7`. This improves source-point distance control; it does not certify topology, regularity, or continuous symmetric Hausdorff distance.

Paired comparisons first take a median over the six matched settings within each handwriting trace. Win/tie/loss counts use the combined curve-distance error budgets. They are descriptive outcomes, not significance tests or estimates for unseen writers.

## What the executed results add

The current results support a useful qualification of the synthetic findings: parameterization rankings change with the data source. At centripetal parameters, repeated controls improve whole-curve distance against averaged knots in 34 of 40 trace-level comparisons beyond tolerance, but endpoint comparisons are mixed. Most whole-curve changes against natural boundaries are tiny. These findings support reporting the boundary convention rather than claiming a universally best choice.

Coastline reindexing confirms that matching seam speed does not match acceleration or remove dependence on the chosen starting vertex. Periodic interpolation has the expected derivative matching and reindexing invariance to roundoff in the tested cases. Approximation and interpolation impose different data constraints, so the seam table is not a fidelity ranking.

Three averaged-knot, uniform-parameter handwriting fits have unresolved adaptive-integration warnings: `pen_08_01_n32_index`, `pen_08_02_n32_index`, and `pen_14_01_n32_index`. Their source labels are `l`, `l`, and `r`. Their energy estimates remain in the raw CSV with `energy_warning` status. The affected aggregate energy cell is withheld; their controlled distance results remain included. Do not use the raw warning-marked energy values as reliable measurements.

## Regenerate the results

Run from the separate numerical scripts directory:

```bash
python -m pip install -r requirements.txt
python verify.py
python verify_real.py
python reproduce.py --output ../study_output
```

The first complete run downloads about 11 MB. The default cache is outside the scripts directory, under the output directory. To use the supplied snapshots without downloading, adjust this path to your extraction layout:

```bash
python reproduce.py --output ../study_output \
  --cache ../article/supplement/results/source_cache --offline
```

Run only the new study or only the original synthetic component:

```bash
python reproduce.py --output ../study_output --real-only \
  --cache ../article/supplement/results/source_cache --offline
python reproduce.py --output ../study_output --synthetic-only
```

The standalone `download_real_data.py`, `run_real_experiments.py`, and `report_real_results.py` also provide `--help`. The complete pipeline records 20 core checks and 10 real-data checks. Its scripts never compile the manuscript. Copy generated `figures/` and `tables/` into the article directory, then compile `main.tex` yourself.

## Next research steps

1. Add more writers or acquisition devices with a separate source-level evaluation split. The present same-writer subset cannot support population generalization.
2. Evaluate noisy measurements with an approximation or smoothing objective. Interpolation alone should not be presented as denoising.
3. Test projection and map-scale sensitivity, and topology-aware reconstruction on geographic contours. Source coastlines are generalized cartographic representations.
4. Extend the refinement pilot to real data with a fixed tuning/evaluation separation and matched constraints; do not tune on withheld reconstruction errors and then report them as independent evidence.
5. Resolve difficult curvature integration using interval or higher-precision methods before making fairness claims about near-stationary fits.
