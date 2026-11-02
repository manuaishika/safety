# Reproducing results

Every reported number comes from one chain. Each step is deterministic given its inputs, and a
re-run resumes rather than repeats work.

| Step | Command | Produces | Reproduce check |
|---|---|---|---|
| 0. Data | `indicsafe data validate` / `data build` | `data/built/manifest.json` (dataset hash) | hash is stamped into every log |
| 1. Generate | `indicsafe run --config configs/<run>.yaml` | `results/runs/<run_id>/logs`, `run_meta.json` | same config + dataset gives the same `run_id`; re-run resumes |
| 2. Grade | `indicsafe grade --logs <run> --grader-config configs/graders/<g>.yaml` | `grades/<rubric>/<g>/grades.jsonl` | cached; re-grade is byte-identical |
| 3. Validate | `indicsafe validate export / import / report` | `report.json` with grader error per language | seeded bootstrap |
| 4. Analyse | `indicsafe analyze --grades <grades folder> --out <dir>` | tidy table, tables per scoring policy, figures | `indicsafe repro-check --grades <grades folder>` |
| 5. Report | `indicsafe report --analysis <dir> --validation <validation folder>` | `report.md` and a manifest of file hashes | `indicsafe verify-report --report <folder>` |

Checks you can run any time:

    indicsafe repro-check --grades <grades folder> --include-unreviewed   # analysis twice, byte-compare
    indicsafe verify-report --report results/report                        # report vs its manifest
    indicsafe licenses --config configs/<run>.yaml --grader-config configs/graders/<g>.yaml

What pins a number: the dataset hash, rubric version and grader config (analysis refuses to mix
them), the seed, and the package versions recorded in `analysis_manifest.json` (`uv.lock` pins
them). Bootstrap and the Bayesian fit are seeded; figure metadata is stripped; no timestamps are
written to analysis outputs.

Re-running with different package versions can legitimately change numbers in the last digits or
change figure bytes. `repro-check` proves reproducibility on one machine and one environment, not
across environments.

Real-model runs cannot be made bit-for-bit repeatable by this repo: local models can differ across
hardware and library versions even at temperature 0. Record the run folder (`run_meta.json` has the
model ids, config, git state) and publish the graded labels, not the model.

## Releasing the data

    indicsafe data export-public --out release/                 # dev split only
    indicsafe data export-public --out release/ --include-test --reason "camera-ready"

The test split needs a prior `data freeze`, the flag and a reason; each release is logged to
`data/release_log.jsonl`. Model responses are never part of a public export; keep
`results/runs/*/logs` private (responses can contain harmful-compliance text).
