# LitReview-SCRIBE

[![arXiv](https://img.shields.io/badge/arXiv-2609.32318-b31b1b.svg)](https://arxiv.org/abs/2609.32318)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for **[What Can a Leaderboard Certify? Compositional Controllability for Fair Evaluation and Training of Biomedical Literature-Review Agents](https://arxiv.org/abs/2609.32318)**.

Zhaowei Han†, Xiang Zhang†, Lingxiao Guan†, Danqi Hu, Kai Liu, Kevin Chang, Jie Liu — University of Michigan, Ann Arbor  
<sub>† Equal contribution.</sub>

A comparison window covers one stage, several stages, or the whole agent. The gap between
observed and controlled score differences is bounded using only nuisance outside the window,
which gives an admissibility test applied before scores are inspected: inadmissible comparisons
are refused, and an admissible ordering is certified only when the score gap exceeds the
combined sampling and nuisance radii. This repository holds **BioLitBench** and its scorer, the
certification pipeline, and **SCRIBE**, trained with rewards measured at each stage's exit.

Commands for each table of the paper. Run them from the repository root; `<...>` marks a value to fill in.

## Requirements

Copy `configs/site.env.example` to `site.env`, set its paths, and load it and the scorer environment in every shell:

```bash
set -a; . ./site.env; set +a
. "$CCBENCH_ENV"
export PYTHONPATH=$LITREVIEW_ROOT/evaluation:$PYTHONPATH PYTHONHASHSEED=0
```

- `CCBENCH_ENV` is a script you write that sets `PY`, `CCBENCH_ROOT`, `CCBENCH_PARENT` and `CCBENCH_OUT`. `PY` runs the scorer, the benchmark builders and the analysis: torch, sentence-transformers, vllm, pandas, pyarrow, scipy, scikit-learn, statsmodels, networkx, POT, rapidfuzz, PyStemmer, pyyaml, matplotlib.
- `SCRIBE_RUN_PY` harness, `SCRIBE_SERVE_PY` vLLM, `SCRIBE_TRAIN_PY` kvskill training (also pandas, pyarrow, rapidfuzz, pyyaml), `SCRIBE_POOL_PY` pool service (fastapi, uvicorn), `SCRIBE_EMBED_PY` FAISS embedders.
- Slurm for the `.slurm` files; GPU jobs need `CUDA_HOME` or a `SCRIBE_TOOLCHAIN` script.
- Tests: `PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -p no:cacheprovider -q tests evaluation/windowbench/tests`.

Backbone, used by every server:

```bash
huggingface-cli download Qwen/Qwen3.8-27B --revision <revision in configs/models/Qwen3.8-27B-nothink.json>
$SCRIBE_TRAIN_PY scribe/training/backbone.py prepare $HF_HOME/hub/models--Qwen--Qwen3.8-27B/snapshots/<revision> $SCRIBE_MODEL_DIR
```

## Data

`SCRIBE_DATA_ROOT` is one data root: `test50/` (the evaluation tasks and the reference files of their peers) or `train1842/` (the training split). Set `CCBENCH_PARENT=$SCRIBE_DATA_ROOT` and `SCRIBE_RUNS_ROOT=$SCRIBE_DATA_ROOT/<sub>/runs`, where `<sub>` is `CCBENCH_EVAL_SUBDIR` (empty: the root itself). `CCBENCH_ROOT` is searched first and holds the shared score tables in `out/`. Point the other variables of `site.env` at their files in the root.

```
$SCRIBE_DATA_ROOT/
  tasks.txt                                     task ids
  inputs/<task>/                                task_spec.json, task.json, bundle.json
  data/pool_corpus/                             corpus shards, pool_manifest.json, cutoffs.json, campaign50.json
  data/pool_index/bm25/, data/pool_index/edges/ BM25 index, citation-edge store
  data/models/nomic-embed-text-v1/
  data/reference_corpus/                        enriched references, claim grounding
  data_2000/dataset_2000/                       graphs/, refs_enriched/, manifests/
  biolit-bench/results_2000/benchmark_claims/   claims and review structure per task
  <sub>/runs/                                   taskspecs/, gold/, canonical/, oracle/ and the system runs
  <sub>/analysis/, <sub>/schemas/, <sub>/prereg/
  same_pool/                                    budget_caps.json, kcap.json
  out_seed/                                     initial content of $CCBENCH_OUT
```

## Benchmark construction

Set `SCRIBE_BENCH_RESULTS`, `SCRIBE_BGE_MODEL_DIR` and `SCRIBE_NLI_MODEL_DIR`; the `candidates`, `label` and `verify` steps need a GPU.

```bash
$PY biolitbench/build/build_reference_corpus.py --source <claims dir> --out <reference corpus>
for s in candidates label verify assemble; do $PY biolitbench/build/biolitreview_bench/extract/build_enhanced_graph.py --step $s --paper <paper> --work <dataset dir>/graphs/<paper> --model <Qwen3.5-9B dir> --refs <reference corpus>/refs_enriched --grounding-dir <reference corpus>/grounding; done
$PY biolitbench/build/make_splits.py --dataset <dataset dir> --out <manifests>
GLKB_URL=<url> GLKB_USER=<user> GLKB_PASS=<secret> $PY biolitbench/pool/export/export_pool_corpus.py --out <pool corpus>
GLKB_URL=<url> GLKB_USER=<user> GLKB_PASS=<secret> $PY biolitbench/pool/export_pool_edges.py --out <edges>
GLKB_URL=<url> GLKB_USER=<user> GLKB_PASS=<secret> $PY biolitbench/pool/export_pool_meta.py --out <meta>
$PY biolitbench/pool/freeze_manifest_streaming.py --pool <pool corpus> --tasks <dir of <task>/refs.json> --tmp <dir>
$PY biolitbench/pool/build_bm25.py --shards '<pool corpus>/corpus_*.jsonl.gz' --out <pool index>/bm25
$PY biolitbench/pool/build_edges_db.py --edges '<edges>/*.jsonl.gz' --meta '<meta>/*.jsonl.gz' --out <pool index>/edges
$PY biolitbench/pool/select_eval_tasks.py --tasks <dir of <task>/refs.json> --out <eval tasks json>
$PY biolitbench/tasks/build_task_inputs.py --manifest <manifests>/biolit_dev.jsonl --xml <dir of <task>.xml>
$PY biolitbench/tasks/make_taskspecs.py --pool-index <pool index>/bm25
$PY -m ccbench.fair.make_allowlists --out <allowlists>
$PY biolitbench/tasks/build_pool_meta_sqlite.py --shards '<pool corpus>/corpus_*.jsonl.gz'
$PY biolitbench/tasks/build_reference_evidence.py
$PY biolitbench/tasks/build_canonical_bundles.py
```

Per-task retrieval budgets, after the same-pool runs of the published pipelines:

```bash
$PY biolitbench/pool/budget/build_caps.py --runs $SCRIBE_RUNS_ROOT/campaign50 --arm-dir drtulu=$SCRIBE_RUNS_ROOT/campaign50_native/drtulu --tasks $SCRIBE_EVAL_TASKS --out <budget_caps.json>
$PY biolitbench/pool/budget/build_kcap.py --caps <budget_caps.json> --rollouts $CCBENCH_OUT/rollouts --out <kcap.json>
```

## Training the planning carrier (RL from skill: init, seed, GRPO)

On the `test50/` root:

```bash
$PY scribe/training/build_train_tasks.py --check --train_gold <train1842 runs>/gold/train
OUT_ROOT=<dir> bash scribe/training/launch/01_init_theta.sh
export SCRIBE_THETA0=<dir>/initial_carrier
```

On the `train1842/` root, with `SCRIBE_EVAL_TASKS` still the `test50/` task list:

```bash
$PY biolitbench/tasks/build_canonical_bundles.py --split train
set -a; . scribe/launchers/configs/scribe_untrained.env; set +a
TASKS_FILE=$LITREVIEW_ROOT/configs/train_tasks/train_candidates.txt LEVEL=<t> sbatch --gres=gpu:1 --export=ALL scribe/launchers/run_fixed_input.slurm
export BASE_CACHE=$SCRIBE_RUNS_ROOT/level<t>/SCRIBE/bundle_entry TASKS_FILE=<train list>
$SCRIBE_TRAIN_PY scribe/training/launch/finalize_train_list.py --base_cache $BASE_CACHE --out $TASKS_FILE
$SCRIBE_EMBED_PY scribe/training/embed_service.py --model $SCRIBE_EMBED_MODEL_DIR --port <port> &
$PY scribe/training/precompute_train_bands.py --tasks $TASKS_FILE --embed_url http://127.0.0.1:<port> --out <bands.json>
OUT_ROOT=<dir> sbatch --export=ALL scribe/training/launch/02_seed_init.slurm
```

On the `test50/` root, after SCRIBE (untrained) is run and scored under fixed input (sections below):

```bash
$PY -m windowbench.run --ours <untrained fixed-input key> --opponents <published-pipeline fixed-input keys> --windows planning --out <cells dir>
$PY scribe/training/reward_weights.py --cells <cells dir>/cells.csv --out <targets.json>
```

On the `train1842/` root:

```bash
OUT_ROOT=<dir> READOUT_TARGETS=<targets.json> BANDS=<bands.json> sbatch --export=ALL scribe/training/launch/03_grpo.slurm
export SCRIBE_PLANNING_CARRIER=<dir>/grpo_<job>/final_planning_carrier
```

## SCRIBE, SCRIBE (untrained) and SCRIBE-Luna

`SCRIBE_THETA0` and `SCRIBE_PLANNING_CARRIER` come from the training block.

```bash
set -a; . scribe/launchers/configs/scribe_trained.env; set +a        # SCRIBE (untrained): scribe_untrained.env
LEVEL=<n> sbatch --gres=gpu:2 --export=ALL scribe/launchers/run_fixed_input.slurm        # fixed input; SCRIBE (untrained): SEEDS=0,1,2 and --gres=gpu:1
LEVEL=<a> SEEDS=<s> sbatch --export=ALL scribe/launchers/run_same_pool_acquisition.slurm    # same pool: acquisition, one seed per level
$PY scribe/launchers/same_pool_prepare.py pin --src-root $SCRIBE_RUNS_ROOT/level<a> --tasks <task ids file> --seed <s> --out <pins.json>
LEVEL=<b> SRC_PINS=<pins.json> bash scribe/launchers/run_same_pool.sh                    # same pool: generation
RUN_DIR=<dir> LEVEL=<n> TASKS_FILE=<task ids file> LUNA_API_KEY_FILE=<key file> LUNA_BUDGET_USD=<usd> bash scribe/luna/eval_luna.sh fixed   # SCRIBE-Luna; same pool: pool
```

## Published pipelines and commercial agents

```bash
$PY scribe/harness/baselines/make_baseline_inputs.py --tasks $SCRIBE_EVAL_TASKS --taskspecs <taskspecs dir> --out $BASELINE_INPUTS
$PY scribe/harness/baselines/make_lira_input.py --inputs $BASELINE_INPUTS --out <lira data>
for m in nomic gte; do for f in title abs; do $SCRIBE_EMBED_PY biolitbench/pool/embed/embed_shard.py --model $m --field $f --shards '<pool corpus>/corpus_*.jsonl.gz' --out <emb>; done; done
$SCRIBE_EMBED_PY biolitbench/pool/embed/assemble_db.py --system both --emb-root <emb> --shards '<pool corpus>/corpus_*.jsonl.gz' --out <faiss dir> --meta-dir <pool meta> --survey-ids <survey pmid file>
$SCRIBE_EMBED_PY biolitbench/pool/embed/cutoff_views.py --task-cutoffs $SCRIBE_POOL_CUTOFFS --gold $POOL_GOLD_DIR --db <faiss dir> --out <faiss dir>/cutoff_ids
$PY scribe/harness/baselines/make_allowlists_idx.py --arm autosurvey --allowlists $SCRIBE_ALLOWLISTS --pool-index <faiss dir> --cutoff-views <faiss dir>/cutoff_ids
export ALLOW_DIR=$SCRIBE_ALLOWLISTS CUTOFF_DIR=<faiss dir>/cutoff_ids AUTOSURVEY_DB=<faiss dir>/autosurvey_db SURVEYFORGE_DB=<faiss dir>/surveyforge_db AUTOSURVEY_EMBED_MODEL=$SCRIBE_EMBED_MODEL_DIR SURVEYFORGE_EMBED_MODEL=$SCRIBE_GTE_MODEL_DIR LIRA_INPUT=<lira data>/scireviewgen/full_data_abs.json
sbatch --gres=gpu:1 --export=ALL,BL=autosurvey,EVIDENCE=pool scribe/harness/baselines/campaign.slurm   # surveyforge also --gres=gpu:1; other arms without --gres; fixed input: EVIDENCE=ref
SHARD=<n> SLICE=<lo-hi> sbatch --export=ALL scribe/harness/baselines/drtulu_pool.slurm
```

Per-arm settings: `scribe/harness/baselines/README.md`. Commercial agents, one `COMMERCIAL_OUT` bundle per agent: `scribe/harness/commercial/README.md`.

## Scoring

```bash
mkdir -p $CCBENCH_OUT && cp -rL $SCRIBE_DATA_ROOT/out_seed/. $CCBENCH_OUT/
$PY -m ccbench.build                                    # same pool
$PY -m ccbench.build --campaign campaign50_ref          # fixed input
$PY -m windowbench.draft_rollouts build
$PY -m windowbench.draft_rollouts conformance
$PY -m ccbench.score_units --peers
$PY -m ccbench.score_windows --peers --truncate
$PY -m ccbench.experiments.E10_dimensions
$PY -m windowbench.allocation_extract extract                                           # writes $WINDOWBENCH_OUT/allocation
$PY -m windowbench.planning_exit --controls                                             # writes $WINDOWBENCH_OUTLINE_WINDOW/scores_arms.jsonl, scores_controls.jsonl
$PY -m windowbench.admission                                                            # writes $WINDOWBENCH_MEMBERSHIP
sbatch evaluation/windowbench/slurm/outline_axis_gpu.slurm                              # writes $WINDOWBENCH_OUT/outline_axis
```

Our runs and the commercial agents enter the score tables and the boards with:

```bash
SYSTEMS=<TAG>.bundle_entry sbatch evaluation/board/score_new_arms.slurm                 # our fixed-input runs; SCRIBE (untrained) adds <TAG>.bundle_entry.seed1,<TAG>.bundle_entry.seed2
$PY -m windowbench.allocation_extract extract --arms "" --ours <key>=<run tree>        # <run tree> holds <task>/seed0 of the run
$PY -m windowbench.planning_exit
$PY -m windowbench.outline_axis_add --tag <TAG>
evaluation/board/score_iso.sh native <key> <level>                                       # our same-pool runs (generation LEVEL)
evaluation/board/build_view.sh native <key> $ISO_ROOT/<key>_iso_<job>
COMMERCIAL_OUT=<agent bundle> COMMERCIAL_TASKS=<task list> $PY evaluation/board/ingest/<agent>.py convert --out <agent runs>   # elicit also COMMERCIAL_ALLOWLISTS, claude_science also CS_PROMPTS
COMMERCIAL_OUT=<agent bundle> evaluation/board/score_iso.sh agent <agent> <agent runs>
evaluation/board/build_view.sh <agent> $ISO_ROOT/<agent>_iso_<job>
```

`<agent>` is `openai_tool_loop`, `claude_code`, `claude_science`, `gemini_deep_research` or `elicit`. Declare each `<TAG>` in `$WINDOWBENCH_ROSTER_EXTRA`: `{"version": 1, "families": [{"tag": "<TAG>", "keys": ["<TAG>.bundle_entry"]}]}`; add `"seeds": [1, 2]` for SCRIBE (untrained), `"template": "native"` for same-pool keys and `"model_group": "gpt56"` for SCRIBE-Luna.

## tab:leaderboard

```bash
evaluation/board/build_board.sh --ref <our fixed-input keys> --extra <KEY>=<view>,<KEY>=<view> --out <board>
$PY evaluation/board/check_paper_tables.py --tex <paper.tex> --board-a <board> --board-b <board> --rows-a 'SCRIBE=<key>,SCRIBE (untrained)=<key>,SCRIBE-Luna=<key>' --rows-b 'SCRIBE=<key>,SCRIBE (untrained)=<key>,SCRIBE-Luna=<key>'
```

`--extra` takes one `<KEY>=<view>` per key of each view: the agents' keys are in `evaluation/board/view_agents.py`, a native view's key is its `<key>`. The table is `<board>/summary.csv` at `tier=reference_fed`, `obs=system_trunc`.

## tab:rules

```bash
$PY -m ccbench.contrast_certified --root <board> --radius <refusals.json>        # <refusals.json>: see tab:refusals
```

## Entry disclosures and rebuild assertions (app:validation)

```bash
$PY evaluation/analysis/citation_route.py --pool-logs $SCRIBE_RUNS_ROOT/campaign50/sgi --gold $POOL_GOLD_DIR --tasks $SCRIBE_EVAL_TASKS --board <board>
$PY -m windowbench.validate --previous <cells.csv> --independent <window_scores.parquet> --replicates <untrained fixed-input key>
```

`<cells.csv>` is the cells table to reproduce, written by `$PY -m windowbench.run --ours <key> --opponents fixed_input --out <dir>`; `<window_scores.parquet>` holds the reference-fed reports scored by an independent implementation.

## tab:leaderboard-pool, tab:leaderboard-full

From the board of tab:leaderboard: `<board>/summary.csv` at `tier=self_retrieving`, `obs=system_trunc` (same pool), and at `obs=system` (full length).

## tab:capability, tab:capability-pool

```bash
$PY -m windowbench.axes table --dataset fixed_input --with-ours <untrained fixed-input key> --obs system_trunc,draft_trunc,system,draft,retrieval,planning
$PY -m windowbench.axes table --dataset same_pool --with-ours <untrained same-pool key> --obs system_trunc,draft_trunc,system,draft,retrieval,planning
$PY evaluation/analysis/mechanism.py --ccbench-out $CCBENCH_OUT --row <label>=<key>[,<seed key>] --row <label>=<key>
$PY -m windowbench.allocation_score
$PY -m windowbench.allocation_pairs
$PY -m windowbench.conditioned_synthesis build --view system
$PY -m windowbench.conditioned_synthesis build --view system_trunc
$PY -m windowbench.conditioned_synthesis table --dataset same_pool --with-ours <untrained same-pool key> --view system_trunc
$PY -m windowbench.conditioned_synthesis table --dataset fixed_input --with-ours <untrained fixed-input key> --view system_trunc   # full length: --view system
```

## tab:certified

```bash
$PY -m ccbench.merge_windowbench --root <board>
$PY -m ccbench.merge_diagnostics --root <board>
```

## Theory checks (tab:theory-checks, tab:designs, tab:lmtests, tab:refusals, tab:synthetic, tab:calibration, tab:ctrl, tab:thm43)

```bash
$PY -m ccbench.fair.units
$PY -m ccbench.fair.radius --out $CCBENCH_OUT/E13
$PY -m ccbench.fair.filter
$PY -m ccbench.fair.radius_panel_a                                                  # tab:designs
$PY -m ccbench.fair.lm_falsification                                                # tab:lmtests
$PY evaluation/analysis/refusals.py --radius-dir $CCBENCH_OUT/E13 --json <refusals.json>   # tab:refusals
$PY -m ccbench.experiments.E0_synthetic                                             # tab:synthetic
$PY -m windowbench.aa_test --root <board>                                           # tab:calibration; <board> built without the trained SCRIBE and Elicit
$PY -m ccbench.fair.ctrl                                                            # tab:ctrl
$PY -m ccbench.experiments.E3_distance                                              # tab:thm43
```

tab:theory-checks collects these outputs.

## Citation

```bibtex
@misc{han2026leaderboardcertifycompositionalcontrollability,
      title={What Can a Leaderboard Certify? Compositional Controllability for Fair Evaluation and Training of Biomedical Literature-Review Agents}, 
      author={Zhaowei Han and Xiang Zhang and Lingxiao Guan and Danqi Hu and Kai Liu and Kevin Chang and Jie Liu},
      year={2026},
      eprint={2609.32318},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.32318}, 
}
```

## License

MIT — see [`LICENSE`](LICENSE).
