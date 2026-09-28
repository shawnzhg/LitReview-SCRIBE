# Published pipelines

Clone each pipeline at its commit and apply its deviation patch. `campaign.slurm` and `drtulu_pool.slurm` refuse to run unless the clone's `HEAD` is the commit below and the sha256 of its `git diff` and of its patch file both equal the patch hash below.

| Pipeline | `BL` | Upstream | Commit | Deviation patch (sha256) | Python |
|---|---|---|---|---|---|
| AutoSurvey | `autosurvey` | https://github.com/AutoSurveys/AutoSurvey | `5e8f389f3d51b29bad16dc6ae75db3e8a45a3b65` | `patches/autosurvey.patch` (`d9d6637005bb76a0800e6fe56062aa6999ffb77e39eeb7812b8a3cc7e37ad850`) | 3.12, `env/autosurvey_env_lock.txt` |
| SurveyForge | `surveyforge` | https://github.com/Alpha-Innovator/SurveyForge (`APP_SURVEYFORGE` is its `code/` directory) | `9114a0b7895a0f7eb614938d9bc0c956cf25245b` | none (the `git diff` must be empty) | 3.11, `env/surveyforge_env_lock.txt` |
| SurveyGen-I | `sgi` | https://github.com/SurveyGens/SurveyGen-I | `92f84ba3bbfb6ea3437d23fa773d90e6c96e1997` | same pool: `patches/sgi_same_pool.patch` (`791e8821798d3922d98de72e142e23eb143edce7df13a59d3a79e498c22d2d05`); fixed input: `patches/sgi_fixed_input.patch` (`73c3d44038ca19355a70580154c8035d1d2d2f900cdf628c1ee51d1ca497fee2`) | 3.10, `env/sgi_env_lock.txt` |
| SurveyG | `surveyg` | https://github.com/akBear23/SurveyG | `089a05ba0fd870cc4c2285e762e1db3bd489884c` | `patches/surveyg.patch` (`6ccbe8c1f36d84ee59742168ab7060dd94f68d77fd9dd4a8c3ed2050ef19a431`) | 3.11, `env/surveyg_env_lock.txt` |
| LLM×MapReduce | `llmxmr` | https://github.com/thunlp/LLMxMapReduce | `0e93cc9ef8b89d6b0c75032b24d219dc15117908` | `patches/llmxmr.patch` (`e987aaa333b82462c80f196dde5668a284543e5060de02e3bdd95ec6eb64272f`) | 3.10, `env/llmxmr_env_lock.txt` |
| LiRA | `lira` | https://github.com/lira-workflow/auto-review-writing | `2bc77a6e7b30586343119bd104cc07ebaec380a1` | `patches/lira.patch` (`c6bcd821306533130206e6427683b452a4684c81872698a53459271a5d936a7b`) | 3.10, `env/lira_env_lock.txt` |
| DR-Tulu | `drtulu_pool.slurm` | https://github.com/rlresearch/dr-tulu | `9d7b0371c085e9311ddec483ed39768c0bd9fe99` | `patches/drtulu.patch` (`2b8ab863c2db50f60bfc0bf56668afb97170c494417452286f7bbb7f90fad50f`) | 3.10, `env/drtulu_env_lock.txt`, model `rl-research/DR-Tulu-8B` |

For each arm set `PY_<ARM>` (its interpreter), `APP_<ARM>` (its clone) and `PATCH_<ARM>` (its patch), with `<ARM>` one of `AUTOSURVEY`, `SURVEYFORGE` (no patch), `SGI`, `SURVEYG`, `LLMXMR`, `LIRA`; for DR-Tulu set `DRTULU_AGENT_DIR` (the clone's `agent/` directory) and `PATCH_DRTULU`.
Also set `POOL_GOLD_DIR` (the tasks' gold records, which name the evaluated review), `AUTOSURVEY_DB`, `SURVEYFORGE_DB`, `AUTOSURVEY_EMBED_MODEL`, `SURVEYFORGE_EMBED_MODEL`, `CUTOFF_DIR` (the `<cutoff>.<task>.npy` views of `biolitbench/pool/embed/cutoff_views.py`), `ALLOW_DIR` (fixed input), `SGI_MODELS_DIR` and `LIRA_INPUT`.
SurveyForge runs under same pool only. Submit `autosurvey` and `surveyforge` with `--gres=gpu:1`; the other arms need no GPU.
