# Pilot status (2026-10-02)

> Historical log written while the experiment was being set up. The final results are in the [README](../README.md) and `runs/final_*/summary.md`.

## Done (no API key needed)

| Item | Result |
| --- | --- |
| Data download and conversion | BrowseComp-Plus (830 questions, 100,195 documents, questions decrypted with the official script); RAGBench covidqa / techqa / hotpotqa. Raw files in `raw/`, unified format in `data/<dataset>/` (both can be regenerated) |
| Pilot step 2 (RAGBench format) | **Passed**: documents can be merged into a per-subset corpus and document-level recall computed, but a content hash must be used as doc id (see below) |
| BrowseComp-Plus chunking | `work/bcp/chunks.sqlite`: **1,702,356 chunks**, about 879M embedding tokens (title included), 516 tokens/chunk on average |
| Offline tests | 30 pass; the `demo` end-to-end flow passes; `prepare` -> fp16 index -> retrieval verified on the real RAGBench structure |

## Dataset decisions (spec TBDs I settled; can be changed)

- **RAGBench subsets are covidqa (biomedical), techqa (technical docs) and hotpotqa (multi-hop).** The spec's example CUAD and FinQA are unsuitable for corpus-level retrieval: CUAD has only 41 generic question templates over 509 contracts in 2,061 questions, so a question cannot point to a specific contract; FinQA questions depend on a specific table and carry no context ("What is the average expected volatility?"). The spec's "legal / finance" domains are therefore absent from RAGBench; covering them requires another data source.
- **500 questions sampled per subset** (fixed seed, full document set kept) -> dev 100 / test 400. BrowseComp-Plus: all 830 questions -> dev 166 / test 664.
- covidqa documents have titles; the RAGBench versions of techqa and hotpotqa have none (title field empty, so serialization and the embedding prefix omit it automatically). BrowseComp-Plus titles come from the document front matter.
- BrowseComp-Plus has 2.9 gold and 6.1 evidence documents per question on average, matching the spec, plus 76 hard negatives per question.

## Cost and feasibility estimates (section 9 formulas, real chunk counts)

| Item | Value |
| --- | --- |
| BrowseComp-Plus embedding (one-off) | about **$17.6** (879M tokens x $0.02/M; check against the current OpenAI price) |
| BrowseComp-Plus index memory | float32 is about 10.4 GB, but only about 3-5 GB of RAM is available (15.7 GB total). Embeddings are stored as fp16 (5.2 GB, score error <= 2e-5, identical top-10 sets). **The index uses streaming block scans** (`index.mode: auto` picks `memory` / `stream` from free RAM): it is not resident in memory, reads the file in 20k-row blocks and keeps a running top-k; concurrent queries are merged into one batch so a single scan serves many queries. It is still an exact full scan, not ANN |
| Streaming index measurements (1.7M x 1536, fp16) | peak memory **0.79 GB** (5.25 GB for the memmap version); one query takes about 3.4 s; with 32 concurrent queries merged the average is **0.15 s/query**. For BrowseComp-Plus test runs, `--workers` >= 16 amortizes the disk scan |
| RAGBench grading | No accuracy; faithfulness and completeness are reported instead (see SPEC_REVIEW.md). Converted questions carry annotated relevant sentences (covidqa 5.3 on average, hotpotqa 3.1, techqa 15.0); a problem with trailing periods in some annotation keys was fixed, and no subset has questions without relevant sentences |
| G5 over the **full** BrowseComp-Plus corpus | **About 1.7M requests per "query x round": about 11.8 hours and $70-100 at the 40 req/s limit.** Infeasible -> **go/no-go verdict: a reduced candidate pool is required** |
| G5 reduced pool (gold + evidence + all hard negatives, 1,387 chunks on average) | About 1,000 tokens per request (chunk about 580 + **Jev question and rubric text about 420**) -> about 1.4M tokens per query round = **$0.06, about 35 s**. Test 664 questions x about 3 rounds x 3 repeats = about 6,000 query rounds = **$350, about 58 hours**. G5's only randomness is query generation (Jev is deterministic), so a single repeat (about 19 hours, $120) or a 500-chunk pool (about 12 s/round) is enough |
| G5 over the full RAGBench corpus (measured / projected) | covidqa (4,944 chunks) **measured $0.109, 124 s/round** (bound by 40 req/s); hotpotqa about $0.24 / 263 s; techqa about $0.34 / 255 s. 400 test questions x about 2.5 rounds x 1 repeat = about 34 hours -> **full scans are not recommended for RAGBench either; G5 uses a reduced pool** (500 chunks: about 12 s/round, about 3.5 hours per repeat) |

**Measured correction (2026-10-02, covidqa 3-question G5 trial):** two things had been underestimated. (1) Each request carries Jev's question and rubric text in addition to the chunk; measured at 521 tokens/request on average (chunks average only about 100), so cost is about 3x the earlier estimate. (2) With small requests **the bottleneck is 40 req/s, not 100K tokens/s**, so time = requests / 40. The table above uses the measured values.

Key point: in the reduced-pool variant, **G4 and G5 must be compared on the same pool** (the code controls this with `--pool-size`; pool = gold + evidence + negatives, topped up with fixed-seed distractor chunks). A pool size of 1500 is recommended for BrowseComp-Plus. If all of G1-G7 run on reduced pools, also report a full-corpus G1-G4 (embedding only, cheap).

## Blocked on "you" (cannot be done on your behalf)

1. **API keys**: environment variables are currently unset -> `OPENAI_API_KEY` (embedding), `GLM_API_KEY`, `TYPESAFE_API_KEY` (Jev), `JUDGE_API_KEY`. Set the keys in environment variables yourself; do not paste them into chat or files.
2. **GLM provider / base_url / model id / price** and the **judge model** (fill into `configs/experiment.yaml`).
3. **Budget cap** (the embedding of about $17.6 is certain; G5 about $120-350 depending on repeats and pool size, plus GLM cost). Once confirmed, pilot steps 1, 3, 4, 5 can run.

## Run order once keys are available

```bash
python -m ragexp.cli prepare --dataset covidqa  --data-dir data/covidqa     # validate the real APIs on the small corpus first
python -m ragexp.cli calibrate raw/ragbench_covidqa_test.jsonl --n 100   # first confirm the judge reproduces RAGBench annotations (faithfulness / completeness)
python -m ragexp.cli traj    --dataset covidqa  --run-id pilot --split dev --limit 20 --repeats 1
python -m ragexp.cli tune    --dataset covidqa  --run-id pilot               # judge AUC / calibration (step 3)
python -m ragexp.cli groups  --dataset covidqa  --run-id pilot --split dev --limit 20 --repeats 1   # 20-question dry run (step 5)
python -m ragexp.cli prepare --dataset bcp      --data-dir data/bcp          # about $17.6, about 1.7M chunks, resumable
```

## Decision log: BrowseComp-Plus reduced to 100 questions (2026-10-02)

- **Decision**: the OpenAI embedding account is capped at 40,000 TPM and will not be upgraded; BrowseComp-Plus uses only a fixed-seed sample of **100 questions** (dev 20 / test 80), dataset name `bcp100`. The official prebuilt Qwen3 document-level vectors are not used (they would break the chunk design and the G4/G5 comparison).
- **Corpus**: the `bcp100` corpus = gold + evidence + all hard negatives of those 100 questions (8,095 documents, 145,198 chunks, about 74.3M embedding tokens), roughly $1.49 in total; about 31 hours at 40k TPM. This is a materialized version of the spec section 9 "reduced pool", not the full 100k-document corpus.
- **How to read the comparisons**:
  - G5 and G1-G4 (for the G4-vs-G5 comparison) retrieve inside the **per-question candidate pool** (`--pool-size`: gold + evidence + negatives, topped up with distractors).
  - G1-G4 can additionally retrieve over the whole `bcp100` corpus (145k chunks, the union pool) as a reference for a larger search space; **there are no results on the full corpus (1.7M chunks)**, and reports must say so.
  - Every document in the union corpus is some question's gold / evidence / hard negative, so it is harder than a random corpus; absolute recall numbers cannot be compared with full-corpus numbers.
- **Statistical power**: the test split has only 80 questions, so McNemar / bootstrap intervals are wide and only large effects can be detected. This is especially true because BrowseComp-Plus accuracy is low; state conclusions conservatively.
- **G5 cost (80 test questions, pools of about 1,400 chunks)**: about $0.06 and 35 s per query round; 80 questions x about 3 rounds x 1 repeat = about $14, 2.3 hours (40 req/s limit).
- **Embedding order** (the account TPM is shared, so runs are sequential): covidqa -> hotpotqa -> techqa (about 5.6M tokens in total, about 2.5 hours) -> bcp100 (about 31 hours). Shards are small and resumable: if the process is interrupted, rerun the same `prepare` command to continue.

## Pilot results (covidqa, 2026-10-02)

| Step | Result |
| --- | --- |
| 1 Call stability | **Passed**: embedding, Jev (noul / score), GLM (plain / tool calling / JSON) and judge all call reliably. The dry run logged 1,538 records with only 1 retry (very low 429 rate). The OpenAI embedding account limit is 40k TPM (see above). z.ai's content filter (code 1301) occasionally blocks a question; retry and failure isolation were added |
| 2 RAGBench format | Passed (see above) |
| 3 Judges vs `covered_all` | **Failed "clearly better than random"**: on 100 dev questions / 600 round samples, Jev AUC 0.58 (ECE 0.18) and LLM binary judgement balanced accuracy 0.52. Switching to a "relevant-sentence coverage" label gives only AUC 0.56-0.62; giving only the top 3 / 5 / 10 chunks did not help (context rot ruled out). Jev's probability varies mainly across questions (between-question SD 0.27) and barely across rounds of the same question (SD 0.04). `covered_any` AUC was 0.99 at 20 questions and 0.69 at 100, a small-sample artifact. tau tentatively 0.79, R_fixed 4. **Verified on covidqa only; hotpotqa and bcp100 not yet validated** |
| 4 G5 vs embedding | 20 questions, 300-chunk candidate pool: G5 recall_first@5 0.75 vs embedding 0.92, NDCG@10 0.74 vs 0.80 (n=20, differences not significant); G5 costs $0.034 per question, about 10x the other groups, p50 latency 155 s. Full-scan cost is above |
| 5 All-group dry run | **Passed**: 20 questions x 7 groups, no missing record fields, all eight components present, stop reasons and round counts reasonable (G7 averages 1.2 rounds, 70% of G4 truncated by R_max) |

### 20-question dry-run numbers (pipeline check only; n=20 is too small to compare groups)

faithfulness 0.94-0.97 for every group (near the ceiling, low discrimination); completeness G1 0.51, G2 0.38 (45% abstain), G3 0.54, G4 0.55, G5 0.37, G6 0.57, G7 0.53. Cost per question: G1 $0.001, G3/G4/G6 about $0.003-0.004, G5 $0.034.

### Open items (affect every group; must be frozen before the real test run)

- **GLM reasoning setting**: reasoning is **84%** of GLM output tokens; `glm_query_gen`, which produces a single query sentence, uses about 547 reasoning tokens per call, with p50 latency 9.5 s. Whether to disable it (`glm.extra_body`, z.ai's thinking switch) changes cost, latency and accuracy together and must be compared on dev before freezing. *(Later frozen as `reasoning_effort: low`; see `configs/experiment.yaml`.)*
- **The Jev judge is near random on RAGBench**: whether to accept this result or change labels / settings was left for discussion. *(The final report keeps it and reports the AUC per dataset.)*
