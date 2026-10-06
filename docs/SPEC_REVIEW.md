# Spec v0.1 review notes (2026-10-02)

Conclusion: **no blocking flaws**; implementation started. The items below were found by checking external facts and walking through the design. They are split into "needs your decision" and "default already adopted in code".

## Verified correct

- Jev: endpoint, pinned `jev-1.13.0`, price $0.042/Mtok (output free), 40 req/s, 100K tok/s, 64k/32k context, `noul`/`score` response formats and `usage.input_tokens/output_tokens` all match the official documentation (docs.typesafe.ai). The `score` in a `score` response is the probability-weighted expected level, which matches the "expected level" in spec section 5.6.
- BrowseComp-Plus: 830 questions, about 100,195 documents, fields `query_id/query/answer/gold_docs/evidence_docs/negative_docs`; the corpus is not encrypted, but the query fields must first be decrypted with the official script.
- The cost example in section 9 is arithmetically correct (560M tokens -> $23.52, 5,600 s).

## Needs your decision (affects conclusions, does not block infrastructure)

1. **RAGBench has no document ids.** `documents` is a plain list of strings per question; there are no doc ids and no human gold answers (`response` is LLM-generated).
   - Adopted: doc id = content hash (identical content is de-duplicated automatically); gold = the documents referenced by `all_relevant_sentence_keys`; samples with no relevant sentence are dropped.
   - Also: RAGBench's CUAD and FinQA are unsuitable for corpus-level retrieval, so covidqa / techqa / hotpotqa were chosen instead (reasons in [PILOT_STATUS.md](PILOT_STATUS.md)). techqa and hotpotqa documents have no title.
   - **Decided:** RAGBench does not report answer accuracy. It reports **faithfulness (claim level) and completeness**; BrowseComp-Plus still reports accuracy.
     - faithfulness: the judge splits the answer into atomic claims and checks each against the group's final evidence set; the score is the supported share (a binary "all claims supported" variant is also reported, for McNemar). Answers with no claims (e.g. "insufficient evidence") are excluded from the mean and reported as `abstain_rate`, so groups that abstain often do not get an inflated faithfulness.
     - completeness: uses RAGBench's annotated relevant sentences (`all_relevant_sentence_keys`) as the reference; the judge decides how many of them the answer covers. This matches RAGBench's own definition and is independent of the retrieval group, so groups that fail to retrieve the key information score low.
     - The two complement each other: faithfulness alone rewards abstention, and completeness closes that gap. `premature_stop` has no "wrong answer" to use on RAGBench, so it is defined as "the judge decided to stop but not all gold documents are in the evidence".
     - judge calibration: the `calibrate` command checks the judge against RAGBench's `adherence_score` / `completeness_score` (re-grading the original response) and reports accuracy, kappa, AUC, MAE, Pearson / Spearman. The judge model was TBD and had to pass calibration in the pilot before use.
     - Known limitation: the completeness reference sentences are the original sample's annotations and do not necessarily cover everything in the corpus that could answer the question; techqa has 15 relevant sentences per question on average (up to 100), which is a heavier load for the judge.
2. **BrowseComp-Plus has no separate title field**, but the document text starts with front matter (`title:`). Adopted: use the front-matter title as the title and strip the front matter from the body (to avoid a duplicate title); fall back to the first line / URL when absent.
3. **NDCG@10 conflicts with top-k=5.** Round 1 retrieves only 5 chunks, so NDCG@10 cannot be computed. Adopted: the retriever keeps an extra top-10 ranking used only for round-1 MRR/NDCG@10; evidence still gets the top 5.
4. **E_max=8k interacts with R_max=6.** Each chunk is about 560 tokens (with title and id), so 8k holds about 14 chunks, while 6 rounds can retrieve up to 30. Evidence is kept by score, so new chunks from later rounds that score lower than existing ones are squeezed out, meaning "multi-round" may add no new information after round 3, and `covered_all` is computed after truncation. When running the dev pilot, check `covered_all_by_round` and evidence tokens; raise E_max (Jev's 32k limit allows it) or switch to "new evidence first" if needed.
5. **G5 shares tau.** tau is tuned on G4's dev trajectories (embedding evidence) and then reused for G5 (Jev-retrieved evidence). This keeps the "only retrieval changes" comparison clean, but G5's evidence distribution differs and its probability calibration may drift. Reporting a sensitivity analysis with a tau tuned on G5's own dev trajectories is recommended.
6. **G1 can be derived from round 1 of the shared trajectory** (same query, same top-5); the spec lists it as an independent run, and it is still run independently (cost is one retrieval plus one generation).

## Implementation details handled in code (not in the spec)

- **The cache key for repeated runs must include the replicate id**, otherwise "3 repeats per question" would get the same sample because the request hash is identical. GLM cache keys include `salt=rep{n}`; Jev and embedding are deterministic and share the cache across repeats. A cache hit reuses the original call's token / latency / retry records, so re-analysis is not distorted.
- With 3 repeats, G5's randomness comes only from upstream query generation (Jev itself is deterministic); G1 has almost no variance at temperature 0.
- Evidence selection: sort by score descending and take the longest prefix that fits in E_max; chunks squeezed out stay in the candidate pool and come back if retrieved later with a higher score.
- Tie handling for equal scores: `is_relevant` first, then arrival order; G5 records the number of tied pairs (`n_tied_pairs`) so saturation can be checked in the pilot.
- `stop_reason` has an extra value `fixed` (G6) beyond the spec's `{judge, r_max, answered}`.
- G2's "rounds" = number of searches <= R_max, after which an answer is forced; truncation by R_max is determined by the forced answer with `tool_choice="none"`.
- Statistics: repeats are averaged per question first; McNemar uses the majority vote (>= 0.5); cost and accuracy differences use a paired bootstrap.

## Still TBD (config slots exist; unfilled values raise a clear error or show `-`)

GLM provider / base_url / model id / price, judge model, official BrowseComp-Plus grading (the official one uses Qwen3-32B as judge), RAGBench subsets and question counts, reduced pool size, budget cap.
