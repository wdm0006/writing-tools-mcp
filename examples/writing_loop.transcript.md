Committed transcript of examples/writing_loop.py against the live stdio server
(uv run writing-tools-mcp, 15 tools, brown_corpus baseline). Generated on the
feat/writing-loop-example branch, 2026-09-13. Server logs (stderr) omitted for brevity.

# Happy path: verification clear, exit 0

$ uv run python examples/writing_loop.py
$ echo $?   # 0

WRITING LOOP - analyze -> revise -> verify (server: uv run writing-tools-mcp, 15 tools)

DRAFT (85 words):
The quarterly report for the quarter was prepared by the analytics team. Several data gaps were identified during the course of the review of the process. It was decided by the leadership that the rollout would be postponed for the duration of the quarter. Concerns about the migration timeline were raised by the engineering staff of the organization. A plan for additional testing was proposed by the subcommittee. The findings were documented in a shared repository. Follow-up actions are being tracked by the program manager.

---- STAGE 1 - ANALYZE: stylometric_analysis (0.6s) ----
  baseline_used: brown_corpus
  findings (4):
  1. [pos_anomalies @ document] part-of-speech patterns deviate from the baseline (pos_det +4.97, pos_adp +3.30, pos_pron -2.92) -> rework sentences built on the same clause pattern — vary how phrases attach to the verb
  2. [hapax_legomena_rate_outlier @ document] hapax_legomena_rate sits 6.45 standard deviations from the baseline mean -> compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
  3. [punct_density_outlier @ document] punct_density sits 4.16 standard deviations from the baseline mean -> compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
  4. [comma_ratio_outlier @ document] comma_ratio sits 5.25 standard deviations from the baseline mean -> compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
---- STAGE 2 - REVISE: guided_revision brief (0.0s) ----
You are revising the document below. Work through the findings in order — highest impact first — and change only what a finding justifies.

## Findings (impact order)
1. [pos_anomalies] at document — part-of-speech patterns deviate from the baseline (pos_det +4.97, pos_adp +3.30, pos_pron -2.92)
   Fix: rework sentences built on the same clause pattern — vary how phrases attach to the verb
2. [hapax_legomena_rate_outlier] at document — hapax_legomena_rate sits 6.45 standard deviations from the baseline mean
   Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
3. [punct_density_outlier] at document — punct_density sits 4.16 standard deviations from the baseline mean
   Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
4. [comma_ratio_outlier] at document — comma_ratio sits 5.25 standard deviations from the baseline mean
   Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis

## Document

The quarterly report for the quarter was prepared by the analytics team. Several data gaps were identified during the course of the review of the process. It was decided by the leadership that the rollout would be postponed for the duration of the quarter. Concerns about the migration timeline were raised by the engineering staff of the organization. A plan for additional testing was proposed by the subcommittee. The findings were documented in a shared repository. Follow-up actions are being tracked by the program manager.

## Revision protocol
1. Fix each finding in order, keeping the author's meaning.
2. Re-run the analysis tools on the revised text and compare against the numbers in the findings.
3. Stop when the findings clear or plateau — do not polish past the evidence.
---- STAGE 3 - VERIFY: revised draft vs draft (0.5s) ----
  REVISED (74 words):
The analytics team wrote the quarterly report for the quarter, and they found several data gaps in it while reviewing the process. They raised concerns about the timeline for the migration, so leadership pushed the rollout back for the quarter. The subcommittee proposed more testing for the gaps, and they wrote them down for the team for the quarter. The program manager tracks the follow-up actions. Everyone expects a decision about them next month.

  baseline_used: brown_corpus
  verdicts: 14 improved / 0 unchanged / 0 regressed
  avg_sentence_len: z -0.67 -> -0.34 (delta +0.33, increased) => improved
  avg_word_len: z 0.47 -> 0.24 (delta -0.23, decreased) => improved
  comma_ratio: z -5.25 -> -1.08 (delta +4.17, increased) => improved
  function_word_ratio: z 1.23 -> 0.87 (delta -0.36, decreased) => improved
  hapax_legomena_rate: z 6.45 -> 5.29 (delta -1.16, decreased) => improved
  pos_adj: z -2.26 -> -1.33 (delta +0.93, increased) => improved
  pos_adp: z 3.30 -> 1.33 (delta -1.97, decreased) => improved
  pos_det: z 4.97 -> 4.50 (delta -0.47, decreased) => improved
  pos_noun: z 2.10 -> 1.92 (delta -0.18, decreased) => improved
  pos_pron: z -2.92 -> 1.17 (delta +4.09, increased) => improved
  pos_verb: z -1.46 -> -0.89 (delta +0.57, increased) => improved
  punct_density: z -4.16 -> -4.00 (delta +0.16, increased) => improved
  sentence_len_std: z -1.56 -> -0.23 (delta +1.33, increased) => improved
  ttr: z 1.64 -> 1.33 (delta -0.31, decreased) => improved

  verify_revision prompt verdict:
    You are verifying a revision. The `stylometric_delta` tool profiled a draft (text_a) and its revision (text_b) against the same baseline; the verdict below says what the revision actually moved.
    How to read it: a z-score of 0 sits exactly in the baseline's range, so a statistic improves when the revision moved its z-score closer to zero and regresses when the movement pushed it further out.
    
    ## Verdict (baseline: brown_corpus)
    14 improved, 0 regressed, 0 unchanged.
    Improved: avg_sentence_len, avg_word_len, comma_ratio, function_word_ratio, hapax_legomena_rate, pos_adj, pos_adp, pos_det, pos_noun, pos_pron, pos_verb, punct_density, sentence_len_std, ttr
    Regressed: nothing
    
    ## Improved — the revision moved these toward the baseline
    - avg_sentence_len: z -0.67 -> -0.34 (delta +0.33, increased)
    - avg_word_len: z 0.47 -> 0.24 (delta -0.23, decreased)
    - comma_ratio: z -5.25 -> -1.08 (delta +4.17, increased)
    - function_word_ratio: z 1.23 -> 0.87 (delta -0.36, decreased)
    - hapax_legomena_rate: z 6.45 -> 5.29 (delta -1.16, decreased)
    - pos_adj: z -2.26 -> -1.33 (delta +0.93, increased)
    - pos_adp: z 3.30 -> 1.33 (delta -1.97, decreased)
    - pos_det: z 4.97 -> 4.50 (delta -0.47, decreased)
    - pos_noun: z 2.10 -> 1.92 (delta -0.18, decreased)
    - pos_pron: z -2.92 -> 1.17 (delta +4.09, increased)
    - pos_verb: z -1.46 -> -0.89 (delta +0.57, increased)
    - punct_density: z -4.16 -> -4.00 (delta +0.16, increased)
    - sentence_len_std: z -1.56 -> -0.23 (delta +1.33, increased)
    - ttr: z 1.64 -> 1.33 (delta -0.31, decreased)
    
    ## Findings on the revised text
    1. [pos_anomalies] at document — part-of-speech patterns deviate from the baseline (pos_det +4.50, pos_adv -2.33, pos_noun +1.92)
       Fix: rework sentences built on the same clause pattern — vary how phrases attach to the verb
    2. [hapax_legomena_rate_outlier] at document — hapax_legomena_rate sits 5.29 standard deviations from the baseline mean
       Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
    3. [punct_density_outlier] at document — punct_density sits 4.00 standard deviations from the baseline mean
       Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
    
    ## Verification protocol
    1. Fix the regressed statistics first — they are what the revision broke; leave the improved ones alone.
    2. Re-run `stylometric_delta` after the next pass and render this prompt again with the fresh response.
    3. Stop when the regressions clear or plateau — do not polish past the evidence.

RESULT: verification clear - 14 statistic(s) improved, none regressed against brown_corpus. Exit 0.

# Failure mode (--demo-failure): regression caught, exit 1

$ uv run python examples/writing_loop.py --demo-failure
$ echo $?   # 1

WRITING LOOP - analyze -> revise -> verify (server: uv run writing-tools-mcp, 15 tools)

DRAFT (85 words):
The quarterly report for the quarter was prepared by the analytics team. Several data gaps were identified during the course of the review of the process. It was decided by the leadership that the rollout would be postponed for the duration of the quarter. Concerns about the migration timeline were raised by the engineering staff of the organization. A plan for additional testing was proposed by the subcommittee. The findings were documented in a shared repository. Follow-up actions are being tracked by the program manager.

---- STAGE 1 - ANALYZE: stylometric_analysis (0.6s) ----
  baseline_used: brown_corpus
  findings (4):
  1. [pos_anomalies @ document] part-of-speech patterns deviate from the baseline (pos_det +4.97, pos_adp +3.30, pos_pron -2.92) -> rework sentences built on the same clause pattern — vary how phrases attach to the verb
  2. [hapax_legomena_rate_outlier @ document] hapax_legomena_rate sits 6.45 standard deviations from the baseline mean -> compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
  3. [punct_density_outlier @ document] punct_density sits 4.16 standard deviations from the baseline mean -> compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
  4. [comma_ratio_outlier @ document] comma_ratio sits 5.25 standard deviations from the baseline mean -> compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
---- STAGE 2 - REVISE: guided_revision brief (0.0s) ----
You are revising the document below. Work through the findings in order — highest impact first — and change only what a finding justifies.

## Findings (impact order)
1. [pos_anomalies] at document — part-of-speech patterns deviate from the baseline (pos_det +4.97, pos_adp +3.30, pos_pron -2.92)
   Fix: rework sentences built on the same clause pattern — vary how phrases attach to the verb
2. [hapax_legomena_rate_outlier] at document — hapax_legomena_rate sits 6.45 standard deviations from the baseline mean
   Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
3. [punct_density_outlier] at document — punct_density sits 4.16 standard deviations from the baseline mean
   Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
4. [comma_ratio_outlier] at document — comma_ratio sits 5.25 standard deviations from the baseline mean
   Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis

## Document

The quarterly report for the quarter was prepared by the analytics team. Several data gaps were identified during the course of the review of the process. It was decided by the leadership that the rollout would be postponed for the duration of the quarter. Concerns about the migration timeline were raised by the engineering staff of the organization. A plan for additional testing was proposed by the subcommittee. The findings were documented in a shared repository. Follow-up actions are being tracked by the program manager.

## Revision protocol
1. Fix each finding in order, keeping the author's meaning.
2. Re-run the analysis tools on the revised text and compare against the numbers in the findings.
3. Stop when the findings clear or plateau — do not polish past the evidence.
---- STAGE 3 - VERIFY: revised draft vs draft (0.5s) ----
  REVISED (88 words):
It was determined by the analytics team of the organization that the preparation of the quarterly report for the quarter was completed by the team. The identification of the data gaps was carried out during the course of the review of the process by the staff. The postponement of the rollout for the duration of the quarter was decided by the leadership group of the organization. The raising of concerns about the timeline of the migration was done by the engineering staff of the organization for the quarter.

  baseline_used: brown_corpus
  verdicts: 4 improved / 1 unchanged / 9 regressed
  avg_sentence_len: z -0.67 -> 0.51 (delta +1.18, increased) => improved
  avg_word_len: z 0.47 -> -0.03 (delta -0.50, decreased) => improved
  comma_ratio: z -5.25 -> -5.25 (delta +0.00, none) => unchanged
  function_word_ratio: z 1.23 -> 2.82 (delta +1.59, increased) => regressed
  hapax_legomena_rate: z 6.45 -> 4.99 (delta -1.46, decreased) => improved
  pos_adj: z -2.26 -> -3.43 (delta -1.17, decreased) => regressed
  pos_adp: z 3.30 -> 6.50 (delta +3.20, increased) => regressed
  pos_det: z 4.97 -> 8.14 (delta +3.17, increased) => regressed
  pos_noun: z 2.10 -> 2.49 (delta +0.39, increased) => regressed
  pos_pron: z -2.92 -> -2.93 (delta -0.01, decreased) => regressed
  pos_verb: z -1.46 -> -3.44 (delta -1.98, decreased) => regressed
  punct_density: z -4.16 -> -4.41 (delta -0.25, decreased) => regressed
  sentence_len_std: z -1.56 -> -2.06 (delta -0.50, decreased) => regressed
  ttr: z 1.64 -> -0.96 (delta -2.60, decreased) => improved

  verify_revision prompt verdict:
    You are verifying a revision. The `stylometric_delta` tool profiled a draft (text_a) and its revision (text_b) against the same baseline; the verdict below says what the revision actually moved.
    How to read it: a z-score of 0 sits exactly in the baseline's range, so a statistic improves when the revision moved its z-score closer to zero and regresses when the movement pushed it further out.
    
    ## Verdict (baseline: brown_corpus)
    4 improved, 9 regressed, 1 unchanged.
    Improved: avg_sentence_len, avg_word_len, hapax_legomena_rate, ttr
    Regressed: function_word_ratio, pos_adj, pos_adp, pos_det, pos_noun, pos_pron, pos_verb, punct_density, sentence_len_std
    
    ## Regressed — what the revision broke; fix these first
    - function_word_ratio: z 1.23 -> 2.82 (delta +1.59, increased)
    - pos_adj: z -2.26 -> -3.43 (delta -1.17, decreased)
    - pos_adp: z 3.30 -> 6.50 (delta +3.20, increased)
    - pos_det: z 4.97 -> 8.14 (delta +3.17, increased)
    - pos_noun: z 2.10 -> 2.49 (delta +0.39, increased)
    - pos_pron: z -2.92 -> -2.93 (delta -0.01, decreased)
    - pos_verb: z -1.46 -> -3.44 (delta -1.98, decreased)
    - punct_density: z -4.16 -> -4.41 (delta -0.25, decreased)
    - sentence_len_std: z -1.56 -> -2.06 (delta -0.50, decreased)
    
    ## Improved — the revision moved these toward the baseline
    - avg_sentence_len: z -0.67 -> 0.51 (delta +1.18, increased)
    - avg_word_len: z 0.47 -> -0.03 (delta -0.50, decreased)
    - hapax_legomena_rate: z 6.45 -> 4.99 (delta -1.46, decreased)
    - ttr: z 1.64 -> -0.96 (delta -2.60, decreased)
    
    ## Unchanged
      comma_ratio
    
    ## Findings on the revised text
    1. [high_ai_probability] at document — stylometric confidence 0.9 (high) from 3 converging indicator(s)
       Fix: revise for human cadence first — sentence-length variety and concrete specifics — then re-run stylometric_analysis to verify the confidence score drops
    2. [uniform_sentences] at document — sentence lengths are unusually uniform (-2.06 z-score vs the baseline on sentence_len_std)
       Fix: break the rhythm: split one long sentence and merge two short ones so lengths vary
    3. [pos_anomalies] at document — part-of-speech patterns deviate from the baseline (pos_det +8.14, pos_adp +6.50, pos_verb -3.44)
       Fix: rework sentences built on the same clause pattern — vary how phrases attach to the verb
    4. [function_word_anomaly] at document — function-word usage (articles, prepositions, conjunctions) is unusual (+2.82 z-score vs the baseline on function_word_ratio)
       Fix: swap some connective-heavy phrasing for direct verbs, or the reverse, to move function-word usage back toward the baseline
    5. [hapax_legomena_rate_outlier] at document — hapax_legomena_rate sits 4.99 standard deviations from the baseline mean
       Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
    6. [punct_density_outlier] at document — punct_density sits 4.41 standard deviations from the baseline mean
       Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
    7. [comma_ratio_outlier] at document — comma_ratio sits 5.25 standard deviations from the baseline mean
       Fix: compare this aspect of the draft with the baseline register, revise toward its range, then re-run the analysis
    
    ## Verification protocol
    1. Fix the regressed statistics first — they are what the revision broke; leave the improved ones alone.
    2. Re-run `stylometric_delta` after the next pass and render this prompt again with the fresh response.
    3. Stop when the regressions clear or plateau — do not polish past the evidence.

RESULT: REGRESSION CAUGHT - the revision moved function_word_ratio, pos_adj, pos_adp, pos_det, pos_noun, pos_pron, pos_verb, punct_density, sentence_len_std away from the baseline. Exit 1.
