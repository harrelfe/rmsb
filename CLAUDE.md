# CLAUDE.md — rmsb package (Bayesian regression modeling strategies)

Maintainer: Frank E. Harrell Jr. Companion to the `rms` package
(`~/R/rms`, https://github.com/harrelfe/rms). rmsb fits Bayesian models
(chiefly `blrm`, Bayesian binary/ordinal logistic regression) in Stan.

Read this file first every session. Then read `~/R/rms/R/orm.fit.r`,
`orm.rfit.r`, `orm.s`, `Ocens.r` and `src/ormidx.f90` as the reference
implementation. Treat `rms` as read-only unless Frank says otherwise.

## Standing rules (same as rms)

- Base R only. No tidyverse, ever. Use data.table where it clearly helps.
- Minimal, surgical diffs. This is a mature package with reverse dependencies.
  Justify each change by correctness, safety, or a large line reduction.
  No broad sweeps or gratuitous refactoring.
- Flag anything that could affect CRAN compliance or reverse dependencies.
- Do not refactor Fortran/C++ inner loops unless explicitly asked.
- Do not call private `pkg:::` APIs. Prefer high-level functions of other
  packages over low-level ones.
- Preserve existing base R style. Roxygen2 for documentation.
- Every Claude-authored code block is preceded by a comment of the form
  `# Claude <model name> High YYYY-MM-DD          L lines`
  (use the model actually running this session).
- Change reports use: file name, line number, existing code, new code. Do not
  reproduce whole functions.

## How to work with Frank (verification first)

- Present analysis and confirm understanding before proposing edits.
- Evidence beats inference. In the rms work, two plausible, reasoned fixes
  (Hessian dynamic-range pre-damping; an exact-zero-density hypothesis) were
  both disproven by real trace output after being proposed. Do not ship a fix
  for a hypothesis that has not been confirmed by output from a real run. If
  you cannot confirm it, add diagnostics (gated behind a `trace`/debug
  option, zero cost when off) and ask Frank to run them.
- Working code is verified against an independent reference before
  delivery. Build test harnesses for non-trivial changes.
- Workflow: edit -> `R CMD check` -> run all test files -> rebuild the
  books/vignettes that use the package. Frank runs the books.
- Environment: macOS, zsh, R >= 4.4. Apple Silicon (aarch64) in use.

## What a Stan-based package changes

- rmsb needs only the log-likelihood, written in Stan. No score vector or
  Hessian. That is the main simplification relative to `rms::orm.fit`.
- Goal: compute the log-likelihood components with exceptional numerical
  accuracy, as was done in `orm.fit`. Inspect the Stan code for
  cancellation, differences of nearly equal CDF values, log of tiny
  probabilities, and underflow in the tails. Use stable forms
  (log-scale arithmetic, `log_diff_exp`, `log1m`, `log_sum_exp`, etc.) and
  confirm each against a high-precision or independent reference before
  adopting it. Report measured error, not just the new code.
- Replace the many conditionals now in the Stan code with index vectors
  pre-computed in R, using `rms`'s `ormidx.f90` subroutine (the same
  subscripts `orm.fit` uses for its likelihood). Stan receives the
  subscripts as data and does no branching on data type. Check how
  `ormidx.f90` is exposed from `rms` (registered entry point vs. a wrapper
  in `rms`) before deciding how rmsb calls it, and flag any new dependency
  or export this implies for CRAN.

## Lessons from the rms `orm` work that carry over

- Large number of distinct Y values (k on the order of 1e5) is a different
  regime. In `rms`, `Matrix::solve` segfaulted in native sparse LU for
  probit/cloglog at k ~ 290,000 (R 4.6.1, Matrix 1.7.6, aarch64). A segfault
  or bus error is a fatal OS signal; `try()`/`tryCatch()` cannot catch it.
  The crash tracked matrix dimension, not Hessian values. Stan avoids the
  solve, but sampling speed and memory at large k still matter.
- `y.precision` (rounding Y to fewer distinct levels) is the existing
  mechanism for reducing the number of intercepts. Keep it consistent with
  `orm`.
- `maxetastep` (cap on the largest single-step shift in the linear
  predictor) was needed for `orm` Newton-Raphson stability. It is an
  optimizer feature and will not apply to Stan sampling, but initial values
  passed to Stan for large k may benefit from similar care. Test before
  assuming.
- In an R package all files in `R/` share one namespace. A top-level helper
  defined in one file is visible in the others. Do not define shared helpers
  inside a function body.
- Censoring: `rms::orm` handles censored Y through `Ocens()` / `Ocens2ord`
  using exact integer-grid arithmetic (no floating-point `eps` offsets).
  Read `Ocens.r` and `orm.fit.r` for the mechanism.
- Random intercepts in `orm.fit`/`ormrfit` use nested AGQ with SQUAREM
  acceleration. Frank validated these against `clmm2` and direct numerical
  integration. They are a numerical reference for rmsb random-effect
  results, not code to port.
- Other `rms` invariants worth knowing: `Design()` short variable names are
  hash keys (`age`, `age'`, `age''`) into `parms`/`limits`/`values`/
  `datadist`, so renaming is impractical. Formulas that must survive
  re-evaluation by `predict`/`update` need `bquote`-spliced literals, not
  `eval` in a callee frame.

## Roadmap (in this order)

### Phase 1 — improve `blrm` (must be fully tested before Phase 2)

1. Audit the current `blrm` R code and Stan code. First report, for each
   item below, what the code does today. Change nothing until Frank agrees.
2. Accuracy: stable log-likelihood components (see above), with measured
   error against references.
3. Pre-computed subscripts from `ormidx.f90` passed to Stan; remove the
   data-dependent conditionals from the Stan code.
4. Censoring: `blrm` must handle censored Y with exactly the mechanism used
   by `rms::orm` (`Ocens`, `Ocens2ord`). Determine whether it already does.
   If it does not, implement it. If it partly does, list the differences.
5. Keep `cppo` working. Do not break existing `blrm` behavior, saved fits,
   or reverse dependencies.
6. Test plan: with very weak (flat-like) priors, posterior modes/means
   should agree with `rms::orm` estimates on the same data, including
   censored, partial-PO and clustered cases. Compare log-likelihood values
   at identical parameters between `blrm` and `orm`. Add regression tests
   to the package test suite. Phase 1 is done only when these pass and
   `R CMD check` is clean.

### Phase 2 — new function `borm` (Bayesian analog of `orm`)

- Handles every family (link function) that `rms::orm` handles (logistic,
  probit, loglog, cloglog, cauchit, and whatever `orm`'s
  `probabilityFamilies` lists at the time you read it).
- Intercept prior: implement only one, the Dirichlet prior on the
  multinomial (cell) probabilities. Simulations showed that other intercept
  priors (e.g. scaled t) slow sampling when the number of distinct Y values
  is large. Do not add alternatives.
- Implements `mix_re` to augment `cluster`, which can be brought over from
  `blrm`. Read how `blrm` does `cluster`/random effects first.
- Implements `ydep` (below).
- Same censoring mechanism as `orm`.

### `ydep` specification (borm only; not in `blrm` yet)

- `cppo` in `blrm` is a function of Y that gives one scalar
  non-proportional-odds parameter per predictor in `ppo`.
- `ydep` is the analog of `cppo` that returns a matrix instead of a scalar,
  giving more than one non-proportional-odds parameter per predictor
  parameter, i.e. a Y-dependent effect shape per predictor.
- The predictors it applies to are specified in the `ppo` argument, which is
  a formula object (partial proportional odds).
- `blrm` is unaffected. Do not add `ydep` to `blrm`. Design the data
  structures and Stan interface in Phase 1 so that Phase 2 does not require
  undoing work (e.g. pass the Y-dependent design as a matrix block with a
  width, even if the width is 1 for `cppo`).
- Open design questions to settle with Frank before coding: how the matrix
  columns are parameterized and given priors, how `ydep` interacts with the
  `ormidx` subscripts and with censored observations, and how `contrast`/
  `Predict`/`summary` should report a Y-dependent effect.

## Things Claude must not assume

- That `blrm` lacks or has censoring support, `cluster`, `mix_re`, or
  `ormidx` use. Check the code and report.
- That a Stan change is faster or more accurate. Measure it.
- Details of the original `ydep` design beyond what is written here. Ask.
