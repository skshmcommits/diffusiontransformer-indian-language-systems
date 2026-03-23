# Quantifiable Improvements Report

Date: 2026-03-23

This file lists measurable improvements verified from project artifacts and run logs.

## 1) Code/Runtime Stability

| Area | Before | After | Quantifiable Change | Evidence |
|---|---:|---:|---:|---|
| Workspace diagnostics | Import/runtime issues were present earlier in the session history | No current IDE errors | Reduced to **0 reported errors** | `get_errors` result: "No errors found" |
| Checkpoint inference load | `main.py infer` failed due to state_dict shape mismatch | Same command succeeds after fix in `inference.py` | Command success changed from **0/1 to 1/1** on the same test sentence | Session logs + fix in `OKMInferenceEngine.from_checkpoint` |
| End-to-end training execution | Not yet validated for this repo state | Completed full smoke epoch | **4250/4250 steps completed (100%)** | `training_log.jsonl` entries up to step 4250 |

## 2) Training Metrics (Smoke Run: 1 epoch, batch_size=2)

Source: `checkpoints/training_log.jsonl`

- Log entries: **85**
- First logged step: **50**
- Last logged step: **4250**

### Loss improvements

| Metric | Early Value | Late/Best Value | Improvement |
|---|---:|---:|---:|
| Total loss (first -> last) | 5.3977 | 2.4823 | **54.01% decrease** |
| Total loss (first -> best seen) | 5.3977 | 1.8113 | **66.44% decrease** |
| Node diffusion loss (first -> last) | 0.7652 | 0.4836 | **36.80% decrease** |
| Edge diffusion loss (first -> last) | 0.3397 | 0.2408 | **29.11% decrease** |

### Validation trend (from recorded run output)

- Validation loss improved from **0.7408** to **0.6363**.
- Relative improvement: **14.11% decrease**.

## 3) Artifact Production

Generated artifacts in `checkpoints/`:

- `best_model.pt`
- `final_model.pt`
- `ckpt_step_1000.pt`
- `ckpt_step_2000.pt`
- `ckpt_step_3000.pt`
- `ckpt_step_4000.pt`
- `training_log.jsonl`
- `vocab.json`

Quantifiable result: **8 training artifacts** produced, including **6 model checkpoints**.

## 4) Current Model Status (Important)

### What is working now

- Data loading, training, checkpoint saving, and checkpoint-based inference are operational.
- System-level pipeline reliability is significantly improved versus earlier failures.

### What is not yet strong

- Semantic output quality is still weak after smoke training (1 epoch).
- Example inference output can still show low-quality structure/lexical artifacts.

## 5) Practical Conclusion

- **Engineering quality improved measurably** (fewer errors, successful training/inference path, decreasing losses).
- **Model semantic quality is improving but not production-ready yet**; more training steps and dataset cleanup are still required.
