# Agent Configuration

## Agent skills

### Issue tracker

Issues are tracked as Local Markdown files in `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` at repo root + `docs/adr/`. See `docs/agents/domain.md`.

## Installed skills

The following skills have been loaded and are available for use by the agent:

- **writing-plans** — Create comprehensive implementation plans with bite-sized tasks. Use when planning new features or stages.
- **executing-plans** — Execute implementation plans systematically with verification. Use with writing-plans.
- **verification-before-completion** — Verify work before claiming completion. Use before any completion claim.
- **improve-codebase-architecture** — Surface architectural friction, propose deepening opportunities. Use for codebase refactoring.
- **systematic-debugging** — Root cause investigation before fixes. Use for bugs, test failures, unexpected behavior.

## Session Context — May 24 2026

### srgan-full training notebook (`notebooks/srgan/srgan-full.ipynb`)
- Fixed bugs: LPIPS=nan caused no `_best.pt` → crash on `load_best_generator`
- **Changes applied:**
  - Save checkpoint every epoch (not just when validation runs)
  - Handle NaN LPIPS: `if not math.isnan(lpips_val) and lpips_val < best_lpips`
  - Validate only at last epoch (`if epoch == self.n_epochs`) instead of epochs 1 and 5
  - Fallback: `load_best_generator` already tries `.pt` if `_best.pt` missing
  - Reduced epochs: `n_epochs: 5 → 3`, expected ~1h total for 9 models
  - `full_evaluation` batch_size: `1 → 4` for speed
- **Status:** Training was interrupted at M2 epoch 3. User needs to rerun cell 16 (Kernel → Restart & Run All)
- **WARNING:** Old checkpoint files from interrupted run may exist in `models/Super-resolution/srgan-full/checkpoints/` — clear them before restarting if they'd interfere

### Diagnostic preservation notebook (`notebooks/evaluation/diagnostic-preservation.ipynb`)
- Created: inference-only evaluation notebook (27 cells) for comparing classifier accuracy between original vs SR-reconstructed images
- Passes syntax check: all 13 code cells valid
- Ready to run once srgan-full checkpoints exist
- Uses: ResNet50 ordinal (QWK=0.8764) + VGG16 CE pipelines
- Evaluates all SRGAN families: srgan2 (9), srgan31 (9), srgan32 (9), srgan-full (9)

### Known issues / bugs fixed May 24 2026
- **Bug: `validate()` returned `None`** — `return np.mean(...)` was accidentally dropped during edit. Fixed by adding it back.
- **Bug: checkpoint saved inside validate block** — `save_checkpoint` was inside `if epoch == self.n_epochs:` instead of before it. Fixed by moving it outside.
- M1 (×2 scale, Sc.A) produces LPIPS=nan and PSNR=3.63 dB — generator collapses for ×2. Needs investigation if it recurs.
- M2 (×4 scale, Sc.A) works: PSNR=22.67, LPIPS=0.5171 at epoch 1 prior to interrupt
