# TONE — Remaining Work

Status snapshot (2026-04-25): Phases A–D done. Saturator and auto-EQ smoke-trained and exporting. LA-2A LSTM still training. Phase E (composite CLAP) and full-length training runs are the open items.

---

## 1. LA-2A LSTM training — decide on max_steps cap

**State:** step ~63.7k / 451k, val loss ~0.473 stable, GPU 89% util, ETA ~4 days at 1.3 it/s.

**Action:**
- Pull the latest `metric/val/*` from the run logs and compare against the TCN-param checkpoint baseline.
- If LSTM val loss has plateaued near TCN parity, stop the run early and keep the best checkpoint — don't burn 4 days for marginal gains.
- If still descending, let it run but set a hard cutoff at ~150k steps.
- Decision criterion: "is val loss still improving by >2% per 20k steps?" If no → stop.

**Files:** `conf/trainer/bb.yaml` (max_steps), checkpoint dir under `/shared/artifacts/`.

---

## 2. Real auto-EQ training run

**State:** only an 80-step smoke run was done. Synth corpus is in `/shared/datasets/tone_auto_eq` (300 trainval, 30 test).

**Action:**
- Kick off a full training run on the synthetic corpus first to validate the loss curve and model capacity before chasing real data.
- Config: `conf/model/gb/tone_auto_eq/model_gb_tone_auto_eq_peq.d.yaml`
- Target: ~50k steps, monitor `metric/val/brown` and `metric/val/mrstft`.
- Verify spectra qualitatively: pick 3 held-out test clips, run them through the trained model, plot input vs. output mel spectrum vs. −6 dB/oct reference.

**Pass criterion:** output log-mel slope within ±1.5 dB of −6 dB/oct on average across held-out test set.

---

## 3. Full saturator training

**State:** only 800-step smoke run. Default max_steps is 15k.

**Action:**
- Re-run with full 15k steps on `/shared/datasets/tone_sat`.
- Verify transfer-curve fit on a DC sweep `[-1, 1]` after training: max abs error < 0.02 vs. `SaturatorCurveSynth` reference.
- Re-export ONNX, re-verify roundtrip vs. PyTorch.

---

## 4. MUSDB18 ingestion (real music for auto-EQ)

**State:** synth corpus stand-in only. MUSDB18 stems available at `/shared/datasets/musdb18/`.

**Action:**
- Extend `scripts/prepare_auto_eq_data.py --src /shared/datasets/musdb18` to walk the MUSDB stems, sum `mixture.wav` (or sum the four stems), resample to 44.1 kHz mono, segment into ~10 s windows, and emit (dry, brown-EQ'd) pairs.
- Target: ~10 h material → ~3.6k pairs at 10 s each.
- Re-run task #2 on this corpus once it's prepared.

---

## 5. End-to-end Python chain test

**File to create:** `scripts/test_tone_chain.py`

**Action:**
- Load 3 ONNX sessions (`auto_eq_model.onnx`, `saturator_model.onnx`, `la2a_model.onnx`) via `onnxruntime`.
- Reimplement LUFS leveler + true-peak ceiling in numpy (mirroring the C++ logic — small enough to be a faithful port).
- Pipeline: `audio → LUFS leveler → auto-EQ → saturator → LA-2A → ceiling → trim`.
- Run a representative test mix through it.
- Validate:
  - output integrated LUFS within ±0.5 dB of −14
  - true-peak ≤ −1 dBTP measured by independent 4× oversampler
  - no NaN / Inf on extreme inputs (silence, DC, +20 dB sine, 0 dBFS white noise)
- Sweep `Amount` ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, confirm monotonic loudness/character change without artefacts.

---

## 6. Phase E — Composite CLAP plugin (macOS required)

**Files to create:**
- `nablafx/export/composite.py` — `CompositePluginMeta` + `export_composite_bundle()` that calls existing `export_bundle` 3× and writes `tone_meta.json`.
- `scripts/export_tone.py` — CLI: `--auto-eq-run`, `--saturator-run`, `--la2a-run` → produces staging bundle with all three ONNX + meta files plus top-level `tone_meta.json`.
- `native/clap/src/composite_meta.{hpp,cpp}` — loads `tone_meta.json`, declares stage ring-buffer sizes and Amount-knob → (sat pre-gain, LA-2A PR, auto-EQ wet/dry) mappings.
- `native/clap/src/tone_plugin.cpp` — separate dylib target. Wires: `lufs_leveler → ort_session(autoeq) → ort_session(sat) → ort_session(la2a) → true_peak_ceiling → output_trim`. Reuses `ort_session.{hpp,cpp}` and the ring-buffer pattern from `nablafx_plugin.cpp`. Reports `latency = autoeq_rf − 1 + la2a_rf − 1 + ceiling_lookahead`.

**Build wiring:**
- `native/clap/CMakeLists.txt` — add `tone.clap` target listing the new sources + linking `lufs_leveler.cpp`, `true_peak_ceiling.cpp`, `composite_meta.cpp`, `ort_session.cpp`.
- `native/clap/build.sh` — accept `tone` argument routing to that target.

**DAW verification (must run on macOS):**
- Build: `native/clap/build.sh tone`
- Install: `~/Library/Audio/Plug-Ins/CLAP/`
- Load on a Bitwig / Reaper master bus.
- Sweep Amount knob — listen for clicks, confirm reported latency matches sum of stage RFs + ceiling lookahead.
- Verify integrated LUFS converges to −14 on a real mix.
- Verify downstream true-peak meter never exceeds −1 dBTP.
- Stress: silence, DC, +20 dB sine, 0 dBFS white noise — no crashes, no NaN.

**Blocker:** current host is Linux. Phase E either needs to be deferred until a macOS host is available, or needs a Linux CLAP build path validated first (CLAP itself is cross-platform — feasible to build/test on Linux with Bitwig Linux or Reaper Linux).

---

## Suggested order

1. Decide LA-2A cap (task 1) — frees up GPU and unblocks Phase E inputs.
2. Full saturator training (task 3) — short, cheap, gets us a real saturator.onnx.
3. Full auto-EQ training on synth (task 2) — validates the architecture before MUSDB.
4. MUSDB ingestion + retrain (task 4).
5. Python chain test (task 5) — verifies all three trained models compose correctly.
6. Phase E composite CLAP (task 6) — only after the three ONNX models are final.
