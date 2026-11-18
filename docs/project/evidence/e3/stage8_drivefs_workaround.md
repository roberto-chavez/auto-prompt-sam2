# Stage 8 DriveFS workaround

The first two Stage 8 attempts stopped before reading reserved images or labels. The
runner blocked while opening a small Stage 5 cached artifact marker through Colab's
DriveFS mount (`stage5/.../leave_paraguay_out_points_3_0_030/.../COMPLETE.json`).
The process was in `folio_wait_bit_common`; the run created no Stage 8 predictions or
report.

Stage 8 recomputes DINO features for reserved chips and does not call the Stage 7
`ValidationEngine.reuse()` path. It therefore does not consume Stage 5's cached feature
payloads. The runner now continues to validate the pinned Stage 5 validation marker,
report hash, and completed run state, while skipping only the per-payload cache audit.
All other Stage 6/7 lineage checks, the human-review gate, the frozen protocol hash,
reserved-ID checks, and per-image/per-label hashes remain enabled. General callers of
`read_dependency` retain full artifact verification by default.

The first retry then exposed a second pre-data issue: `configs/training.yaml` had the
same content as the Stage 6 snapshot but different LF/CRLF bytes after Git checkout.
Source verification now accepts only exact byte matches or hashes of the same file
with line endings normalized to LF/CRLF; any other source change still fails closed.

This is an explicit implementation limitation: Stage 5 cache payload integrity is not
rechecked during Stage 8. It does not alter the model, selected rows, data splits,
thresholds, prompt budgets, or evaluation metrics. The exact source commit and frozen
protocol hash must be recorded in the eventual Stage 8 report. No reserved data was
read by the interrupted attempts.
