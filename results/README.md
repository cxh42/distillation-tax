# results/ — per-video records

`<cell>/<pid>_s<seed>.json` holds every metric for one video (schema in `data/README.md`); the
`_done_*` flags are bookkeeping for `scripts/metrics_watch.py`. `<stem>.iqa.json` is the IQA part
(also merged into the main JSON). `<stem>.feat.npz` (VideoMAE 768-d, DINOv2 1024-d, per-frame
DINOv2 16×1024) is not in git: download `features_npz.tar.gz.*.part` from the `v0.1-data` release,
`cat features_npz.tar.gz.*.part > features_npz.tar.gz && sha256sum -c features_npz.tar.gz.sha256 &&
tar -xzf features_npz.tar.gz` in the repo root, or regenerate with `scripts/metrics_watch.py --once`.
