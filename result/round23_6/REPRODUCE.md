# Reproduction and continuation

Canonical server project: /home/mxy/Stock/课题/一进二策略 (same inode as /data1/stock-mxy/课题/一进二策略). Use the independent worktree of the corresponding branch under research/round23_6_26_execution. The frozen input is research/literature_round23_6_26/MASTER_PROMPT.md.

Run `python tools/run_round23_6.py --phase reference --method SuperLearner --resume`, then the implemented aggregation stages `--phase adapt`, `--phase backtest`, `--phase stats`, `--phase report`. Use the isolated env/bin/python for Torch/arch and rsr-env/bin/python for TensorFlow 1.15.5. The CLI exposes explicit NOT_IMPLEMENTED or UNSUPPORTED_FILTER for unfinished adapters; it never silently substitutes an algorithm. Its DAG covers only implemented tasks, and is not a claim that all required stages are implemented.

Read source_lock.json and resolved_author_configs before training. Third-party source is obtained from locked commits through external vendor downloads, not vendored without license. Raw supplier tables, supplier credentials, environments and large public data are server-only. Download failures and partial public datasets are retained locally; do not load a partial archive as successful data. The exact server resource/download recovery state is in the master directory.

Standalone reproduction without the supplier datasets and canonical source dependencies is not claimed. source_dependency_manifest.json and source_dependency_manifest.csv identify required source files and hashes. Full missing requirements are listed in missing_artifacts.json and execution_status_by_paper.csv. Do not interpret a successful individual CLI task as COMPLETE for a paper or research line.

Source recovery is pinned by commit/blob hashes. Canonical dependency snapshot is under research/canonical_dependencies; production files were not overwritten. No vendor SDK binaries, raw market databases or credentials are included.
