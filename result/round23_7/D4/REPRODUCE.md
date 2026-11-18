在现有授权服务器的项目根目录运行，使用现有env及已锁定作者源码/原始数据，不升级ZyQuant。

```bash
PYTHONDONTWRITEBYTECODE=1 research/round23_6_26_execution/env/bin/python -B tools/continue_round23_7_round26_1.py --deliverable D4 --resume --jobs 2 --seed 17 --publish
```

固定17全路径→42→73，不能选优种子；恢复前检查无重复控制进程。模型缓存核验完整源码、数据/标签/成熟时间、参数、执行契约和全部完成工件。完整参考先于目标训练。实际server-only输入路径与SHA见target_data/server_only_inputs_manifest.json。
