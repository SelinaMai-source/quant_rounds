"""复用 fs01122 replay 的 overlay/train 清单；不笛卡尔展开。"""

from core.replay_fs01122.catalog import ReplayJob, overlay_jobs, train_r5_jobs, train_r6_jobs

__all__ = ["ReplayJob", "overlay_jobs", "train_r5_jobs", "train_r6_jobs"]
