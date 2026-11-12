"""研究完整性 v3：门禁必须带产物与哈希，禁止缺证据默认通过。"""

from core.research_integrity_v3.gates import GateResult, fingerprint

__all__ = ["GateResult", "fingerprint"]
