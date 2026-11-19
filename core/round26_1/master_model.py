"""Locked upstream MASTER; target-only identity attention and legal label loss."""
from pathlib import Path
import hashlib
import importlib.util
import sys
import types
import numpy as np
import torch

VENDOR = Path(__file__).resolve().parents[2] / 'research/round23_6_26_execution/vendor/MASTER'
EXPECTED = {'master.py': 'e5e798952c9636c8c43fc91cff51a6dbbebc009b99d16a1ae090be32d0744fe4',
            'base_model.py': '116322431b83194234390a5929cd98883afb8fac2b1eaffb3d01556b9f5d691a'}

def upstream():
    for name, digest in EXPECTED.items():
        assert hashlib.sha256((VENDOR/name).read_bytes()).hexdigest() == digest
    if '_locked_master_author' in sys.modules:
        return sys.modules['_locked_master_author']
    spec = importlib.util.spec_from_file_location('_locked_master_base', VENDOR/'base_model.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    prior = sys.modules.get('base_model')
    sys.modules['base_model'] = base
    try:
        spec = importlib.util.spec_from_file_location('_locked_master_author', VENDOR/'master.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['_locked_master_author'] = module
    finally:
        if prior is None:
            del sys.modules['base_model']
        else:
            sys.modules['base_model'] = prior
    return module

def identity_spatial_forward(self, x):
    # Exact author SAttention operations, with only fixed diagonal logits mask.
    x = self.norm1(x)
    q, k, v = (layer(x).transpose(0, 1) for layer in (self.qtrans, self.ktrans, self.vtrans))
    dim = int(self.d_model / self.nhead)
    outputs = []
    for i in range(self.nhead):
        sl = slice(i*dim, None if i == self.nhead-1 else (i+1)*dim)
        qh, kh, vh = q[:, :, sl], k[:, :, sl], v[:, :, sl]
        logits = torch.matmul(qh, kh.transpose(1, 2)) / self.temperature
        mask = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
        attention = torch.softmax(logits.masked_fill(~mask, float('-inf')), dim=-1)
        if self.attn_dropout:
            attention = self.attn_dropout[i](attention)
        outputs.append(torch.matmul(attention, vh).transpose(0, 1))
    xt = self.norm2(x + torch.concat(outputs, dim=-1))
    return xt + self.ffn(xt)

def create(arm, seed):
    if arm not in ('M0', 'M1'):
        raise ValueError(arm)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    m = upstream().MASTER(d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
        T_dropout_rate=.5, S_dropout_rate=.5,
        gate_input_start_index=158, gate_input_end_index=221, beta=5)
    if arm == 'M0':
        m.layers[3].forward = types.MethodType(identity_spatial_forward, m.layers[3])
    return m

def legal_loss(prediction, targets, legal):
    """All nodes already forwarded; target availability masks loss only."""
    mask = legal.bool() & torch.isfinite(targets)
    indices = torch.where(mask)[0]
    if len(indices) < 2:
        return None, {'legal': len(indices), 'used': 0, 'reason': 'LESS_THAN_TWO'}
    values = targets[indices]
    p = int(.025 * len(values))
    order = values.sort().indices
    if p:
        indices = indices[order[p:-p]]
        values = targets[indices]
    std = values.std()
    if not torch.isfinite(std) or std <= 0:
        return None, {'legal': int(mask.sum()), 'used': 0, 'reason': 'ZERO_STD'}
    standardized = (values-values.mean()) / std
    return ((prediction[indices]-standardized)**2).mean(), {
        'legal': int(mask.sum()), 'used': len(indices), 'reason': 'TRAIN'}
