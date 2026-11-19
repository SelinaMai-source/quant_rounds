"""Architecture integration only; never substitutes for real reference/OOF fits."""
import json
from pathlib import Path
import numpy as np
import torch
from core.round26_1.master_model import create, upstream, legal_loss

def run():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = create('M1', 17)
    author = upstream().MASTER(158,256,4,2,.5,.5,158,221,5)
    author.load_state_dict(model.state_dict())
    x = torch.randn(47,8,221)
    # Identical stochastic dropout stream, loss, gradient and optimizer update.
    first = torch.optim.Adam(model.parameters(),lr=1e-5)
    second = torch.optim.Adam(author.parameters(),lr=1e-5)
    losses=[]
    for m,opt in ((model,first),(author,second)):
        torch.manual_seed(901)
        opt.zero_grad()
        y=m(x)
        loss=y.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_value_(m.parameters(),3.)
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[0] == losses[1]
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),author.state_dict().values()))
    m0=create('M0',17);m1=create('M1',17)
    assert all(torch.equal(a,b) for a,b in zip(m0.state_dict().values(),m1.state_dict().values()))
    records=[]
    for arm,m in [('M0',m0),('M1',m1)]:
        m.eval()
        with torch.no_grad():
            p=m(x);perm=torch.randperm(len(x));p2=m(x[perm])[torch.argsort(perm)]
            assert torch.allclose(p,p2,atol=1e-5,rtol=1e-5)
            changed=x.clone();changed[1:]+=3; altered=m(changed)
            if arm=='M0': assert torch.equal(p[:1],altered[:1])
        records.append({'arm':arm,'permutation_max_abs':float((p-p2).abs().max()),'first_node_changed_by_other_nodes':bool(not torch.equal(p[:1],altered[:1]))})
    pred=m1(x);labels=torch.arange(47,dtype=torch.float32);legal=torch.zeros(47,dtype=torch.bool);legal[:9]=True
    loss,audit=legal_loss(pred,labels,legal)
    assert audit['used']==9 and torch.isfinite(loss)
    labels[9:]=float('nan');other,b=legal_loss(pred,labels,legal)
    assert torch.equal(loss,other)
    zero,why=legal_loss(pred,torch.ones(47),legal)
    assert zero is None and why['reason']=='ZERO_STD'
    p=Path(__file__).resolve().parents[2]/'result/round26_1/D5/MODEL_INTEGRATION.json'
    p.write_text(json.dumps({'status':'PASS','scope':'architecture integration; random inputs, NOT real-data training or public reference','M1_author_forward_backward_update_bitexact':True,'M0_M1_initial_parameters_bitexact':True,'label_only_mask_N_lt40_keeps9':audit,'permutation':records,'parameter_count':sum(v.numel() for v in m1.parameters())},indent=2))
    print(p.read_text())
if __name__=='__main__':run()
