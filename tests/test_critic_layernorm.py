from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scripts.train_sac import SoftQNetwork


def environment():
    return SimpleNamespace(single_observation_space=SimpleNamespace(shape=(42,)),
                           single_action_space=SimpleNamespace(shape=(4,)))


def test_original_checkpoint_outputs_unchanged_on_real_inputs():
    torch.set_num_threads(2)
    snapshot=torch.load('releases/u0-analysis-precollapse-001/weights.pt',map_location='cpu')
    critics=[]
    for key in ['qf1','qf2']:
        q=SoftQNetwork(environment());q.load_state_dict(snapshot[key]);critics.append(q)
    with np.load('artifacts/f-u0-value-drift-002/fixed_batches.npz') as data,np.load('artifacts/f-u0-value-drift-001/values.npz') as expected:
        for source in ['expert','online']:
            obs=torch.from_numpy(data[source+'_obs']);action=torch.from_numpy(data[source+'_actions'])
            with torch.no_grad():actual=torch.minimum(*(q(obs,action) for q in critics)).squeeze(-1).numpy()
            np.testing.assert_array_equal(actual,expected['before_'+source+'_soft_q_recorded'])


def test_layernorm_keeps_dense_initialization_and_restores_optimizer():
    torch.manual_seed(7);plain=SoftQNetwork(environment());plain_rng=torch.get_rng_state()
    torch.manual_seed(7);normalized=SoftQNetwork(environment(),layer_norm=True)
    assert torch.equal(plain_rng,torch.get_rng_state())
    old=[m for m in plain.modules() if isinstance(m,nn.Linear)]
    new=[m for m in normalized.modules() if isinstance(m,nn.Linear)]
    for a,b in zip(old,new):
        assert torch.equal(a.weight,b.weight) and torch.equal(a.bias,b.bias)
    norms=[m for m in normalized.modules() if isinstance(m,nn.LayerNorm)]
    assert len(norms)==3 and all(m.eps==1e-6 for m in norms)
    assert sum(p.numel() for p in normalized.parameters())-sum(p.numel() for p in plain.parameters())==1536
    target=deepcopy(normalized);optimizer=torch.optim.Adam(normalized.parameters(),lr=3e-4)
    obs=torch.randn(8,42);action=torch.randn(8,4,requires_grad=True)
    loss=normalized(obs,action).square().mean();loss.backward()
    assert torch.isfinite(action.grad).all()
    assert all(m.weight.grad is not None and torch.isfinite(m.weight.grad).all() for m in norms)
    optimizer.step()
    with torch.no_grad():
        for online,frozen in zip(normalized.parameters(),target.parameters()):frozen.lerp_(online,.01)
    restored=deepcopy(normalized);restored_optimizer=torch.optim.Adam(restored.parameters(),lr=3e-4)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for model,opt in [(normalized,optimizer),(restored,restored_optimizer)]:
        opt.zero_grad();model(obs,action.detach()).square().mean().backward();opt.step()
    for a,b in zip(normalized.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
