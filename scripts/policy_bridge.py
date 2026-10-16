"""Explicit deterministic BC mean -> official stochastic SAC initialization."""
import copy
import hashlib
import torch


def fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_il(bc):
    frozen = copy.deepcopy(bc).eval()
    frozen.requires_grad_(False)
    assert fingerprint(frozen) == fingerprint(bc)
    assert not {p.data_ptr() for p in frozen.parameters()} & {p.data_ptr() for p in bc.parameters()}
    return frozen


def initialize_sac_mean(bc, sac):
    if not torch.equal(sac.action_scale, torch.ones_like(sac.action_scale)) or not torch.equal(sac.action_bias, torch.zeros_like(sac.action_bias)):
        raise ValueError('This bridge requires the verified normalized [-1,1] action space')
    variance_before = fingerprint(sac.fc_logstd)
    for index in (0, 2, 4):
        sac.backbone[index].load_state_dict(bc.net[index].state_dict())
    sac.fc_mean.load_state_dict(bc.net[6].state_dict())
    assert fingerprint(sac.fc_logstd) == variance_before, 'Do not copy a BC variance branch'
    assert not {p.data_ptr() for p in sac.parameters()} & {p.data_ptr() for p in bc.parameters()}


def check_frozen(frozen, optimizer, expected_hash):
    parameters = list(frozen.parameters())
    assert all(not module.training for module in frozen.modules())
    assert not any(p.requires_grad or p.grad is not None for p in parameters)
    assert not {id(p) for p in parameters} & {id(p) for group in optimizer.param_groups for p in group['params']}
    assert fingerprint(frozen) == expected_hash
