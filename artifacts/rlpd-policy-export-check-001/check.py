import json
from pathlib import Path
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from external.rlpd.components import make_components
from external.rlpd.policy import export,FrozenPolicy
source=Path('checkpoints/e3-rlpd-smoke-001/final')
out=Path('releases/rlpd-smoke-001')
m=export(source,out)
policy=FrozenPolicy(out)
ac,_=make_components(m['config'],np.zeros(m['obs_dim'],np.float32),np.zeros(m['action_dim'],np.float32))
ac=flax.serialization.from_bytes(ac,(source/'learner.msgpack').read_bytes())
with np.load(source/'replay.npz') as replay:
 obs=replay['env_obs'].reshape(-1,m['obs_dim'])[:64]
a,_=ac.act(jax.random.PRNGKey(0),ac.actor,jnp.asarray(obs));b=policy.act(jnp.asarray(obs))
np.testing.assert_array_equal(np.array(a),np.array(b))
np.savez('artifacts/rlpd-policy-export-check-001/actions.npz',obs=obs,original=np.array(a),portable=np.array(b))
Path('artifacts/rlpd-policy-export-check-001/result.json').write_text(json.dumps(dict(valid=True,states=len(obs),max_abs_error=float(np.max(np.abs(np.array(a)-np.array(b)))),actor_bytes=m['bytes'],no_critic_or_optimizer_in_package=True),indent=2))
