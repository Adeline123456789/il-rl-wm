# U0 fixed-final inference package

This 1.26 MB package contains SAC actor, H2 world model and pure-task value weights. No IL, replay or optimizer. It deliberately exports the fixed final model: **development validation failed (0/16)**. Do not interpret successful loading as task success.

From the repository root using the locked main environment:

```python
import torch
from scripts.deployment import DeploymentPolicy
package = torch.load("releases/u0-final-001/policy.pt", map_location="cpu")
policy = DeploymentPolicy(package, device="cuda")
# observation: batched float32 state from the manifest's ManiSkill protocol.
with torch.no_grad():
    action = policy.get_eval_action(observation)
```

9 stochastic SAC candidates, H=2, execute one action then replan. Externally seed Torch for repeatability. Package SHA, source checkpoint SHA and protocol are in manifest.json; CPU parity on 64 saved real states passed with the IL constructor forbidden. Main environment and repository modules are required. Full training recovery checkpoints remain separate and are not included here.
