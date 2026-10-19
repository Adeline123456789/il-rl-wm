# Adapted from official ManiSkill SAC at 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3.
# Apache-2.0; original source and licenses in vendor/maniskill/.
# Keep official actor, twin-Q, entropy loss, Polyak update and collection loop.
# Changes: explicit sparse continuing-task protocol, evidence/replay checkpoints,
# fixed validation seeds and optional independent BC mean initialization.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from scripts.sac_evidence import RawLabels, append_json, evaluate_policy, save_snapshot, restore_new_segment
from scripts.expert_replay import ExpertReplay
from scripts.interaction_counter import InteractionCounter, CountedEnv
from scripts.policy_bridge import freeze_il, fingerprint, initialize_sac_mean, check_frozen
from scripts.train_bc import Actor as BCActor

from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs


@dataclass
class Args:
    output: str = ""
    retention_config: Optional[str] = None
    actor_init_checkpoint: Optional[str] = None
    actor_init_kind: str = 'mlp'
    initial_log_std: Optional[float] = None
    matched_eval_resets: bool = False
    bc_checkpoint: Optional[str] = None
    expert_dataset: Optional[str] = None
    expert_audit: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    wm_checkpoint: Optional[str] = None
    task_value_checkpoint: Optional[str] = None
    critic_layer_norm: bool = False
    """U1 only: LayerNorm before each critic hidden ReLU; see U1_CHANGE.md."""
    inherited_training_interactions: int = 0
    max_episode_steps: int = 200
    eval_seed: int = 10000
    torch_threads: int = 4
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = 1
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in real environment transitions (upstream implementation semantics)"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.raw_terminated = torch.zeros_like(self.dones, dtype=torch.bool)
        self.raw_truncated = torch.zeros_like(self.dones, dtype=torch.bool)
        self.episode_ids = torch.zeros_like(self.dones, dtype=torch.int64)
        self.timesteps = torch.zeros_like(self.dones, dtype=torch.int32)
        self.action_sources = torch.full_like(self.dones,3,dtype=torch.int8)
        self.planner_selected = torch.full_like(self.dones,-1,dtype=torch.int16)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, raw_terminated, raw_truncated, episode_ids, timesteps, action_sources=None, planner_selected=None):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        for name, value in [('raw_terminated', raw_terminated), ('raw_truncated', raw_truncated), ('episode_ids', episode_ids), ('timesteps', timesteps)]:
            getattr(self, name)[self.pos] = value.to(self.storage_device)
        self.action_sources[self.pos]=3 if action_sources is None else action_sources.to(self.storage_device)
        self.planner_selected[self.pos]=-1 if planner_selected is None else planner_selected.to(self.storage_device)
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

    def sample_sequences(self,batch_size):
        count=self.per_env_buffer_size if self.full else self.pos
        i=torch.randint(count,(batch_size,),device=self.storage_device);e=torch.randint(self.num_envs,(batch_size,),device=self.storage_device);j=(i+1)%count
        valid=(self.episode_ids[i,e]==self.episode_ids[j,e])&(self.timesteps[j,e]==self.timesteps[i,e]+1)&~self.raw_truncated[i,e]
        i,j,e=i[valid],j[valid],e[valid]
        return dict(states=torch.stack([self.obs[i,e],self.next_obs[i,e],self.next_obs[j,e]],1).to(self.sample_device),
                    actions=torch.stack([self.actions[i,e],self.actions[j,e]],1).to(self.sample_device),
                    rewards=torch.stack([self.rewards[i,e],self.rewards[j,e]],1).to(self.sample_device))

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, layer_norm=False):
        super().__init__()
        dims=[np.array(env.single_observation_space.shape).prod()+np.prod(env.single_action_space.shape),256,256,256]
        layers=[]
        for source,target in zip(dims[:-1],dims[1:]):
            layers.append(nn.Linear(source,target))
            if layer_norm:layers.append(nn.LayerNorm(target,eps=1e-6))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256,1))
        self.net=nn.Sequential(*layers)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
        value = float(scalar_value)
        if not np.isfinite(value):
            raise FloatingPointError(f'Nonfinite metric {tag}: {value}')
        append_json(output / 'metrics.jsonl', dict(step=step, metric=tag, value=value))
    def close(self):
        self.writer.close()

def validate_args(args):
    """Validate the registered SAC interfaces before constructing any environment."""
    assert args.actor_init_kind in {'mlp','act'}
    if args.retention_config:
        assert args.actor_init_checkpoint and args.actor_init_kind == 'act'
        assert not (args.expert_dataset or args.resume_checkpoint or args.wm_checkpoint or args.bc_checkpoint)
    assert not (args.actor_init_checkpoint and (args.bc_checkpoint or args.resume_checkpoint or args.wm_checkpoint)), 'Offline ACT initialization is a separate no-planning experiment'
    assert bool(args.expert_dataset) == bool(args.expert_audit)
    assert bool(args.wm_checkpoint)==bool(args.task_value_checkpoint)
    if args.wm_checkpoint:
        assert args.resume_checkpoint and args.expert_dataset and args.bc_checkpoint
    if args.resume_checkpoint:
        assert args.autotune and args.learning_starts==0, 'Resume as a new segment with retained replay and immediate updates'
    if args.expert_dataset:
        assert args.batch_size % 2 == 0, 'Expert mixture requires exactly 50% expert samples'
    assert args.output and not args.track and not args.capture_video and not args.save_trajectory
    assert not args.evaluate and args.checkpoint is None, 'Use explicit evaluation/restart entry; no implicit optimizer reset'
    assert not args.partial_reset and not args.eval_partial_reset and args.bootstrap_at_done == 'always'
    assert args.env_vectorization == 'gpu' and args.cuda
    assert args.env_id in {'PickCube-v1','PegInsertionSide-v1','StackCube-v1'}
    assert args.num_envs > 0 and args.training_freq % args.num_envs == 0
    assert args.total_timesteps % args.training_freq == 0 and args.buffer_size >= args.training_freq
    assert args.training_freq * args.utd >= 1 and args.policy_frequency == 1
    assert args.num_eval_steps % args.max_episode_steps == 0 and args.eval_freq > 0
    assert args.total_timesteps >= args.learning_starts


if __name__ == "__main__":
    args = tyro.cli(Args)
    validate_args(args)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'metrics.jsonl').exists(), 'Do not append a new run to old evidence'
    torch.set_num_threads(args.torch_threads)
    (output / 'sac_config.json').write_text(json.dumps(vars(args), indent=2))
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", reward_mode="sparse", max_episode_steps=args.max_episode_steps)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    interaction_counter = InteractionCounter(output / 'interactions.json')
    envs = CountedEnv(envs, interaction_counter, 'training')
    eval_envs = CountedEnv(eval_envs, interaction_counter, 'validation')
    raw_env = RawLabels(envs)
    envs = ManiSkillVectorEnv(raw_env, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id,  env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id,  env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["sac", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    from scripts.sac_actor_initialization import initialize_actor
    actor, actor_initialization = initialize_actor(actor, args.actor_init_checkpoint,
        args.actor_init_kind, args.initial_log_std)
    frozen_il = None
    if args.bc_checkpoint:
        # Loading BC must not shift fresh critic/variance initialization or rollout RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            bc = BCActor(int(np.prod(envs.single_observation_space.shape)), int(np.prod(envs.single_action_space.shape))).to(device)
            bc.load_state_dict(torch.load(args.bc_checkpoint, map_location=device)['actor'])
        frozen_il = freeze_il(bc)
        teacher_hash = fingerprint(frozen_il)
        initialize_sac_mean(bc, actor)
        del bc
    qf1 = SoftQNetwork(envs,args.critic_layer_norm).to(device)
    qf2 = SoftQNetwork(envs,args.critic_layer_norm).to(device)
    qf1_target = SoftQNetwork(envs,args.critic_layer_norm).to(device)
    qf2_target = SoftQNetwork(envs,args.critic_layer_norm).to(device)
    if args.actor_init_checkpoint or args.initial_log_std is not None:
        actor_initialization.update(actor_hash=fingerprint(actor),critic1_hash=fingerprint(qf1),
            critic2_hash=fingerprint(qf2),critic_layer_norm=args.critic_layer_norm,
            bc_checkpoint=args.bc_checkpoint,teacher_calls_in_sac_updates=False)
        (output/'actor_initialization.json').write_text(json.dumps(actor_initialization,indent=2))
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )


    expert = ExpertReplay(args.expert_dataset, args.expert_audit, device) if args.expert_dataset else None
    if expert is not None:
        assert expert.manifest['env_info']['env_id'] == args.env_id
        assert expert.manifest['env_info']['env_kwargs']['control_mode'] == args.control_mode
        assert expert.data['obs'].shape[1:] == envs.single_observation_space.shape
        (output / 'expert_manifest.json').write_text(json.dumps(dict(**expert.manifest, expert_fraction=0.5),indent=2))
    sampled_expert = 0
    sampled_online = 0
    retention = None
    if args.retention_config:
        from scripts.act_retention import MeanRetention
        retention = MeanRetention.from_config(args.retention_config, device, output)

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.eval_seed)
    parent_u0=None;episode_start=0
    if args.resume_checkpoint:
        args.inherited_training_interactions,episode_start,parent_u0=restore_new_segment(args.resume_checkpoint,actor,qf1,qf2,qf1_target,qf2_target,actor_optimizer,q_optimizer,log_alpha,a_optimizer,rb,frozen_il,output,args)
        alpha=log_alpha.exp().item()
    if frozen_il is not None:
        with torch.no_grad():
            if not args.resume_checkpoint:torch.testing.assert_close(frozen_il(obs), actor.get_eval_action(obs), rtol=0, atol=0)
        check_frozen(frozen_il, actor_optimizer, teacher_hash)
        (output / 'bridge.json').write_text(json.dumps(dict(initial_mean_bit_exact=None if args.resume_checkpoint else True, actor_from_resume=bool(args.resume_checkpoint), frozen_hash=teacher_hash)))
    u0=None
    if args.wm_checkpoint:
        from scripts.u0_modules import U0
        u0=U0(args,actor,frozen_il,expert,device,output)
        if parent_u0 is not None:u0.restore(parent_u0)
    episode_ids = episode_start+torch.arange(args.num_envs, device=device, dtype=torch.int64)
    timesteps = torch.zeros(args.num_envs, device=device, dtype=torch.int32)
    validation_interactions = 0
    started_at = time.perf_counter()
    parameter_count=sum(p.numel() for m in [actor,qf1,qf2] for p in m.parameters())+int(args.autotune)+(0 if u0 is None else u0.parameters_count)
    assert parameter_count<10000000
    (output / 'model.json').write_text(json.dumps(dict(trainable_parameters=parameter_count,additional_u0_parameters=0 if u0 is None else u0.parameters_count,frozen_il_parameters=0 if frozen_il is None else sum(p.numel() for p in frozen_il.parameters()))))
    (output / 'sac_config.json').write_text(json.dumps(vars(args),indent=2))
    global_step = 0
    global_update = 0
    learning_has_started = bool(args.resume_checkpoint)

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:
        if (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            validation_interactions += evaluate_policy(actor, eval_envs, args, output, global_step) if u0 is None else u0.evaluate(eval_envs,args,global_step)
            if args.save_model:
                save_snapshot(output, run_name, global_step, global_update, actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer, log_alpha if args.autotune else None, a_optimizer if args.autotune else None, rb, args, frozen_il,u0=None if u0 is None else u0.snapshot())

        # Collect samples from environemnts
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            action_sources=planner_selected=None
            if u0 is not None:
                actions,action_sources,planner_selected=u0.act(obs,global_step)
            elif not learning_has_started:
                actions = 2 * torch.rand(size=envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            assert torch.isfinite(real_next_obs).all() and torch.isfinite(rewards).all()
            assert torch.equal(rewards.float(), raw_env.success.float()), 'Unexpected sparse reward semantics'
            if "final_info" in infos:
                # Auto-reset replaces outside info; raw labels were captured before the wrapper.
                mask = infos['_final_info']
                append_json(output / 'reset_checks.jsonl', dict(step=global_step, count=int(mask.sum()), final_vs_reset_l2=(real_next_obs[mask]-next_obs[mask]).norm(dim=-1).cpu().tolist(), bootstrap_stopped=int(stop_bootstrap[mask].sum())))
                for i in torch.where(mask)[0].tolist():
                    append_json(output / 'train_episodes.jsonl', dict(step=global_step, env=i, episode_id=int(episode_ids[i]), **{k:float(v[i]) for k,v in infos['final_info']['episode'].items()}))
            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap, raw_env.terminated, raw_env.truncated, episode_ids, timesteps,action_sources,planner_selected)
            timesteps += 1
            reset_mask = truncations | terminations
            episode_ids[reset_mask] += args.num_envs
            timesteps[reset_mask] = 0

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            if expert is None:
                data = rb.sample(args.batch_size)
                sampled_online += args.batch_size
            else:
                count = args.batch_size // 2
                online = rb.sample(count)
                expert_batch = expert.sample(count)
                data = ReplayBufferSample(**{k:torch.cat([getattr(online,k),v],dim=0) for k,v in expert_batch.items()})
                sampled_expert += count
                sampled_online += count

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                if retention is not None:
                    retention_mse, retention_coefficient = retention.loss(actor, global_step)
                    actor_loss = actor_loss + retention_coefficient * retention_mse

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            if u0 is not None:u0.update(data,rb,global_update)
        if frozen_il is not None:
            check_frozen(frozen_il, actor_optimizer, teacher_hash)
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if u0 is not None:u0.log(global_step)
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            if retention is not None:
                logger.add_scalar("retention/mse", retention_mse.item(), global_step)
                logger.add_scalar("retention/coefficient", retention_coefficient, global_step)
                logger.add_scalar("retention/samples", retention.samples, global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    validation_interactions += evaluate_policy(actor, eval_envs, args, output, global_step) if u0 is None else u0.evaluate(eval_envs,args,global_step)
    if args.save_model:
        save_snapshot(output, run_name, global_step, global_update, actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer, log_alpha if args.autotune else None, a_optimizer if args.autotune else None, rb, args, frozen_il, final=True,u0=None if u0 is None else u0.snapshot())
    if frozen_il is not None:
        check_frozen(frozen_il, actor_optimizer, teacher_hash)
    (output / 'sampling.json').write_text(json.dumps(dict(expert_samples=sampled_expert,online_samples=sampled_online,expert_fraction=0.5 if expert is not None else 0.0),indent=2))
    if u0 is not None:
        u0.log(global_step)
        assert int(u0.counts.sum())==global_step
        (output/'executed_sources.json').write_text(json.dumps(dict(il=int(u0.counts[0]),planned_sac=int(u0.counts[1]),random_sac=int(u0.counts[2]),wm_updates=u0.wm_updates,task_value_updates=u0.value_updates),indent=2))
    (output / 'result.json').write_text(json.dumps(dict(training_interactions=global_step,inherited_training_interactions=args.inherited_training_interactions, gradient_updates=global_update, validation_interactions=validation_interactions, final_test_evaluated=False, elapsed_seconds=time.perf_counter()-started_at, frozen_il_verified=frozen_il is not None, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()), indent=2))
    interaction_counter.flush()
    if retention is not None:
        (output/'retention_result.json').write_text(json.dumps(dict(updates=retention.updates,
            samples=retention.samples, final_coefficient=retention.coefficient(global_step),
            teacher_calls_in_sac_updates=False, teacher_required_at_deployment=False), indent=2)+'\n')
    assert interaction_counter.counts['training']['completed_transitions'] == global_step
    assert interaction_counter.counts['validation']['completed_transitions'] == validation_interactions
    logger.close()
    envs.close()
    eval_envs.close()
