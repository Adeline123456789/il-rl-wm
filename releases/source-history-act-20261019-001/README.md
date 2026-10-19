# Incremental original experiment history

This archive extends `releases/source-history-20261016-001` through local source head `9f1536cf857892f0e4db301377b05db4e7c9518c`. It preserves all 45 distinct original code commits found in the 145 GPU-job manifests at capture, including the running U2 code commit. Both bundles were restored in a fresh bare repository without network access; `git fsck --full` passed. The current U2 run was not stopped or restarted.

The increment is 148,878,288 bytes split into 4 MiB pieces; its exact file list, SHA256 values and prerequisite commit are in `manifest.json`. The previous bundle is required. This is committed source/evidence history, including prior committed archive parts; it is not a dataset or training-resume checkpoint. Dirty execution code is preserved separately in each run's source snapshot and patch. The historically missing plotting-script version remains documented in `artifacts/source-history-preflight-001/unresolved.json`; no replacement bytes are invented.

From the repository root, run the restoration script below with a new output directory:

```bash
python releases/source-history-act-20261019-001/restore.py /tmp/il-rl-wm-restored-history
```

The script validates every part and both aggregate bundles, restores both into a new bare repository, checks all recorded original code commits, and checks object integrity. It does not use the network or GPU. To read code after restore:

```bash
git --git-dir=/tmp/il-rl-wm-restored-history/history.git show 922077e293ee74f14ec37a389caa5bd2a2725daa:scripts/act_retention.py
```

The history capture excludes publication of its own parts, avoiding self-inclusion. While U2 and the registered final evaluations run, local source HEAD remains frozen; evidence can be published as a Git tree snapshot without changing that source HEAD. Final delivery must recheck that every later experiment code_commit is covered; additional source commits, if required by a real fix, need a separate small increment before claiming full coverage.

2026-10-19 终态覆盖：214份实际作业manifest涉及46个code_commit，终态审计均在原base+increment离线恢复仓库验证存在，详见final_execution_coverage.json。归档无需递归重打；每次最终评测的实际source tar/patch单独交付。原145作业/45commit检验是归档生成时的快照，不替代本次终态覆盖；旧未提交plot文件字节缺口仍保留。
