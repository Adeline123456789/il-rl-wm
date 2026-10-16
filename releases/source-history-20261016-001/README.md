# Original experiment Git history

Concatenate the numbered parts in lexical order and verify the size/SHA256 in manifest.json. This bundle preserves the original local experiment code commits, independently of the GitHub publication commit SHA.

```bash
cat releases/source-history-20261016-001/source-history.bundle.part* > /tmp/il-rl-wm-source-history.bundle
sha256sum /tmp/il-rl-wm-source-history.bundle
git bundle verify /tmp/il-rl-wm-source-history.bundle
git clone /tmp/il-rl-wm-source-history.bundle /tmp/il-rl-wm-original-history
```

The archive was restored in a fresh bare repository without network access, and all code_commit identities found in experiment manifests at capture were verified there. Its source head precedes the progress-upload documentation and archive-parts commit; this avoids recursive inclusion. Check manifest.json for the exact head and verified commits. The bundle contains committed source/evidence history, not ignored datasets, environments, full training replay/checkpoints, or the unresolved historical untracked plotting-script bytes documented in artifacts/source-history-preflight-001/unresolved.json.
