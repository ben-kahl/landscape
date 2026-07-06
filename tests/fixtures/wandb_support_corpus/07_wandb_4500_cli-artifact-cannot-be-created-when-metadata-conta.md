# [CLI]: Artifact cannot be created when metadata contains Tensors with wandb 0.13.5

- Source: https://github.com/wandb/wandb/issues/4500
- Repo: wandb/wandb · Issue #4500 · State: closed (closed 2023-01-13)
- Labels: ty:bug, a:sdk
- Topic: serialization · Difficulty: easy

## Report

### Describe the bug

<!--- Description of the issue below  -->

When I try to create an Artifact with wandb 0.13.5, I got the error
```
TypeError: Object of type Tensor is not JSON serializable
```
, while the same code works with wandb 0.13.4. 

After a quick check, I found the behavior change in `_normalize_metadata`

in 0.13.4:
https://github.com/wandb/wandb/blob/d622ee37b232e54addcd48e9f92d9198a3e2790b/wandb/sdk/wandb_artifacts.py#L92-L99

in 0.13.5:
https://github.com/wandb/wandb/blob/e85ba97972308091d22f34e275cd58fea4b774dc/wandb/sdk/wandb_artifacts.py#L92-L99

In 0.13.4, `json_friendly_val` will check and convert the tensor type, and this is removed in 0.13.5. 

It this change intentionally?

### Additional Files

_No response_

### Environment

WandB version: 0.13.5

OS: Ubuntu 20.04

Python version: 3.8

Versions of relevant libraries:


### Additional Context

_No response_

## Discussion / resolution

**MBakirWB:** Hi @jchengai , thank you for writing in. I will review the change on our end and get back to you.

**konstantinjdobler:** I can confirm the same issue, `wandb==0.13.5` `python==3.10.6` on Ubuntu 20.04 as well. `wandb==0.13.4` works fine.

**RoyHEyono:** I had the same issue, but this was resolved by downgrading to `wandb==0.13.4`
