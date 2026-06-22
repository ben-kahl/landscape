# Error while calling W&B API: permission denied

- Source: https://github.com/wandb/wandb/issues/673
- Repo: wandb/wandb · Issue #673 · State: closed (closed 2021-02-22)
- Labels: none
- Topic: auth · Difficulty: med

## Report

Hello, 

I am training a model on a remote server and I got this error after running on the terminal `wandb login` and running the training script itself. 

```
wandb: Tracking run with wandb version 0.8.15
wandb: Run data is saved locally in wandb/run-20191107_074736-rjih2419
wandb: ERROR Error while calling W&B API: permission denied (<Response [403]>)
wandb: ERROR Launch exception: Permission denied to access wandb/Dressyou Recommender system/rjih2419
wandb: ERROR To disable wandb syncing set WANDB_MODE=dryrun
wandb: ERROR W&B process (PID 19599) did not respond
wandb: ERROR W&B process failed to launch, see: wandb/debug.log
Traceback (most recent call last):
  File "script/train.py", line 168, in <module>
    wandb.init(entity="wandb", project="Dressyou Recommender system")
  File "/home/amal/miniconda3/envs/all_dep_cuda/lib/python3.7/site-packages/wandb/__init__.py", line 1051, in init
    _init_headless(run)
  File "/home/amal/miniconda3/envs/all_dep_cuda/lib/python3.7/site-packages/wandb/__init__.py", line 288, in _init_headless
    "W&B process failed to launch, see: {}".format(path))
wandb.run_manager.LaunchError: W&B process failed to launch, see: wandb/debug.log
```
Do you have an idea how to solve this please ?

## Discussion / resolution

**issue-label-bot:** Issue Label Bot is not confident enough to auto-label this issue. See [dashboard](https://mlbot.net/data/wandb/client) for more details.

**vanpelt:** Hey @bamal it looks like you're trying to launch runs into the "wandb" team.  The quickest fix is to set entity="bamal" (or your wandb username) when you call wandb.init

**radoye:** @vanpelt This indeed solves the problem. It is a bit confusing when the `sample-project` instructions fail, though. Maybe mention this there?

**shawnlewis:** Where did you get the sample project instructions? From the UI? Can you paste what you're seeing?

It looks like that run was from an old version of wandb, 0.8.15. The latest version shouldn't have this issue.

**radoye:** Sure, here's the output

```bash
wandb: Appending key for api.wandb.ai to your netrc file: /home/rrs/.netrc
Successfully logged in to Weights & Biases!

[rrs@/home/rrs/tf.sif]-[jobs:0]-[~/docks/dev/learn/wandb/tutorial]
$ python tutorial.py
Using TensorFlow backend.
2020-04-14 16:33:47.382557: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
2020-04-14 16:33:47.383710: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
wandb: Tracking run with wandb version 0.8.31
wandb: Run data is saved locally in wandb/run-20200414_233348-2kd35ww6
wandb: ERROR Error while calling W&B API: permission denied (<Response [403]>)
wandb: ERROR Launch exception: Permission denied to access wandb/set-run-name-to-id/2kd35ww6
wandb: ERROR To disable wandb syncing set WANDB_MODE=dryrun
wandb: ERROR W&B process (PID 16419) did not respond
wandb: ERROR W&B process failed to launch, see: wandb/debug.log
Traceback (most recent call last):
  File "tutorial.py", line 33, in <module>
    wandb.init(config=hyperparameter_defaults)#, entity="rrs")
  File "/home/rrs/.local/lib/python3.6/site-packages/wandb/__init__.py", line 1090, in init
    _init_headless(run)
  File "/home/rrs/.local/lib/python3.6/site-packages/wandb/__init__.py", line 306, in _init_headless
    "W&B process failed to launch, see: {}".format(path))
wandb.run_manager.LaunchError: W&B process failed to launch, see: wandb/debug.log

[rrs@/home/rrs/tf.sif]-[jobs:0]-[~/docks/dev/learn/wandb/tutorial]
```

Notes: 
- Installed ~10 mins ago
- My TF environment is in a singularity container (so `pip install --user` was used)
- C-c C-v the instructions from the `sample-project` page under my Projects on wandb

**vanpelt:** Hey @radoye that's a permission error.  You're trying to save results to `entity="wandb", project="set-run-name-to-id"` which you don't have permission to.  In your call to init, add `entity="YOUR_USERNAME"` and try again.

…(further comments trimmed)
