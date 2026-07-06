# [CLI] matplotlib Figure object not JSON serializable

- Source: https://github.com/wandb/wandb/issues/2063
- Repo: wandb/wandb · Issue #2063 · State: closed (closed 2024-11-22)
- Labels: ty:bug, c:sdk:media, a:sdk
- Topic: serialization · Difficulty: easy

## Report

**Description**
[You documentation](https://docs.wandb.ai/library/log#matplotlib) says "You can pass a matplotlib pyplot or figure object to `wandb.log()`". However, when I do this, at the moment of the commit, I get the following error.

```python
Traceback (most recent call last):
  File "src/main.py", line 236, in <module>
    main(cfg, train_ns_cfg)
  File "src/main.py", line 149, in main
    trainer.fit(model, datamodule=data_module)
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 499, in fit
    self.dispatch()
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 546, in dispatch
    self.accelerator.start_training(self)
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py", line 73, in start_training
    self.training_type_plugin.start_training(trainer)
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 114, in start_training
    self._results = trainer.run_train()
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 637, in run_train
    self.train_loop.run_training_epoch()
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py", line 560, in run_training_epoch
    self.trainer.logger_connector.log_train_epoch_end_metrics(
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py", line 439, in log_train_epoch_end_metrics
    self.log_metrics(epoch_log_metrics, {})
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py", line 236, in log_metrics
    self.trainer.logger.save()
  File "/cw/liir/NoCsBack/testliir/

…(trimmed)

## Discussion / resolution

**ariG23498:** Hey @rubencart 

I have tried reproducing the ticket. This is the [GIST](https://gist.github.com/ariG23498/dd5c0449166beac5e2bb68a11809d1a3) that is a full-fledged repro of your issue. The wandb dashboard corresponding to the GIST can be found [here]()https://wandb.ai/repro/GH-2063.

**My observations**:
1. Sending a `plt.figure` as a list is harmful as the lists are serialised and the `plt.figure` cannot be serialised.
2. Upon sending the `plt.figure` as a solo entity, wandb complained about `ValueError: min() arg is an empty sequence`. This might be related to the fact that the figure is not filled with anything yet.
3. Later when I created a plot and then sent the `plt` object over to be logged, everything was working perfectly.

I would like for you to comment of the steps that I have taken and let us know if this helps you. We might need to change the docs as you have suggested. Thanks for the ticket! 👍

**rubencart:** I see, thank you! 

It would be useful to me if I could submit `plt.Figure` instances though and not just the `plt` object, because I'm generating multiple figures at once. Maybe I'm not understanding it correctly, but I think that if you can only submit `plt`, you'd need to generate a figure, log it, close it, generate the next one, log it,... and so on.  Is that correct? So then you wouldn't be able to generate in batch and then log in batch?

Also, when I fill the `plt.Figure` object and log it not as a list but as a single entity, I get the following error: `wandb.errors.Error: plotly is required to log interactive plots, install with: pip install plotly or convert the plot to an image with 'wandb.Image(plt)'`. If `plotly` is required as an extra dependency that you need to install for this yourself, maybe it would also make sense to mention this in the docs?

After installing `plotly`, when logging a single (non-empty) `plt.Figure` like in the snippet below, I get the following error (both at the time of the log and at the end of the script).
```python
import matplotlib.pyplot as plt
import seaborn
import torch
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='MMMAli', save_dir='./output/tmp', offline=True)

fig1: plt.Figure = plt.figure()
ax1 = seaborn.heatmap(torch.randn((10, 10)),
                      square=True,
                      vmin=0.0, vmax=1.0,
                      cbar=False,
                      )
plt.subplots_adjust(bottom=0.2, left=0.2, hspace=0.8)
fig1.add_axes(ax1)

wandb_logger.experiment.log({'examples': fig1})
```
```python
Traceback (most recent call last):
  File "src/main.py", line 236, in <module>
    main(cfg, train_ns_cfg)
  File "src/main.py", line 149, in main
    trainer.fit(model, datamodule=data_module)
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 499, in fit
    self.dispatch()
  File "/cw/liir/NoCsBack/testliir/rubenc/miniconda3/envs/alienv/l

…(trimmed)
