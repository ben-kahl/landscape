# [CLI]: `TypeError: Object of type Error is not JSON serializable` when using `wandb.integration.openai.fine_tuning`

- Source: https://github.com/wandb/wandb/issues/7275
- Repo: wandb/wandb · Issue #7275 · State: closed (closed 2024-04-02)
- Labels: c:sdk:integration
- Topic: serialization · Difficulty: easy-med

## Report

### Describe the bug

<!--- Description of the issue below  -->
I followed the instructions from the [guide](https://docs.wandb.ai/guides/integrations/openai), but got the below.

<!--- A minimal code snippet between the quotes below  -->
```python
from wandb.integration.openai.fine_tuning import WandbLogger
WandbLogger.sync()
```

<!-- Stack trace -->
```ipython
wandb: Currently logged in as: foobar. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.5
wandb: Run data is saved locally in /Users/itamar/work/rsgpt-train/wandb/run-20240401_193441-ftjob-123abc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ftjob-123abc
wandb: ⭐️ View project at https://wandb.ai/foobar/OpenAI-Fine-Tune
wandb: 🚀 View run at https://wandb.ai/foobar/OpenAI-Fine-Tune/runs/ftjob-123abc/workspace
wandb: Waiting for the OpenAI fine-tuning job to be finished...
wandb: Fine-tuning finished, logging metrics, model metadata, and more to W&B
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[4], line 1
----> 1 WandbLogger.sync()

File /xxx/wandb/integration/openai/fine_tuning.py:147, in WandbLogger.sync(cls, fine_tune_job_id, openai_client, num_fine_tunes, project, entity, overwrite, wait_for_job_success, **kwargs_wandb_init)
    144     if wait_for_job_success:
    145         fine_tune = cls._wait_for_job_success(fine_tune)
--> 147     cls._log_fine_tune(
    148         fine_tune,
    149         project,
    150         entity,
    151         overwrite,
    152         show_individual_warnings,
    153         **kwargs_wandb_init,
    154     )
    156 if not show_individual_warnings and not any(fine_tune_logged):
    157     wandb.termwarn("No new successful fine-tunes were found")

File /xxx/wandb/integration/openai/fine_tuning.py:236, in WandbLogger._log_fine_tune(cls, fine_tune, project, entity, overwrite, show_individual_warnings, **kwargs_wandb_init)
    233     cls._run.summary["fine_tuned_model"] = fine_tuned_model
    235 # training/validation files and fine-tune details
--> 236 cls._log_artifacts(fine_tune, project,

…(trimmed)

## Discussion / resolution

**umakrishnaswamy:** hey @itamarhaber - this is a known issue that is currently being worked on. we currently have a fix in with plans to include it in an upcoming SDK release, but if you would like to install this dev branch and try it out, feel free to do so using the following:

`pip install git+https://github.com/wandb/wandb.git@openai_ft_fix`

and I can write back into this thread once the fix is merged :)

**itamarhaber:** Hi @umakrishnaswamy - thanks for the prompt reply :) Will keep this open until the fix is merged.
