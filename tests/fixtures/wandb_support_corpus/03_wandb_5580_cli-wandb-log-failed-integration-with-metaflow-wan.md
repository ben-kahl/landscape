# [CLI]: @wandb_log failed integration with metaflow - wandb.errors.AuthenticationError

- Source: https://github.com/wandb/wandb/issues/5580
- Repo: wandb/wandb · Issue #5580 · State: closed (closed 2025-05-13)
- Labels: a:app
- Topic: auth · Difficulty: med

## Report

### Current Behavior

When creating metaflow flow, following https://docs.wandb.ai/guides/integrations/metaflow:
```
export WANDB_API_KEY=<YOUR KEY>
```

```python
from metaflow import FlowSpec, Parameter, step, batch, environment
from wandb.integration.metaflow import wandb_log
import wandb

@wandb_log(datasets=True, models=True, settings=wandb.Settings(api_key=os.getenv("WANDB_API_KEY"))
class WandbExampleFlow(FlowSpec):
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.mid)

  @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }
    )
  @batch(gpu=1)
  @retry(times=2)
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.end)

  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```

Met the following error when it is running the step with @batch:
```bash
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: ERROR Error while calling W&B API: user is not logged in (<Response [401]>)
<flow XXX step train> failed:
    Internal error
Traceback (most recent call last):
  File "/home/code/metaflow/metaflow/cli.py", line 1171, in main
    start(auto_envvar_prefix="METAFLOW", obj=state)
  File "/home/code/metaflow/metaflow/_vendor/click/core.py", line 829, in __call__
    return self.main(args, kwargs)
  File "/home/code/metaflow/metaflow/_vendor/click/core.py", line 782, in main
    rv = self.invoke(ctx)
  File "/home/code/metaflow/metaflow/_vendor/click/core.py", line 1259, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/home/code/metaflow/metaflow/_vendor/click/core.py", line 1066, in invoke
    return ctx.invoke(self.callback, ctx.params)
  File "/home/code/metaflow/metaflow/_vendor/click/core.py", line 610, in invoke
    return callback(args, kwargs)
  File "/home/code/metaflow/metaflow/_vendor/click/decorators.py", line 21, in new

…(trimmed)

## Discussion / resolution

**luisbergua:** Hi @erichhhhho, thanks for reporting this! I have tested with the [code from the docs](https://docs.wandb.ai/guides/integrations/metaflow#decorate-your-flows-and-steps) and it works properly for me. Could you try setting your api key in the script as `os.environ["WANDB_API_KEY"] = <your_key>` and see if you get the same error?

**luisbergua:** Hi @erichhhhho , I wanted to follow up on this request. Please let us know if we can be of further assistance or if your issue has been resolved.

**erichhhhho:** > Hi @erichhhhho, thanks for reporting this! I have tested with the [code from the docs](https://docs.wandb.ai/guides/integrations/metaflow#decorate-your-flows-and-steps) and it works properly for me. Could you try setting your api key in the script as `os.environ["WANDB_API_KEY"] = <your_key>` and see if you get the same error?

Yes, I got the same error. Adding WANDB_API_KEY in local environment variable will not be able to resolve this issue. Basically, it might need @wandb_log to pass the local WANDB_API_KEY into the metaflow step with `@batch`, since the step is running in a different container. Could you try adding step with `@batch`?

**erichhhhho:** Hi @luisbergua , just to follow up, any update on this issue?

**luisbergua:** Hi @erichhhhho, apologies for the delay on this! As you said, the `@batch`step is running  in a different container so that's why you need to explicitly set `WANDB_API_KEY ` again. I'll report this to have this automated, thanks for sharing your feedback!
