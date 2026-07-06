# [Bug]: Login error `Object has no attribute 'disabled'`

- Source: https://github.com/wandb/wandb/issues/9580
- Repo: wandb/wandb · Issue #9580 · State: closed (closed 2025-03-24)
- Labels: ty:bug, a:sdk
- Topic: login · Difficulty: med

## Report

### Describe the bug

I just `pip install wandb` (version `0.19.8`) and got an error when running `wandb login <API-KEY>`.

```
Traceback (most recent call last):
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/bin/wandb", line 8, in <module>
    sys.exit(cli())
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/click/core.py", line 1161, in __call__
    return self.main(*args, **kwargs)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/click/core.py", line 1082, in main
    rv = self.invoke(ctx)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/click/core.py", line 1697, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/click/core.py", line 1443, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/click/core.py", line 788, in invoke
    return __callback(*args, **kwargs)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/wandb/cli/cli.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/wandb/cli/cli.py", line 246, in login
    wandb.setup(
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/wandb/sdk/wandb_setup.py", line 382, in setup
    return _setup(settings=settings)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/wandb/sdk/wandb_setup.py", line 318, in _setup
    _singleton = _WandbSetup(settings=settings, pid=pid)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/wandb/sdk/wandb_setup.py", line 96, in __init__
    self._settings = self._settings_setup(settings)
  File "/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/ply_env/lib/python3.10/site-packages/wandb/sdk/wandb_setup.py", line 123, in _settings_setup
    s.update_from_workspace_config_file()
  File "/home/GRAMES.POLYMTL.CA/p11

…(trimmed)

## Discussion / resolution

**murnanedaniel:** I also have just started seeing this error.

**ArtsiomWB:** Hey guys! 
Thank you for writing in!

Could you please run `pip show pydantic` and send the output of the cli command?

**NathanMolinier:** The issue appears to have disappeared... I can't reproduce the error now, which is odd. I'll let you know if it occurs again.

**ArtsiomWB:** Thank you for the update!

I'll keep this thread open until next week incase you see this behavior again. If you cannot reproduce it anymore in a few days, we'll go ahead and close it out

**hyeobiiii:** @ArtsiomWB 
I just got the same error when I run `wandb login`, and here's my `pip show pydantic` result.
wandb==0.19.8

```sh
Name: pydantic
Version: 2.10.6
Summary: Data validation using Python type hints
Home-page: 
Author: 
Author-email: Samuel Colvin <s@muelcolvin.com>, Eric Jolibois <em.jolibois@gmail.com>, Hasan Ramezani <hasan.r67@gmail.com>, Adrian Garcia Badaracco <1755071+adriangb@users.noreply.github.com>, Terrence Dorsey <terry@pydantic.dev>, David Montague <david@pydantic.dev>, Serge Matveenko <lig@countzero.co>, Marcelo Trylesinski <marcelotryle@gmail.com>, Sydney Runkle <sydneymarierunkle@gmail.com>, David Hewitt <mail@davidhewitt.io>, Alex Hall <alex.mojaki@gmail.com>, Victorien Plot <contact@vctrn.dev>
License: 
Location: /usr/local/lib/python3.10/dist-packages
Requires: annotated-types, pydantic-core, typing-extensions
Required-by: compressed-tensors, deepspeed, fastapi, gradio, lm-format-enforcer, mistral_common, openai, outlines, vllm, wandb, xgrammar
```

**ArtsiomWB:** Checking our req.txt:
https://github.com/wandb/wandb/blob/530768399c1f912a1147d8b21cc7ce6a48b10d34/requirements_dev.txt#L7C1-L8C1

It looks like we currently only support `pydantic~=2.9`. Could you please uninstall your pydantic, uninstall your wandb, and then install wandb back? This should reinstall the supported version.

**ArtsiomWB:** Hi there, I wanted to follow up on this request. Please let us know if we can be of further assistance or if your issue has been resolved.

**ArtsiomWB:** If you're all set for now, I will close this ticket for tracking purposes. If you have any follow-up questions, feel free to add them here, and it will reopen the thread
