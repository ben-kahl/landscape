# [CLI]: Can't resume run with correct id

- Source: https://github.com/wandb/wandb/issues/5396
- Repo: wandb/wandb · Issue #5396 · State: closed (closed 2024-11-19)
- Labels: c:docs, c:sdk:internal-process, a:sdk, c:sdk:resume
- Topic: resume · Difficulty: med

## Report

### Describe the bug

<!--- Description of the issue below  -->
I'm trying to resume a run that finished successfully. I got the ID from the W&B Web UI and have my training mechanism set up to resume from a checkpoint (so it will spit out correct step numbers, etc). What I want is to log more data to that same run as though it had never stopped.

So I do this:

<!--- A minimal code snippet between the quotes below  -->
```python
run = wandb.init(id="g9jgwkjp", resume="must")
```

But then I get this:

<!--- A full traceback of the exception in the quotes below -->
```shell
wandb: ERROR resume='must' but run (g9jgwkjp) doesn't exist
Traceback (most recent call last):
  File "/home/davidg/.pycharm_helpers/pydev/pydevconsole.py", line 364, in runcode
    coro = func()
  File "<input>", line 204, in <module>
  File "/home/davidg/.virtualenvs/learning/lib/python3.10/site-packages/wandb/sdk/wandb_init.py", line 1144, in init
    run = wi.init()
  File "/home/davidg/.virtualenvs/learning/lib/python3.10/site-packages/wandb/sdk/wandb_init.py", line 773, in init
    raise error
wandb.errors.UsageError: resume='must' but run (g9jgwkjp) doesn't exist
```


### Additional Files

This ID does exist, I can confirm by looking in the web UI:
![image](https://user-images.githubusercontent.com/4443482/233529735-adb48258-c8c5-40e7-bf07-2ce9a0526840.png)



### Environment

WandB version: 0.14.0

OS: Windows 11

Python version: 3.10



### Additional Context

Perhaps I misunderstand what `resume` is supposed to do, the docs are a bit unclear. It seems like the assumption is that I'd only ever like to resume a crashed run. But what if I finished a run for a few epochs, and now I want to run it for a few epochs more?

I also tried `resume="allow"` and that doesn't seem to do what I expect.

## Discussion / resolution

**davidgilbertson:** In case my intent isn't clear, I've managed to continue training and just log a second run to W&B with the same name, and set a very similar colour so that it looks like one run on the charts:

![image](https://user-images.githubusercontent.com/4443482/233551079-f8f1382d-e857-4341-be3b-3b27bd7f570d.png)

What I'm trying to achieve is all the same logged values, but in one run instead of two. Is that possible?

**thanos-wandb:** Hi @davidgilbertson thanks for reporting this issue. Could you please also provide the `entity` and `project` too? It will attempt to use the default values, and it won't find this specific run. Would the following work for you?

`run = wandb.init(entity="gilbertson-david", project="rich-vocab-2d", id="g9jgwkjp", resume="must")`

**davidgilbertson:** Thanks Thanos, I'll try that next time. May I suggest updating [the docs](https://docs.wandb.ai/guides/runs/resuming) to reflect this, currently they state:
> when you resume (if you want to be sure that it is resuming, you do `wandb.init(id=run_id, resume="must")`.

**thanos-wandb:** Thanks @davidgilbertson for the feedback. I have made a Docs update request as that would be indeed useful to include. Please let me know if you're still having any issues with resuming your run.

**JohannesTheo:** Hey @thanos-wandb, I just ran into the very same issue and wasted some time figuring this out myself before I found your conversation. May I suggest to give the docs update request a bumb? :) It's such an easy fix but not yet in the docs as far as I can tell?

**thanos-wandb:** Hi @JohannesTheo sorry to hear about the trouble this issue caused you. I've just checked the internal ticket I created for the Docs update request, and it appears that the pull request (PR) has not been merged yet; it's currently under review. The Docs team collaborated with our SDK team to rewrite this part, and they will be overhauling the entire "resuming run" section.

**JohannesTheo:** All good :) Great to hear it's going to be in the docs soon.

**stellarpower:** I also feel the way runs, sweeps, projects, and switching back and forth between IDs and names gets confusing rapidly. The web interface makes it quite difficult to find a run/sweep's ID (I have to copy it from the URL often), and uses names instead; yet the code examples use the ID. Does a run belong to a sweep? Is it just loosely associated with one? If I'm logged in and have access to the API (Which is nice without having to add code to do this explicitly), then needing to specify the username half of the time seems to defeat the reason for being logged in already.

If the docs use the run ID, and I am opening it via a unique ID, I would expect that to be sufficient to get a handle on it. If it expects a path-like name, that needs separators, then this would be as expected. The examples seem to flip between the two - if I open it from the API, I'm expected to give a string separated by slashes. I think it would make more s

…(trimmed)
