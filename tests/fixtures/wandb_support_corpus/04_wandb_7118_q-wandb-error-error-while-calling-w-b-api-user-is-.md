# [Q]wandb: ERROR Error while calling W&B API: user is not logged in (<Response [401]>)

- Source: https://github.com/wandb/wandb/issues/7118
- Repo: wandb/wandb · Issue #7118 · State: closed (closed 2024-04-18)
- Labels: a:sdk, c:sdk:login
- Topic: auth · Difficulty: easy-med

## Report

I'm using wandb in Visual Code Studio and for some reason I have issue logging in, I even tried entity= 'myusername' but it didn't work. I checked and I have the latest update of wandb. My project folder location in in Desktop 

```
os.environ["WANDB_API_KEY"] = 'API_KEY'

!wandb login API_KEY
```

/Users/myname/.netrc
```
machine api.wandb.ai
  login user
  password API_KEY
```


when I run wandb.init  I get this error 

```
Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to [the W&B docs](https://wandb.me/wandb-init).
wandb: ERROR Error while calling W&B API: user is not logged in (<Response [401]>)

```

## Discussion / resolution

**fmamberti-wandb:** Hi @qasara, thank you for reaching out. 

Would you mind sharing some additional information to help us troubleshoot this:

- What version of the SDK are you currently running?
- Are you running this as a Jupyter Notebook in VSC (I assume so since you are running !wandb but would like to confirm).
- Do you have a `./wandb` folder created in your project folder? If so, you should have a `./wandb/run-date_time-runid/logs/` folder containing a `debug.log` and `debug-internal.log` files. Could you share those? 

If you are running this in a Notebook, could you try replacing 

```python
!wandb login API_KEY
```

with:

```python
wandb.login(key=os.environ["WANDB_API_KEY"])
```

making sure this is the first command running after `import wandb` in your notebook.

Thanks!

**fmamberti-wandb:** Hi @qasara , I wanted to follow up on this request. Please let us know if we can be of further assistance or if your issue has been resolved.

**fmamberti-wandb:** Hi @qasara , since we have not heard back from you we are going to close this request. If you would like to re-open the conversation, please let us know!
