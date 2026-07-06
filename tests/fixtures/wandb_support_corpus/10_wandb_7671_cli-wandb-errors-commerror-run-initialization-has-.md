# [CLI]: wandb.errors.CommError: Run initialization has timed out after 90.0 sec. “Is it wandb'server error?”

- Source: https://github.com/wandb/wandb/issues/7671
- Repo: wandb/wandb · Issue #7671 · State: closed (closed 2024-09-03)
- Labels: a:app, s:nexus-fix
- Topic: init · Difficulty: med

## Report

### Describe the bug

<!--- Description of the issue below  -->

wandb: Network error (ReadTimeout), entering retry loop.
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: ERROR Run initialization has timed out after 90.0 sec. 


### Additional Files

_No response_

### Environment

WandB version:wandb, version 0.16.6

OS:python

Python version:3.8

Versions of relevant libraries:


### Additional Context

_No response_

## Discussion / resolution

**umakrishnaswamy:** hey @Echhoo - few questions to help me dig into this:

- are you running wandb locally or utilizing https://wandb.ai/ ?
- Could you go ahead and try to refresh your login credentials:
```
rm ~/.netrc
wandb login --relogin --host=<host_url>
```

- do you have any firewalls set up? or are you trying to access wandb through a proxy? would love any details about this as well

**Echhoo:** > hey @Echhoo - few questions to help me dig into this:
> 
> * are you running wandb locally or utilizing https://wandb.ai/ ?
> * Could you go ahead and try to refresh your login credentials:
> 
> ```
> rm ~/.netrc
> wandb login --relogin --host=<host_url>
> ```
> 
> * do you have any firewalls set up? or are you trying to access wandb through a proxy? would love any details about this as well

I use wandb utilizing in  https://wandb.ai/  and after I try this commend 
'''
rm ~/.netrc
wandb login --relogin --host=<host_url>
'''
Wandb still cannot be used and reports the following error message：
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: \ Waiting for wandb.init()...
wandb: | Waiting for wandb.init()...
wandb: / Waiting for wandb.init()...
wandb: - Waiting for wandb.init()...
wandb: ERROR Run initialization has timed out after 90.0 sec.

**umakrishnaswamy:** hey @Echhoo - <host_url> corresponds to "https://api.wandb.ai" 

Could you try the above steps with that url replaced? please let me know if you reach the hanging then

**umakrishnaswamy:** Hi @Echhoo, since we have not heard back from you we are going to close this request. If you would like to re-open the conversation, please let us know!

**exalate-issue-sync:** WandB Internal User commented: 
umakrishnaswamy commented: 
hey @Echhoo - few questions to help me dig into this:

- are you running wandb locally or utilizing https://wandb.ai/ ?
- Could you go ahead and try to refresh your login credentials:
```
rm ~/.netrc
wandb login --relogin --host=<host_url>
```

- do you have any firewalls set up? or are you trying to access wandb through a proxy? would love any details about this as well

**Echhoo:** > WandB Internal User commented: umakrishnaswamy commented: hey @Echhoo - few questions to help me dig into this:
> 
> * are you running wandb locally or utilizing https://wandb.ai/ ?
> * Could you go ahead and try to refresh your login credentials:
> 
> ```
> rm ~/.netrc
> wandb login --relogin --host=<host_url>
> ```
> 
> * do you have any firewalls set up? or are you trying to access wandb through a proxy? would love any details about this as well

I still failed to connect，but when i use ping：
![image](https://github.com/wandb/wandb/assets/52142940/16310e30-bc43-4

…(trimmed)
