# [Bug]: wandb hangs at init if entity is different to logged in user

- Source: https://github.com/wandb/wandb/issues/11532
- Repo: wandb/wandb · Issue #11532 · State: closed (closed 2026-03-30)
- Labels: ty:bug, a:sdk
- Topic: login · Difficulty: med-hard

## Report

### Describe the bug

wandb-0.19.10

If the logged in user has a different entity to that provided to wandb.init() then wandb simply hangs.

A message is printed:
`Currently logged in as: ........ (entity....) to ...... Use `wandb login --relogin` to force relogin`
but this is also printed with a successful init and gives no indication as to the error.

My preference would be to throw an exception in this case with a reasonable error message.

## Discussion / resolution

**willtryagain:** Hello @JamieTattersallZenseact ,
Could you please share a few additional details so we can narrow this down?
* Are you using wandb.ai (SaaS) or a Dedicated Cloud instance?
* Would you mind sharing debug.log and debug-internal.log for an affected run? These files are usually located under wandb/run-<date>_<time>-<run-id>/logs in the same directory where you’re running your code.
* Could you share everything printed between calling wandb.init() and the point where it hangs?

**exalate-issue-sync:** Aman Atman commented: 
Hi,

We wanted to follow up with you regarding your support request as we have not heard back from you. Please let us know if we can be of further assistance or if your issue has been resolved.

Best,
Weights & Biases

**exalate-issue-sync:** Aman Atman commented: 
Hi, since we have not heard back from you we are going to close this request. If you would like to re-open the conversation, please let us know!
