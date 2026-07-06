# [CLI]: wandb login in GithubAction workflow for unit tests

- Source: https://github.com/wandb/wandb/issues/3845
- Repo: wandb/wandb · Issue #3845 · State: closed (closed 2023-03-08)
- Labels: a:sdk, c:sdk:console
- Topic: auth · Difficulty: med

## Report

### Describe the bug

<!--- Description of the issue below  -->
As my code contains several features using wandb and I want to conduct tests in my CI workflow with GithubActions, I am trying to use `wandb login` in the `.github/workflow` script, however even after setting secret github variables and trying both approaches of `wandb login API_KEY` and setting the environment variable, the login is unsuccessful. Unfortunately, the debug log is in a temp file with github actions and I am not sure how to trace the error exactly. 


Workflow yaml file:
```yaml
name: tests
on:
  push:
    branches:
    - main
  pull_request:
    branches:
    - main
  
jobs:
  pytest:
    name: pytest
    runs-on: ${{ matrix.os }}
    env:
      MPLBACKEND: Agg
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.8', '3.9']
    steps:
    - name: Clone repo
      uses: actions/checkout@v2
    - name: Set up python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install apt dependencies (Linux)
      run: sudo apt-get install unrar
      if: ${{ runner.os == 'Linux' }}
    - name: Install brew dependencies (macOS)
      run: brew install rar
      if: ${{ runner.os == 'macOS' }}
    - name: Install poetry (ubuntu and macOS)
      run: |
        python -m ensurepip
        python -m pip install --upgrade pip
        python -m pip install poetry
    - name: Install dependencies (ubuntu and macOS)
      shell: bash
      run: |
        python -m poetry lock --no-update
        python -m poetry install
    - name: Run pytest checks
      shell: bash
      env: # wandb api key
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      run: |
        poetry run wandb login "$WANDB_API_KEY"
        poetry run python -m pytest -v tests
```

Github Action Error output.
```
Run poetry run wandb login "$WANDB_API_KEY"
  poetry run wandb login "$WANDB_API_KEY"
  poetry run python -m pytest -v tests 
  shell: /usr/bin/bash -e {0}
  env:
    MPLBACKEND: Agg
    pythonLocation: /opt/hostedtoolcache/Python/3.8.12/x64
    LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.8.12

…(trimmed)

## Discussion / resolution

**MBakirWB:** Hi @nilsleh , thank you for writing in.  

Can you uninstall and reinstall wandb and see if that helps? If not, please respond to the following:

What are the contents of your .netrc file [ cat ~/.netrc  ]
If you set the WANDB_API_KEY env var you don't need wandb.login . Can you run this without the wandb.login call?

**nilsleh:** Installation happens in the virtual environment of github Actions, so I'm not sure how I would uninstall and reinstall wandb there. I have tried both just setting the env variable or wandb login.

If I change the workflow yaml file like this

```yaml
- name: Run pytest checks
      shell: bash
      env: # wandb api key
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      run: |
        cat ~/.netrc
        poetry run python -m pytest -v tests
```
I get the following:
`cat: /home/runner/.netrc: No such file or directory`

If I just run wandb login:
```yaml
- name: Run pytest checks
      shell: bash
      env: # wandb api key
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      run: |
        poetry run python -m pytest -v tests
```
I get the following:
`wandb.errors.UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key])`

**ayulockin:** Hey @nilsleh can you give this method a try: https://github.com/ayulockin/wandb2kaggle/blob/65203f07249b1e52bec153bab10fc7524b3b0675/.github/workflows/main.yml#L18

These are the secrets: 
<img width="1181" alt="image" src="https://user-images.githubusercontent.com/31141479/176097549-b4b3437b-07fc-4a47-be03-c8486a0ec98c.png">

**nilsleh:** That worked! Thank you both so much for the fast response!

**nilsleh:** Actually, the tests all run and pass now. However, I receive `Error: Process completed with exit code 255.`. This did not happen before.

this is the error being printed:
```
with open(summary_path, "w") as f:
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/testp78z0eby/wandb/run-20220628_100117-8xtt70eu/files/wandb-summary.json'
wandb: ERROR Internal wandb error: file data was not synced
```

**vanpelt:** Hmm, it seems like the runner is removing the temporary directory out from underneath the script.  I'm not sure about your specific tests, but you might want to try setting the `WANDB_MODE=off` environment variable.  This will stub out all of the wandb calls and not actually hit any of our api's which is usually what you want for automated unit tests.

**nilsleh:** If I do the following:
```yaml
- name: Run pytest checks
      shell: bash
      env:
        WANDB_MODE: offline
        WANDB_API: ${{ secrets.WANDB_API_KEY }}
      run: |
        poetry run wandb login "$WANDB_API"
        poetry run python -m pytest -v tests
```
the `Error: Process completed with exit code 255` persists. Basically, the pytests I constructed are to launch a training job and then retrieve configurations and metrics from the local `wandb` directory in the save directory to check if training is running as intended.

…(further comments

…(trimmed)
