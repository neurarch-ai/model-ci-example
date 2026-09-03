# model-ci-example

A PyTorch model repo with two Neurarch CI gates: [neurarch-lint](https://github.com/neurarch-ai/neurarch-lint) reads the changed files, and [neurarch-bot](https://github.com/neurarch-ai/neurarch-bot) traces the model and posts a plan.

On every pull request that touches a `.py` file, the Action reads the changed files and checks the model structure, not the syntax: attention heads that do not divide the embedding width, a `Softmax` feeding `CrossEntropyLoss`, zero-sized layers, and 17 other defects that `ruff` and `mypy` pass. A blocking finding fails the check, so the defect is caught in the PR instead of one GPU hour into training. Nothing is installed and no model is loaded; the check takes a few seconds.

## The workflow

This is the whole gate, at [`.github/workflows/neurarch-lint.yml`](.github/workflows/neurarch-lint.yml):

```yaml
name: Structural lint
on:
  pull_request:
    paths: ['**/*.py']

permissions:
  contents: read
  pull-requests: write   # lets the Action post its findings as a PR comment

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # full history so the Action can diff the PR
      - uses: neurarch-ai/neurarch-lint@v1
        with:
          comment-on-pr: true
          fail-on-warn: false
```

## The second gate: a plan on every model change

[`.github/workflows/neurarch-bot.yml`](.github/workflows/neurarch-bot.yml) runs [neurarch-bot](https://github.com/neurarch-ai/neurarch-bot) on every pull request. Where the lint reads text, the bot imports the model and traces a forward pass, on this branch and on the base commit, and comments with what the change costs: params, cost, GPU fit, whether it will run, and the per-field diff against base. A blocker fails the check.

```yaml
name: Neurarch plan
on: pull_request

permissions:
  contents: read
  pull-requests: write   # lets the bot post and update its plan comment

jobs:
  plan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # the bot needs the base commit to trace it and diff
      - uses: neurarch-ai/neurarch-bot@v0
```

The models it plans, and the input each one traces with, are listed in [`.neurarch.yml`](.neurarch.yml).

## What is in here

- [`models/tiny_gpt.py`](models/tiny_gpt.py): a small decoder-only transformer, `embed_dim=384`, `num_heads=6`, raw logits into `CrossEntropyLoss`.
- [`models/small_cnn.py`](models/small_cnn.py): a CIFAR-10 CNN, Conv, BatchNorm, ReLU per stage.

Both lint clean on `main`. To see the gate fire, look at the open pull request that tries 5 heads on the 384-wide model and adds a `Softmax` before the loss: the check fails before anything is trained.

## Run it locally

```bash
git clone https://github.com/neurarch-ai/neurarch-lint
node neurarch-lint/lint.mjs --dir=models
```

## Links

- The lint Action and the rule list: https://github.com/neurarch-ai/neurarch-lint
- The plan Action: https://github.com/neurarch-ai/neurarch-bot
- The full verifier the rules come from: https://neurarch.com

MIT licensed.
