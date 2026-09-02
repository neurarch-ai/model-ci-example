# model-ci-example

A PyTorch model repo with [neurarch-lint](https://github.com/neurarch-ai/neurarch-lint) as a CI gate.

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

- The Action and the rule list: https://github.com/neurarch-ai/neurarch-lint
- The full verifier the rules come from: https://neurarch.com

MIT licensed.
