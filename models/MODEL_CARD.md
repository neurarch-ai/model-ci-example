# SmallCNN

A CIFAR-10 classifier: five `Conv2d -> BatchNorm2d -> ReLU` stages, two max pools, a global
average pool, and a linear head over 10 classes. The conv bias is off because BatchNorm carries
its own shift term.

| | |
|---|---|
| Task | CIFAR-10, 10 classes |
| Input | `3 x 32 x 32` |
| Parameters | 1,200,000 |
| Optimizer | Adam, lr 1e-3 |
| License | MIT |

## Why this card is wrong on purpose

The parameter count above is stale. It was written when the model was four times wider, and
nobody updated it when the widths came down; a card is prose, so nothing in CI has ever read it.

That is the point of the weekly report. `neurarch-bot` in `mode: report` traces the model, gets
the real count, and compares it against any parameter count written in a `MODEL_CARD.md` or
`README.md` next to the model file. A number that is off by more than 1 percent shows up in the
report's "Documentation drift" section, with both figures side by side. Fix the number here and
the line disappears from next week's report.
