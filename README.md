# Minesweeper symmetry

A small convolutional network learns, for each covered cell of an 8x8 Minesweeper board with 10 mines, the probability that the cell is a mine. It is trained and scored against the exact probability: the share of all mine layouts that fit the visible board. The board has 8 symmetries, and the exact probabilities turn with the board. This repository compares a plain network with several ways of giving it that symmetry, on the same data, training schedule and scoring.

The exact probabilities come from the Minesweeper solver at github.com/Evasion-OC/minesweeper-solver. That solver's design comes from my BSc final project at the University of Leeds, and its code is the 2026 implementation of that design. `exact_symmetry/minesweeper/` is a pinned copy of the solver's exact core (SAT for forced moves, counting for probabilities), taken from solver commit 63d178a. Only the module paths differ.

## The question

The symmetries of a square board form the dihedral group D4: four rotations, each with or without a reflection. Turn a board by one of them and its exact probabilities turn with it, so the exact map is equivariant. A trained network is not equivariant unless something makes it so. The first question is how much each way of using the symmetry lowers the error, at 1,000, 10,000 and 100,000 training positions.

The second question is what happens at a board that is its own mirror image. If a symmetry h fixes a board x (hx = x), an equivariant f gives f(x) = f(hx) = h f(x). So cells that h swaps must get the same probability. This is the fixed-subspace fact, a form of Curie's principle, that Wang et al. (ICLR 2024) use for point clouds of rank below three. Here it is applied to a finite group that permutes cells. A method that is equivariant only at boards with no symmetry of their own can split such cells.

## Setup

Data (`exact_symmetry/data.py`, `scripts/make_data.py`):

- Positions come from games the exact player plays. It makes a forced move when there is one. Otherwise it reveals the covered cell with the lowest exact probability, with ties going to the lowest cell index. With probability 0.2 a move is replaced by revealing a random safe covered cell, so late positions appear too. The first move reveals a random cell, and that cell and its neighbours never hold a mine.
- The random reveal uses the hidden layout to pick a safe cell, but it does not bias the labels. Every mine layout that fits the visible board gives the moves made the same chance, so all such layouts stay equally likely.
- Every position met before the game ends is kept, with its exact probabilities. Flags are shown to the network as covered.
- Training set 100,000 positions (5,702 games), validation 10,000 (567 games), test 10,000 (573 games). The training sizes 1,000, 10,000 and 100,000 are the first n positions of the training set. Each split draws its boards from its own random stream.
- The exact probabilities of a turned board are the turned probabilities. So the target respects the symmetry whatever positions the games produce (`tests/test_data.py` checks this).

Input (`exact_symmetry/net.py`): 13 planes. They are the covered cells, the clues 0 to 8 one-hot on revealed cells, and a plane of ones so that the zero padding marks the edge. The last two planes are mines per covered cell and the share of cells still covered, each one number spread over the board. All the planes turn with the board.

Networks (`exact_symmetry/net.py`, `exact_symmetry/arms.py`):

- Plain network: a 3x3 convolution to 64 channels, 6 residual blocks of two 3x3 convolutions, and a 1x1 convolution to one logit per cell. There is no normalisation layer. Each block's second convolution starts at zero, so every block starts as the identity. 450,753 parameters and 28.8 million multiply-adds per board.
- Group network: a lifting convolution, then 6 residual blocks of group convolutions over D4, the p4m layers of Cohen and Welling (2016). Each layer has 8 channels per group element (64 feature maps). The group axis is averaged out before the 1x1 head. On a bounded board it is equivariant to D4 by construction. 56,345 parameters and 28.8 million multiply-adds per board, the same compute as the plain network.

Training (`exact_symmetry/net.py`, `scripts/train.py`, `scripts/run_grid.py`):

- 10,000 steps, batch 256, Adam, learning rate 0.001 with cosine decay, batches drawn uniformly with replacement.
- Loss: binary cross-entropy against the exact probability, averaged over covered cells. Its excess over its minimum is exactly the KL divergence from the exact probability to the prediction.
- The validation KL is measured every 250 steps (every 25 in the first 250), and the weights with the lowest validation KL are kept.
- Five seeds (0 to 4) per method and size. At a given seed every method draws the same batches, and plain, augmentation and the canonical form also start from the same weights. Tests check, on short CPU runs, that training repeats exactly from its seed.
- Most runs took 6 to 12 minutes on an Apple laptop GPU (MPS).

Scoring (`exact_symmetry/evaluate.py`, `scripts/summarise.py`, `scripts/symmetry_analysis.py`):

- KL: the KL divergence between the exact and the predicted probability of each covered cell (nats), averaged over all covered cells of the 10,000 test positions. It is computed from the logits in float64, with no cap.
- Equivariance error: the mean of |f(gx) - g f(x)| over covered cells and the 7 non-identity symmetries, on the first 2,000 test positions.
- Symmetric boards: 400 positions with a symmetry of their own, collected from 54,492 extra games on a separate stream. They are rare in play: 5 of the 10,000 test positions have one. Each of the 400 has exactly 2 symmetries, the identity and one reflection. Together they hold 9,110 pairs of covered cells that the reflection swaps, and within each pair the exact probabilities are equal. The gap is the larger predicted probability of a pair minus the smaller, averaged over the pairs. The KL on these boards is reported as well.

## The methods

- Plain: nothing enforces the symmetry.
- Augmentation: each training board and its labels are turned by a random one of the 8 symmetries.
- Averaging: at test time the network runs on all 8 turned boards and the outputs are turned back. The arithmetic mean of the 8 probabilities is taken, computed in log space so that it does not round to 0 or 1. It is exactly equivariant, costs 8 times the compute and needs no retraining. It is applied to the trained plain and augmented networks.
- Canonical form: each board is turned to one fixed representative, the first of its 8 images in lexicographic order of the cell codes. The network predicts there and the output is turned back. It is trained on canonical boards. It is equivariant at boards with no symmetry of their own. At a board with a symmetry of its own, several symmetries reach the same representative, the first is used, and nothing makes swapped cells agree.
- Canonical form + frame: at test time only, the trained canonical network's output is averaged over every symmetry that reaches the representative. These symmetries form a coset of the board's stabiliser, so this is the minimal frame of Lin et al. (ICML 2024) within the frame averaging of Puny et al. (ICLR 2022). It needs one network pass, is exactly equivariant at every board, and equals the canonical form wherever the board has no symmetry of its own.
- Group network: described above.

Kaba et al. (ICML 2023) make a network equivariant by canonicalisation, with a canonicalisation function that is learned. The canonical form here is fixed. Baker, Wang, de Fernex and Wang (ICML 2024) construct frames for 3D point clouds. They note that, at symmetric inputs, a frame and a canonicalised model can be equivariant only up to the stabiliser. The canonical form here meets the same issue on the grid, and the frame version applies the known remedy.

## Results

KL per covered cell on the test positions, mean ± standard deviation over 5 seeds (lower is better).

| Method | 1,000 | 10,000 | 100,000 |
|---|---|---|---|
| plain | 0.1775 ± 0.0051 | 0.0473 ± 0.0029 | 0.0074 ± 0.0006 |
| plain + averaging | 0.1463 ± 0.0142 | 0.0306 ± 0.0013 | 0.0051 ± 0.0005 |
| augmentation | 0.0602 ± 0.0028 | 0.0104 ± 0.0008 | 0.0029 ± 0.0002 |
| augmentation + averaging | 0.0464 ± 0.0011 | 0.0070 ± 0.0004 | 0.0022 ± 0.0001 |
| canonical form | 0.1465 ± 0.0041 | 0.0356 ± 0.0016 | 0.0061 ± 0.0005 |
| canonical form + frame | 0.1465 ± 0.0041 | 0.0356 ± 0.0016 | 0.0061 ± 0.0005 |
| group network | 0.1166 ± 0.0540 * | 0.0243 ± 0.0025 | 0.0250 ± 0.0387 * |

\* One of the five seeds has a probability floor, described below.

Augmentation, averaging and both canonical versions lower the KL at every training size. Each of their differences from plain has a 95% paired t interval over seeds that excludes zero. Augmentation lowers the KL by 0.117 nats at 1,000 positions and by 0.0045 at 100,000, so its absolute gain is largest with little data. In relative terms it removes 66%, 78% and 61% of the plain network's KL at the three sizes, so the relative gain does not fall steadily. Averaging never raised the test KL in any of the 45 runs it was applied to, and it costs 8 times the compute at test time. The canonical form helps less than augmentation at every size. The frame version leaves its test KL unchanged to four decimals, because only 5 of the 10,000 test positions have a symmetry of their own.

The group network trails augmentation at every size, under the one untuned schedule that all methods share. It beats plain in every seed at 10,000 positions. Seed 1 fails at 1,000 and 100,000 positions with a probability floor. Its predictions on covered cells never go below 0.11 and 0.24 respectively, where a working network goes close to 0. In both runs the 1x1 head's only negative weight sits on a pooled channel that never fires on covered cells, so the logit cannot go below the head's bias. At 100,000 positions that seed also collapsed during training, and early stopping kept weights from before the collapse. Without seed 1, the group network's mean KL is 0.093 at 1,000 positions and 0.0077 at 100,000, level with plain there. Why it trails augmentation is not established here. Averaging the augmented network is also exactly equivariant and does better at every size, at 8 times the parameters and 8 times the inference compute of the group network.

Boards with a symmetry of their own, mean over 5 seeds:

| Method | Gap, 1,000 | Gap, 10,000 | Gap, 100,000 | KL, 1,000 | KL, 10,000 | KL, 100,000 |
|---|---|---|---|---|---|---|
| plain | 0.044 | 0.026 | 0.0099 | 0.0463 | 0.0112 | 0.0013 |
| plain + averaging | about 1e-8 | about 1e-8 | about 1e-8 | 0.0354 | 0.0053 | 0.0007 |
| augmentation | 0.023 | 0.011 | 0.0066 | 0.0157 | 0.0022 | 0.0006 |
| augmentation + averaging | about 1e-8 | about 1e-8 | about 1e-8 | 0.0118 | 0.0012 | 0.0004 |
| canonical form | 0.043 | 0.021 | 0.0087 | 0.0386 | 0.0067 | 0.0010 |
| canonical form + frame | 0 | 0 | 0 | 0.0322 | 0.0046 | 0.0007 |
| group network | below 1e-6 | below 1e-6 | below 1e-6 | 0.0341 | 0.0051 | 0.0099 |

At these boards the exact probabilities of two swapped cells are equal, so any gap between them is error. Averaging and the group network keep the gap at float32 rounding, since both are equivariant by construction. The group network's KL at 100,000 positions is dominated by the seed with the floor.

The canonical form is the case the fixed-subspace fact predicts. At 1,000 positions, plain + averaging and the canonical form reach almost the same test KL (0.1463 and 0.1465). Yet the canonical form's gap between swapped cells is 0.043, against zero for averaging. Its equivariance error on the test positions is 2e-5 at 1,000 positions, and it comes from the one position with a symmetry of its own among the 2,000 tested. So it turns with the board wherever the board has no symmetry, and it can split swapped cells where the board has one.

Averaging the canonical form over the symmetries that reach its representative removes the gap exactly. It lowers the KL on these boards by 16%, 31% and 29% at the three sizes, with no seed worse. It can never raise a board's KL: within a swapped pair the exact probabilities are equal, and the KL is convex in the prediction. A test checks this.

Full results per run are in `results/<arm>_8x8x10_n<positions>_seed<seed>.json`, where `<arm>` is `plain`, `augment`, `canonical` or `p4m8`. The averaged and frame scores are inside the plain, augment and canonical files. The summary is in `results/summary_8x8x10.json` and the symmetric-board analysis in `results/symmetry_analysis_8x8x10.json`. Runs were scored again from their saved checkpoints with `scripts/rescore.py` after the scoring code changed. Each file records the commit that scored it, and the training is unchanged.

## Limits

- One board size (8x8, 10 mines) and five seeds. The intervals are over seeds, on one training sample and one test set.
- Positions from one game share a board and are not independent. The first 1,000 training positions come from 55 games and the first 10,000 from 578.
- One training schedule for every method, not tuned per method. The kept weights are chosen on the same 10,000 validation positions at every size. At 1,000 positions, plain and the canonical form keep weights from within the first 200 steps, while the learning rate is still close to its starting value.
- The group network matches the plain network in compute, not in parameters. Its gap to augmentation is observed under one schedule, and its cause is not established.
- The games are not balanced across orientations. The exact player plays forced moves in cell-index order, so upper rows tend to be revealed earlier. The target is still symmetric, so this changes which boards are seen, not what the right answer is. On the test set as generated, the skew favours methods that are not equivariant and can learn it.
- Symmetric boards are rare in play, and mostly early and simple. Of the 400, 1%, 6% and 29% also appear, in some orientation, among the first 1,000, 10,000 and 100,000 training positions. So at the larger sizes the gaps are partly measured on boards seen in training.
- The canonical form is one fixed lexicographic rule. A learned or smoother canonicalisation could behave differently.

## Running it

Tested with Python 3.13 and PyTorch 2.8. The data and checkpoints are not in the repository, and the commands below make them.

```
pip install -r requirements.txt
python scripts/make_data.py --split train --positions 100000
python scripts/make_data.py --split validation --positions 10000
python scripts/make_data.py --split test --positions 10000
python scripts/run_grid.py
python scripts/summarise.py
python scripts/symmetry_analysis.py
python -m pytest
```

`make_data.py` writes each split to `data/`. `run_grid.py` trains every method at every size and seed (60 runs, since averaging and the frame version need no training). It writes `results/<arm>_8x8x10_n<positions>_seed<seed>.json` and the weights to `checkpoints/`. It skips a run only when both its result and its checkpoint exist. `summarise.py` writes `results/summary_8x8x10.json`, and `symmetry_analysis.py` writes `results/symmetry_analysis_8x8x10.json`.

`python -m pytest` runs 44 tests (43 on a machine without Apple's GPU). They check the labels against brute-force counting of every layout on small boards, and that the labels turn with the board. They also check each method's symmetry property, the frame version's guarantee, and that training repeats exactly from its seed.

## Layout

- `exact_symmetry/`
  - `data.py`: games, positions and their exact probabilities
  - `net.py`: input planes, the plain network, loss, training
  - `arms.py`: the symmetry group and the methods
  - `evaluate.py`: KL and calibration on a set of positions
  - `provenance.py`: which commit produced a result
  - `minesweeper/`: the pinned solver core (SAT, counting, symmetries)
- `scripts/`
  - `make_data.py`: one split of positions with their exact probabilities
  - `train.py`: one training run, scored on the test positions
  - `run_grid.py`: every method at every size and seed
  - `summarise.py`: the table over seeds, with the paired intervals
  - `symmetry_analysis.py`: the equivariance error and the symmetric boards
  - `rescore.py`: scores saved checkpoints again with the current evaluation code
- `tests/`: the test suite
- `results/`: one file per run, the summary and the symmetric-board analysis

## References

- T. S. Cohen and M. Welling, "Group Equivariant Convolutional Networks", ICML 2016.
- O. Puny, M. Atzmon, H. Ben-Hamu, I. Misra, A. Grover, E. J. Smith and Y. Lipman, "Frame Averaging for Invariant and Equivariant Network Design", ICLR 2022.
- S.-O. Kaba, A. K. Mondal, Y. Zhang, Y. Bengio and S. Ravanbakhsh, "Equivariance with Learned Canonicalization Functions", ICML 2023.
- S.-H. Wang, Y.-C. Hsu, J. Baker, A. L. Bertozzi, J. Xin and B. Wang, "Rethinking the Benefits of Steerable Features in 3D Equivariant Graph Neural Networks", ICLR 2024.
- J. Baker, S.-H. Wang, T. de Fernex and B. Wang, "An Explicit Frame Construction for Normalizing 3D Point Clouds", ICML 2024.
- Y. Lin, J. Helwig, S. Gui and S. Ji, "Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency", ICML 2024.
