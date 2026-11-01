# CS 412 Project: TRACE

Course project for CS 412 (Machine Learning, UIC, Fall 2026) by Shicheng Fan, Jie Yang and Honghui Bao.

TRACE recovers continuously evolving causal mechanisms from time series. Each intermediate mechanism is modeled as a convex combination of atomic mechanisms, a Mixture-of-Experts sequential VAE learns one expert per atomic mechanism (Stage 1), and the mixing trajectory is recovered at test time by least-squares projection (Stage 2).

The code builds on the official TRACE implementation, [shichengf/trace](https://github.com/shichengf/trace) (MIT License), which in turn adapts parts of [TDRL](https://github.com/weirayao/tdrl). This repository currently covers the synthetic and CartPole experiments.

## Repository layout

```
reports/intermediate/   intermediate report (LaTeX source, figures, PDF)
trace_crl/              model, datasets, data generators, configs
scripts/                training scripts (synthetic 5-domain, CartPole)
inference/              Stage 2 trajectory recovery and evaluation
```

## Installation

```bash
conda create -n trace python=3.8
pip install -e .
```

## Usage

### Data generation

```bash
python -m trace_crl.tools.gen_dataset -d 5 -s 222
python -m trace_crl.tools.gen_trajectory -d 5 -k 3
```
`-d` is the number of domains, `-k` the number of active domains, `-s` the random seed.

### Training

```bash
python scripts/train_change_5domain.py -e change_5 -s 222
python scripts/train_cartpole.py
```

### Inference

Set the dataset and checkpoint paths in `trace_crl/configs/inference.yaml`, then run

```bash
python inference/infer.py -d 5 -k 3 -t simple
```

## Reports

- Intermediate report: [`reports/intermediate/`](reports/intermediate/)

## License

MIT License (see `LICENSE`).
