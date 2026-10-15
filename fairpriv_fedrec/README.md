# FairPriv-FedRec — Fair and Privacy-Preserving Federated Recommendation

Reference implementation of the project plan *"Fair and Privacy-Preserving Federated Recommendation"*
(NIT Patna). Lightweight federated NCF with (1) gradient clipping + Gaussian/Laplace noise and
(2) popularity-aware fairness: loss re-weighting + exposure-gap regularisation + a server-side fairness monitor.

**Dataset (one, as agreed):** MovieLens-1M — ratings ≥ 4 become implicit positives, each user is one client,
leave-one-out split by time (last item = test, second-to-last = validation), full-catalogue ranking.
(Amazon Beauty/Electronics and Yelp from the plan are not included; add a loader in `fairpriv/data.py`.)

## Install & run

```bash
pip install -r requirements.txt
pytest -q                                                      # 16 unit tests
python run_experiments.py --exp main --seeds 0 --quick --max-users 1000   # ~ minutes, smoke test
python run_experiments.py --exp main --seeds 0 1 2             # Tables 1-2 + figures
python run_experiments.py --exp all  --seeds 0 1 2             # everything
```
A GPU is strongly recommended for ML-1M (6,040 clients): `--device cuda --client-chunk 256`.

## Mapping plan → code

| Plan | Where |
|---|---|
| Local model: NCF (or MF) + BPR | `models.py` (`NCFShared`, `CentralMF`), loss in `federated.py` |
| Privacy layer: clip `g·min(1, C/‖g‖)`, add `N(0,σ²C²I)` or `Lap(0, C/ε)` | `privacy.py` |
| Popularity-aware weight `w_i = 1 + β(1 − pop/max pop)`, `L_u = Σ w_i L_BPR` | `fairness.py::compute_item_weights` |
| Exposure-gap regularisation | `fairness.py::gap_penalty` (client side, differentiable) |
| Fairness monitor + tuning β | `federated.py::_monitor`, `fairness.py::BetaController` (`--exp ablation` row "Ours + adaptive beta") |
| Aggregation `θ ← θ − η Σ (n_u/Σ n_v) g̃_u` | `federated.py::run_round` (`aggregation="data_size"`, `--server-optimizer sgd` = literal update) |
| Metrics HR/Recall/NDCG/MRR@10 | `metrics.py` |
| Fairness: tail-item coverage, exposure gap, Gini, head–tail recall gap | `metrics.py` |
| Privacy: σ, ε, **attack accuracy**, utility–privacy curve | `privacy.py` (accountant), `federated.py::evaluate_attack`, `--exp privacy/epsilon` |
| Baselines: centralized MF/NCF, vanilla FedRec, DP-FedRec, Fair-FedRec, FairPriv-FedRec | `runner.py::METHODS` |
| Ablation: no clipping, no noise, no fairness weight, different ε, different β | `--exp ablation`, `epsilon`, `beta` |
| Table 1/2/3, Figure 1 | `results/ml-1m/main/table1_utility.tex`, `table2_fairness.tex`, `privacy/table3_privacy_utility.tex`, `beta/fig1_accuracy_vs_fairness.pdf` |

## Definitions used in the paper tables

* **Head/tail**: top 20 % most popular items by train count = head, rest = tail.
* **Exposure gap@K** (normalised): `EG = hs/r − (1−hs)/(1−r)`, `hs` = share of top-K slots given to head items,
  `r` = |head|/|catalogue|. `0` = parity (each item equally exposed), larger = more head-dominated (max `1/r = 5`).
* **Head–tail recall gap**: Recall@K of users whose held-out item is head minus that of users whose held-out item is tail.
* **Recall@K vs HR@K**: identical under leave-one-out (one relevant item per user); both are reported as in the plan.
* **Attack accuracy** (`priv_attack_auc`, `priv_attack_prec`): a curious server sees one client's uploaded update and
  ranks items by the energy of the item-embedding rows; label = item is in the client's training set.
  AUC 0.5 = no leakage, 1.0 = perfect. It is a *simple* attack — a lower bound on leakage, not a proof of privacy.
  For `--placement distributed` the attack is run against the update plus that client's *share* of the aggregate noise
  (an adversary that bypasses secure aggregation).


## Layout

```
fairpriv/   config, data, models, fairness, privacy, federated, centralized, metrics, runner, experiments, reporting
tests/      unit tests
run_experiments.py
data/ml-1m/ratings.dat   MovieLens-1M ratings in the official '::' format (see provenance note below)
```
**Data provenance:** the bundled `ratings.dat` was rebuilt from the train/test files of the public NCF repository
(hexiangnan/neural_collaborative_filtering) and verified against the official statistics: 1,000,209 ratings, 6,040 users,
3,706 items, 575,281 ratings ≥ 4. User/item ids are re-indexed. The loader downloads the official GroupLens copy if the
file is missing.
