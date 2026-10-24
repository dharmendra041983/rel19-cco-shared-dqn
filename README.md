# Multi-cell coverage optimization: VCC code revision

Source code for **Shared Deep Q-Learning for Multi-Cell Coverage Optimization:
Transfer and Action-Screening Limits**, by Dharmendra Kumar.

Start with [VCC_README.md](VCC_README.md) for the exact dependencies, fixed seed
schedule, experiment order, and commands to regenerate the study outputs.

This revision publishes implementation code, experiment configurations, numerical
checks, and result-generation scripts. New raw result files, run logs, trained
checkpoints, and manuscript files are not included in this public update.
The scripts generate results and checkpoints locally under `vcc_results/`, and
create figure/table files under `manuscript/`.

| Component | Source |
| --- | --- |
| Shared DQN, graph DQN, VDN, PPO, and topology-transfer experiments | [vcc_experiments.py](vcc_experiments.py) |
| Frozen-policy oracle/learned-gate audit | [vcc_safety.py](vcc_safety.py) |
| Sum/mean VDN training controls | [vcc_ablations.py](vcc_ablations.py) |
| Calibration-population and margin sweep | [vcc_margin_sweep.py](vcc_margin_sweep.py) |
| Native ns-3 propagation-kernel sensitivity | [vcc_ns3_validation.py](vcc_ns3_validation.py), [validation/](validation/) |
| Figure, table, and statistical-summary generation | [scripts/build_vcc_results.py](scripts/build_vcc_results.py), [scripts/build_vcc_ablations.py](scripts/build_vcc_ablations.py) |
| Numerical implementation checks | [scripts/validate_vcc.py](scripts/validate_vcc.py) |
| Dependency versions | [requirements_vcc.txt](requirements_vcc.txt) |

All network inputs are synthetic. No proprietary operator, customer, employer,
or production-network data is used. The simulator retains a simplified antenna,
association, interference, and scheduler abstraction. Native ns-3 checks cover
propagation components, not a full NR packet stack or field validation. The
learned action gate is not a deployment safety guarantee.

## Historical IPCCC materials

The original public files under `results/` and `figures/` remain unchanged.
They belong to the earlier IPCCC evaluation and are not the revised VCC results.
The historical implementation is available at commit
`336fba183de6beef5ad30d87f31446c96e6d125d`. The current PPO script corrects
gradient averaging; reproducing the old PPO numbers requires that base commit.

## License and citation

The software uses the [BSD 3-Clause License](LICENSE). Software citation metadata
is in [CITATION.cff](CITATION.cff). No publication acceptance is claimed.
