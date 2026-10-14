# 20260926T171220Z_1b418a9-dirty

Baselines: all hold (original tests BA 100; always pass, always fail and network probe as required on every task).

## Model × harness (BA with 95% bootstrap interval, macro over tasks and attempts)

| model | claude-code | codex | terminus-2 |
|---|---:|---:|---:|
| sonnet | 50.0 [41.0, 59.0] | n/a | 50.0 [50.0, 50.0] |
| luna | n/a | 50.0 [50.0, 50.0] | 51.0 [50.0, 53.0] |

A model or harness effect is stated only where two intervals do not overlap.

## Writers (macro over tasks and attempts)

| writer | task-attempts | TPR | TNR | BA | flaky | cost |
|---|---:|---:|---:|---:|---:|---:|
| luna-codex | 5 | 20.0 | 80.0 | 50.0 [50.0, 50.0] | 0.0 | $0.11 |
| luna-terminus | 5 | 40.0 | 62.0 | 51.0 [50.0, 53.0] | 20.0 | $0.17 |
| sonnet-claude-code | 5 | 40.0 | 60.0 | 50.0 [41.0, 59.0] | 0.0 | $8.66 |
| sonnet-terminus | 5 | 0.0 | 100.0 | 50.0 [50.0, 50.0] | 80.0 | $7.38 |

## Writer attempts that produced no verifier (scored as rejecting everything)

- pretrain-shard-corruption: sonnet-terminus/write-verifier-pretrain-shard-co__Lg2Rsyo
- production-planning: sonnet-terminus/write-verifier-production-planni__7HrC8nH
- react-lead-form: sonnet-terminus/write-verifier-react-lead-form__hX2Xm9F
- uefi-bootkit: sonnet-terminus/write-verifier-uefi-bootkit__mvyaR4C

## By task

| task | verifier | correct/wrong | TPR | TNR | BA | TNR near miss | TNR real fail | TNR nop | flaky |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hof-topology-interpenetration | always_fail | 2/2 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | always_pass | 2/2 | 100.0 | 0.0 | 50.0 | 0.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | luna-codex/write-verifier-hof-topology-inte__W53S8XR | 11/10 | 100.0 | 0.0 | 50.0 | 0.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | luna-terminus/write-verifier-hof-topology-inte__iUbUvZB | 11/10 | 100.0 | 0.0 | 50.0 | 0.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | network_probe | 2/2 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | original | 11/10 | 100.0 | 100.0 | 100.0 | 100.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | sonnet-claude-code/write-verifier-hof-topology-inte__qHMV2kJ | 11/10 | 0.0 | 70.0 | 35.0 | 70.0 | n/a | n/a | 0.0 |
| hof-topology-interpenetration | sonnet-terminus/write-verifier-hof-topology-inte__yLq29cz | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | always_fail | 2/2 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | always_pass | 2/2 | 100.0 | 0.0 | 50.0 | 0.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | luna-codex/write-verifier-pretrain-shard-co__Sk7pRTc | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | luna-terminus/write-verifier-pretrain-shard-co__bLagQDv | 11/10 | 100.0 | 10.0 | 55.0 | 10.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | luna-terminus/write-verifier-pretrain-shard-co__bRLPp7o | 11/10 | 100.0 | 10.0 | 55.0 | 10.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | network_probe | 2/2 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | original | 11/10 | 100.0 | 100.0 | 100.0 | 100.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | sonnet-claude-code/write-verifier-pretrain-shard-co__njvYwJS | 11/10 | 100.0 | 0.0 | 50.0 | 0.0 | n/a | n/a | 0.0 |
| pretrain-shard-corruption | sonnet-terminus/write-verifier-pretrain-shard-co__Lg2Rsyo | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 100.0 |
| production-planning | always_fail | 2/2 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| production-planning | always_pass | 2/2 | 100.0 | 0.0 | 50.0 | 0.0 | n/a | n/a | 0.0 |
| production-planning | luna-codex/write-verifier-production-planni__VisbYr7 | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| production-planning | luna-terminus/write-verifier-production-planni__LpK8Rdj | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 100.0 |
| production-planning | network_probe | 2/2 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| production-planning | original | 11/10 | 100.0 | 100.0 | 100.0 | 100.0 | n/a | n/a | 0.0 |
| production-planning | sonnet-claude-code/write-verifier-production-planni__UYzuuW4 | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 0.0 |
| production-planning | sonnet-terminus/write-verifier-production-planni__7HrC8nH | 11/10 | 0.0 | 100.0 | 50.0 | 100.0 | n/a | n/a | 100.0 |
| react-lead-form | always_fail | 2/2 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | n/a | 0.0 |
| react-lead-form | always_pass | 2/2 | 100.0 | 0.0 | 50.0 | n/a | 0.0 | n/a | 0.0 |
| react-lead-form | luna-codex/write-verifier-react-lead-form__VNCTMeV | 11/11 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 0.0 |
| react-lead-form | luna-terminus/write-verifier-react-lead-form__vk843rr | 11/11 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 0.0 |
| react-lead-form | network_probe | 2/2 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | n/a | 0.0 |
| react-lead-form | original | 11/11 | 100.0 | 100.0 | 100.0 | n/a | 100.0 | 100.0 | 0.0 |
| react-lead-form | sonnet-claude-code/write-verifier-react-lead-form__uoCA6Vs | 11/11 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 0.0 |
| react-lead-form | sonnet-terminus/write-verifier-react-lead-form__hX2Xm9F | 11/11 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 100.0 |
| uefi-bootkit | always_fail | 2/2 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | n/a | 0.0 |
| uefi-bootkit | always_pass | 2/2 | 100.0 | 0.0 | 50.0 | n/a | 0.0 | n/a | 0.0 |
| uefi-bootkit | luna-codex/write-verifier-uefi-bootkit__cieuXLf | 11/10 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 0.0 |
| uefi-bootkit | luna-terminus/write-verifier-uefi-bootkit__TUSNsgY | 11/10 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 0.0 |
| uefi-bootkit | network_probe | 2/2 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | n/a | 0.0 |
| uefi-bootkit | original | 11/10 | 100.0 | 100.0 | 100.0 | n/a | 100.0 | 100.0 | 0.0 |
| uefi-bootkit | sonnet-claude-code/write-verifier-uefi-bootkit__NRxuErf | 11/10 | 100.0 | 30.0 | 65.0 | n/a | 22.2 | 100.0 | 0.0 |
| uefi-bootkit | sonnet-terminus/write-verifier-uefi-bootkit__mvyaR4C | 11/10 | 0.0 | 100.0 | 50.0 | n/a | 100.0 | 100.0 | 100.0 |
