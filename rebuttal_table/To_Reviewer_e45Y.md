**[W2] Concerns about the strength of empirical improvements.**

All results reported below are based on the 100th percentile evaluations.

**Table 1**: Additional comparison with a recent method (PGD) on Off-MOO-Bench. We also report the average rank (lower is better) across different task categories using both HV and $\text{IGD}_\text{offline}$ metrics. 
| Method            | Synthetic       | MO-NAS       | MORL         | Sci-Design   | RE Suite     | Avg. Rank    |
| ----------------- | --------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| PGD (HV)          | 7.76 ± 0.40     | 12.43 ± 0.94 | 15.20 ± 0.51 | 11.88 ± 1.17 | 11.86 ± 0.36 | 10.79 ± 0.39 |
| PGD ($\text{IGD}_\text{offline}$) | 6.97 ± 0.23 | 12.05 ± 1.87 | 15.20 ± 0.51 | 9.78 ± 1.37  | 8.03 ± 0.96  | 9.21 ± 0.94  |


**Table 2**: Detailed results of PGD on MO-NAS subtasks. We report the performance of PGD using HV (higher is better) and $\text{IGD}_\text{offline}$​ (lower is better) metrics.
| Methods               | C10MOP1        | C10MOP2        | C10MOP3        | C10MOP8        | C10MOP9        | IN1KMOP1       | IN1KMOP2       | IN1KMOP3       | IN1KMOP4       | IN1KMOP5       | IN1KMOP6       | IN1KMOP7       | IN1KMOP8       | NASBench201Test |
| --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| PGD (HV)              | 4.69 ± 0.09    | 10.44 ± 0.01   | 8.56 ± 0.00    | 4.18 ± 0.05    | 9.23 ± 0.15    | 3.48 ± 0.02    | 3.45 ± 0.01    | 7.62 ± 0.02    | 4.52 ± 0.06    | 4.61 ± 0.05    | 10.05 ± 0.09   | 4.10 ± 0.08    | 7.69 ± 0.05    | -               |
| PGD ($\text{IGD}_\text{offline}$) | 0.12 ± 0.01 | 0.10 ± 0.00 | 0.52 ± 0.00 | 0.43 ± 0.02 | 0.48 ± 0.02 | 0.49 ± 0.01 | 0.50 ± 0.00 | 0.60 ± 0.00 | 0.26 ± 0.01 | 0.22 ± 0.01 | 0.32 ± 0.00 | 0.44 ± 0.02 | 0.73 ± 0.01 | -               |

**Table 3**: Detailed results of PGD on MORL subtasks. We report the performance of PGD using HV (higher is better) and $\text{IGD}_\text{offline}$​ (lower is better) metrics.
| Methods                     | MOSwimmerV2   | MOHopperV2   |
| --------------------------- | ------------- | ------------ |
| PGD (HV) | 1.35 ± 0.00   | 2.36 ± 0.00  |
| PGD ($\text{IGD}_\text{offline}$) | 1.56 ± 0.00   | 0.91 ± 0.00  |


**Table 4**: Detailed results of PGD on RE Suite subtasks. We report the performance of PGD using HV (higher is better) and $\text{IGD}_\text{offline}$​ (lower is better) metrics.
| Methods                     | RE21        | RE22        | RE23        | RE24        | RE25        | RE31        | RE32        | RE33        | RE34        | RE35        | RE36        | RE37        | Portfolio   |
| --------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| PGD (HV) | 4.43 ± 0.04 | 4.83 ± 0.01 | 4.84 ± 0.00 | 4.84 ± 0.00 | 4.84 ± 0.00 | 10.59 ± 0.02 | 10.65 ± 0.00 | 10.51 ± 0.07 | 9.60 ± 0.09 | 10.29 ± 0.10 | 9.46 ± 0.24 | 6.05 ± 0.14 | 1.96 ± 0.58 |
| PGD ($\text{IGD}_\text{offline}$) | 0.47 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.02 | 0.01 ± 0.00 | 0.02 ± 0.00 | 0.04 ± 0.00 | 0.33 ± 0.01 | 0.13 ± 0.02 | 0.44 ± 0.04 | 0.57 ± 0.01 | 1.90 ± 0.28 |

**Table 5**: Detailed results of PGD on Sci-Design subtasks. We report the performance of PGD using HV (higher is better) and $\text{IGD}_\text{offline}$​ (lower is better) metrics.
| Methods                     | Molecule     | Regex        | ZINC         | RFP          |
| --------------------------- | ------------ | ------------ | ------------ | ------------ |
| PGD (HV) | 3.22 ± 0.81  | 4.39 ± 0.00  | 3.31 ± 0.00  | 1.44 ± 0.00  |
| PGD ($\text{IGD}_\text{offline}$) | 0.66 ± 0.30  | 1.03 ± 0.00  | 0.60 ± 0.00  | 1.39 ± 0.00  |


**Table 6**: Detailed results of PGD on Synthetic subtasks. We report the performance of PGD using HV (higher is better) and $\text{IGD}_\text{offline}$​ (lower is better) metrics.
| Methods                     | ZDT1        | ZDT2        | ZDT3        | ZDT4        | ZDT6        | OmniTest    | VLMOP1      | VLMOP2      | VLMOP3       | DTLZ1       | DTLZ2       | DTLZ3       | DTLZ4       | DTLZ5       | DTLZ6       | DTLZ7       |
| --------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| PGD (HV) | 4.51 ± 0.08 | 5.36 ± 0.05 | 5.59 ± 0.06 | 5.00 ± 0.04 | 4.81 ± 0.02 | 4.71 ± 0.05 | 0.32 ± 0.00 | 2.13 ± 0.20 | 45.32 ± 0.62 | 10.64 ± 0.00 | 10.56 ± 0.00 | 10.63 ± 0.00 | 10.65 ± 0.01 | 10.07 ± 0.01 | 10.17 ± 0.04 | 9.80 ± 0.14 |
| PGD ($\text{IGD}_\text{offline}$) | 0.26 ± 0.02 | 0.23 ± 0.01 | 0.31 ± 0.01 | 0.29 ± 0.02 | 0.12 ± 0.03 | 0.24 ± 0.01 | 0.03 ± 0.00 | 1.22 ± 0.06 | 0.06 ± 0.01 | 0.17 ± 0.00 | 0.26 ± 0.01 | 0.15 ± 0.01 | 0.41 ± 0.01 | 0.37 ± 0.01 | 0.41 ± 0.02 | 0.36 ± 0.01 |