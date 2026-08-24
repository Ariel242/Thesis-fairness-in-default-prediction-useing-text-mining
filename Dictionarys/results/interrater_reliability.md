# Inter-Rater Reliability -- LLM vs Human Expert (Ariel)

- Executed: 2026-08-23T15:40:24
- Sample size compared: 147
- Blind labels: `Dictionarys\results\human_validation_blind.csv`
- LLM labels (key, not seen during labeling): `Dictionarys\results\human_validation_key.csv`

## 5-category agreement

- **Cohen's kappa: 0.556**
- Raw agreement: 0.660

### Confusion matrix (rows = human label, columns = LLM label)

|                             |   llm:DEBT_PRESSURE |   llm:LIQUIDITY_SHORTAGE |   llm:SHOCK_EMERGENCY |   llm:HIGH_RISK_REFINANCING |   llm:NEUTRAL_BENIGN |
|:----------------------------|--------------------:|-------------------------:|----------------------:|----------------------------:|---------------------:|
| human:DEBT_PRESSURE         |                   9 |                        2 |                     3 |                           2 |                    0 |
| human:LIQUIDITY_SHORTAGE    |                   2 |                       26 |                     2 |                           1 |                    2 |
| human:SHOCK_EMERGENCY       |                   0 |                        0 |                    27 |                           0 |                    0 |
| human:HIGH_RISK_REFINANCING |                   1 |                        0 |                     0 |                           2 |                    0 |
| human:NEUTRAL_BENIGN        |                  23 |                        7 |                     3 |                           2 |                   33 |

### Per-category precision / recall / F1 (human label = ground truth)

| category              |   precision |   recall |    f1 |   support (human n) |
|:----------------------|------------:|---------:|------:|--------------------:|
| DEBT_PRESSURE         |       0.257 |    0.562 | 0.353 |                  16 |
| LIQUIDITY_SHORTAGE    |       0.743 |    0.788 | 0.765 |                  33 |
| SHOCK_EMERGENCY       |       0.771 |    1.000 | 0.871 |                  27 |
| HIGH_RISK_REFINANCING |       0.286 |    0.667 | 0.400 |                   3 |
| NEUTRAL_BENIGN        |       0.943 |    0.485 | 0.641 |                  68 |

## Binary is_distress view (NEUTRAL_BENIGN vs any distress category)

- Precision: 0.688
- Recall: 0.975
- F1: 0.806
- Agreement: 0.748
