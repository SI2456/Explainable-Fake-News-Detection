# Explainable, Uncertainty-Aware Fake News Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SI2456/explainable-fake-news-detection/blob/main/Fake_News_Detection_XAI_Uncertainty.ipynb)

A fake news classifier built on fine-tuned **DistilBERT**. It does more than output FAKE or REAL:

- **Confidence you can trust.** It uses Monte Carlo Dropout and temperature scaling, and checks calibration with ECE.
- **An `UNCERTAIN` option.** Low-confidence predictions are sent for human review instead of being forced into FAKE or REAL.
- **Explanations.** LIME shows which words drove each prediction, and a deletion test checks that those explanations are faithful.
- **An honest generalization test.** The model is trained on political news (ISOT) and tested on COVID-19 health misinformation.

> Machine Learning subject project, B.Tech Information Technology, BVM Engineering College (GTU).

![Pipeline](docs/pipeline.png)

---

## Why this project is different

The popular Kaggle *Fake and Real News* dataset has a **leakage problem**. The raw text reveals the *source* of each article:

| Pattern in raw text | % of REAL | % of FAKE |
|---|---|---|
| `(Reuters)` dateline | 99.2 | 0.0 |
| "Featured image" credit | 0.0 | 34.8 |
| "Getty Images" | 0.0 | 16.8 |
| URL / pic.twitter link | 0.0 | 23.5 |
| Twitter @handle | 1.3 | 26.1 |
| Curly apostrophe `’` | 47.4 | 0.0 |

Because of this, even TF-IDF + Logistic Regression scores about 99% without learning anything about misinformation. This project removes those artifacts and de-duplicates the data. It then uses LIME to *show* the shortcut and a cross-domain test to measure what the model really learned.

## What's inside

| Component | Method |
|---|---|
| Baseline | TF-IDF (1–2 grams) + Logistic Regression, on raw vs cleaned text |
| Model | `distilbert-base-uncased`, fine-tuned for 2 epochs, lr 2e-5, max 256 tokens, mixed precision |
| Uncertainty | Softmax vs Temperature Scaling vs MC Dropout (20 passes). Reports predictive entropy and mutual information (BALD) |
| Calibration | Expected Calibration Error (ECE), NLL, Brier score, reliability diagrams |
| Selective prediction | Risk–coverage curve, AURC, error-detection AUROC, `UNCERTAIN` flag (threshold tuned on the validation set) |
| Explainability | LIME word importance, aggregated top words, deletion (faithfulness) test |
| Domain shift | ISOT (politics, 2016–17) → COVID-19 posts (health, 2020): accuracy drop, ECE, OOD-detection AUROC |

## Datasets

| Dataset | Use | Size |
|---|---|---|
| [Fake and Real News Dataset (ISOT)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | train / validation / in-domain test | 44,898 articles → 38,243 after cleaning |
| [COVID-19 Fake News Dataset (Constraint@AAAI 2021)](https://github.com/parthpatwa/covid19-fake-news-detection) | cross-domain test | 2,140 posts |

The notebook downloads both automatically. If a link fails, upload `True.csv`, `Fake.csv` and `english_test_with_labels.csv` to the Colab file panel. Local files are used first.

## How to run

1. Click the **Open in Colab** badge above, or upload the notebook to [Colab](https://colab.research.google.com).
2. Go to `Runtime → Change runtime type → T4 GPU`.
3. Choose `Runtime → Run all`. A full run takes about 40–60 minutes. For a 10-minute test, set `quick_run=True` in the settings cell first.
4. All tables and figures are saved in `results/`. Download that folder before the Colab session ends.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook Fake_News_Detection_XAI_Uncertainty.ipynb
```

A CUDA GPU is strongly recommended.

## Results

> Fill this table in from `results/all_results.xlsx` after running the notebook.

| Model | In-domain Acc | In-domain F1 | ECE | COVID-19 Acc | COVID-19 F1 |
|---|---|---|---|---|---|
| TF-IDF + LR (raw text) | | | – | – | – |
| TF-IDF + LR (clean text) | | | – | | |
| DistilBERT + Softmax | | | | | |
| DistilBERT + Temperature scaling | | | | | |
| DistilBERT + MC Dropout | | | | | |

| UNCERTAIN flag | ISOT test | COVID-19 |
|---|---|---|
| % flagged | | |
| Accuracy on confident predictions | | |
| % of all errors caught by the flag | | |

<!-- Add figures after running, e.g.:
![Reliability diagrams](results/figures/reliability_in_domain.png)
![Risk-coverage](results/figures/risk_coverage_in_domain.png)
![Domain shift](results/figures/domain_shift_uncertainty.png)
-->

## Try it on your own text

After the notebook has run, use the `analyse()` function:

```python
analyse("BREAKING: Doctors are FURIOUS after this one weird trick cures diabetes overnight!")
# Verdict   : FAKE / REAL / UNCERTAIN — needs human review
# P(fake)   : ...   (MC Dropout mean of 20 passes)
# Confidence: ...   threshold tau = ...
# Key words : ...
```

## Project structure

```
explainable-fake-news-detection/
├── Fake_News_Detection_XAI_Uncertainty.ipynb   # full pipeline, runs end to end
├── requirements.txt
├── docs/
│   └── pipeline.png
└── results/                                     # created by the notebook
    ├── figures/                                 # reliability, risk-coverage, LIME, domain shift plots
    ├── *.csv                                    # every results table
    ├── all_results.xlsx                         # all tables in one workbook
    └── distilbert_fakenews/                     # saved model (git-ignored, ~260 MB)
```

## Limitations

- Every REAL article comes from Reuters, so the model partly learns *source style*, even after cleaning.
- The model judges how text is written. It does not check facts.
- Only the first 256 tokens of each article are used.
- MC Dropout is an approximation. It can still be over-confident on very different data.

## References

1. Sanh et al. (2019). [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
2. Gal & Ghahramani (2016). [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)
3. Guo et al. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
4. Geifman & El-Yaniv (2017). [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
5. Ribeiro et al. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
6. Geirhos et al. (2020). [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780)
7. Patwa et al. (2021). [Fighting an Infodemic: COVID-19 Fake News Dataset](https://arxiv.org/abs/2011.03327)
8. Ahmed, Traore & Saad (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. ISDDC 2017.

## Author

**Siddhraj Rajput**, B.Tech IT, BVM Engineering College · [GitHub](https://github.com/SI2456) · [Portfolio](https://si2456.github.io/myrasume)
