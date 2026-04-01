\# AML Transaction Detector

\### Anti-Money Laundering Detection using GraphSAGE + Transformer Ensemble



A deep learning system that detects illicit Bitcoin transactions by modeling the transaction network as a graph and applying a hybrid GraphSAGE + Transformer architecture.



\---



\## Results



| Model | F1 Score | Precision | Recall |

|---|---|---|---|

| MLP Baseline | 0.5955 | 0.4401 | 0.9208 |

| GCN Baseline | 0.4855 | 0.3211 | 0.9153 |

| V1 (GraphSAGE + Transformer) | 0.7842 | 0.6908 | 0.9142 |

| V2 (3x SAGE + Transformer) | 0.8677 | 0.8127 | 0.9307 |

| \*\*Ensemble (V1 + V2)\*\* | \*\*0.8684\*\* | \*\*0.8235\*\* | \*\*0.9186\*\* |

| \*\*Ensemble — Temporal Split\*\* | \*\*0.9143\*\* | \*\*0.8690\*\* | \*\*0.9645\*\* |



\- \*\*+53.6% F1 improvement\*\* over MLP baseline

\- \*\*+88.3% F1 improvement\*\* over GCN baseline

\- \*\*96.4% of illicit transactions caught\*\* on future unseen timesteps

\- Temporal split evaluation confirms \*\*no overfitting or temporal bias\*\*



\---



\## Architecture



\### V1 — GraphSAGE + Transformer

```

Input (166 features)

→ GraphSAGE Layer (mean aggregation, 1-hop)

→ Batch Norm + ReLU

→ Transformer Encoder (2 layers, 4 heads)

→ Linear Classifier (128 → 2)

```



\### V2 — Deep GraphSAGE + Transformer

```

Input (166 features)

→ Input Projection (166 → 128)

→ GraphSAGE Layer 1 (mean, 1-hop)

→ GraphSAGE Layer 2 (mean, 2-hop)

→ GraphSAGE Layer 3 (max aggregation)

→ Temporal Positional Encoding (learned embeddings)

→ Transformer Encoder (3 layers, 4 heads, 512 FFN)

→ Two-layer Classifier (128 → 64 → 2)

```



\### Why GraphSAGE + Transformer?

\- \*\*GraphSAGE\*\* captures network topology — a transaction's neighborhood reveals laundering patterns invisible to per-transaction models

\- \*\*Max aggregation\*\* ensures one suspicious neighbor dominates the signal instead of being diluted

\- \*\*Temporal encoding\*\* gives the model learned representations of each 2-week time window

\- \*\*Transformer\*\* attends across all transactions in the same timestep, finding global patterns within time windows



\---



\## Dataset



\*\*Elliptic Bitcoin Dataset\*\* — real Bitcoin blockchain data

\- 203,769 transactions (nodes)

\- 234,355 payment flows (edges)

\- 166 anonymized features per transaction

\- 49 timesteps (\~2 weeks each)

\- Labels: 4,545 illicit, 42,019 licit, 157,205 unknown



\---



\## Project Structure

```

aml-transaction-detector/

├── app.py                  # Streamlit demo

├── phase1.ipynb            # Data exploration

├── phase2.ipynb            # Graph construction  

├── phase3.ipynb            # Model training

├── phase4.ipynb            # Evaluation + baselines

└── README.md

```



\---



\## Demo



Interactive Streamlit app showing:

\- Real-time transaction network visualization

\- Red nodes = flagged illicit transactions

\- Suspicion scores per transaction

\- Illicit activity over time chart

\- Adjustable detection threshold



\---



\## Setup

```bash

git clone https://github.com/YOUR\_USERNAME/aml-transaction-detector.git

cd aml-transaction-detector

python -m venv venv

venv\\Scripts\\activate

pip install torch torch-geometric streamlit plotly networkx pandas scikit-learn

```



Download the Elliptic dataset from \[Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place CSVs in `data/`.

```bash

streamlit run app.py

```



\---



\## Tech Stack



\- \*\*PyTorch\*\* — model training

\- \*\*PyTorch Geometric\*\* — graph neural networks

\- \*\*Streamlit\*\* — interactive demo

\- \*\*Plotly\*\* — network visualization

\- \*\*Scikit-learn\*\* — evaluation metrics



\---



\## Key Design Decisions



\*\*Why not plain GCN?\*\* GCN's symmetric normalization dilutes suspicious signals from minority neighbors. GraphSAGE with max aggregation preserves them.



\*\*Why Transformer over time?\*\* Money laundering happens in coordinated bursts. Transformer attention across a timestep captures coordinated activity that per-node models miss.



\*\*Why ensemble?\*\* V1 and V2 have different inductive biases — V1 is simpler and faster, V2 is deeper with richer temporal encoding. Averaging their probabilities consistently outperforms either alone.



\*\*Why temporal split evaluation?\*\* Random splits leak future patterns into training. Temporal split (train on t=1-35, test on t=36-49) is the honest production-realistic benchmark.

