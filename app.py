import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, precision_score, recall_score

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AML Transaction Detector",
    page_icon="🔍",
    layout="wide"
)

# ===== MODEL DEFINITIONS =====
class AMLDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.5):
        super(AMLDetector, self).__init__()
        self.sage = SAGEConv(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, timesteps):
        x = self.sage(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        out = torch.zeros_like(x)
        for t in timesteps.unique():
            mask = (timesteps == t)
            x_t = x[mask].unsqueeze(0)
            x_t = self.transformer(x_t)
            out[mask] = x_t.squeeze(0)
        out = self.dropout(out)
        return self.classifier(out)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_timesteps=50):
        super(TemporalPositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_timesteps, hidden_dim)

    def forward(self, x, timesteps):
        return x + self.embedding(timesteps)


class AMLDetectorV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.3):
        super(AMLDetectorV2, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.sage3 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.temporal_enc = TemporalPositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, timesteps):
        x = F.relu(self.input_proj(x))
        x = self.dropout(F.relu(self.bn1(self.sage1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.sage2(x, edge_index))))
        x = self.dropout(F.relu(self.bn3(self.sage3(x, edge_index))))
        x = self.temporal_enc(x, timesteps)
        out = torch.zeros_like(x)
        for t in timesteps.unique():
            mask = (timesteps == t)
            x_t = x[mask].unsqueeze(0)
            x_t = self.transformer(x_t)
            out[mask] = x_t.squeeze(0)
        return self.classifier(self.dropout(out))


# ===== LOAD DATA AND MODELS =====
@st.cache_resource
def load_everything():
    # Load CSVs
    features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
    edges    = pd.read_csv('data/elliptic_txs_edgelist.csv')
    classes  = pd.read_csv('data/elliptic_txs_classes.csv')
    features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]

    # Build mappings
    tx_ids   = features['txId'].values
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    idx_to_id = {idx: tx_id for tx_id, idx in id_to_idx.items()}

    # Build graph tensors
    src = edges['txId1'].map(id_to_idx).values
    dst = edges['txId2'].map(id_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    feature_cols = [f'feat_{i}' for i in range(165)]
    x = torch.tensor(features[feature_cols].values, dtype=torch.float)

    timesteps_raw = features['timestep'].values
    timesteps_norm = (timesteps_raw - timesteps_raw.min()) / (timesteps_raw.max() - timesteps_raw.min())
    timestep_tensor = torch.tensor(timesteps_norm, dtype=torch.float).unsqueeze(1)
    x = torch.cat([x, timestep_tensor], dim=1)

    timesteps_long = torch.tensor(timesteps_raw, dtype=torch.long)

    # Labels
    merged = features[['txId']].merge(classes, on='txId', how='left')
    def convert_label(label):
        if label == '1': return 0
        elif label == '2': return 1
        else: return -1
    labels = merged['class'].astype(str).map(convert_label).values
    y = torch.tensor(labels, dtype=torch.long)

    # Load models
    model_v1 = AMLDetector(input_dim=166, hidden_dim=128, output_dim=2)
    model_v1.load_state_dict(torch.load('data/best_model.pt', map_location='cpu', weights_only=False))
    model_v1.eval()

    model_v2 = AMLDetectorV2(input_dim=166, hidden_dim=128, output_dim=2)
    model_v2.load_state_dict(torch.load('data/best_model_v2.pt', map_location='cpu', weights_only=False))
    model_v2.eval()

    return features, edges, x, edge_index, timesteps_long, y, id_to_idx, idx_to_id, model_v1, model_v2


# ===== GET ENSEMBLE PREDICTIONS =====
@st.cache_data
def get_predictions():
    features, edges, x, edge_index, timesteps, y, id_to_idx, idx_to_id, model_v1, model_v2 = load_everything()
    with torch.no_grad():
        out_v1 = model_v1(x, edge_index, timesteps)
        probs_v1 = torch.softmax(out_v1, dim=1)
        out_v2 = model_v2(x, edge_index, timesteps)
        probs_v2 = torch.softmax(out_v2, dim=1)
    ensemble_probs = (probs_v1 + probs_v2) / 2
    ensemble_pred  = ensemble_probs.argmax(dim=1)
    illicit_prob   = ensemble_probs[:, 0].numpy()
    return ensemble_pred.numpy(), illicit_prob, y.numpy()


# ===== MAIN APP =====
st.title("🔍 AML Transaction Detection System")
st.markdown("**GraphSAGE + Transformer ensemble for detecting illicit Bitcoin transactions**")

# Sidebar
st.sidebar.title("Controls")
selected_timestep = st.sidebar.slider(
    "Select Timestep", min_value=1, max_value=49, value=25,
    help="Each timestep = ~2 weeks of Bitcoin transactions"
)
confidence_threshold = st.sidebar.slider(
    "Suspicious Threshold", min_value=0.3, max_value=0.9, value=0.5, step=0.05,
    help="Transactions above this probability are flagged as suspicious"
)

# Load data
with st.spinner("Loading models and data..."):
    features, edges, x, edge_index, timesteps, y, id_to_idx, idx_to_id, model_v1, model_v2 = load_everything()
    predictions, illicit_probs, true_labels = get_predictions()

st.success("Models loaded!")

# ===== METRICS ROW =====
col1, col2, col3, col4 = st.columns(4)

labeled_mask = true_labels != -1
y_true = true_labels[labeled_mask]
y_pred = predictions[labeled_mask]

col1.metric("F1 Score",        f"{f1_score(y_true, y_pred, pos_label=0):.4f}")
col2.metric("Precision",       f"{precision_score(y_true, y_pred, pos_label=0, zero_division=0):.4f}")
col3.metric("Recall",          f"{recall_score(y_true, y_pred, pos_label=0, zero_division=0):.4f}")
col4.metric("Illicit Flagged", f"{(predictions[labeled_mask] == 0).sum():,}")

st.divider()

# ===== TIMESTEP ANALYSIS =====
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"Transaction Network — Timestep {selected_timestep}")

    # Get nodes in this timestep
    timestep_mask = (features['timestep'] == selected_timestep).values
    timestep_indices = np.where(timestep_mask)[0]

    if len(timestep_indices) > 200:
        timestep_indices = timestep_indices[:200]

    timestep_set = set(timestep_indices)

    # Build subgraph
    G = nx.DiGraph()
    for idx in timestep_indices:
        G.add_node(idx)

    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    for s, d in zip(edge_src, edge_dst):
        if s in timestep_set and d in timestep_set:
            G.add_edge(s, d)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Color nodes by prediction
    node_colors = []
    node_texts  = []
    for node in G.nodes():
        prob = illicit_probs[node]
        pred = predictions[node]
        true = true_labels[node]
        if pred == 0 and prob >= confidence_threshold:
            node_colors.append('#e74c3c')   # red = flagged illicit
        elif true == 0:
            node_colors.append('#e67e22')   # orange = actually illicit but not flagged
        else:
            node_colors.append('#2ecc71')   # green = licit
        node_texts.append(f"TX: {idx_to_id[node]}<br>Illicit prob: {prob:.3f}<br>Pred: {'Illicit' if pred==0 else 'Licit'}<br>True: {'Illicit' if true==0 else 'Licit' if true==1 else 'Unknown'}")

    # Draw edges
    edge_traces = []
    for s, d in G.edges():
        if s in pos and d in pos:
            edge_traces.append(go.Scatter(
                x=[pos[s][0], pos[d][0], None],
                y=[pos[s][1], pos[d][1], None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ))

    # Draw nodes
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers',
        hoverinfo='text',
        text=node_texts,
        marker=dict(size=8, color=node_colors, line=dict(width=0.5, color='white'))
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Flagged illicit  🟠 Missed illicit  🟢 Licit")

with col_right:
    st.subheader("Suspicious Transactions")

    # Show top suspicious transactions for this timestep
    ts_df = features[features['timestep'] == selected_timestep][['txId']].copy()
    ts_df = ts_df.reset_index()
    ts_df['index_val'] = ts_df['index']
    ts_df['illicit_probability'] = ts_df['index_val'].apply(
        lambda i: float(illicit_probs[i]) if i < len(illicit_probs) else 0
    )
    ts_df['prediction'] = ts_df['index_val'].apply(
        lambda i: 'ILLICIT' if predictions[i] == 0 else 'Licit'
    )
    ts_df['true_label'] = ts_df['index_val'].apply(
        lambda i: 'Illicit' if true_labels[i] == 0 else 'Licit' if true_labels[i] == 1 else 'Unknown'
    )

    suspicious = ts_df[ts_df['illicit_probability'] >= confidence_threshold].sort_values(
        'illicit_probability', ascending=False
    ).head(20)

    if len(suspicious) > 0:
        st.dataframe(
            suspicious[['txId', 'illicit_probability', 'prediction', 'true_label']].rename(columns={
                'txId': 'Transaction ID',
                'illicit_probability': 'Suspicion Score',
                'prediction': 'Prediction',
                'true_label': 'True Label'
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.info(f"No transactions above {confidence_threshold} threshold at timestep {selected_timestep}")

st.divider()

# ===== ILLICIT ACTIVITY OVER TIME =====
st.subheader("Illicit Activity Over Time")

timestep_stats = []
for t in range(1, 50):
    mask = (features['timestep'] == t).values
    indices = np.where(mask)[0]
    if len(indices) == 0:
        continue
    flagged    = (predictions[indices] == 0).sum()
    total      = len(indices)
    true_ill   = (true_labels[indices] == 0).sum()
    timestep_stats.append({
        'timestep': t,
        'flagged': int(flagged),
        'total': int(total),
        'true_illicit': int(true_ill),
        'flag_rate': flagged / total * 100
    })

stats_df = pd.DataFrame(timestep_stats)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=stats_df['timestep'], y=stats_df['true_illicit'],
    name='True Illicit', line=dict(color='#e74c3c', width=2),
    fill='tozeroy', fillcolor='rgba(231,76,60,0.1)'
))
fig2.add_trace(go.Scatter(
    x=stats_df['timestep'], y=stats_df['flagged'],
    name='Model Flagged', line=dict(color='#9b59b6', width=2, dash='dash')
))
fig2.add_vline(
    x=selected_timestep, line_dash="dot",
    line_color="white", annotation_text=f"T={selected_timestep}"
)
fig2.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation='h'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Timestep'),
    yaxis=dict(title='Count')
)
st.plotly_chart(fig2, use_container_width=True)