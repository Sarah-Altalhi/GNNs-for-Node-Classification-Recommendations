# Graph Neural Networks for Node Classification & Recommendations

This project applies **Graph Neural Networks (GNNs)** to:
- **Node classification** on the **Planetoid** (Cora) citation network.
- **Recommendation / rating prediction** on the **MovieLens** dataset using a user–movie bipartite graph.

---

## ✨ Highlights
- Implemented with **PyTorch Geometric** (PyG).
- **GCN** and **GraphSAGE** layers for graph representation learning.
- **Planetoid (Cora)**: node classification over citation graph structure.
- **MovieLens**: GNN-based rating prediction (graph signal over user–movie interactions).
- Clear, reproducible notebooks.

---

## 📂 Project Structure

gnn-node-recommendation/
├── LICENSE
├── README.md
│
├── Planetoid_Sarah_Altalhi.ipynb # Planetoid (Cora) node classification
├── Movielens_Sarah Altalhi.ipynb # MovieLens rating prediction (GNN)
│
├── configs/ # YAML configs (optional)
├── data/ # Datasets/cache (ignored by git)
├── experiments/ # Logs, metrics, figures 
├── models/ # Saved checkpoints/embeddings
├── notebooks/ # Extra notebooks (optional)
├── scripts/ # Setup/run helper scripts
│
├── src/
│ ├── datasets/ # Data loaders & preprocessing 
│ ├── models/ # GNN architectures (GCN, GraphSAGE)
│ └── utils/ # Training loops, metrics, helpers
│
└── tests/ # Optional unit tests

---

## ⚙️ Setup

### 1) Clone
```bash
git clone https://github.com/Sarah-Altalhi/gnn-node-recommendation.git
cd gnn-node-recommendation
```
### 2) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

source .venv/bin/activate # Linux/Mac:

### 3) Install dependencies
pip install --upgrade pip

pip install -r requirements.txt

Make sure PyTorch Geometric is installed with the right CUDA/CPU wheels per your environment.

See: https://pytorch-geometric.readthedocs.io/

--- 
## 🚀 Running the Notebooks
### Planetoid (Cora) — Node Classification
jupyter notebook "Planetoid_Sarah_Altalhi.ipynb"


The notebook covers:

- Loading Cora (features, edges, labels).

- Building GCN / GraphSAGE.

- Training & validation loop.



### MovieLens — Rating Prediction
jupyter notebook "Movielens_Sarah Altalhi.ipynb"


The notebook covers:

- Building a user–movie bipartite graph.

- Training a GNN to predict ratings.

- Evaluating with RMSE.


---
## 📊 Results

### Planetoid (Cora) — Node Classification

| Model     | Test Accuracy |
|-----------|---------------|
| GCN       | **0.82**      |
| GraphSAGE | **0.84**      |

---

### MovieLens — Rating Prediction (GNN)

| Model | Test RMSE |
|-------|-----------|
| GNN   | **0.8826** |

---
## 🧠 Key Ideas

- Message passing lets nodes aggregate neighbor information (graph structure → learned embeddings).

- For Cora, homophily in citation networks helps GCN/GraphSAGE reach strong accuracy.

- For MovieLens, the user–item bipartite structure captures collaborative signals for rating prediction.

---

## 📚 References

Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks.


Hamilton et al. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE).


PyTorch Geometric Documentation.
