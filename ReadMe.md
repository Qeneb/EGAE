# Embedding Graph Auto-Encoder for Graph Clustering

This is a repository for Data Mining Mid-term Homework in SJTU

The core idea of EGAE is to **design a GNN to find an ideal space for the relaxed *k*-means** on graph data. We prove that the relaxed *k*-means will obtain a precise clustering result under some strong assumptions. So we attempt to use GNNs to map the data into an ideal space that satisfies the strong assumptions.  


## How to Run EGAE
```
python run.py
```
### Requirements 
pytorch >= 1.3.1

scipy 1.3.1

scikit-learn 0.21.3

numpy 1.16.5

### Remark

- model.py: An efficient implementation which can be used when datasets are not too large. 
- sparse_model.py: It is a sparse implementation of EGAE for large scale datasets, *e.g.,* PubMed. 
