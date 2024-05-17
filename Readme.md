# SLIVER: Unveiling Large Scale Gene Regulatory Networks of Single-Cell Transcriptomic Data through Causal Structure Learning and Modules Aggregation

## Introduction
Prevalent Gene Regulatory Network (GRN) construction methods rely on generalized correlation analysis. However, in biological systems, regulation is essentially a causal relationship that cannot be adequately captured solely through correlation. Therefore, it is more reasonable to infer GRNs from a causal perspective. Existing causal discovery algorithms typically rely on Directed Acyclic Graphs (DAGs) to model causal relationships, but it often requires traversing the entire network, which result in computational demands skyrocketing as the number of nodes grows and make causal discovery algorithms only suitable for small networks with one or two hundred nodes or fewer. In this study, we propose the SLIVER (cauSaL dIscovery Via dimEnsionality Reduction) algorithm which integrates causal structural equation model and graph decomposition. SLIVER introduces a set of factor nodes, serving as abstractions of different functional modules to integrate the regulatory relationships between genes based on their respective functions or pathways, thus reducing the GRN to the product of two low-dimensional matrices. Subsequently, we employ the structural causal model (SCM) to learn the GRN within the gene node space, enforce the DAG constraint in the low-dimensional space, and guide each factor to aggregate various functions through cosine similarity. We evaluate the performance of the SLIVER algorithm on 12 real single cell transcriptomic datasets, and demonstrate it outperforms other 11 widely used methods both in GRN inference performance and computational resource usage. The analysis of the gene information integrated by factor nodes also demonstrate the biological explanation of factor nodes in GRNs. 

## Authors

- Hongyang Jiang, Yuezhu Wang, Chaoyi Yin, Hao Pan, Liqun Chen, Ke Feng, Yi Chang, Huiyan Sun


## Requirements

- Python 3.8.16
- scikit-learn==1.1.3
- torch==1.12.0
- GPUtil==1.4.0
- numpy==1.24.3
- pandas==1.4.2
- matplotlib==3.6.3

## Example

You can run SLIVER with the following command-line arguments:

```
python sliver.py
```

### Command-line Arguments:

- `-notcuda`: Flag to indicate running the algorithm without CUDA support.
- `-cache_dir`: Path to the directory for caching intermediate results. Default is "./cache/".
- `-data`: Name of the dataset to be used. Default is "hESC-CellType-500".
- `-nodes`: Number of nodes to be used in the algorithm. Default is 64.
