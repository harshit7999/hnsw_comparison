# HNSW Vector Database Comparison

A comprehensive benchmarking framework for comparing different HNSW (Hierarchical Navigable Small World) vector search algorithm implementations. This project evaluates performance metrics across multiple vector database libraries to help you choose the best HNSW implementation for your use case.

## 🎯 Overview

This benchmark compares popular HNSW implementations across multiple dimensions:
- **Indexing Performance**: Time to build the index and memory consumption
- **Query Performance**: Search speed and accuracy (recall@k)
- **Scalability**: Performance across different vector dimensions and dataset sizes
- **Parameter Sensitivity**: Impact of HNSW parameters (ef_construction, M) on performance

## 🔧 Supported Libraries

| Library | Implementation | Status |
|---------|---------------|--------|
| **HNSWLib** | `hnsw_lib` | ✅ Active |
| **FAISS HNSW** | `faiss_hnsw` | ✅ Active |
| **NMSLIB HNSW** | `nmslib_hnsw` | ✅ Active |

## 📊 Metrics Measured

- **Indexing Time**: Time required to build the vector index
- **Memory Usage**: RAM consumption during indexing
- **Query Time**: Average time per query across multiple searches
- **Recall@K**: Accuracy of top-k nearest neighbor retrieval

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hnsw_comparison
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

Run a benchmark with the default configuration:

```bash
python main.py --config config/config.json --report_path results/benchmark_results.csv
```

### Custom Configuration

Create your own configuration file:

```json
{
    "run_count": 3,
    "models": ["hnsw_lib", "faiss_hnsw", "nmslib_hnsw"],
    "dataset_params": [
        { "dim": 128, "num_vectors": 10000 },
        { "dim": 256, "num_vectors": 50000 }
    ],
    "model_params": [
        { "ef_construction": 100, "M": 16 },
        { "ef_construction": 200, "M": 32 }
    ],
    "query_params": [
        { "num_queries_percent": 10, "k": 10 }
    ]
}
```

## ⚙️ Configuration Options

### Dataset Parameters
- `dim`: Vector dimension (e.g., 100, 200, 300, 500)
- `num_vectors`: Number of vectors in the dataset (e.g., 10000, 50000, 100000)
- `seed`: Random seed for reproducible results (default: 42)

### Model Parameters (HNSW-specific)
- `ef_construction`: Size of the dynamic candidate list (higher = better recall, slower indexing)
- `M`: Number of bi-directional links created for each node (higher = better recall, more memory)

### Query Parameters
- `num_queries_percent`: Percentage of dataset to use as queries (e.g., 10 = 10% of vectors)
- `k`: Number of nearest neighbors to retrieve

### General Settings
- `run_count`: Number of benchmark runs per configuration (for statistical significance)
- `models`: List of HNSW implementations to benchmark

## 📈 Results and Visualization

Results are automatically saved in two formats:

### CSV Results
- Location: `results/[model_name]_output.csv`
- Contains: All metrics for each configuration combination
- Headers: run_count, model, vector_dimension, total_vectors, ef_construction, M, index_time, memory_used, num_queries_percent, K, query_time, recall_at_k

### Visualization Graphs
- Location: `results/graphs/`
- Generated using the Jupyter notebook: `results/graphs.ipynb`
- Available plots:
  - Indexing time vs. parameters
  - Memory usage vs. parameters  
  - Query time vs. parameters
  - Recall@k vs. parameters

## 📁 Project Structure

```
hnsw_comparison/
├── main.py                     # Main benchmarking script
├── benchmark.py               # Core benchmarking logic
├── config/
│   └── config.json           # Default configuration
├── model/
│   ├── model.py              # Abstract base model
│   ├── model_factory.py      # Model factory pattern
│   ├── hnsw_model.py         # HNSWLib implementation
│   ├── faiss_hnsw_model.py   # FAISS HNSW implementation
│   └── nmslib_hsnw_model.py  # NMSLIB implementation
├── results/
│   ├── *.csv                 # Benchmark results
│   ├── graphs/               # Performance visualizations
│   └── graphs.ipynb          # Plotting notebook
└── requirements.txt          # Python dependencies
```

## 🔍 Example Results Analysis

The benchmark helps answer questions like:
- Which HNSW implementation is fastest for your vector dimension?
- What's the memory vs. accuracy trade-off for different parameters?
- How do libraries scale with dataset size?
- What are optimal ef_construction and M values for your use case?

## 🤝 Contributing

1. Fork the repository
2. Improve existing implementations or add new features
3. Add corresponding tests and documentation
4. Submit a pull request


## 🙏 Acknowledgments

This benchmark framework builds upon the excellent work of:
- [HNSWLib](https://github.com/nmslib/hnswlib)
- [FAISS](https://github.com/facebookresearch/faiss)
- [NMSLIB](https://github.com/nmslib/nmslib)

---

**Note**: This benchmark uses synthetic random vectors for fair comparison. For production use cases, consider testing with your actual data distribution. 