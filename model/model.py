import abc
import faiss
import numpy as np
import os
import psutil
import logging

logger = logging.getLogger(__name__)

class Model(abc.ABC):
    """Abstract base class for vector search models"""

    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def load_dataset(self, num_vectors, dim, seed=42):
        """
        Generate synthetic dataset.
        
        Args:
            num_vectors (int): Number of vectors to generate
            dim (int): Dimension of each vector
            seed (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Generated dataset of shape (num_vectors, dim)
        """
        try:
            # Check memory requirements (rough estimate)
            estimated_memory_gb = (num_vectors * dim * 4) / (1024**3)  # 4 bytes per float32
            if estimated_memory_gb > 8:  # Arbitrary threshold
                logger.warning(f"Large dataset: estimated {estimated_memory_gb:.2f} GB")
            
            np.random.seed(seed)
            data = np.random.rand(num_vectors, dim).astype('float32')
            
            logger.info(f"Generated dataset: {data.shape} with seed {seed}")
            return data
            
        except MemoryError:
            raise MemoryError(f"Cannot allocate memory for dataset of size ({num_vectors}, {dim})")
        except Exception as e:
            raise RuntimeError(f"Failed to generate dataset: {e}")
    
    def generate_ground_truth(self, data, queries, k):
        """
        Generate ground truth using FAISS.
        
        Args:
            data (np.ndarray): Database vectors
            queries (np.ndarray): Query vectors
            k (int): Number of nearest neighbors
            
        Returns:
            np.ndarray: Ground truth neighbor indices
        """
        try:
            # Ensure data types are correct
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            if queries.dtype != np.float32:
                queries = queries.astype(np.float32)
            
            ground_truth = faiss.IndexFlatL2(data.shape[1])
            ground_truth.add(data)
            distances, neighbors = ground_truth.search(queries, k)
            
            logger.info(f"Generated ground truth for {queries.shape[0]} queries with k={k}")
            return neighbors
            
        except Exception as e:
            if "FAISS" in str(e) or "faiss" in str(e):
                raise RuntimeError(f"FAISS operation failed: {e}")
            raise
    
    @abc.abstractmethod
    def generate_index(self, data, model_params):
        """
        Generate index for the given data.
        
        Args:
            data (np.ndarray): Data to index
            model_params (dict): Model-specific parameters
            
        Returns:
            tuple: (index, index_time, memory_used)
        """
        pass
    
    @abc.abstractmethod
    def run_queries(self, index, queries, k):
        """
        Run queries against the index.
        
        Args:
            index: The search index
            queries (np.ndarray): Query vectors
            k (int): Number of nearest neighbors
            
        Returns:
            tuple: (neighbors, distances, query_time)
        """
        pass
    
    def get_recall_at_k(self, ground_truth_neighbors, found_neighbors, k):
        """
        Calculate recall@k.
        
        Args:
            ground_truth_neighbors (np.ndarray): True nearest neighbors
            found_neighbors (np.ndarray): Found nearest neighbors
            k (int): Number of neighbors
            
        Returns:
            float: Recall@k value between 0 and 1
        """
        try:
            correct = 0
            total_queries = len(ground_truth_neighbors)
            
            # Calculate recall
            for i in range(total_queries):
                ground_truth_set = set(ground_truth_neighbors[i])
                found_set = set(found_neighbors[i])
                correct += len(ground_truth_set & found_set)
            
            recall = correct / (total_queries * k)
            
            logger.info(f"Calculated recall@{k}: {recall:.4f}")
            return recall
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate recall: {e}")
