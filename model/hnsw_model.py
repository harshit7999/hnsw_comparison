import hnswlib
import numpy as np
import time
import logging
from model.model import Model

logger = logging.getLogger(__name__)

class HnswModel(Model):
    """HNSW model implementation using hnswlib"""

    def generate_index(self, data, model_params):
        """
        Generate HNSW index using hnswlib.
        
        Args:
            data (np.ndarray): Data to index
            model_params (dict): Model parameters including ef_construction, M, max_elements
            
        Returns:
            tuple: (index, index_time, memory_used)
        """
        try:
            # Extract parameters
            max_elements = model_params.get('max_elements', data.shape[0])
            ef_construction = model_params.get('ef_construction', 100)
            M = model_params.get('M', 16)
            dim = data.shape[1]
            total_vectors = data.shape[0]
            
            # Ensure data type is correct
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            logger.info(f"Creating HNSW index: dim={dim}, vectors={total_vectors}, ef_construction={ef_construction}, M={M}")
            
            # Create index
            index = hnswlib.Index(space='l2', dim=dim)
            index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            index.set_num_threads(4)
            
            # Measure memory and time
            before = self.get_memory_usage_mb()
            start = time.time()
            
            # Add data to index
            index.add_items(data)
            
            index_time = time.time() - start
            after = self.get_memory_usage_mb()
            memory_used = after - before
            
            # Handle negative memory usage
            if memory_used < 0:
                logger.warning("Negative memory usage detected, setting to 0")
                memory_used = 0.0
            
            logger.info(f"HNSW index created successfully: {index_time:.4f}s, {memory_used:.2f}MB")
            return index, index_time, memory_used
            
        except Exception as e:
            if "hnswlib" in str(e).lower():
                raise RuntimeError(f"HNSWLib indexing failed: {e}")
            raise RuntimeError(f"Failed to create HNSW index: {e}")

    def run_queries(self, index, queries, k):
        """
        Run queries against the HNSW index.
        
        Args:
            index: HNSW index
            queries (np.ndarray): Query vectors
            k (int): Number of nearest neighbors
            
        Returns:
            tuple: (neighbors, distances, query_time)
        """
        try:
            # Ensure data type is correct
            if queries.dtype != np.float32:
                queries = queries.astype(np.float32)
            
            logger.info(f"Running {queries.shape[0]} queries with k={k}")
            
            # Run queries
            start = time.time()
            found_neighbors, distances = index.knn_query(queries, k=k)
            query_time = time.time() - start
            
            logger.info(f"Queries completed: {query_time:.4f}s")
            return found_neighbors, distances, query_time
            
        except Exception as e:
            if "hnswlib" in str(e).lower():
                raise RuntimeError(f"HNSWLib querying failed: {e}")
            raise RuntimeError(f"Failed to run queries: {e}")
