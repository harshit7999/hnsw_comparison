import faiss
import numpy as np
import time
import logging
from model.model import Model

logger = logging.getLogger(__name__)

class FaissHnswModel(Model):
    """FAISS HNSW model implementation"""

    def generate_index(self, data, model_params):
        """
        Generate HNSW index using FAISS.
        
        Args:
            data (np.ndarray): Data to index
            model_params (dict): Model parameters including ef_construction, M
            
        Returns:
            tuple: (index, index_time, memory_used)
        """
        try:
            # Extract parameters
            ef_construction = model_params.get('ef_construction', 100)
            M = model_params.get('M', 16)
            dim = data.shape[1]
            
            # Ensure data type is correct
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            logger.info(f"Creating FAISS HNSW index: dim={dim}, vectors={data.shape[0]}, ef_construction={ef_construction}, M={M}")
            
            # Create index
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = ef_construction
            
            # Measure memory and time
            before = self.get_memory_usage_mb()
            start = time.time()
            
            # Add data to index
            index.add(data)
            
            index_time = time.time() - start
            after = self.get_memory_usage_mb()
            memory_used = after - before
            
            # Handle negative memory usage
            if memory_used < 0:
                logger.warning("Negative memory usage detected, setting to 0")
                memory_used = 0.0
            
            logger.info(f"FAISS HNSW index created successfully: {index_time:.4f}s, {memory_used:.2f}MB")
            return index, index_time, memory_used
            
        except Exception as e:
            if "faiss" in str(e).lower():
                raise RuntimeError(f"FAISS indexing failed: {e}")
            raise RuntimeError(f"Failed to create FAISS HNSW index: {e}")

    def run_queries(self, index, queries, k):
        """
        Run queries against the FAISS HNSW index.
        
        Args:
            index: FAISS HNSW index
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
            distances, found_neighbors = index.search(queries, k)
            query_time = time.time() - start
            
            logger.info(f"Queries completed: {query_time:.4f}s")
            return found_neighbors, distances, query_time
            
        except Exception as e:
            if "faiss" in str(e).lower():
                raise RuntimeError(f"FAISS querying failed: {e}")
            raise RuntimeError(f"Failed to run queries: {e}")
