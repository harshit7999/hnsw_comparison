# benchmark.py
from model.model_factory import ModelFactory
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)



def run_benchmark(model_type, dataset_params, model_params, query_params):
    """Run benchmark with comprehensive error handling."""
    
    try:
        logger.info(f"Starting benchmark for {model_type}")
        print(f"Running benchmark for {model_type}...")

        # Basic parameter extraction
        
        # Extract parameters
        vector_dim = dataset_params.get("dim")
        vector_count = dataset_params.get("num_vectors")
        seed = dataset_params.get("seed", 42)
        
        num_queries_percent = query_params.get("num_queries_percent")
        k = query_params.get("k")
        
        # Calculate number of queries
        num_queries = int((num_queries_percent / 100) * vector_count)

        print("\nPreparing model...")
        try:
            model = ModelFactory.get_model(model_type)
            logger.info(f"Model {model_type} created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create model '{model_type}': {str(e)}")

        # Load/generate dataset
        print(f"Loading {vector_count} vectors of dimension {vector_dim} with seed {seed}...")
        try:
            data = model.load_dataset(vector_count, vector_dim, seed)
            
            # Basic data validation
            
            logger.info(f"Loaded {data.shape[0]} vectors of dimension {data.shape[1]}")
            print(f"Loaded {data.shape[0]} vectors of dimension {data.shape[1]}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

        # Generate ground truth
        print(f"Preparing ground truth for {num_queries} queries with k={k}...")
        try:
            query_vectors = data[:num_queries]
            ground_truth_neighbors = model.generate_ground_truth(data, query_vectors, k)
            
            logger.info(f"Generated ground truth for {num_queries} queries")
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate ground truth: {str(e)}")

        # Build index
        print(f"\nIndexing with {model_type}...")
        try:
            index_start_time = time.time()
            index, index_time, mem_used = model.generate_index(data, model_params)
            actual_index_time = time.time() - index_start_time
            
            # Use actual timing if model timing seems unreliable
            if abs(index_time - actual_index_time) > actual_index_time * 0.5:
                logger.warning(f"Model reported index time ({index_time:.4f}s) differs significantly from measured time ({actual_index_time:.4f}s)")
                index_time = actual_index_time
            
            logger.info(f"Index created in {index_time:.4f}s, memory used: {mem_used:.2f}MB")
            
        except Exception as e:
            raise RuntimeError(f"Failed to build index: {str(e)}")

        # Run queries
        print(f"\nQuerying {num_queries} queries with k={k}...")
        try:
            query_start_time = time.time()
            found_neighbors, distances, query_time = model.run_queries(index, query_vectors, k)
            actual_query_time = time.time() - query_start_time
            
            # Use actual timing if model timing seems unreliable
            if abs(query_time - actual_query_time) > actual_query_time * 0.5:
                logger.warning(f"Model reported query time ({query_time:.4f}s) differs significantly from measured time ({actual_query_time:.4f}s)")
                query_time = actual_query_time
            
            logger.info(f"Queries completed in {query_time:.4f}s")
            
        except Exception as e:
            raise RuntimeError(f"Failed to run queries: {str(e)}")

        # Calculate recall
        print("Calculating Recall@K...")
        try:
            recall_at_k = model.get_recall_at_k(ground_truth_neighbors, found_neighbors, k)
            
            # Basic recall validation
            
            logger.info(f"Recall@{k} calculated: {recall_at_k:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate recall: {str(e)}")

        # Final results
        results = {
            "indexing": {
                "time": float(index_time),
                "memory_used": float(mem_used)
            },
            "querying": {
                "time": float(query_time),
                "recall_at_k": float(recall_at_k)
            }
        }
        
        # Report
        print("\nBenchmark Results")
        print(f"Indexing Time     : {index_time:.4f} s")
        print(f"Query Time        : {query_time:.4f} s for {num_queries} queries")
        print(f"Recall@{k}        : {recall_at_k:.4f}")
        print(f"Memory Used       : {mem_used:.2f} MB")
        
        logger.info(f"Benchmark completed successfully for {model_type}")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed for {model_type}: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller