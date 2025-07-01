import argparse
import os
import json
import csv
import time
import sys
import logging
from benchmark import run_benchmark

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_results_to_csv(results, output_path):
    """Save benchmark results to CSV file"""
    headers = [
        "run_count",
        "model",
        "vector_dimension",
        "total_vectors",
        "ef_construction",
        "M",
        "index_time",
        "memory_used",
        "num_queries_percent",
        "K",
        "query_time",
        "recall_at_k"
    ]

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        with open(output_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"Results successfully saved to {output_path}")
    except PermissionError:
        logger.error(f"Permission denied: Cannot write to {output_path}")
        raise
    except OSError as e:
        logger.error(f"OS error when saving results: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when saving results: {e}")
        raise

def validate_config_structure(config):
    """Validate the structure and content of the configuration."""
    required_keys = ["models", "dataset_params", "query_params", "model_params"]
    
    # Check required keys exist
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate models list
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("'models' must be a non-empty list")
    
    # Validate dataset_params
    if not isinstance(config["dataset_params"], list) or not config["dataset_params"]:
        raise ValueError("'dataset_params' must be a non-empty list")
    
    for i, dataset_param in enumerate(config["dataset_params"]):
        if not isinstance(dataset_param, dict):
            raise ValueError(f"dataset_params[{i}] must be a dictionary")
        if "dim" not in dataset_param or "num_vectors" not in dataset_param:
            raise ValueError(f"dataset_params[{i}] must contain 'dim' and 'num_vectors'")
        if not isinstance(dataset_param["dim"], int) or dataset_param["dim"] <= 0:
            raise ValueError(f"dataset_params[{i}]['dim'] must be a positive integer")
        if not isinstance(dataset_param["num_vectors"], int) or dataset_param["num_vectors"] <= 0:
            raise ValueError(f"dataset_params[{i}]['num_vectors'] must be a positive integer")
    
    # Validate model_params
    if not isinstance(config["model_params"], list) or not config["model_params"]:
        raise ValueError("'model_params' must be a non-empty list")
    
    for i, model_param in enumerate(config["model_params"]):
        if not isinstance(model_param, dict):
            raise ValueError(f"model_params[{i}] must be a dictionary")
        if "ef_construction" not in model_param or "M" not in model_param:
            raise ValueError(f"model_params[{i}] must contain 'ef_construction' and 'M'")
    
    # Validate query_params
    if not isinstance(config["query_params"], list) or not config["query_params"]:
        raise ValueError("'query_params' must be a non-empty list")
    
    for i, query_param in enumerate(config["query_params"]):
        if not isinstance(query_param, dict):
            raise ValueError(f"query_params[{i}] must be a dictionary")
        if "num_queries_percent" not in query_param or "k" not in query_param:
            raise ValueError(f"query_params[{i}] must contain 'num_queries_percent' and 'k'")
        if not (0 < query_param["num_queries_percent"] <= 100):
            raise ValueError(f"query_params[{i}]['num_queries_percent'] must be between 1 and 100")
        if not isinstance(query_param["k"], int) or query_param["k"] <= 0:
            raise ValueError(f"query_params[{i}]['k'] must be a positive integer")
    
    # Validate run_count
    run_count = config.get("run_count", 1)
    if not isinstance(run_count, int) or run_count <= 0:
        raise ValueError("'run_count' must be a positive integer")
    
    return True

def load_and_validate_config(config_path):
    """Load and validate configuration file."""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not os.access(config_path, os.R_OK):
            raise PermissionError(f"Cannot read configuration file: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        validate_config_structure(config)
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def main():
    """Main function"""
    start_time = time.time()
    logger.info(f"Starting vector DB comparison at {time.ctime()}")
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run vector DB comparison with specified config file')
        parser.add_argument('--config', '-c', type=str, required=True, 
                          help='Path to the configuration file')
        parser.add_argument('--report_path', '-r', type=str, required=True, 
                          help='Path to the report file (CSV format)')
        args = parser.parse_args()
        
        config_path = args.config
        report_path = args.report_path
        
        # Load and validate configuration
        config = load_and_validate_config(config_path)
        
        # Extract configuration parameters
        models = config["models"]
        dataset_params = config["dataset_params"]
        query_params = config["query_params"]
        model_params = config["model_params"]
        run_count = config.get("run_count", 1)
        
        logger.info(f"Starting benchmark with {len(models)} models, "
                   f"{len(dataset_params)} dataset configs, "
                   f"{len(model_params)} model configs, "
                   f"{len(query_params)} query configs, "
                   f"{run_count} runs each")
        
        final_results = []
        failed_runs = []
        total_configurations = len(models) * len(dataset_params) * len(model_params) * len(query_params) * run_count
        completed_runs = 0
        
        # Run benchmarks
        for model in models:
            for dataset_param in dataset_params:
                for model_param in model_params:
                    for query_param in query_params:
                        for run_index in range(run_count):
                            completed_runs += 1
                            config_id = f"{model}_{dataset_param['dim']}d_{dataset_param['num_vectors']}v_ef{model_param['ef_construction']}_M{model_param['M']}_run{run_index + 1}"
                            
                            try:
                                logger.info(f"[{completed_runs}/{total_configurations}] Running configuration: {config_id}")
                                print("-" * 80)
                                print(f"Run {run_index + 1}/{run_count} - Model: {model}")
                                print(f"Dataset: {dataset_param}, Model params: {model_param}, Query: {query_param}")
                                
                                res = run_benchmark(model, dataset_param, model_param, query_param)
                                
                                final_results.append({
                                    "run_count": run_index + 1,
                                    "model": model,
                                    "vector_dimension": dataset_param.get("dim"),
                                    "total_vectors": dataset_param.get("num_vectors"),
                                    "ef_construction": model_param.get("ef_construction"),
                                    "M": model_param.get("M"),
                                    "index_time": res["indexing"]["time"],
                                    "memory_used": res["indexing"]["memory_used"],
                                    "num_queries_percent": query_param.get("num_queries_percent"),
                                    "K": query_param.get("k"),
                                    "query_time": res["querying"]["time"],
                                    "recall_at_k": res["querying"]["recall_at_k"]
                                })
                                
                                logger.info(f"Completed configuration: {config_id}")
                                
                            except Exception as e:
                                error_msg = f"Failed configuration {config_id}: {str(e)}"
                                logger.error(error_msg)
                                failed_runs.append({
                                    "config_id": config_id,
                                    "error": str(e),
                                    "model": model,
                                    "dataset_param": dataset_param,
                                    "model_param": model_param,
                                    "query_param": query_param,
                                    "run_index": run_index + 1
                                })
                                print(f"Warning: {error_msg}")
                                
                                # Continue with next configuration instead of stopping
                                continue
        
        # Save results
        if final_results:
            save_results_to_csv(final_results, report_path)
            logger.info(f"Saved {len(final_results)} successful results to {report_path}")
        else:
            logger.warning("No successful results to save!")
        
        # Report summary
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total configurations attempted: {total_configurations}")
        print(f"Successful runs: {len(final_results)}")
        print(f"Failed runs: {len(failed_runs)}")
        print(f"Success rate: {len(final_results)/total_configurations*100:.1f}%")
        print(f"Total execution time: {execution_time:.2f} seconds")
        
        if failed_runs:
            print(f"\nFailed configurations:")
            for failed in failed_runs:
                print(f"  - {failed['config_id']}: {failed['error']}")
        
        logger.info(f"Benchmark completed in {execution_time:.2f} seconds")
        
        # Exit with appropriate code
        if len(final_results) == 0:
            logger.error("All benchmark runs failed!")
            sys.exit(1)
        elif failed_runs:
            logger.warning(f"Some benchmark runs failed ({len(failed_runs)}/{total_configurations})")
            sys.exit(2)  # Partial success
        else:
            logger.info("All benchmark runs completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        print("\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()