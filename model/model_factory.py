from model.hnsw_model import HnswModel
from model.faiss_hnsw_model import FaissHnswModel
from model.nmslib_hsnw_model import NmslibModel

class ModelFactory:
    """Factory class for creating HNSW model instances"""
    
    # Supported model types
    SUPPORTED_MODELS = {
        'hnsw_lib': HnswModel,
        'faiss_hnsw': FaissHnswModel,
        'nmslib_hnsw': NmslibModel
    }
    
    @staticmethod
    def get_model(db_type):
        """
        Create and return a model instance for the specified type.
        
        Args:
            db_type (str): The type of model to create
            
        Returns:
            Model: An instance of the requested model type
            
        Raises:
            ValueError: If db_type is invalid or unsupported
            ImportError: If required dependencies are missing
            Exception: If model instantiation fails
        """
        if not isinstance(db_type, str) or not db_type.strip():
            raise ValueError("Model type must be a non-empty string")
        
        db_type = db_type.strip()
        
        if db_type not in ModelFactory.SUPPORTED_MODELS:
            supported_list = ', '.join(ModelFactory.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unknown model type: '{db_type}'. Supported types: {supported_list}")
        
        try:
            model_class = ModelFactory.SUPPORTED_MODELS[db_type]
            return model_class()
        except ImportError as e:
            raise ImportError(f"Failed to import dependencies for '{db_type}': {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model '{db_type}': {str(e)}")
    
    @staticmethod
    def get_supported_models():
        """Return a list of supported model types."""
        return list(ModelFactory.SUPPORTED_MODELS.keys())