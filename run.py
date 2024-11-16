import fire
import os
from src.logger import logger
from src.model import DetM


class CliParser:
    def __init__(self):
        """Initialize the CLI wrapper and segmentator."""
        try:
            self.segmentator = DetM()
            self.segmentator.setup_env()
            logger.info("CliParser initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize CliParser: {e}")
            raise RuntimeError(f"Initialization error: {e}")

    def train(
        self,
        dataset: str,
        model_type: str = "YOLO",
        batch_size: int = 4,
        n_epochs: int = 1,
    ):
        """
        Train the model with the given dataset.

        :param dataset: Path to the dataset.
        :param model_type: Type of model to train (default: YOLO).
        :param batch_size: Batch size for training (default: 4).
        :param n_epochs: Number of epochs for training (default: 1).
        """
        if not dataset:
            raise ValueError("Dataset path is required.")

        try:
            self.segmentator.train(dataset, model_type, batch_size, n_epochs)
            models_folder_path = os.environ.get(
                "YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH", "./models"
            )
            logger.info(f"Model training complete. Model saved to {models_folder_path}")
            print(f"Model saved to {models_folder_path}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training error: {e}")

    def evaluate(
        self,
        dataset: str,
        model_type: str = "YOLO",
        use_default_model: bool = False,
        model_path: str = None,
    ):
        """
        Evaluate the model on the given dataset.

        :param dataset: Path to the evaluation dataset.
        :param model_type: Type of model to evaluate (default: YOLO).
        :param use_default_model: Whether to use the default model.
        :param model_path: Path to the custom model (if not using the default).
        """
        if not dataset:
            raise ValueError("Dataset path is required.")

        try:
            results = self.segmentator.evaluate(
                dataset, model_type, use_default_model, model_path
            )
            logger.info(f"Evaluation complete. Results: {results}")
            print(f"Model evaluated. Results: {results}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(f"Evaluation error: {e}")

    def load(
        self,
        model_type: str = "YOLO",
        model_path: str = None,
        use_default_model: bool = False,
    ):
        """
        Load a model into the environment.

        :param model_type: Type of model to load (default: YOLO).
        :param model_path: Path to the custom model (if not using the default).
        :param use_default_model: Whether to use the default model.
        """
        try:
            self.segmentator.load_model(
                model_type=model_type,
                use_default_model=use_default_model,
                model_path=model_path,
            )
            self.segmentator.clear_cache()
            logger.info(f"{model_type} model loaded successfully.")
            print(f"Successfully loaded {model_type} model.")
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise RuntimeError(f"Model loading error: {e}")


if __name__ == "__main__":
    fire.Fire(CliParser)
