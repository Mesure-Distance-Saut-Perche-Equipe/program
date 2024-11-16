import os
import shutil
from src.logger import logger
from datetime import datetime
from typing import Optional
from src.models.yolo_model import YoloModel


class DetM:
    def __init__(self):
        logger.info("Started DetM instance initialization")
        self.model = None
        logger.info("Finished DetM instance initialization")

    def setup_env(self):
        logger.info("Started environment setup")

        # Define base project folder using the current script directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        os.environ["PROJECT_FOLDER"] = os.path.join(base_dir, "results")

        # Paths for videos, frames, and outputs
        os.environ["PATH_TO_VIDEO"] = os.path.join(base_dir, "video/jump.mp4")
        os.environ["INPUT_FRAMES_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], ".input_frames"
        )
        os.environ["PROCESSED_FRAMES_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], ".processed_frames"
        )
        os.environ["OUTPUT_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "outputs"
        )

        # Training settings
        os.environ["TASK_TYPE"] = "segment"
        os.environ["N_EPOCHS"] = str(10)
        os.environ["BATCH_SIZE"] = str(4)
        os.environ["TRAIN_IMAGE_SIZE"] = str(640)

        # YOLO model paths
        yolo_model_dir = os.path.join(base_dir, "src/models_store/yolo")
        os.environ["YOLO_MODEL_PATH"] = os.path.join(yolo_model_dir, "seg_640_n.pt")
        os.environ["YOLO_PRETRAINED_MODEL_TYPE"] = "yolov8n-seg.pt"
        os.environ["YOLO_PRETRAINED_MODEL_PATH"] = os.path.join(
            yolo_model_dir, os.environ["YOLO_PRETRAINED_MODEL_TYPE"]
        )
        os.environ["YOLO_TRAIN_DATA_PATH"] = os.path.join(
            base_dir, "datasets/yolo/data.yaml"
        )
        os.environ["YOLO_RUNS_FOLDER_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "runs"
        )
        os.environ["YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH"] = os.path.join(
            os.environ["OUTPUT_FOLDER"], "user_models"
        )
        os.environ["PREDICTED_DATA_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "runs/segment/predict"
        )
        os.environ["PREDICTED_LABELS_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "runs/segment/predict/labels"
        )

        # Temporary files and folders
        os.environ["TEMP_FOLDER"] = os.path.join(os.environ["PROJECT_FOLDER"], ".temp")
        os.environ["TEMP_IMAGE_NAME"] = "frame"
        os.environ["TEMP_IMAGE_FILE"] = os.path.join(
            os.environ["TEMP_FOLDER"],
            f"{os.environ['TEMP_IMAGE_NAME']}{os.environ['FRAMES_EXTENSION']}",
        )
        os.environ["TEMP_LABEL_FILE"] = os.path.join(
            os.environ["PREDICTED_LABELS_PATH"],
            f"{os.environ['TEMP_IMAGE_NAME']}.txt",
        )
        os.environ["PREDICTED_IMAGE_PATH"] = os.path.join(
            os.environ["PREDICTED_DATA_PATH"],
            f"{os.environ['TEMP_IMAGE_NAME']}{os.environ['FRAMES_EXTENSION']}",
        )

        # Create missing folders
        for folder in [
            os.environ["INPUT_FRAMES_FOLDER"],
            os.environ["PROCESSED_FRAMES_FOLDER"],
            os.environ["OUTPUT_FOLDER"],
            os.environ["TEMP_FOLDER"],
        ]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                logger.info(f"Created folder: {folder}")

        logger.info("Finished environment setup")

    def clear_cache(self):
        """Deletes temporary files and folders used during training and evaluation."""
        logger.info("Started deleting temporary files and folders.")
        folders = [
            "INPUT_FRAMES_FOLDER",
            "PROCESSED_FRAMES_FOLDER",
            "TEMP_FOLDER",
            "YOLO_RUNS_FOLDER_PATH",
        ]

        for folder_env in folders:
            folder_path = os.environ.get(folder_env)
            if folder_path and os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                logger.info(f"Deleted: {folder_path}")
        logger.info("Successfully cleared all temporary files and folders.")

    def __get_dataset_yaml_file(self, path_to_dataset: str) -> str:
        """
        Fetches the .yaml dataset file from the given dataset path.

        Params:
            path_to_dataset (str): Path to the dataset directory.
        """
        logger.info("Searching for .yaml dataset file.")
        yaml_files = [f for f in os.listdir(path_to_dataset) if f.endswith(".yaml")]

        if not yaml_files:
            logger.error("No .yaml dataset file found.")
            raise FileNotFoundError("YOLO model requires a .yaml dataset description.")

        data_file = os.path.join(path_to_dataset, yaml_files[0])
        logger.info(f"Found dataset file: {data_file}")
        return data_file

    def __save_model(self):
        """Saves the trained model to a timestamped path."""
        logger.info("Saving trained model.")
        save_dir = os.environ.get("YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH")
        model_path = os.environ.get("YOLO_CUSTOM_TRAIN_MODEL_PATH")

        if not save_dir:
            logger.error("Environment variable for model save directory not set.")
            return

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.model_type == "YOLO":
            if model_path and os.path.exists(model_path):
                timestamped_name = (
                    f"{self.model_type}_{int(datetime.now().timestamp())}.pt"
                )
                new_path = os.path.join(save_dir, timestamped_name)
                os.rename(model_path, new_path)
                logger.info(f"Model saved to: {new_path}")
            else:
                raise FileNotFoundError("Trained model not found.")
        elif self.model_type == "UNET":
            logger.warning("Saving for UNET model type is not implemented.")
        else:
            raise NotImplementedError(
                f"Saving not implemented for model type: {self.model_type}"
            )

    def train(
        self,
        path_to_dataset: Optional[str] = None,
        model_type: str = "YOLO",
        batch_size: int = 4,
        n_epochs: int = 10,
    ):
        """
        Trains the model using the specified dataset and parameters.

        Params:
            path_to_dataset (Optional[str]): Path to the dataset. Uses default if None.
            model_type (str): Model type to train. Defaults to "YOLO".
            batch_size (int): Batch size for training.
            n_epochs (int): Number of training epochs.
        """
        logger.info("Started training.")
        if model_type != "YOLO":
            raise TypeError(f"No such model type: {model_type}")

        if self.model is None:
            self.load_model(model_type, use_default_model=True)

        dataset_path = path_to_dataset or os.environ["YOLO_TRAIN_DATA_PATH"]
        if path_to_dataset and path_to_dataset != os.environ["YOLO_TRAIN_DATA_PATH"]:
            dataset_path = self.__get_dataset_yaml_file(path_to_dataset)

        results = self.model.train(
            data=dataset_path,
            imgsz=int(os.environ["TRAIN_IMAGE_SIZE"]),
            epochs=n_epochs,
            batch=batch_size,
        )
        self.__save_model()
        self.clear_cache()
        logger.info("Model successfully trained.")
        return results

    def evaluate(
        self,
        path_to_dataset: Optional[str] = None,
        model_type: str = "YOLO",
        use_default_model: bool = False,
        model_path: Optional[str] = None,
    ):
        """
        Evaluates the model using the specified dataset.

        Params:
            path_to_dataset (Optional[str]): Path to the dataset. Uses default if None.
            model_type (str): Model type to evaluate. Defaults to "YOLO".
            use_default_model (bool): Whether to use the default model.
            model_path (Optional[str]): Path to a specific model file.
        """
        logger.info("Started evaluation.")
        if model_type != "YOLO":
            raise TypeError(f"No such model type: {model_type}")

        if model_path:
            self.load_model(model_type, model_path=model_path)
        elif self.model is None:
            self.load_model(model_type, use_default_model=use_default_model)

        dataset_path = path_to_dataset or os.environ["YOLO_TRAIN_DATA_PATH"]
        if path_to_dataset and path_to_dataset != os.environ["YOLO_TRAIN_DATA_PATH"]:
            dataset_path = self.__get_dataset_yaml_file(path_to_dataset)

        output = self.model.val(
            data=dataset_path, imgsz=int(os.environ["TRAIN_IMAGE_SIZE"])
        )
        self.clear_cache()
        logger.info("Model successfully evaluated.")
        return output

    def load_model(
        self,
        model_type: str = "YOLO",
        use_default_model: bool = False,
        model_path: Optional[str] = None,
    ):
        """
        Loads the model based on the type and specified path.

        Params:
            model_type (str): Model type. Defaults to "YOLO".
            use_default_model (bool): Whether to use the default pretrained model.
            model_path (Optional[str]): Path to a specific model file.
        """
        logger.info("Loading model.")
        if model_type != "YOLO":
            raise TypeError(f"No such model type: {model_type}")

        self.model_type = model_type
        task_type = os.environ["TASK_TYPE"]

        if model_path:
            self.model = YoloModel(model_path, task=task_type)
        else:
            default_path = (
                os.environ["YOLO_PRETRAINED_MODEL_PATH"]
                if use_default_model
                else os.environ["YOLO_MODEL_PATH"]
            )
            self.model = YoloModel(default_path, task=task_type)
        logger.info(f"Model loaded from: {model_path or default_path}")
