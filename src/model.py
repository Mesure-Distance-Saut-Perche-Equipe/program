import os
import shutil
from datetime import datetime
from src.models.yolo_model import YoloModel
from src.logger import logger


class DetM:
    def __init__(self):
        logger.info("started DetM instance init")
        self.model = None
        logger.info("finished DetM instance init")

    def setup_env(self):
        logger.info("started venv setup")

        # folders
        os.environ["PATH_TO_VIDEO"] = "video/jump.mp4"
        os.environ["PROJECT_FOLDER"] = (
            os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + "/results"
        )
        os.environ["INPUT_FRAMES_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], ".input_frames"
        )
        os.environ["PROCESSED_FRAMES_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], ".processed_frames"
        )
        os.environ["OUTPUT_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "outputs"
        )

        # train settings
        os.environ["TASK_TYPE"] = "segment"
        os.environ["N_EPOCHS"] = str(10)
        os.environ["BATCH_SIZE"] = str(4)
        os.environ["TRAIN_IMAGE_SIZE"] = str(640)

        # input video settings
        os.environ["IMAGE_SIZE"] = str(640)
        os.environ["FRAMES_EXTENSION"] = ".jpg"
        os.environ["CONVERT_BRIGHTNESS"] = str(100)
        os.environ["CONVERT_CONTRAST"] = str(65)
        os.environ["CONVERT_SHARPENING_CYCLE"] = str(1)
        os.environ["FPS"] = "5.0"

        # models settings

        # YOLO settings
        os.environ["YOLO_MODEL_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "./src/models_store/yolo/seg_640_n.pt"
        )
        os.environ["YOLO_PRETRAINED_MODEL_TYPE"] = "yolov8n-seg.pt"
        os.environ["YOLO_PRETRAINED_MODEL_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"],
            f'./src/models_store/yolo/{os.environ["YOLO_PRETRAINED_MODEL_TYPE"]}',
        )
        os.environ["YOLO_TRAIN_DATA_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "./datasets/yolo/data.yaml"
        )
        os.environ["YOLO_RUNS_FOLDER_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "./runs"
        )
        os.environ["YOLO_CUSTOM_TRAIN_MODEL_PATH"] = os.path.join(
            os.environ["YOLO_RUNS_FOLDER_PATH"],
            f'./{os.environ["TASK_TYPE"]}/train/weights/best.pt',
        )
        os.environ["YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH"] = os.path.join(
            os.environ["OUTPUT_FOLDER"], "./user_models"
        )
        os.environ["PREDICTED_DATA_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "runs/segment/predict"
        )
        os.environ["PREDICTED_LABELS_PATH"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "runs/segment/predict/labels"
        )

        # temp files and folders
        os.environ["TEMP_FOLDER"] = os.path.join(
            os.environ["PROJECT_FOLDER"], "./.temp"
        )
        os.environ["TEMP_IMAGE_NAME"] = "frame"
        os.environ["TEMP_IMAGE_FILE"] = os.path.join(
            os.environ["TEMP_FOLDER"],
            f"./{os.environ['TEMP_IMAGE_NAME']}{os.environ['FRAMES_EXTENSION']}",
        )
        os.environ["TEMP_LABEL_FILE"] = os.path.join(
            os.environ["PREDICTED_LABELS_PATH"],
            f"./{os.environ['TEMP_IMAGE_NAME']}.txt",
        )
        os.environ["PREDICTED_IMAGE_PATH"] = os.path.join(
            os.environ["PREDICTED_DATA_PATH"],
            f"./{os.environ['TEMP_IMAGE_NAME']}{os.environ['FRAMES_EXTENSION']}",
        )

        # Check and create unexisting folders
        isExist = os.path.exists(os.environ["INPUT_FRAMES_FOLDER"])
        if not isExist:
            os.makedirs(os.environ["INPUT_FRAMES_FOLDER"])
            logger.info(f"created folder {os.environ['INPUT_FRAMES_FOLDER']}")
        isExist = os.path.exists(os.environ["OUTPUT_FOLDER"])
        if not isExist:
            os.makedirs(os.environ["OUTPUT_FOLDER"])
            logger.info(f"created folder {os.environ['OUTPUT_FOLDER']}")
        isExist = os.path.exists(os.environ["PROCESSED_FRAMES_FOLDER"])
        if not isExist:
            os.makedirs(os.environ["PROCESSED_FRAMES_FOLDER"])
            logger.info(f"created folder {os.environ['PROCESSED_FRAMES_FOLDER']}")
        isExist = os.path.exists(os.environ["TEMP_FOLDER"])
        if not isExist:
            os.makedirs(os.environ["TEMP_FOLDER"])
            logger.info(f"created folder {os.environ['TEMP_FOLDER']}")
        logger.info("finished venv setup")

    def clear_cache(self):
        logger.info("started deleting temp files and folders")
        if os.path.exists(os.environ["INPUT_FRAMES_FOLDER"]):
            shutil.rmtree(os.environ["INPUT_FRAMES_FOLDER"])

        if os.path.exists(os.environ["PROCESSED_FRAMES_FOLDER"]):
            shutil.rmtree(os.environ["PROCESSED_FRAMES_FOLDER"])

        if os.path.exists(os.environ["TEMP_FOLDER"]):
            shutil.rmtree(os.environ["TEMP_FOLDER"])

        if os.path.exists(os.environ["YOLO_RUNS_FOLDER_PATH"]):
            shutil.rmtree(os.environ["YOLO_RUNS_FOLDER_PATH"])
        logger.info("successfully finished deleting temp files and folders")

    def __get_dataset_yaml_file(self, path_to_dataset):
        logger.info("started __get_dataset_yaml_file method")
        yaml_files = [f for f in os.listdir(path_to_dataset) if f.endswith(".yaml")]
        if yaml_files:
            data_file = os.path.join(path_to_dataset, yaml_files[0])
            logger.info("__get_dataset_yaml_file method successfully executed")
            return data_file
        else:
            logger.error("No .yaml dataset file")
            raise FileNotFoundError("YOLO model requares .yaml dataset description")

    def __save_model(self):
        logger.info("started __save_model method")
        if not os.path.exists(os.environ["YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH"]):
            os.mkdir(os.environ["YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH"])
        if self.model_type == "YOLO":
            if os.path.exists(os.environ["YOLO_CUSTOM_TRAIN_MODEL_PATH"]):
                new_path = (
                    str(
                        "_".join(
                            [
                                self.model_type,
                                str(datetime.now().timestamp()).split(".")[0],
                            ]
                        )
                    )
                    + f".pt"
                )
                os.rename(
                    os.environ["YOLO_CUSTOM_TRAIN_MODEL_PATH"],
                    os.path.join(
                        os.environ["YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH"], new_path
                    ),
                )
            else:
                raise FileNotFoundError("trained model haven't saved")
        elif self.model_type == "UNET":
            pass
        else:
            raise NotImplementedError("attempt to save wrong model type")
        logger.info("__save_model method successfully executed")

    def train(
        self,
        path_to_dataset: str = None,
        model_type: str = "YOLO",
        batch_size: int = 4,
        n_epochs: int = 10,
    ):
        logger.info("started train method")
        if model_type == "YOLO":
            if self.model is None:
                self.load_model("YOLO", use_default_model=True)
            if path_to_dataset is None:
                path_to_dataset = os.environ["YOLO_TRAIN_DATA_PATH"]
            elif path_to_dataset != os.environ["YOLO_TRAIN_DATA_PATH"]:
                path_to_dataset = self.__get_dataset_yaml_file(path_to_dataset)
            # print(os.environ['YOLO_PRETRAINED_MODEL_PATH'], os.environ['TASK_TYPE'])
            results = self.model.train(
                data=path_to_dataset,
                imgsz=int(os.environ["TRAIN_IMAGE_SIZE"]),
                epochs=n_epochs,
                batch=batch_size,
            )
            self.__save_model()
            self.clear_cache()
            logger.info("Model sucessfully trained")
            return results
        elif model_type == "UNET":
            pass
        else:
            raise TypeError("no such model")
        logger.info("train method successfully executed")

    def evaluate(
        self,
        path_to_dataset: str = None,
        model_type: str = "YOLO",
        use_default_model: bool = False,
        model_path: str = None,
    ):
        logger.info("started evaluate method")
        if model_type == "YOLO":
            if model_path is not None:
                self.load_model("YOLO", model_path=model_path)
            elif self.model is None:
                self.load_model("YOLO", use_default_model)

            if path_to_dataset is None:
                path_to_dataset = os.environ["YOLO_TRAIN_DATA_PATH"]
            elif path_to_dataset != os.environ["YOLO_TRAIN_DATA_PATH"]:
                path_to_dataset = self.__get_dataset_yaml_file(path_to_dataset)

            output = self.model.val(
                data=path_to_dataset, imgsz=int(os.environ["TRAIN_IMAGE_SIZE"])
            )

            # output = {
            #     'mAP50': results.results_dict['metrics/mAP50(M)'],
            #     'precision' : results.results_dict['metrics/precision(B)'],
            #     'recall' : results.results_dict['metrics/recall(B)'],
            #     'f1' : results.box.f1[0]
            # }
            self.clear_cache()
            logger.info("Model sucessfully evaluated")
            return output
        elif model_type == "UNET":
            pass
        else:
            raise TypeError("no such model")
        logger.info("evaluate method successfully executed")

    def load_model(
        self,
        model_type: str = "YOLO",
        use_default_model: bool = False,
        model_path: str = None,
    ):
        logger.info("started load_model method")
        if model_type == "YOLO":
            self.model_type = "YOLO"
            if model_path is not None:
                self.model = YoloModel(model_path, task=os.environ["TASK_TYPE"])
            elif not use_default_model:
                self.model = YoloModel(
                    os.environ["YOLO_MODEL_PATH"], task=os.environ["TASK_TYPE"]
                )
                logger.info(f"loaded model: {os.environ['YOLO_MODEL_PATH']}")
            else:
                self.model = YoloModel(
                    os.environ["YOLO_PRETRAINED_MODEL_PATH"],
                    task=os.environ["TASK_TYPE"],
                )
                logger.info(f"loaded model: {os.environ['YOLO_PRETRAINED_MODEL_PATH']}")
        else:
            raise TypeError("no such model")
        logger.info("load_model method successfully executed")
