from src.models.base_model import BaseModel
from ultralytics import YOLO
from src.logger import logger


class YoloModel(BaseModel):
    def __init__(self, model_path: str, task: str):
        self.estimator = YOLO(model_path, task=task)
        logger.info(f"YOLO inited: model_path: {model_path}; task: {task}")

    def train(self, data: str, imgsz: int, epochs: int, batch: int):
        logger.info("YOLO started train")
        self.estimator.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch)
        logger.info("YOLO finished train")

    def val(self, data: str, imgsz: int):
        logger.info("YOLO started validation")
        results = self.estimator.val(data=data, imgsz=imgsz)
        output = {
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "mAP50": results.results_dict["metrics/mAP50(M)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(M)"],
            "f1": results.box.f1[0],
        }
        logger.info("YOLO finished validation")
        return output

    def predict(self, source: str, task: str, save: bool, save_txt: bool, stream: bool):
        logger.info(f"YOLO started prediction: {source}")
        generators = self.estimator.predict(
            source=source, task=task, save=save, save_txt=save_txt, stream=stream
        )
        for _ in generators:
            pass
        logger.info(f"YOLO predicted: {source}")
