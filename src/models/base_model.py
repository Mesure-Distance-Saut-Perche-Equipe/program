from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, data: str, imgs: int, epochs: int, batch: int):
        pass

    @abstractmethod
    def val(self, data: str, imgs: int):
        pass

    @abstractmethod
    def predict(self, source: str):
        pass
