from threading import Lock, Semaphore
from typing import List

import onnxruntime


class OnnxBaseModel:
    _instances = {}

    @classmethod
    def get_instance(cls, model_path: str, execution_providers: List[str]):
        if cls not in cls._instances:
            cls._instances[cls] = cls(model_path, execution_providers)
        return cls._instances[cls]
    
    
    def __init__(self, model_path: str, execution_providers: List[str]):
        self.lock: Lock = Lock()
        self.semaphore: Semaphore = Semaphore()
        self.model_path = model_path
        self.execution_providers = execution_providers
        self.session: onnxruntime.InferenceSession = None
        with self.lock:
            if self.session is None:
                self.session = onnxruntime.InferenceSession(self.model_path, providers = self.execution_providers)
        inputs = self.session.get_inputs()
        self.input_names = []
        for input in inputs:
            self.input_names.append(input.name)
        outputs = self.session.get_outputs()
        self.output_names = []
        for output in outputs:
            self.output_names.append(output.name)
        onnxruntime.set_default_logger_severity(3)