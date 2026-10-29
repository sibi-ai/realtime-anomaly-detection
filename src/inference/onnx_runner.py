"""ONNX Runtime wrapper for low-latency model inference."""
import numpy as np
import onnxruntime as rt
import structlog

log = structlog.get_logger()


class ONNXInferenceRunner:
    """Thin wrapper around ONNX Runtime for INT8-quantised model serving."""

    def __init__(self, model_path: str, providers: list[str] | None = None):
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = rt.InferenceSession(model_path, opts, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info("onnx_model_loaded", path=model_path,
                 provider=self.session.get_providers()[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference. features shape: (batch, n_features). Returns probabilities."""
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: features})
        return outputs[0][:, 1]  # positive class probability

    def predict_single(self, features: np.ndarray) -> float:
        return float(self.predict(features[np.newaxis, :])[0])
