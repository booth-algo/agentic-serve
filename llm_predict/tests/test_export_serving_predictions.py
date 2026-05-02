from llm_predict.export_serving_predictions import _predictor_gpu_key, _serving_gpu_key


def test_serving_gpu_key_preserves_h100_tensor_parallel_labels():
    assert _serving_gpu_key("H100x2") == "H100x2"
    assert _serving_gpu_key("H100x4") == "H100x4"


def test_predictor_gpu_key_normalizes_h100_tensor_parallel_labels():
    assert _predictor_gpu_key("H100x2") == "H100"
    assert _predictor_gpu_key("H100x4") == "H100"


def test_serving_gpu_key_preserves_single_gpu_labels():
    assert _serving_gpu_key("H100") == "H100"
    assert _serving_gpu_key("A100-40GB") == "A100"
    assert _serving_gpu_key("3090") == "RTX3090"
