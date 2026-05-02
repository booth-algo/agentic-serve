from llm_predict.export_serving_predictions import _predictor_gpu_key, _serving_gpu_key


def test_serving_gpu_key_preserves_h100_tensor_parallel_labels():
    assert _serving_gpu_key("H100x2") == "H100x2"
    assert _serving_gpu_key("H100x4") == "H100x4"


def test_serving_gpu_key_normalizes_tensor_parallel_base_labels():
    assert _serving_gpu_key("A100-40GBx4") == "A100x4"
    assert _serving_gpu_key("3090x4") == "RTX3090x4"
    assert _serving_gpu_key("2080Tix4") == "RTX2080Tix4"


def test_predictor_gpu_key_normalizes_tensor_parallel_labels():
    assert _predictor_gpu_key("H100x2") == "H100"
    assert _predictor_gpu_key("H100x4") == "H100"
    assert _predictor_gpu_key("A100x4") == "A100"
    assert _predictor_gpu_key("RTX3090x4") == "RTX3090"
    assert _predictor_gpu_key("RTX2080Tix4") == "RTX2080Ti"


def test_serving_gpu_key_preserves_single_gpu_labels():
    assert _serving_gpu_key("H100") == "H100"
    assert _serving_gpu_key("A100-40GB") == "A100"
    assert _serving_gpu_key("3090") == "RTX3090"
