from scripts.coreml_fastkill import _evaluate_gate


def test_coreml_gate_requires_precision_and_speed():
    rows = [
        {
            "benchmark_state": "complete",
            "compute_unit": "ALL",
            "latency_p50_seconds": 0.031,
            "summary_cosine": 0.999,
            "spatial_cosine": 0.999,
        },
        {
            "benchmark_state": "complete",
            "compute_unit": "CPU_AND_GPU",
            "latency_p50_seconds": 0.031,
            "summary_cosine": 0.999,
            "spatial_cosine": 0.999,
        },
    ]

    gate = _evaluate_gate(rows, baseline_p50_ms=42.0, variant="h")

    assert gate["precision_pass"] is True
    assert gate["speed_pass"] is True
    assert gate["ane_inferred"] is False
    assert gate["decision"] == "continue"


def test_coreml_gate_kills_when_speed_misses():
    rows = [
        {
            "benchmark_state": "complete",
            "compute_unit": "ALL",
            "latency_p50_seconds": 0.025,
            "summary_cosine": 0.999,
            "spatial_cosine": 0.999,
        }
    ]

    gate = _evaluate_gate(rows, baseline_p50_ms=28.0, variant="so400m")

    assert gate["precision_pass"] is True
    assert gate["speed_pass"] is False
    assert gate["decision"] == "kill"
