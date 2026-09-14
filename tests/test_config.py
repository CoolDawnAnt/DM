from pathlib import Path

from dm.config import load_config
from dm.method import make_schedule


ROOT = Path(__file__).resolve().parents[1]


def test_release_configs_load():
    expected = {
        "sd35_distill.yaml": "distill",
        "sd35_joint.yaml": "joint",
        "sd35_seq.yaml": "seq",
    }
    for filename, mode in expected.items():
        config = load_config(ROOT / "configs" / filename)
        assert config.method.mode == mode
        assert config.model.pretrained_model.startswith("stabilityai/")


def test_piecewise_schedule():
    schedule = make_schedule("steps:1,1,0,49")
    assert schedule(0) == 1
    assert schedule(1) == 0
    assert schedule(49) == 0
    assert schedule(50) == 1
