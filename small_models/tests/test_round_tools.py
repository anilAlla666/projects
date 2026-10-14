import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate import abstain_band, fit_temperature, sigmoid  # noqa: E402
from mine import weights_for  # noqa: E402


def test_mining_weights():
    scored = [{"example_id": "a", "p": 0.95}, {"example_id": "b", "p": 0.6}, {"example_id": "c", "p": 0.1}, {"example_id": "d", "p": 0.4}]
    labels = {"a": 0, "b": 0, "c": 0, "d": 1}
    # a: confidently wrong; b and d: wrong, either direction (d rejects a correct attempt); c: right, unweighted
    assert weights_for(scored, labels) == {"a": 3.0, "b": 2.0, "d": 2.0}


def test_temperature_recovers_a_known_scale():
    # labels drawn as the probability sigmoid(z / 2); the fit must land near 2, not at either end of the grid
    logits, labels = [], []
    for i in range(-200, 201):
        z = i / 20
        p = sigmoid(z / 2)
        n_pos = round(p * 10)
        logits += [z] * 10
        labels += [1] * n_pos + [0] * (10 - n_pos)
    assert 1.8 <= fit_temperature(logits, labels) <= 2.2
    # teeth: overconfident labels (the sign alone) push the temperature to the low end
    assert fit_temperature(logits, [int(z > 0) for z in logits]) == 0.5


def test_abstain_band_is_the_smallest_that_reaches_the_target():
    probs = [0.9] * 90 + [0.1] * 90 + [0.55] * 10 + [0.45] * 10
    labels = [1] * 90 + [0] * 90 + [0] * 10 + [1] * 10
    lo, hi, cover = abstain_band(probs, labels)
    assert math.isclose(hi - 0.5, 0.06) and math.isclose(cover, 180 / 200)
    assert abstain_band([0.9, 0.1], [1, 0])[2] == 1.0  # already perfect: no band


def test_hard_wrong_factor_multiplies_the_mined_weight():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_weights import combine
    assert combine({"a": 3.0, "b": 2.0}, {"a", "c"}, 4.0) == {"a": 12.0, "b": 2.0, "c": 4.0}
