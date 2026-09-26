import sys
import types

import pytest

from moore_web import score_comet_qe
from moore_web.dedup_aligned_comet import deduplicate_by_comet


@pytest.fixture
def fake_comet(monkeypatch):
    """A stand-in `comet` module that counts checkpoint loads."""
    loads = []

    class Model:
        def predict(self, data, batch_size, gpus, num_workers=None):
            return types.SimpleNamespace(scores=[float(len(d["mt"])) for d in data])

    module = types.ModuleType("comet")
    module.download_model = lambda name: f"/cache/{name}"
    module.load_from_checkpoint = lambda path: loads.append(path) or Model()
    monkeypatch.setitem(sys.modules, "comet", module)
    score_comet_qe.load_model.cache_clear()
    yield loads
    score_comet_qe.load_model.cache_clear()


def test_dedup_and_annotation_share_one_model(fake_comet):
    pairs = [
        {"fr": "Le Conseil.", "mo": "A."},
        {"fr": "Le Conseil.", "mo": "Longer."},  # same French: a duplicate group
        {"fr": "Autre.", "mo": "B."},
    ]
    kept = deduplicate_by_comet(pairs, gpus=0)
    assert [p["mo"] for p in kept] == ["Longer.", "B."]

    # The annotation step loads again through the same cached function.
    assert score_comet_qe.load_model() is score_comet_qe.load_model()
    assert fake_comet == ["/cache/McGill-NLP/ssa-comet-qe"]
