"""Offline NPZ regression tests; no models, audio files or datasets are needed."""
import numpy as np
import pytest

from AudioRetrieval import eval_core
from AudioRetrieval.evaluation import metrics
from AudioRetrieval.evaluation.runners.from_embeddings import PrecomputedEmbeddingRunner


def write_inputs(tmp_path, audio=None, captions=None, uiq=None, uiq_name="uiq_pragmatic_embeddings.npz"):
    ids = np.array(["a", "b"])
    for name, embeddings in [("audio.npz", audio), ("captions.npz", captions), (uiq_name, uiq)]:
        np.savez(tmp_path / name, embeddings=np.eye(2) if embeddings is None else embeddings,
                 clip_ids=ids, texts=np.array(["first", "second"]))
    return PrecomputedEmbeddingRunner(tmp_path / "audio.npz", tmp_path / "captions.npz", tmp_path)


@pytest.mark.parametrize("source", ["audio", "captions", "uiq"])
@pytest.mark.parametrize("value,label", [(np.nan, "NaN"), (np.inf, "+inf"), (-np.inf, "-inf")])
def test_rejects_nonfinite_embeddings(tmp_path, source, value, label):
    bad = np.eye(2)
    bad[0, 1] = value
    runner = write_inputs(tmp_path, **{source: bad})
    path = tmp_path / ("uiq_pragmatic_embeddings.npz" if source == "uiq" else source + ".npz")
    with pytest.raises(ValueError) as error:
        if source == "uiq":
            runner.run_uiq(["Pragmatic"])
        else:
            runner.run_baseline()
    message = str(error.value)
    assert str(path) in message
    assert "embeddings" in message and label in message
    # A rejected array must not enter the lazy cache; repair and retry the same runner.
    np.savez(path, embeddings=np.eye(2), clip_ids=np.array(["a", "b"]))
    result = runner.run_uiq(["Pragmatic"]) if source == "uiq" else runner.run_baseline()
    assert result["Pragmatic" if source == "uiq" else "text_to_audio"]["R@1"] == 100


def test_diagnostic_counts_each_invalid_value(tmp_path):
    runner = write_inputs(tmp_path, audio=np.array([[np.nan, np.inf], [-np.inf, 0.]]))
    with pytest.raises(ValueError, match=r"NaN=1, \+inf=1, -inf=1"):
        runner.run_baseline()


@pytest.mark.parametrize("name", ["uiq_pragmatic_embeddings.npz", "uiq_pragmatic.npz", "pragmatic_embeddings.npz"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_uiq_filename_variants_are_validated(tmp_path, name, value):
    runner = write_inputs(tmp_path, uiq=np.full((2, 2), value), uiq_name=name)
    with pytest.raises(ValueError):
        runner.load_uiq_embeddings("Pragmatic")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("zeros", [False, True])
def test_finite_inputs_preserve_perfect_metrics(tmp_path, dtype, zeros):
    array = (np.zeros((2, 2)) if zeros else np.eye(2)).astype(dtype)
    runner = write_inputs(tmp_path, audio=array, captions=array, uiq=array)
    perfect = {"R@1": 100., "R@5": 100., "R@10": 100., "MRR": 1., "DCG": 1.}
    assert runner.run_baseline() == {"text_to_audio": perfect, "audio_to_text": perfect}
    assert runner.run_uiq(["Pragmatic"]) == {"Pragmatic": perfect, "Average": perfect}
    np.testing.assert_array_equal(runner.audio_data["embeddings"], array)
    np.testing.assert_array_equal(runner.caption_data["embeddings"], array)


def test_finite_nontrivial_matching_and_multiple_captions(tmp_path):
    runner = write_inputs(tmp_path)
    np.savez(tmp_path / "captions.npz", embeddings=np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),
             clip_ids=np.array(["a", "b", "a", "unmatched"]))
    result = runner.run_baseline()
    # Matched T2A ranks [1,2,2]; best of multiple captions gives A2T ranks [1,3].
    dcg = 1. / np.log2(3.)
    assert result["text_to_audio"] == pytest.approx({"R@1": 100./3, "R@5": 100., "R@10": 100.,
                                                    "MRR": 2./3, "DCG": (1.+2*dcg)/3})
    assert result["audio_to_text"] == pytest.approx({"R@1": 50., "R@5": 100., "R@10": 100.,
                                                    "MRR": 2./3, "DCG": .75})


@pytest.mark.parametrize("rank", [metrics.ranks_from_scores, eval_core.ranks_from_scores_row])
@pytest.mark.parametrize("scores,gt,ignore,expected", [
    ([.5, .5, .2], 1, None, 1),
    ([.9, .5, .2], 1, None, 2),
    ([.9, .5, .2], 1, 0, 1),
    ([-np.inf, .5, .2], 1, None, 1),
    ([.9, .5, .2], 1, 1, 2),
])
def test_optimistic_ties_and_internal_exclusions(rank, scores, gt, ignore, expected):
    array = np.array(scores)
    original = array.copy()
    assert rank(array, gt, ignore) == expected
    np.testing.assert_array_equal(array, original)


def test_missing_optional_uiq_is_still_skipped(tmp_path):
    runner = write_inputs(tmp_path)
    assert runner.load_uiq_embeddings("Missing") is None
    assert runner.run_uiq(["Missing"]) == {}
