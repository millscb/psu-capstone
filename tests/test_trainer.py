"""Tests for Trainer brain creation with proper InputFields, OutputFields, and ColumnFields.

``TestSaveLoadBrain`` (below) checks that checkpoints preserve learned HTM state, not only
structure. It uses ``_learned_htm_snapshot`` to fingerprint proximal + distal weights; see that
helper and the class docstring for what is and is not asserted. Encoder internals are still
serialized with pickle but are not part of the snapshot tuple.
"""

import pickle
from pathlib import Path

import pytest

import htmrl.encoder_layer as el
from htmrl.agent_layer.brain import Brain
from htmrl.agent_layer.HTM import CONNECTED_PERM, ColumnField, InputField, OutputField
from htmrl.agent_layer.train import Trainer


def _learned_htm_snapshot(brain: Brain) -> tuple[int, float, int, float]:
    """Summarize HTM weights so save/load can be checked for true learning state.

    Returns:
        (connected_proximal_synapses, sum_proximal_permanence, distal_segment_count,
         sum_distal_permanence)
    """

    prox_connected = 0
    prox_perm_sum = 0.0
    distal_segments = 0
    distal_perm_sum = 0.0

    for cf in brain.column_fields:
        for column in cf.columns:
            for syn in column.potential_synapses:
                prox_perm_sum += syn.permanence
                if syn.permanence >= CONNECTED_PERM:
                    prox_connected += 1
        for cell in cf.cells:
            for seg in cell.segments:
                distal_segments += 1
                for syn in seg.synapses:
                    distal_perm_sum += syn.permanence

    return (
        prox_connected,
        round(prox_perm_sum, 9),
        distal_segments,
        round(distal_perm_sum, 9),
    )


@pytest.fixture
def trainer():
    brain = Brain()
    return Trainer(brain)


# ---------------------------------------------------------------------------
# build_brain
# ---------------------------------------------------------------------------


class TestBuildBrain:
    def test_single_input_field_is_present_in_brain(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert "value_input" in brain.fields

    def test_single_input_field_is_instance_of_input_field(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["value_input"], InputField)

    def test_column_field_is_created_automatically(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert len(brain.column_fields) == 1

    def test_column_field_is_instance_of_column_field(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.column_fields[0], ColumnField)

    def test_multiple_input_fields_all_present(self, trainer):
        fields = [
            ("temp_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
            ("humidity_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        assert "temp_input" in brain.fields
        assert "humidity_input" in brain.fields

    def test_multiple_input_fields_are_all_input_field_instances(self, trainer):
        fields = [
            ("temp_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
            ("humidity_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        assert all(
            isinstance(brain.fields[k], InputField) for k in ("temp_input", "humidity_input")
        )

    def test_input_field_size_matches_requested_size(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert len(brain.fields["value_input"].cells) == 512

    def test_brain_has_no_output_fields_when_none_defined(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert len(brain.output_fields) == 0

    def test_invalid_field_name_raises(self, trainer):
        fields = [("value", 512, el.RDSEParameters(size=512, resolution=1.0))]
        with pytest.raises(ValueError):
            trainer.build_brain(fields)

    def test_column_field_num_columns_equals_max_field_size(self, trainer):
        fields = [
            ("small_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
            ("large_input", 1024, el.RDSEParameters(size=1024, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        assert brain.column_fields[0].num_columns == 1024

    def test_build_brain_returns_brain_instance(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain, Brain)

    def test_built_brain_stored_as_main_brain(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert trainer.main_brain is brain

    def test_category_encoder_input_field_is_created(self, trainer):
        fields = [
            ("label_input", 512, el.CategoryParameters(size=512, category_list=["a", "b", "c"]))
        ]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["label_input"], InputField)
        assert isinstance(brain.fields["label_input"].encoder, el.CategoryEncoder)

    def test_scalar_encoder_input_field_is_created(self, trainer):
        fields = [
            ("score_input", 512, el.ScalarEncoderParameters(size=512, minimum=0, maximum=100))
        ]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["score_input"], InputField)
        assert isinstance(brain.fields["score_input"].encoder, el.ScalarEncoder)

    def test_date_encoder_input_field_is_created(self, trainer):
        fields = [("date_input", 512, el.DateEncoderParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["date_input"], InputField)
        assert isinstance(brain.fields["date_input"].encoder, el.DateEncoder)

    def test_fourier_encoder_input_field_is_created(self, trainer):
        fields = [("fourier_input", 512, el.FourierEncoderParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["fourier_input"], InputField)
        assert isinstance(brain.fields["fourier_input"].encoder, el.FourierEncoder)

    def test_geospatial_encoder_input_field_is_created(self, trainer):
        fields = [("geo_input", 512, el.GeospatialParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["geo_input"], InputField)
        assert isinstance(brain.fields["geo_input"].encoder, el.GeospatialEncoder)

    def test_coordinate_encoder_input_field_is_created(self, trainer):
        fields = [("coord_input", 512, el.CoordinateParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["coord_input"], InputField)
        assert isinstance(brain.fields["coord_input"].encoder, el.CoordinateEncoder)

    def test_rdse_encoder_input_field_is_created(self, trainer):
        fields = [("rdse_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["rdse_input"], InputField)
        assert isinstance(brain.fields["rdse_input"].encoder, el.RandomDistributedScalarEncoder)

    def test_multiple_encoder_types_all_input_fields_created(self, trainer):
        fields = [
            ("category_input", 512, el.CategoryParameters(size=512, category_list=["a", "b", "c"])),
            ("scalar_input", 512, el.ScalarEncoderParameters(size=512, minimum=0, maximum=100)),
            ("date_input", 512, el.DateEncoderParameters(size=512)),
            ("fourier_input", 512, el.FourierEncoderParameters(size=512)),
            ("geo_input", 512, el.GeospatialParameters(size=512)),
            ("coord_input", 512, el.CoordinateParameters(size=512)),
            ("rdse_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        for name in [
            "category_input",
            "scalar_input",
            "date_input",
            "fourier_input",
            "geo_input",
            "coord_input",
            "rdse_input",
        ]:
            assert isinstance(brain.fields[name], InputField)
            assert isinstance(
                brain.fields[name].encoder,
                (
                    el.CategoryEncoder,
                    el.ScalarEncoder,
                    el.DateEncoder,
                    el.FourierEncoder,
                    el.GeospatialEncoder,
                    el.CoordinateEncoder,
                    el.RandomDistributedScalarEncoder,
                ),
            )


# ---------------------------------------------------------------------------
# add_input_field / add_output_field / add_column_field
# ---------------------------------------------------------------------------


class TestAddFields:
    def test_add_input_field_adds_to_brain_fields(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_input_field("extra_input", 512, el.RDSEParameters(size=512, resolution=1.0))
        assert "extra_input" in trainer.main_brain.fields

    def test_add_input_field_is_input_field_instance(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_input_field("extra_input", 512, el.RDSEParameters(size=512, resolution=1.0))
        assert isinstance(trainer.main_brain.fields["extra_input"], InputField)

    def test_add_input_field_bad_name_raises(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        with pytest.raises(ValueError):
            trainer.add_input_field("extra", 512, el.RDSEParameters(size=512, resolution=1.0))

    def test_add_output_field_is_output_field_instance(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_output_field("motor_output", 128, motor_action=(None,))
        assert isinstance(trainer.main_brain.fields["motor_output"], OutputField)

    def test_add_output_field_bad_name_raises(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        with pytest.raises(ValueError):
            trainer.add_output_field("motor", 128, motor_action=(None,))

    def test_add_column_field_is_column_field_instance(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_column_field("extra_column", num_columns=512, cells_per_column=8)
        assert isinstance(trainer.main_brain.fields["extra_column"], ColumnField)

    def test_add_column_field_bad_name_raises(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        with pytest.raises(ValueError):
            trainer.add_column_field("extra", num_columns=512, cells_per_column=8)


# ---------------------------------------------------------------------------
# save_brain / load_brain
# ---------------------------------------------------------------------------


class TestSaveLoadBrain:
    """Trainer persistence: pickle round-trip and ``set_as_main`` behavior.

    **Fingerprint:** ``_learned_htm_snapshot`` returns ``(connected proximal count,
    sum proximal permanence, distal segment count, sum distal permanence)``. That
    summarizes spatial-pooler and temporal-memory learning on ``ColumnField``; it
    does not separately hash encoder/RDSE state (still present in the pickled ``Brain``).

    **Round-trip test:** Asserts (1) the snapshot changes after many ``learn=True`` steps
    so "progress" is real, and (2) the snapshot after ``load_brain`` equals the snapshot
    just before ``save_brain``, so the file fully encodes that progress for what we measure.

    Other tests cover ``set_as_main=False`` and rejecting non-``Brain`` pickle payloads.
    """

    def test_roundtrip_persists_state_and_syncs_trainer_for_testing(self, tmp_path: Path):
        t1 = Trainer(Brain({}))
        brain = t1.build_brain(
            [("value_input", 64, el.RDSEParameters(size=64, resolution=1.0, seed=7))]
        )
        before_train = _learned_htm_snapshot(brain)
        for i in range(120):
            brain.step({"value_input": (i % 40) * 0.01}, learn=True)
        after_train = _learned_htm_snapshot(brain)
        assert after_train != before_train, "training should change HTM synapse / segment state"

        path = tmp_path / "checkpoint.pkl"
        t1.save_brain(brain, path)
        assert path.is_file()

        t2 = Trainer(Brain({}))
        loaded = t2.load_brain(path)
        assert loaded is t2.main_brain
        assert (
            _learned_htm_snapshot(loaded) == after_train
        ), "pickle round-trip must preserve learned weights (proximal + distal)"

        assert len(t2.input_fields) == 1
        assert t2.input_fields[0].name == "value_input"
        loaded.step({"value_input": 0.5}, learn=False)
        t2.train_full_brain(loaded, {"value_input": [0.1, 0.2]}, steps=2)

    def test_load_set_as_main_false_does_not_replace_main(self, tmp_path: Path):
        t = Trainer(Brain({}))
        brain = t.build_brain(
            [("value_input", 64, el.RDSEParameters(size=64, resolution=1.0, seed=3))]
        )
        path = tmp_path / "c.pkl"
        t.save_brain(brain, path)

        empty = Brain({})
        t2 = Trainer(empty)
        loaded = t2.load_brain(path, set_as_main=False)
        assert loaded is not t2.main_brain
        assert t2.main_brain is empty

    def test_load_non_brain_file_raises(self, trainer, tmp_path: Path):
        path = tmp_path / "bad.pkl"
        with path.open("wb") as f:
            pickle.dump({"not": "a brain"}, f)
        with pytest.raises(TypeError, match="Expected a Brain"):
            trainer.load_brain(path)
