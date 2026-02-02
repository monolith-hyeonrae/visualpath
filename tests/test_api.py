"""Tests for the high-level API."""

import numpy as np
import pytest

from visualbase import Frame

import visualpath as vp
from visualpath.api import (
    _extractor_registry,
    _fusion_registry,
    FunctionExtractor,
    FunctionFusion,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registries before each test."""
    _extractor_registry.clear()
    _fusion_registry.clear()
    yield
    _extractor_registry.clear()
    _fusion_registry.clear()


def make_frame(frame_id: int = 0, t_ns: int = 0) -> Frame:
    """Create a test frame."""
    return Frame.from_array(
        np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        t_src_ns=t_ns,
    )


class TestExtractorDecorator:
    """Tests for @vp.extractor decorator."""

    def test_basic_extractor(self):
        """Test simple extractor creation."""
        @vp.extractor("test")
        def my_extractor(frame):
            return {"score": 0.5}

        assert isinstance(my_extractor, FunctionExtractor)
        assert my_extractor.name == "test"

        # Should be registered
        assert "test" in _extractor_registry

    def test_extractor_extract(self):
        """Test extractor extraction."""
        @vp.extractor("brightness")
        def check_brightness(frame):
            return {"brightness": 128.0, "valid": 1.0}

        frame = make_frame()
        obs = check_brightness.extract(frame)

        assert obs is not None
        assert obs.source == "brightness"
        assert obs.signals["brightness"] == 128.0
        assert obs.signals["valid"] == 1.0

    def test_extractor_returns_none(self):
        """Test extractor that returns None."""
        @vp.extractor("conditional")
        def conditional_extractor(frame):
            if frame.frame_id < 5:
                return None
            return {"passed": 1.0}

        frame = make_frame(frame_id=0)
        assert conditional_extractor.extract(frame) is None

        frame = make_frame(frame_id=10)
        obs = conditional_extractor.extract(frame)
        assert obs is not None
        assert obs.signals["passed"] == 1.0

    def test_extractor_with_init_cleanup(self):
        """Test extractor with init and cleanup functions."""
        state = {"initialized": False, "cleaned": False}

        def init():
            state["initialized"] = True

        def cleanup():
            state["cleaned"] = True

        @vp.extractor("stateful", init=init, cleanup=cleanup)
        def stateful_extractor(frame):
            return {"ready": 1.0 if state["initialized"] else 0.0}

        assert not state["initialized"]

        stateful_extractor.initialize()
        assert state["initialized"]

        stateful_extractor.cleanup()
        assert state["cleaned"]

    def test_extractor_context_manager(self):
        """Test extractor as context manager."""
        calls = []

        @vp.extractor(
            "ctx",
            init=lambda: calls.append("init"),
            cleanup=lambda: calls.append("cleanup"),
        )
        def ctx_extractor(frame):
            return {"ok": 1.0}

        with ctx_extractor:
            assert "init" in calls
            assert "cleanup" not in calls

        assert "cleanup" in calls

    def test_extractor_non_scalar_data(self):
        """Test extractor with non-scalar data."""
        @vp.extractor("objects")
        def object_detector(frame):
            return {
                "count": 2.0,
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            }

        frame = make_frame()
        obs = object_detector.extract(frame)

        assert obs.signals["count"] == 2.0
        assert obs.data["boxes"] == [[10, 20, 30, 40], [50, 60, 70, 80]]


class TestFusionDecorator:
    """Tests for @vp.fusion decorator."""

    def test_basic_fusion(self):
        """Test simple fusion creation."""
        @vp.fusion(sources=["face"])
        def smile_detector(face):
            if face.get("happy", 0) > 0.5:
                return vp.trigger("smile")

        assert isinstance(smile_detector, FunctionFusion)
        assert smile_detector.name == "smile_detector"
        assert "smile_detector" in _fusion_registry

    def test_fusion_with_name(self):
        """Test fusion with custom name."""
        @vp.fusion(sources=["face"], name="my_fusion")
        def detector(face):
            pass

        assert detector.name == "my_fusion"

    def test_fusion_triggers(self):
        """Test fusion triggering."""
        @vp.fusion(sources=["face"], cooldown=0.1)
        def happy_trigger(face):
            if face.get("happy", 0) > 0.5:
                return vp.trigger("happy", score=face["happy"])

        # Create observation
        obs = vp.Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"happy": 0.8},
        )

        result = happy_trigger.update(obs)

        assert result.should_trigger
        assert result.trigger is not None
        assert result.reason == "happy"
        assert result.score == 0.8

    def test_fusion_no_trigger(self):
        """Test fusion not triggering."""
        @vp.fusion(sources=["face"])
        def happy_trigger(face):
            if face.get("happy", 0) > 0.5:
                return vp.trigger("happy")

        obs = vp.Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"happy": 0.2},  # Below threshold
        )

        result = happy_trigger.update(obs)
        assert not result.should_trigger

    def test_fusion_cooldown(self):
        """Test fusion cooldown."""
        @vp.fusion(sources=["face"], cooldown=1.0)
        def always_trigger(face):
            return vp.trigger("test")

        # First trigger
        obs1 = vp.Observation(source="face", frame_id=0, t_ns=0, signals={})
        result1 = always_trigger.update(obs1)
        assert result1.should_trigger

        # During cooldown
        obs2 = vp.Observation(source="face", frame_id=1, t_ns=int(0.5e9), signals={})
        result2 = always_trigger.update(obs2)
        assert not result2.should_trigger  # In cooldown

        # After cooldown
        obs3 = vp.Observation(source="face", frame_id=2, t_ns=int(1.5e9), signals={})
        result3 = always_trigger.update(obs3)
        assert result3.should_trigger

    def test_fusion_multiple_sources(self):
        """Test fusion with multiple sources."""
        @vp.fusion(sources=["face", "pose"])
        def interaction(face, pose):
            if face.get("happy", 0) > 0.5 and pose.get("wave", 0) > 0.5:
                return vp.trigger("greeting")

        # Only face observation - should not trigger
        obs_face = vp.Observation(
            source="face", frame_id=0, t_ns=0,
            signals={"happy": 0.8},
        )
        result1 = interaction.update(obs_face)
        assert not result1.should_trigger  # Missing pose

        # Add pose observation
        obs_pose = vp.Observation(
            source="pose", frame_id=0, t_ns=0,
            signals={"wave": 0.9},
        )
        result2 = interaction.update(obs_pose)
        assert result2.should_trigger


class TestTriggerSpec:
    """Tests for trigger() helper."""

    def test_simple_trigger(self):
        """Test simple trigger creation."""
        t = vp.trigger("smile")
        assert t.reason == "smile"
        assert t.score == 1.0
        assert t.metadata == {}

    def test_trigger_with_score(self):
        """Test trigger with custom score."""
        t = vp.trigger("wave", score=0.75)
        assert t.reason == "wave"
        assert t.score == 0.75

    def test_trigger_with_metadata(self):
        """Test trigger with metadata."""
        t = vp.trigger("face", score=0.9, face_id=5, emotion="happy")
        assert t.reason == "face"
        assert t.score == 0.9
        assert t.metadata == {"face_id": 5, "emotion": "happy"}


class TestRegistry:
    """Tests for extractor/fusion registry."""

    def test_get_extractor(self):
        """Test getting registered extractor."""
        @vp.extractor("test_ext")
        def my_ext(frame):
            return {"x": 1.0}

        ext = vp.get_extractor("test_ext")
        assert ext is not None
        assert ext.name == "test_ext"

    def test_get_unknown_extractor(self):
        """Test getting unknown extractor."""
        ext = vp.get_extractor("nonexistent")
        assert ext is None

    def test_get_fusion(self):
        """Test getting registered fusion."""
        @vp.fusion(sources=["a"], name="test_fus")
        def my_fus(a):
            pass

        fus = vp.get_fusion("test_fus")
        assert fus is not None
        assert fus.name == "test_fus"

    def test_list_extractors(self):
        """Test listing extractors."""
        @vp.extractor("ext1")
        def ext1(frame):
            return {}

        @vp.extractor("ext2")
        def ext2(frame):
            return {}

        names = vp.list_extractors()
        assert "ext1" in names
        assert "ext2" in names

    def test_list_fusions(self):
        """Test listing fusions."""
        @vp.fusion(sources=["a"], name="fus1")
        def fus1(a):
            pass

        @vp.fusion(sources=["b"], name="fus2")
        def fus2(b):
            pass

        names = vp.list_fusions()
        assert "fus1" in names
        assert "fus2" in names


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_default_values(self):
        """Test ProcessResult defaults."""
        result = vp.ProcessResult()
        assert result.triggers == []
        assert result.frame_count == 0
        assert result.duration_sec == 0.0


class TestAPIUsability:
    """Tests demonstrating API usability."""

    def test_readme_example(self):
        """Test example that would go in README."""
        # Define extractor
        @vp.extractor("quality")
        def check_quality(frame):
            brightness = float(frame.data.mean())
            return {"brightness": brightness, "is_bright": brightness > 128}

        # Define fusion
        @vp.fusion(sources=["quality"], cooldown=0.5)
        def brightness_spike(quality):
            if quality.get("is_bright") and quality.get("brightness", 0) > 200:
                return vp.trigger("bright_frame", score=quality["brightness"] / 255)

        # Test the extractor
        frame = Frame.from_array(
            np.full((480, 640, 3), 220, dtype=np.uint8),  # Bright frame
            frame_id=0,
            t_src_ns=0,
        )

        obs = check_quality.extract(frame)
        assert obs.signals["brightness"] == 220.0
        assert obs.signals["is_bright"] == 1.0

        # Test the fusion
        result = brightness_spike.update(obs)
        assert result.should_trigger
        assert result.reason == "bright_frame"
        assert 0.8 < result.score < 0.9  # 220/255 ≈ 0.86

    def test_minimal_extractor(self):
        """Test minimal extractor definition."""
        @vp.extractor("simple")
        def simple(frame):
            return {"value": 1.0}

        # That's it - 3 lines to define an extractor
        obs = simple.extract(make_frame())
        assert obs.signals["value"] == 1.0

    def test_minimal_fusion(self):
        """Test minimal fusion definition."""
        @vp.fusion(sources=["simple"])
        def always_fire(simple):
            return vp.trigger("test")

        # That's it - 3 lines to define a fusion
        obs = vp.Observation(source="simple", frame_id=0, t_ns=0, signals={})
        result = always_fire.update(obs)
        assert result.should_trigger
