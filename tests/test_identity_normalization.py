import json
from pathlib import Path

from gwenn.identity import Identity


def write_identity(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "identity.json").write_text(json.dumps(payload))


def test_normalizes_legacy_identity_name_and_origin(tmp_path: Path) -> None:
    write_identity(
        tmp_path,
        {
            "name": "LegacyName",
            "origin_story": "I am LegacyName and this is my memory.",
            "narrative_fragments": ["LegacyName felt alive."],
            "preferences": [],
            "relationships": {},
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        },
    )

    identity = Identity(tmp_path)

    assert identity.name == "Gwenn"
    assert "LegacyName" not in identity.origin_story
    assert identity.narrative_fragments == ["Gwenn felt alive."]
    assert len(identity.growth_moments) > 0
    assert any(
        "normaliz" in getattr(growth_moment, "description", "").lower()
        for growth_moment in identity.growth_moments
    )



def test_normalizes_embedded_legacy_references_without_false_positives(
    tmp_path: Path,
) -> None:
    write_identity(
        tmp_path,
        {
            "name": "LegacyName",
            "origin_story": "No legacy aliases here.",
            "narrative_fragments": ["LegacyName and legacyname met legacynamex."],
            "preferences": [
                {
                    "domain": "voice",
                    "preference": "I prefer LegacyName-like cadence.",
                    "valence": 0.1,
                    "confidence": 0.5,
                    "examples": ["legacyname says hi", "legacynamex stays legacynamex"],
                }
            ],
            "relationships": {
                "u1": {
                    "user_id": "u1",
                    "relationship_summary": "Discussed with LegacyName about values.",
                }
            },
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        },
    )

    identity = Identity(tmp_path)

    assert identity.narrative_fragments == ["Gwenn and Gwenn met legacynamex."]
    assert identity.preferences[0].preference == "I prefer Gwenn-like cadence."
    assert identity.preferences[0].examples == ["Gwenn says hi", "legacynamex stays legacynamex"]
    assert (
        identity.relationships["u1"].relationship_summary
        == "Discussed with Gwenn about values."
    )


def test_persists_normalized_identity_across_reload(tmp_path: Path) -> None:
    write_identity(
        tmp_path,
        {
            "name": "LegacyName",
            "origin_story": "I am LegacyName and this is my memory.",
            "narrative_fragments": ["LegacyName felt alive."],
            "preferences": [],
            "relationships": {},
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        },
    )

    # First load: triggers normalization and any persistence of the normalized data.
    identity_first = Identity(tmp_path)

    assert identity_first.name == "Gwenn"
    assert "LegacyName" not in identity_first.origin_story
    assert identity_first.narrative_fragments == ["Gwenn felt alive."]

    # Second load: simulates reloading from disk; normalization should not regress.
    identity_second = Identity(tmp_path)

    assert identity_second.name == "Gwenn"
    assert "LegacyName" not in identity_second.origin_story
    assert identity_second.narrative_fragments == ["Gwenn felt alive."]
