from pathlib import Path

ROOT = Path(__file__).resolve().parent

def test_required_routes_are_present():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    for route in [
        '"/ingest"', '"/predict"', '"/rollout"', '"/explain"',
        '"/security-zone"', '"/alerts"', '"/alerts/trigger-attack"',
        '"/evidence-pack/{incident_id}"', '"/knowledge-center"', '"/help-resources"'
    ]:
        assert route in source

def test_no_scaler_or_placeholder_implementation():
    source = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.glob("*.py"))
    assert "StandardScaler" not in source
    assert "generate_placeholder_scaler" not in source
