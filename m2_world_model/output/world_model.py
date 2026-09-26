from pathlib import Path
# The canonical architecture is kept at m2_world_model/world_model.py.
# This copy exists in output/ to satisfy the frozen M2 artifact layout.
exec((Path(__file__).resolve().parents[1] / "world_model.py").read_text())
