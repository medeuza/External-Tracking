from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
TRACKING_ASSETS = ROOT / "src" / "tracking_assets"
if not TRACKING_ASSETS.exists():
    TRACKING_ASSETS = ROOT / "tracking_assets"

MODELS_DIR = TRACKING_ASSETS / "models"
TEMPLATES_DIR = MODELS_DIR / "templates"
GENERATED_DIR = MODELS_DIR / "generated"

TEMPLATE_NAME = "turtlebot3_burger_apriltag"
TEMPLATE_DIR = TEMPLATES_DIR / TEMPLATE_NAME


ROBOTS = [
    {
        "model_name": "turtlebot3_burger_apriltag_000",
        "marker_id": "000",
        "pose": "4.0   9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_001",
        "marker_id": "001",
        "pose": "1.0   9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_002",
        "marker_id": "002",
        "pose": "-4.25 9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_003",
        "marker_id": "003",
        "pose": "-4.25 -9.0 0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_004",
        "marker_id": "004",
        "pose": "1.0   -9.0 0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_005",
        "marker_id": "005",
        "pose": "2.5   -9.0 0.01 0 0 0",
    },
]


def read_text(path): return path.read_text(encoding="utf-8")
def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render(text, model_name, marker_id, pose):
    return (text.replace("__MODEL_NAME__", model_name)
                .replace("__MARKER_ID__", marker_id)
                .replace("__POSE__", pose))


def generate_one(model_name, marker_id, pose):
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template dir not found: {TEMPLATE_DIR}")

    template_sdf = TEMPLATE_DIR / "model.sdf"
    template_config = TEMPLATE_DIR / "model.config"

    out_dir = GENERATED_DIR / model_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text(out_dir / "model.sdf",
               render(read_text(template_sdf), model_name, marker_id, pose))
    write_text(out_dir / "model.config",
               render(read_text(template_config), model_name, marker_id, pose))
    print(f"[OK] Generated: {out_dir.name}  (marker_id={marker_id}, pose='{pose}')")


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for r in ROBOTS:
        generate_one(r["model_name"], r["marker_id"], r["pose"])
    print(f"[DONE] All {len(ROBOTS)} AprilTag models generated")


if __name__ == "__main__":
    main()