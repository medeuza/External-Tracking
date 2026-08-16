from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
TRACKING_ASSETS = ROOT / "src" / "tracking_assets"
if not TRACKING_ASSETS.exists():
    TRACKING_ASSETS = ROOT / "tracking_assets"

MODELS_DIR = TRACKING_ASSETS / "models"
TEMPLATES_DIR = MODELS_DIR / "templates"
GENERATED_DIR = MODELS_DIR / "generated"

TEMPLATE_NAME = "turtlebot3_burger_aruco"
TEMPLATE_DIR = TEMPLATES_DIR / TEMPLATE_NAME

# строки, жёстко зашитые в шаблоне ArUco, которые нужно заменить
TEMPLATE_MODEL_LINE = '<model name="turtlebot3_burger">'
TEMPLATE_POSE_LINE  = '<pose>-4.5 0 0.01 0 0 0</pose>'


ROBOTS = [
    {
        "model_name": "turtlebot3_burger_aruco_000",
        "marker_id": "000",
        "pose": "4.0   9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_aruco_001",
        "marker_id": "001",
        "pose": "1.0   9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_aruco_002",
        "marker_id": "002",
        "pose": "-4.25 9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_aruco_003",
        "marker_id": "003",
        "pose": "-4.25 -9.0 0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_aruco_004",
        "marker_id": "004",
        "pose": "1.0   -9.0 0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_aruco_005",
        "marker_id": "005",
        "pose": "2.5   -9.0 0.01 0 0 0",
    },
]


def read_text(path):
    return path.read_text(encoding="utf-8")


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_sdf(text, model_name, marker_id, pose):
    text = text.replace("__MODEL_NAME__", model_name)
    # 1. marker id placeholder (как в apriltag-шаблоне)
    out = text.replace("__MARKER_ID__", marker_id)

    # 2. имя модели — в шаблоне зашито, заменяем строкой
    new_model_line = f'<model name="{model_name}">'
    if TEMPLATE_MODEL_LINE not in out:
        raise RuntimeError(
            f"Model line not found in template, expected: {TEMPLATE_MODEL_LINE}"
        )
    out = out.replace(TEMPLATE_MODEL_LINE, new_model_line, 1)

    # 3. стартовая поза — в шаблоне зашита, заменяем строкой
    new_pose_line = f'<pose>{pose}</pose>'
    if TEMPLATE_POSE_LINE not in out:
        raise RuntimeError(
            f"Pose line not found in template, expected: {TEMPLATE_POSE_LINE}"
        )
    out = out.replace(TEMPLATE_POSE_LINE, new_pose_line, 1)

    return out


def render_config(text, model_name, marker_id, pose):
    # config может содержать __MARKER_ID__ или нет — заменяем на всякий случай
    return text.replace("__MARKER_ID__", marker_id)


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
               render_sdf(read_text(template_sdf), model_name, marker_id, pose))
    write_text(out_dir / "model.config",
               render_config(read_text(template_config), model_name, marker_id, pose))
    print(f"[OK] Generated: {out_dir.name}  (marker_id={marker_id}, pose='{pose}')")


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for r in ROBOTS:
        generate_one(r["model_name"], r["marker_id"], r["pose"])
    print(f"[DONE] All {len(ROBOTS)} ArUco models generated")


if __name__ == "__main__":
    main()