from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]  # tracking_assets

MODELS_DIR = ROOT / "models"
TEMPLATES_DIR = MODELS_DIR / "templates"
GENERATED_DIR = MODELS_DIR / "generated"

TEMPLATE_NAME = "turtlebot3_burger_stag"
TEMPLATE_DIR = TEMPLATES_DIR / TEMPLATE_NAME

MARKERS_DIR = ROOT / "markers" / "stag"


ROBOTS = [
    {
        "model_name": "turtlebot3_burger_stag_000",
        "marker_id": "000",
        "pose": "-4.5 0 0.01 0 0 0",
    },
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_template(text: str, model_name: str, marker_id: str, pose: str) -> str:
    marker_uri = f"file://{MARKERS_DIR / f'tag_{marker_id}.png'}"

    return (
        text.replace("__MODEL_NAME__", model_name)
            .replace("__MARKER_ID__", marker_id)
            .replace("__POSE__", pose)
            .replace("__MARKER_URI__", marker_uri)
    )


def generate_one_robot(model_name: str, marker_id: str, pose: str) -> None:
    template_sdf = TEMPLATE_DIR / "model.sdf"
    template_config = TEMPLATE_DIR / "model.config"
    marker_file = MARKERS_DIR / f"tag_{marker_id}.png"

    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template dir not found: {TEMPLATE_DIR}")
    if not template_sdf.exists():
        raise FileNotFoundError(f"Template SDF not found: {template_sdf}")
    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")
    if not marker_file.exists():
        raise FileNotFoundError(f"STag marker not found: {marker_file}")

    out_dir = GENERATED_DIR / model_name

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        out_dir / "model.sdf",
        render_template(read_text(template_sdf), model_name, marker_id, pose),
    )

    write_text(
        out_dir / "model.config",
        render_template(read_text(template_config), model_name, marker_id, pose),
    )

    print(f"[OK] Generated model: {out_dir}")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for robot in ROBOTS:
        generate_one_robot(
            model_name=robot["model_name"],
            marker_id=robot["marker_id"],
            pose=robot["pose"],
        )

    print("[DONE] All STag models generated")


if __name__ == "__main__":
    main()