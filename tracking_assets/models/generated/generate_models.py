from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "tracking_assets" / "models"
TEMPLATES_DIR = MODELS_DIR / "templates"
GENERATED_DIR = MODELS_DIR / "generated"

TEMPLATE_NAME = "turtlebot3_burger_aruco"
TEMPLATE_DIR = TEMPLATES_DIR / TEMPLATE_NAME


ROBOTS = [
    {
        "model_name": "turtlebot3_burger_aruco_000",
        "marker_id": "000",
        "pose": "-4.5 0 0.01 0 0 0",
    },
    # Примеры для будущего:
    # {
    #     "model_name": "turtlebot3_burger_aruco_001",
    #     "marker_id": "001",
    #     "pose": "0 0 0.01 0 0 0",
    # },
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_template(text: str, model_name: str, marker_id: str, pose: str) -> str:
    return (
        text.replace("__MODEL_NAME__", model_name)
            .replace("__MARKER_ID__", marker_id)
            .replace("__POSE__", pose)
    )


def generate_one_robot(model_name: str, marker_id: str, pose: str) -> None:
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template dir not found: {TEMPLATE_DIR}")

    template_sdf = TEMPLATE_DIR / "model.sdf"
    template_config = TEMPLATE_DIR / "model.config"

    if not template_sdf.exists():
        raise FileNotFoundError(f"Template SDF not found: {template_sdf}")

    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")

    out_dir = GENERATED_DIR / model_name

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdf_text = read_text(template_sdf)
    config_text = read_text(template_config)

    rendered_sdf = render_template(
        sdf_text,
        model_name=model_name,
        marker_id=marker_id,
        pose=pose,
    )
    rendered_config = render_template(
        config_text,
        model_name=model_name,
        marker_id=marker_id,
        pose=pose,
    )

    write_text(out_dir / "model.sdf", rendered_sdf)
    write_text(out_dir / "model.config", rendered_config)

    print(f"[OK] Generated model: {out_dir}")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for robot in ROBOTS:
        generate_one_robot(
            model_name=robot["model_name"],
            marker_id=robot["marker_id"],
            pose=robot["pose"],
        )

    print("[DONE] All models generated")


if __name__ == "__main__":
    main()