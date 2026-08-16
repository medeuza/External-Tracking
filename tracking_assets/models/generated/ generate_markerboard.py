from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
TRACKING_ASSETS = ROOT / "src" / "tracking_assets"
if not TRACKING_ASSETS.exists():
    TRACKING_ASSETS = ROOT / "tracking_assets"

MODELS_DIR = TRACKING_ASSETS / "models"
TEMPLATES_DIR = MODELS_DIR / "templates"
GENERATED_DIR = MODELS_DIR / "generated"

# Папка с готовыми изображениями маркер-бордов (board_00.png ... board_05.png)
MARKERS_DIR = TRACKING_ASSETS / "markers" / "apriltag_board" / "DICT_4X4_50"

TEMPLATE_NAME = "turtlebot3_burger_apriltag_board"
TEMPLATE_DIR = TEMPLATES_DIR / TEMPLATE_NAME


ROBOTS = [
    {
        "model_name": "turtlebot3_burger_apriltag_board_000",
        "board_id": "00",
        "pose": "4.0   9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_board_001",
        "board_id": "01",
        "pose": "1.0   9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_board_002",
        "board_id": "02",
        "pose": "-4.25 9.0  0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_board_003",
        "board_id": "03",
        "pose": "-4.25 -9.0 0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_board_004",
        "board_id": "04",
        "pose": "1.0   -9.0 0.01 0 0 0",
    },
    {
        "model_name": "turtlebot3_burger_apriltag_board_005",
        "board_id": "05",
        "pose": "4.0   -9.0 0.01 0 0 0",
    },
]


def read_text(path): return path.read_text(encoding="utf-8")
def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render(text, model_name, board_id, pose):
    return (text.replace("__MODEL_NAME__", model_name)
                .replace("__BOARD_ID__", board_id)
                .replace("__POSE__", pose))


def generate_one(model_name, board_id, pose):
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template dir not found: {TEMPLATE_DIR}")

    # Проверяем, что соответствующий маркер-борд реально существует.
    # Путь к PNG прописан в model.sdf как текстура материала — здесь
    # просто страхуемся от опечаток в board_id.
    marker_png = MARKERS_DIR / f"board_{board_id}.png"
    if not marker_png.exists():
        raise FileNotFoundError(f"Marker board image not found: {marker_png}")

    template_sdf = TEMPLATE_DIR / "model.sdf"
    template_config = TEMPLATE_DIR / "model.config"

    out_dir = GENERATED_DIR / model_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text(out_dir / "model.sdf",
               render(read_text(template_sdf), model_name, board_id, pose))
    write_text(out_dir / "model.config",
               render(read_text(template_config), model_name, board_id, pose))
    print(f"[OK] Generated: {out_dir.name}  (board_id={board_id}, pose='{pose}')")


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for r in ROBOTS:
        generate_one(r["model_name"], r["board_id"], r["pose"])
    print(f"[DONE] All {len(ROBOTS)} AprilTag board models generated")


if __name__ == "__main__":
    main()