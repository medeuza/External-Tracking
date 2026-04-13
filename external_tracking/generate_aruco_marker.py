from pathlib import Path
import cv2
import numpy as np

def generate_aruco_set(
    dictionary_name: int,
    dictionary_label: str,
    start_id: int,
    end_id: int,
    marker_size_px: int,
    padding_px: int,
) -> None:
    output_dir = (
        Path("/home/katya/wspace/src/tracking_assets/markers/aruco")
        / dictionary_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)

    canvas_size = marker_size_px + 2 * padding_px

    for marker_id in range(start_id, end_id + 1):

        marker = cv2.aruco.drawMarker(
            dictionary,
            marker_id,
            marker_size_px,
        )

        canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

        canvas[
            padding_px:padding_px + marker_size_px,
            padding_px:padding_px + marker_size_px
        ] = marker

        output_file = output_dir / f"marker_{marker_id:03d}.png"

        cv2.imwrite(str(output_file), canvas)

        print(f"Saved: {output_file}")


def main() -> None:
    generate_aruco_set(
        dictionary_name=cv2.aruco.DICT_4X4_50,
        dictionary_label="DICT_4X4_50",
        start_id=0,
        end_id=19,
        marker_size_px=400,
        padding_px=8,
    )


if __name__ == "__main__":
    main()