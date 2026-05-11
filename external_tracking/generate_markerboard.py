from pathlib import Path

import cv2
import numpy as np


def generate_apriltag_boards(
    dictionary_name: int,
    dictionary_label: str,
    num_boards: int,
    markers_per_board: int,
    grid_rows: int,
    grid_cols: int,
    marker_size_px: int,
    marker_separation_px: int,
    padding_px: int,
    start_id: int,
) -> None:
    assert grid_rows * grid_cols == markers_per_board, (
        "grid_rows * grid_cols must equal markers_per_board"
    )

    output_dir = (
        Path("/home/katya/wspace/src/tracking_assets/markers/apriltag_board")
        / dictionary_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)

    board_inner_w = (
        grid_cols * marker_size_px + (grid_cols - 1) * marker_separation_px
    )
    board_inner_h = (
        grid_rows * marker_size_px + (grid_rows - 1) * marker_separation_px
    )
    canvas_w = board_inner_w + 2 * padding_px
    canvas_h = board_inner_h + 2 * padding_px

    for board_idx in range(num_boards):
        canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

        first_id = start_id + board_idx * markers_per_board

        for local_idx in range(markers_per_board):
            row = local_idx // grid_cols
            col = local_idx % grid_cols

            marker_id = first_id + local_idx
            marker = cv2.aruco.generateImageMarker(
                dictionary,
                marker_id,
                marker_size_px,
            )

            top = padding_px + row * (marker_size_px + marker_separation_px)
            left = padding_px + col * (marker_size_px + marker_separation_px)

            canvas[
                top:top + marker_size_px,
                left:left + marker_size_px,
            ] = marker

        output_file = output_dir / f"board_{board_idx:02d}.png"
        cv2.imwrite(str(output_file), canvas)
        print(f"Saved: {output_file}")


def main() -> None:
    generate_apriltag_boards(
        dictionary_name=cv2.aruco.DICT_4X4_50,
        dictionary_label="DICT_4X4_50",
        num_boards=6,
        markers_per_board=4,
        grid_rows=2,
        grid_cols=2,
        marker_size_px=400,
        marker_separation_px=80,
        padding_px=40,
        start_id=0,
    )


if __name__ == "__main__":
    main()