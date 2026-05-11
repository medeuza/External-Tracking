from pathlib import Path
import math

import cv2
import numpy as np


def polar_to_cart(radius: float, radians: float) -> tuple[float, float]:
    return (
        0.5 + math.cos(radians) * radius,
        0.5 - math.sin(radians) * radius,
    )


def fill_code_locations() -> list[tuple[float, float]]:
    code_locs = []

    for i in range(4):
        shift = i * (math.pi / 2)

        code_locs.append(polar_to_cart(0.088363142525988, 0.785398163397448 + shift))

        code_locs.append(polar_to_cart(0.206935928182607, 0.459275804122858 + shift))
        code_locs.append(polar_to_cart(0.206935928182607, (math.pi / 2) - 0.459275804122858 + shift))

        code_locs.append(polar_to_cart(0.313672146827381, 0.200579720495241 + shift))
        code_locs.append(polar_to_cart(0.327493143484516, 0.591687617505840 + shift))
        code_locs.append(polar_to_cart(0.327493143484516, (math.pi / 2) - 0.591687617505840 + shift))
        code_locs.append(polar_to_cart(0.313672146827381, (math.pi / 2) - 0.200579720495241 + shift))

        code_locs.append(polar_to_cart(0.437421957035861, 0.145724938287167 + shift))
        code_locs.append(polar_to_cart(0.437226762361658, 0.433363129825345 + shift))
        code_locs.append(polar_to_cart(0.430628029742607, 0.785398163397448 + shift))
        code_locs.append(polar_to_cart(0.437226762361658, (math.pi / 2) - 0.433363129825345 + shift))
        code_locs.append(polar_to_cart(0.437421957035861, (math.pi / 2) - 0.145724938287167 + shift))

    return code_locs


def load_codes(code_file: Path, no_of_bits: int = 48) -> list[str]:
    with code_file.open("r") as file:
        codes = [line.strip() for line in file if line.strip()]

    for code in codes:
        if len(code) != no_of_bits:
            raise ValueError(f"Invalid code length: {len(code)} in {code_file}")

    return codes


def generate_stag_marker(
    code: str,
    marker_id: int,
    total_markers: int,
    output_file: Path,
    file_size: int = 1000,
) -> None:
    no_of_bits = 48

    border = 0.125
    outer_circle_radius = 0.4
    inner_circle_radius = 0.35
    code_radius = 0.062482177287080
    filler_code_radius = 0.7

    marker_size = file_size / (1 + border * 2)
    border_size = marker_size * border

    outer_circle_diameter_size = 2 * marker_size * outer_circle_radius
    inner_circle_diameter_size = 2 * marker_size * inner_circle_radius

    outer_circle_top_left = (file_size - outer_circle_diameter_size) / 2
    inner_circle_top_left = (file_size - inner_circle_diameter_size) / 2

    code_circle_diameter_size = 2 * inner_circle_diameter_size * code_radius
    filler_circle_diameter_size = code_circle_diameter_size * filler_code_radius

    code_locs = fill_code_locations()

    image = np.full((file_size, file_size), 255, dtype=np.uint8)

    x0 = int(round(border_size))
    y0 = int(round(border_size))
    x1 = int(round(border_size + marker_size))
    y1 = int(round(border_size + marker_size))

    cv2.rectangle(image, (x0, y0), (x1, y1), 0, thickness=-1)

    outer_center = (file_size // 2, file_size // 2)
    outer_radius = int(round(outer_circle_diameter_size / 2))
    inner_radius = int(round(inner_circle_diameter_size / 2))

    cv2.circle(image, outer_center, outer_radius, 255, thickness=-1)

    # Black code circles
    for j in range(no_of_bits):
        if code[j] == "1":
            cx = inner_circle_top_left + inner_circle_diameter_size * code_locs[j][0]
            cy = inner_circle_top_left + inner_circle_diameter_size * code_locs[j][1]
            radius = int(round(code_circle_diameter_size / 2))
            cv2.circle(image, (int(round(cx)), int(round(cy))), radius, 0, thickness=-1)

    # Filler circles between nearby black circles
    for j in range(no_of_bits):
        for k in range(j + 1, no_of_bits):
            if code[j] == "1" and code[k] == "1":
                dx = code_locs[j][0] - code_locs[k][0]
                dy = code_locs[j][1] - code_locs[k][1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < code_radius * 4:
                    cx = inner_circle_top_left + inner_circle_diameter_size * (
                        (code_locs[j][0] + code_locs[k][0]) / 2
                    )
                    cy = inner_circle_top_left + inner_circle_diameter_size * (
                        (code_locs[j][1] + code_locs[k][1]) / 2
                    )
                    radius = int(round(filler_circle_diameter_size / 2))
                    cv2.circle(image, (int(round(cx)), int(round(cy))), radius, 0, thickness=-1)

    # White code circles
    for j in range(no_of_bits):
        if code[j] == "0":
            cx = inner_circle_top_left + inner_circle_diameter_size * code_locs[j][0]
            cy = inner_circle_top_left + inner_circle_diameter_size * code_locs[j][1]
            radius = int(round(code_circle_diameter_size / 2))
            cv2.circle(image, (int(round(cx)), int(round(cy))), radius, 255, thickness=-1)

    # Keep clean white ring between outer and inner circle
    ring_mask = np.zeros_like(image)
    cv2.circle(ring_mask, outer_center, outer_radius, 255, thickness=-1)
    cv2.circle(ring_mask, outer_center, inner_radius, 0, thickness=-1)
    image[ring_mask == 255] = 255

    # Add marker id and HD label, as in reference generator
    id_text = str(marker_id).zfill(len(str(total_markers)))
    display_id = id_text

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_center = int(round(2.375 * border_size))
    hd_center = int(round(1.875 * border_size))

    def draw_rotated_text(base: np.ndarray, text: str, center: tuple[int, int], scale: float, thickness: int) -> None:
        text_layer = np.zeros_like(base)

        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        origin = (center[0] - tw // 2, center[1] + th // 2)

        cv2.putText(
            text_layer,
            text,
            origin,
            font,
            scale,
            255,
            thickness,
            cv2.LINE_AA,
        )

        matrix = cv2.getRotationMatrix2D(center, 315, 1.0)
        rotated = cv2.warpAffine(text_layer, matrix, (file_size, file_size))
        base[rotated > 0] = 255

    draw_rotated_text(image, display_id, (text_center, text_center), 2.4, 4)
    draw_rotated_text(image, "HD11", (hd_center, hd_center), 0.7, 2)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), image)


def generate_stag_set(
    code_file: Path,
    output_dir: Path,
    start_id: int,
    end_id: int,
    marker_size_px: int,
) -> None:
    codes = load_codes(code_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    for marker_id in range(start_id, end_id + 1):
        code = codes[marker_id]
        output_file = output_dir / f"tag_{marker_id:03d}.png"

        generate_stag_marker(
            code=code,
            marker_id=marker_id,
            total_markers=len(codes),
            output_file=output_file,
            file_size=marker_size_px,
        )

        print(f"Saved: {output_file}")


def main() -> None:
    generate_stag_set(
        code_file=Path("/home/katya/wspace/src/stag/ref/marker generator/HD11.txt"),
        output_dir=Path("/home/katya/wspace/src/tracking_assets/markers/stag/HD11"),
        start_id=0,
        end_id=19,
        marker_size_px=1000,
    )


if __name__ == "__main__":
    main()