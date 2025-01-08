from pathlib import Path
from argparse import ArgumentParser

from rich.progress import track
import pandas as pd

from compute_position import compute_position
import os

def compute_amazing_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """
    return compute_position(part_image_path, gripper_image_path)


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        path_to_visualization_image = f"visualizations/visualization-{os.path.basename(part_image_path)}+{os.path.basename(gripper_image_path)}.png"
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle, path_to_visualization_image])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle", "visualization"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
