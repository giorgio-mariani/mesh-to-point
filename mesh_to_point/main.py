import argparse

import numpy as np

from mesh_to_point.utils import create_pointcloud_from_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-path", required=False, type=str, default="pointcloud.npy")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=8192)
    args = parser.parse_args()

    pointcloud = create_pointcloud_from_mesh(
        mesh_path=args.input_path,
        num_points=args.num_points,
        num_views=args.num_images,
    )

    with open(args.output_path, "w") as fp:
        np.save(fp, pointcloud)


if __name__ == "__main__":
    main()
