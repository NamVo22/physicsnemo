import argparse

from common import add_run_args, resolve_run_paths
from vtp_reader import Reader


def main():
    parser = add_run_args(argparse.ArgumentParser())
    args = parser.parse_args()
    paths = resolve_run_paths(args.run_dir)

    srcs, dsts, point_data = Reader()(
        data_dir=str(paths["data_vtp"]),
        num_samples=args.num_samples,
        split="train",
    )
    sample = point_data[0]

    print("run_dir:", paths["run_dir"])
    print("data_vtp:", paths["data_vtp"])
    print("number of samples:", len(point_data))
    print("keys:", sample.keys())
    print("coords shape:", sample["coords"].shape)
    print("thickness shape:", sample["thickness"].shape)
    print("number of edges:", len(srcs[0]))
    print("src:", srcs[0])
    print("dst:", dsts[0])
    print("positions at t0:", sample["coords"][0])
    print("positions at final timestep:", sample["coords"][-1])


if __name__ == "__main__":
    main()
