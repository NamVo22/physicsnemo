import argparse

from common import add_run_args, make_point_cloud_dataset, resolve_run_paths


def main():
    parser = add_run_args(argparse.ArgumentParser())
    args = parser.parse_args()
    paths = resolve_run_paths(args.run_dir)
    dataset = make_point_cloud_dataset(
        args.run_dir,
        num_samples=args.num_samples,
        num_time_steps=args.num_time_steps,
        dt=args.dt,
    )
    sample = dataset[0]

    print("run_dir:", paths["run_dir"])
    print("data_vtp:", paths["data_vtp"])
    print("dataset length:", len(dataset))
    print("sample:", sample)
    print("node_features keys:", sample.node_features.keys())
    print("coords shape:", sample.node_features["coords"].shape)
    print("features shape:", sample.node_features["features"].shape)
    print("node_target shape:", sample.node_target.shape)
    print("first node target:", sample.node_target[0])

    print("\nnode stats:")
    for key, value in dataset.node_stats.items():
        print(key, value)

    print("\nfeature stats:")
    for key, value in dataset.feature_stats.items():
        print(key, value)


if __name__ == "__main__":
    main()
