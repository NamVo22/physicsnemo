import argparse

from common import add_run_args, make_point_cloud_dataset, make_single_sample_loader


def main():
    parser = add_run_args(argparse.ArgumentParser())
    args = parser.parse_args()
    dataset = make_point_cloud_dataset(
        args.run_dir,
        num_samples=args.num_samples,
        num_time_steps=args.num_time_steps,
        dt=args.dt,
    )
    dataloader = make_single_sample_loader(dataset)

    print("dataset length:", len(dataset))
    for idx, batch in enumerate(dataloader):
        sample = batch[0]
        print("\nbatch:", idx)
        print("type(batch):", type(batch))
        print("len(batch):", len(batch))
        print("sample:", sample)
        print("node_features keys:", sample.node_features.keys())
        print("coords shape:", sample.node_features["coords"].shape)
        print("features shape:", sample.node_features["features"].shape)
        print("node_target shape:", sample.node_target.shape)
        print("graph:", sample.graph)


if __name__ == "__main__":
    main()
