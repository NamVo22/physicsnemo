import argparse

import torch

from common import add_run_args, make_point_cloud_dataset, make_single_sample_loader
from rollout import TransolverTimeConditionalRollout


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_stats = {
        "node": {key: value.to(device) for key, value in dataset.node_stats.items()},
        "edge": {},
        "feature": {
            key: value.to(device) for key, value in dataset.feature_stats.items()
        },
    }

    model = TransolverTimeConditionalRollout(
        functional_dim=1,
        embedding_dim=3,
        out_dim=3,
        slice_num=128,
        n_layers=8,
        unified_pos=False,
        structured_shape=None,
        use_te=False,
        time_input=True,
        num_time_steps=args.num_time_steps,
    ).to(device)
    model.eval()

    for batch in dataloader:
        sample = batch[0].to(device)
        with torch.no_grad():
            pred = model(sample=sample, data_stats=data_stats)

        target_steps = args.num_time_steps - 1
        num_nodes = sample.node_target.size(0)
        target = (
            sample.node_target.view(num_nodes, target_steps, 3)
            .transpose(0, 1)
            .contiguous()
        )
        loss = torch.nn.functional.mse_loss(pred, target)

        print("device:", device)
        print("coords:", sample.node_features["coords"].shape)
        print("features:", sample.node_features["features"].shape)
        print("target flat:", sample.node_target.shape)
        print("prediction:", pred.shape)
        print("target:", target.shape)
        print("mse:", loss.item())


if __name__ == "__main__":
    main()
