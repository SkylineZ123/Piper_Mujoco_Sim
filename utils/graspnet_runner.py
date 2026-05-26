"""GraspNet model loading helper."""
import torch

from utils.path_setup import setup_graspnet_path

setup_graspnet_path()
from graspnet import GraspNet  # noqa: E402


def load_graspnet(checkpoint_path: str) -> GraspNet:
    """Instantiate GraspNet and load a checkpoint. Returns eval-mode model on CUDA/CPU."""
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net
