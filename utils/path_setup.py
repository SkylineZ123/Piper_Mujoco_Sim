"""Register graspnet-baseline and manipulator_grasp on sys.path."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_graspnet_path():
    for sub in ('models', 'dataset', 'utils'):
        p = os.path.join(_ROOT, 'graspnet-baseline', sub)
        if p not in sys.path:
            sys.path.append(p)
    mp = os.path.join(_ROOT, 'manipulator_grasp')
    if mp not in sys.path:
        sys.path.append(mp)
