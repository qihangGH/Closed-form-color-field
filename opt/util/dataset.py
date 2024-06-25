from .nerf_dataset import NeRFDataset
from .llff_dataset import LLFFDataset
from .dtu_dataset import DTUDataset
from os import path


def auto_dataset(root: str, *args, **kwargs):
    if path.isfile(path.join(root, 'poses_bounds.npy')):
        print("Detected LLFF dataset")
        return LLFFDataset(root, *args, **kwargs)
    elif path.isfile(path.join(root, 'transforms.json')) or \
            path.isfile(path.join(root, 'transforms_train.json')):
        print("Detected NeRF (Blender) dataset")
        return NeRFDataset(root, *args, **kwargs)
    elif path.isfile(path.join(root, 'Cameras', '00000000_cam.txt')) or \
            path.isfile(path.join(root, 'cameras_sphere.npz')):
        print("Detected DTU dataset")
        return DTUDataset(root, *args, **kwargs)
    else:
        raise ValueError('Dataset has no been defined')


datasets = {
    'nerf': NeRFDataset,
    'llff': LLFFDataset,
    'dtu': DTUDataset,
    'auto': auto_dataset
}
