from __future__ import annotations

import numpy as np
import typing

if typing.TYPE_CHECKING:
    from typing import Optional


def dump_obj_mesh(filename: str, vertices: np.ndarray, faces: Optional[np.ndarray] = None):
    assert vertices.shape[1] == 3 and (faces is None or faces.shape[1] == 3)
    vertices = np.asarray(vertices, dtype=np.float32)
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)
    with open(filename, "w") as obj_file:
        for v in vertices:
            obj_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        if faces is not None:
            for f in faces + 1:
                obj_file.write("f {} {} {}\n".format(f[0], f[1], f[2]))


def load_obj_mesh(filename):
    with open(filename, "r") as obj_file:
        lines = obj_file.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append([float(x) for x in line.strip().split()[1:]])
        elif line.startswith("f "):
            faces.append([int(float(x)) - 1 for x in line.strip().split()[1:]])  # * in case x is a float number
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
