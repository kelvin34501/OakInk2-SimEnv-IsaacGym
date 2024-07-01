import os
from ..scene import GeomCata

_this_dir = os.path.dirname(__file__)
_ws_dir = os.path.normpath(os.path.join(_this_dir, "..", "..", ".."))
_mano_urdf_dir = os.path.normpath(os.path.join(_ws_dir, "asset/mano_mod"))


def load_table_def():
    _asset_dir = os.path.join(_ws_dir, "asset/table")
    _asset_name = "table.urdf"
    assert os.path.exists(os.path.join(_asset_dir, _asset_name))
    return {
        "geom_cata": GeomCata.URDF,
        "geom_param": {
            "asset_dir": _asset_dir,
            "asset_name": _asset_name,
        },
    }


def load_box_def(
    box_dim,
):
    dim = (float(box_dim[0]), float(box_dim[1]), float(box_dim[2]))
    return {
        "geom_cata": GeomCata.BOX,
        "geom_param": {
            "dim": dim,
        },
    }


class AssetLoader(object):
    def __call__(self):
        raise NotImplementedError()
