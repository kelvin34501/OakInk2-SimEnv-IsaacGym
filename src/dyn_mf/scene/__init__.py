import enum


@enum.unique
class FrameConvention(enum.Enum):
    Y_UP = "y_up"
    Z_UP = "z_up"


@enum.unique
class GeomCata(enum.Enum):
    URDF = "urdf"
    BOX = "box"
    SPHERE = "sphere"
