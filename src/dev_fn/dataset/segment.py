from __future__ import annotations

import os
import enum
import json

from .process_upkeep import ProcessDef
from ..util.meta_util import to_snake_case

import typing

if typing.TYPE_CHECKING:
    from typing import Optional


@enum.unique
class SliceType(enum.Enum):
    CombinedSlice = to_snake_case("CombinedSlice")
    PrimitiveSlice = to_snake_case("PrimitiveSlice")
    AtomicPrimtiveSlice = to_snake_case("AtomicPrimtiveSlice")


# within each slice, there are segments
@enum.unique
class SegmentType(enum.Enum):
    ApproachSegment = to_snake_case("ApproachSegment")
    InteractionSegment = to_snake_case("InteractionSegment")
    RetreatSegment = to_snake_case("RetreatSegment")


HAND_SIDE = ["rh", "lh"]


