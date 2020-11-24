from enum import Enum, auto


class EndpointType(Enum):
    """In our code, a connection may exist between either
    of the four pairs of endpoints (A-A, A-B, B-A, B-B),
    or between an endpoint and a centerpoint of two primitives (A-C, B-C, C-A, C-B). """
    ENDPOINT_A = auto()
    ENDPOINT_B = auto()
    CENTERPOINT = auto()
    NONE = auto()


ET_A = EndpointType.ENDPOINT_A      # connects to junction first
ET_B = EndpointType.ENDPOINT_B      # connects to junction second; can be free
ET_C = EndpointType.CENTERPOINT     # midpoint
ET_NONE = EndpointType.NONE         # primitive not connected


class EndpointTypeConfig(Enum):
    """Each junction point is characterized by a combination
    of connected primitive endpoints. """
    # two-line intersections
    L_JUNCTION = [ET_A, ET_A]     # lines intersecting at edges e.g. L, Г
    T_JUNCTION = auto()     # one line intersecting at non-edge, second at edge e.g. T
    X_JUNCTION = auto()     # two lines intersecting at non-edges, e.g. X, +

    # three-line intersections
    Y_JUNCTION = auto()     # three lines intersecting at edge e.g. Y, <-
    K_JUNCTION = auto()     # one line intersecting at non-edge, two others at edge e.g. K, -<-
    XDASH_JUNCTION = auto() # two lines intersecting at non-edge, two others at edge e.g. "X-"
    STAR3_JUNCTION = auto() # three lines intersecting at non-edge e.g. Ж, *


class SnappingType(Enum):
    SNAPPING_NONE = auto()      # lines will intersect ++++
    SNAPPING_BEAM = auto()      # lines will not intersect, but will form T-junctions with outer lines =||=
    SNAPPING_OUTER = auto()     # lines will not intersect, but will form L-junctions with respective lines ||L__
                                #                                                                           |L___

ST_NONE = SnappingType.SNAPPING_NONE
ST_BEAM = SnappingType.SNAPPING_BEAM
ST_OUTER = SnappingType.SNAPPING_OUTER


class OrthoDirType(Enum):
    ORTHO_DIR_RANDOM = 'random'
    ORTHO_DIR_MAXDOT = 'maxdot'


ODT_RANDOM = OrthoDirType.ORTHO_DIR_RANDOM
ODT_MAXDOT = OrthoDirType.ORTHO_DIR_MAXDOT


__all__ = [
    'ET_A', 'ET_B', 'ET_C', 'ET_NONE',
    'ST_NONE', 'ST_BEAM', 'ST_OUTER',
    'ODT_RANDOM', 'ODT_MAXDOT',
]

