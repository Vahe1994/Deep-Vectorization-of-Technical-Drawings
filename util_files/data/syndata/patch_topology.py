from collections import defaultdict
from copy import deepcopy

import numpy as np
from numpy.random import uniform

from util_files.data.graphics_primitives import PT_LINE, Line
from util_files.geometric import direction_from_angle

from .snapping import snap_beam, snap_outer
from .utils import renormalize, choose_with_proba
from .types import ET_A, ET_B, ET_C, ET_NONE
from .types import ST_NONE, ST_BEAM, ST_OUTER
from .types import ODT_MAXDOT, ODT_RANDOM


class PatchTopology:
    NUM_JUNCTIONS = None
    NUM_DIRECTIONS = None
    ENDPOINT_TYPES = None
    SNAPPING_TYPES = None

    def __init__(self, dataset):
        # self.junction_types = junction_types
        self.dataset = dataset

    @classmethod
    def from_dataset(cls, dataset): return cls(dataset)

    def _sample_juncpoints(self): raise NotImplemented

    def _sample_directions(self): raise NotImplemented

    def _sample_direction_props(self):
        # TODO @artonson: same line width for all primitives?
        return [{
            'primitive_type': PT_LINE,
            'num_primitives': choose_with_proba(self.dataset.strokes_probas),
            'gap': uniform(self.dataset.min_primitives_gap, self.dataset.max_primitives_gap),
            'stroke_width': uniform(self.dataset.min_stroke_width, self.dataset.max_stroke_width),
            'length': uniform(self.dataset.min_stroke_length, self.dataset.max_stroke_length),
            'ortho_dir_type': ODT_MAXDOT,
        } for _ in range(self.NUM_DIRECTIONS)]

    def _generate_direction(self, primitive_type=PT_LINE,
                            endpoint_a=None, endpoint_b=None,
                            direction=None,
                            num_primitives=1,
                            gap=0, stroke_width=1, length=1,
                            ortho_dir_type=ODT_RANDOM, snapping_type=ST_NONE,
                            joint_directions_props=None):

        closest_direction_idx = np.argmax([np.dot(props['direction'], direction)
                                           for props in joint_directions_props])
        closest_direction_props = joint_directions_props[closest_direction_idx]
        direction_cl = closest_direction_props['direction']

        ortho_dir_unsigned = np.array([-direction[1], direction[0]])
        if ODT_RANDOM == ortho_dir_type:
            ortho_sign = +1 if uniform() > .5 else -1
        elif ODT_MAXDOT == ortho_dir_type:
            ortho_sign = +1 if np.dot(direction_cl, ortho_dir_unsigned) > 0 else -1
        else:
            raise ValueError('not supported value for ortho_dir_type: "{}"'.format(ortho_dir_type))
        ortho_dir = ortho_sign * ortho_dir_unsigned

        primitives = []
        shift = (gap + stroke_width) * ortho_dir
        for i in range(num_primitives):
            primitive_end_a = endpoint_a + i * shift
            primitive_end_b = primitive_end_a + length * direction
            primitive = Line(tuple(primitive_end_a), tuple(primitive_end_b), stroke_width)
            primitives.append(primitive)
        return {primitive_type: primitives}

    def get_vector(self):
        # select junction point, directions and props within topology
        junction_points = self._sample_juncpoints()
        assert len(junction_points) == self.NUM_JUNCTIONS
        directions = self._sample_directions()
        assert len(directions) == self.NUM_DIRECTIONS
        directions_props = self._sample_direction_props()
        assert len(directions_props) == self.NUM_DIRECTIONS

        primitive_props = self._compute_primitive_props(
            junction_points, directions, directions_props)
        assert len(primitive_props) == self.NUM_DIRECTIONS

        directions = [self._generate_direction(**props) for props in primitive_props]
        vector = self._snap(directions)
        return vector

    def _snap(self, primitives_by_direction):
        vector = defaultdict(list)
        for primitives_by_type in primitives_by_direction:
            for primitive_type, primitive_list in primitives_by_type.items():
                vector[primitive_type].extend(primitive_list)
        return vector

    def _compute_primitive_props(self, junction_points, directions, directions_props):
        raise NotImplemented


class NoJunctionTopology(PatchTopology):
    pass


class OneJunctionTopology(PatchTopology):
    NUM_JUNCTIONS = 1

    def _sample_juncpoints(self):
        patch_width, patch_height = self.dataset.patch_size
        border = self.dataset.border
        return np.array([[
            uniform(border, patch_height - border),
            uniform(border, patch_width - border)
        ]])

    def _sample_directions(self):
        # TODO junction_directions_probas must be offsets, not values
        offset_angle = choose_with_proba(renormalize(self.dataset.offset_directions_probas))
        angles = []
        prohibited_angles = set()
        for angle_id in range(self.NUM_DIRECTIONS):
            direction_probas = renormalize(
                self.dataset.directions_probas, without=prohibited_angles)
            angle = choose_with_proba(direction_probas)
            prohibited_angles.update({angle, angle + np.pi, angle - np.pi})
            angles.append(angle)
        return [direction_from_angle(angle + offset_angle) for angle in angles]

    def _compute_primitive_props(self, junction_points, directions, directions_props):
        primitives_props = deepcopy(directions_props)
        junc_point_coords = junction_points[0]

        for junction_idx in range(self.NUM_JUNCTIONS):
            endpoint_types = self.ENDPOINT_TYPES[junction_idx]
            snapping_types = self.SNAPPING_TYPES[junction_idx]
            for direction_idx, (primitive_props, direction_props, direction, endpoint_type, snapping_type) in \
                    enumerate(zip(primitives_props, directions_props, directions, endpoint_types, snapping_types)):
                joint_directions_props = [props for props_idx, props in enumerate(primitives_props)
                                          if props_idx != direction_idx and ET_NONE != endpoint_types[props_idx]]
                primitive_props.update({
                    'direction': direction,
                    'joint_directions_props': joint_directions_props,
                    'snapping_type': snapping_type,
                })

                if ET_A == endpoint_type:    # endpoint is ET_A, i.e. first endpoint
                    primitive_endpoint = junc_point_coords
                    primitive_props.update({'endpoint_a': primitive_endpoint})

                elif ET_B == endpoint_type:  # endpoint is ET_B, i.e. second endpoint
                    primitive_endpoint = junc_point_coords
                    primitive_props.update({'endpoint_b': primitive_endpoint})

                elif ET_C == endpoint_type:  # endpoint is ET_C, i.e. center point
                    primitive_endpoint = junc_point_coords - .5 * direction_props['length'] * direction
                    primitive_props.update({'endpoint_a': primitive_endpoint})

                elif ET_NONE != endpoint_type:
                    raise ValueError('not supported value for endpoint type: "{}"'.format(endpoint_type))

        return primitives_props

    def _snap(self, primitives_by_direction):
        return self._snap2(primitives_by_direction)

        # vector = {}
        #
        # # iterate over junctions (we know their traits, e.g. endpoints / snappings)
        # for junction_idx, endpoint_types, snapping_types in \
        #         zip(range(self.NUM_JUNCTIONS), self.ENDPOINT_TYPES, self.SNAPPING_TYPES):
        #
        #
        #     first, second = primitives_by_direction
        #     assert {PrimitiveType.PT_LINE} == set(first.keys()) and {PrimitiveType.PT_LINE} == set(second.keys()), \
        #         "can't handle non-lines for now"
        #     first_lines = first[PrimitiveType.PT_LINE]
        #     second_lines = second[PrimitiveType.PT_LINE]
        #     snapped_first = snap_beam(first_lines, second_lines, self.ENDPOINT_TYPES[0][0])
        #
        #
        #     #
        #     # # iterate over primitives in bundle, computing snapping
        #     # for direction_idx, (primitives_by_type, endpoint_type, snapping_type) in \
        #     #         enumerate(zip(primitives_by_direction, endpoint_types, snapping_types)):
        #     #
        #     #     joint_primitives = [primitives for other_idx, primitives in enumerate(primitives_by_direction)
        #     #                         if other_idx != direction_idx and ET_NONE != endpoint_types[props_idx]]
        #     #
        #     #     if snapping_type == ST_BEAM:
        #     #         snap_beam(primitives_by_type, endpoint_type)
        #     #
        #     #     elif snapping_type == ST_OUTER:
        #     #         snap_beam(primitives_by_type, endpoint_type)
        #     #
        #     #     elif snapping_type != ST_NONE:
        #     #         raise ValueError('not supported value for snapping type: "{}"'.format(snapping_type))
        #
        #
        # vector = {
        #     PrimitiveType.PT_LINE: snapped_first + second_lines
        # }
        # return vector

    def _snap2(self, primitives_by_direction):
        first, second = primitives_by_direction
        assert {PT_LINE} == set(first.keys()) == set(second.keys()), "can't handle non-lines for now"
        first_lines = first[PT_LINE]
        second_lines = second[PT_LINE]
        snapped = self._do_snap(first_lines, second_lines)
        vector = {PT_LINE: snapped}
        return vector

    def _do_snap(self, first_lines, second_lines):
        snapping_first, snapping_second = self.SNAPPING_TYPES[0]
        assert ST_NONE == snapping_first == snapping_second
        return first_lines + second_lines


class LTopology(OneJunctionTopology):
    NUM_DIRECTIONS = 2
    ENDPOINT_TYPES = [[ET_A, ET_A]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE]]


class LBeamTopology(LTopology):
    SNAPPING_TYPES = [[ST_BEAM, ST_NONE]]

    def _do_snap(self, first_lines, second_lines):
        et_first, et_second = self.ENDPOINT_TYPES[0]
        snapped_first = snap_beam(first_lines, second_lines, et_first)
        return snapped_first + second_lines


class LBeam2Topology(LTopology):
    SNAPPING_TYPES = [[ST_BEAM, ST_BEAM]]

    def _do_snap(self, first_lines, second_lines):
        et_first, et_second = self.ENDPOINT_TYPES[0]
        snapped_first = snap_beam(first_lines, second_lines, et_first)
        snapped_second = snap_beam(second_lines, first_lines, et_second)
        return snapped_first + snapped_second


class LOuterTopology(LTopology):
    SNAPPING_TYPES = [[ST_OUTER, ST_OUTER]]

    def _do_snap(self, first_lines, second_lines):
        et_first, et_second = self.ENDPOINT_TYPES[0]
        snapped_first, snapped_second = snap_outer(first_lines, second_lines, et_first, et_second)
        return snapped_first + snapped_second


class TTopology(OneJunctionTopology):
    NUM_DIRECTIONS = 2
    ENDPOINT_TYPES = [[ET_A, ET_C]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE]]


class TBeamTopology(TTopology):
    SNAPPING_TYPES = [[ST_BEAM, ST_NONE]]

    def _do_snap(self, first_lines, second_lines):
        et_first, et_second = self.ENDPOINT_TYPES[0]
        snapped_first = snap_beam(first_lines, second_lines, et_first)
        return snapped_first + second_lines


class TOuterTopology(TTopology):
    SNAPPING_TYPES = [[ST_OUTER, ST_OUTER]]

    def _do_snap(self, first_lines, second_lines):
        et_first, et_second = self.ENDPOINT_TYPES[0]

        first_len, second_len = len(first_lines), len(second_lines)
        if first_len == 1:
            snapped_first = snap_beam(first_lines, second_lines, et_first)
            snapped_second = second_lines
        else:
            snapped_first_center = snap_beam(first_lines[1:-1], second_lines, et_first)
            snapped_first_1, snapped_second_1 = snap_outer(first_lines[:1], second_lines[-1:], endpoint_first=ET_A, endpoint_second=ET_B)
            snapped_first_2, snapped_second_2 = snap_outer(first_lines[-1:], second_lines[-1:], endpoint_first=ET_A, endpoint_second=ET_A)
            snapped_first = snapped_first_center + snapped_first_1 + snapped_first_2
            snapped_second = second_lines[:-1] + snapped_second_1 + snapped_second_2
        return snapped_first + snapped_second


class XTopology(OneJunctionTopology):
    NUM_DIRECTIONS = 2
    ENDPOINT_TYPES = [[ET_C, ET_C]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE]]


class XBeamTopology(XTopology):
    SNAPPING_TYPES = [[ST_BEAM, ST_NONE]]

    def _do_snap(self, first_lines, second_lines):
        # et_first, et_second = self.ENDPOINT_TYPES[0]
        snapped_first_1 = snap_beam(first_lines, second_lines[:1], endpoint_type=ET_B)
        snapped_first_2 = snap_beam(first_lines, second_lines[-1:], endpoint_type=ET_A)
        return snapped_first_1 + snapped_first_2 + second_lines


class XOuterTopology(XTopology):
    SNAPPING_TYPES = [[ST_OUTER, ST_OUTER]]

    def _do_snap(self, first_lines, second_lines):
        # et_first, et_second = self.ENDPOINT_TYPES[0]
        first_len, second_len = len(first_lines), len(second_lines)
        if first_len == 1 and second_len == 1:
            snapped_first = first_lines
            snapped_second = second_lines

        elif first_len == 1 and second_len > 1:
            snapped_first_1 = snap_beam(first_lines, second_lines[:1], ET_B)
            snapped_first_2 = snap_beam(first_lines, second_lines[-1:], ET_A)
            snapped_first = snapped_first_1 + snapped_first_2
            snapped_second = second_lines

        elif first_len > 1 and second_len == 1:
            snapped_first = first_lines
            snapped_second_1 = snap_beam(second_lines, first_lines[:1], ET_B)
            snapped_second_2 = snap_beam(second_lines, first_lines[-1:], ET_A)
            snapped_second = snapped_second_1 + snapped_second_2

        else:
            snapped_first_center = first_lines[1:-1]
            snapped_second_center = second_lines[1:-1]

            snapped_first_1, snapped_second_1 = snap_outer(first_lines[:1],  second_lines[:1],  endpoint_first=ET_B, endpoint_second=ET_B)
            snapped_first_2, snapped_second_2 = snap_outer(first_lines[:1],  second_lines[-1:], endpoint_first=ET_A, endpoint_second=ET_B)
            snapped_first_3, snapped_second_3 = snap_outer(first_lines[-1:], second_lines[:1],  endpoint_first=ET_B, endpoint_second=ET_A)
            snapped_first_4, snapped_second_4 = snap_outer(first_lines[-1:], second_lines[-1:], endpoint_first=ET_A, endpoint_second=ET_A)

            snapped_first = snapped_first_center + snapped_first_1 + snapped_first_2 + snapped_first_3 + snapped_first_4
            snapped_second = snapped_second_center + snapped_second_1 + snapped_second_2 + snapped_second_3 + snapped_second_4

        return snapped_first + snapped_second


class YTopology(OneJunctionTopology):
    NUM_DIRECTIONS = 3
    ENDPOINT_TYPES = [[ET_A, ET_A, ET_A]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE, ST_NONE]]


class KTopology(OneJunctionTopology):
    NUM_DIRECTIONS = 3
    ENDPOINT_TYPES = [[ET_C, ET_A, ET_A]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE, ST_NONE]]


class XDashTopology(OneJunctionTopology):
    NUM_DIRECTIONS = 3
    ENDPOINT_TYPES = [[ET_C, ET_C, ET_A]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE, ST_NONE]]


class Star3Topology(OneJunctionTopology):
    NUM_DIRECTIONS = 3
    ENDPOINT_TYPES = [[ET_C, ET_C, ET_C]]
    SNAPPING_TYPES = [[ST_NONE, ST_NONE, ST_NONE]]




# class TwoJunctionTopology(PatchTopology):
#     class JunctionType(Enum):
#         H_JUNCTION = auto()     #
#         F_JUNCTION = auto()     #
#         X_JUNCTION = auto()     #
#
#         K_JUNCTION = auto()     #
#         Y_JUNCTION = auto()     #
#         RUSZH_JUNCTION = auto()
#     pass
#
#
# class HTopology(TwoJunctionTopology):
#     NUM_JUNCTIONS = 2
#     NUM_DIRECTIONS = 3
#     ENDPOINT_TYPES = [ET_MIDPOINT, ET_MIDPOINT, ]
#
#     def _sample_directions(self):
#         alpha = 0.
#         beta = choose_with_proba(self.dataset.directions_probas)
#         return [direction_from_angle(alpha),
#                 direction_from_angle(alpha + beta),
#                 direction_from_angle(alpha - beta)]
#
#     def _compute_primitive_props(self, junction_points, directions, directions_props):
#         primitives_props = deepcopy(directions_props)
#         endpoint = junction_points[0]
#         for primitive_props, direction_props, direction in zip(
#                 primitives_props, directions_props, directions):
#             primitive_props.update({
#                 'endpoint': endpoint,
#                 'direction': direction
#             })
#         return primitives_props
#
#
# class ThreeJunctionArrangement(PatchTopology):
#     pass




TOPOLOGY_BY_NAME = {
    'l': LTopology,
    'l-beam': LBeamTopology,
    'l-beam2': LBeam2Topology,
    'l-outer': LOuterTopology,
    't': TTopology,
    't-beam': TBeamTopology,
    't-outer': TOuterTopology,
    'x': XTopology,
    'x-beam': XBeamTopology,
    'x-outer': XOuterTopology,
    'y': YTopology,
    'k': KTopology,
    'x-dash': XDashTopology,
    'star3': Star3Topology,
}

__all__ = [
    'LTopology', 'LBeamTopology', 'LBeam2Topology', 'LOuterTopology',
    'TTopology', 'TBeamTopology', 'TOuterTopology',
    'XTopology', 'XBeamTopology', 'XOuterTopology',
    'YTopology',
    'KTopology',
    'XDashTopology',
    'Star3Topology',
    'TOPOLOGY_BY_NAME'
]
