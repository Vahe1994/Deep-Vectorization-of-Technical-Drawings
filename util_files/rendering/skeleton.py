import numpy as np

from util_files.graphics.graphics_primitives import PrimitiveType


def make_skeleton_vahe(primitive_sets, line_width, node_size, control_line_width, control_node_size):
    edges = {prim_type: [_get_edges_vahe[prim_type](prim, line_width) for prim in prims] for prim_type, prims in
             primitive_sets.items()}
    nodes = {PrimitiveType.PT_POINT: [node for prim_type, prims in primitive_sets.items() for prim in prims for node in
                                      _get_nodes_vahe[prim_type](prim, node_size)]}

    control_points = []
    control_lines = []
    for prim_type, prims in primitive_sets.items():
        for prim in prims:
            controls = _get_controls_vahe[prim_type](prim, control_line_width, control_node_size)
            control_points += controls[PrimitiveType.PT_POINT]
            control_lines += controls[PrimitiveType.PT_LINE]
    controls = {PrimitiveType.PT_LINE: control_lines, PrimitiveType.PT_POINT: control_points}
    return edges, nodes, controls


def get_line_edge_vahe(line, line_width):
    edge = line.copy()
    edge[4] = line_width
    return edge


def get_bezier_edge_vahe(bezier, line_width):
    edge = bezier.copy()
    edge[8] = line_width
    return edge


def get_qbezier_edge_vahe(bezier, line_width):
    edge = bezier.copy()
    edge[6] = line_width
    return edge


def get_line_nodes_vahe(line, node_size):
    start = np.array([*line[:2], node_size])
    end = np.array([*line[2:4], node_size])
    return start, end


def get_bezier_nodes_vahe(bezier, node_size):
    p0 = np.array([*bezier[:2], node_size])
    p3 = np.array([*bezier[6:8], node_size])
    return p0, p3


def get_qbezier_nodes_vahe(bezier, node_size):
    p0 = np.array([*bezier[:2], node_size])
    p2 = np.array([*bezier[4:6], node_size])
    return p0, p2


def get_line_controls_vahe(line, line_width, node_size):
    return {PrimitiveType.PT_POINT: [], PrimitiveType.PT_LINE: []}


def get_bezier_controls_vahe(bezier, line_width, node_size):
    p0 = np.array([*bezier[:2], node_size])
    p1 = np.array([*bezier[2:4], node_size])
    p2 = np.array([*bezier[4:6], node_size])
    p3 = np.array([*bezier[6:8], node_size])
    return {PrimitiveType.PT_POINT: [p0, p1, p2, p3],
            PrimitiveType.PT_LINE: [
                np.array([*bezier[:4], line_width]),
                np.array([*bezier[2:6], line_width]),
                np.array([*bezier[4:8], line_width])]}


def get_qbezier_controls_vahe(bezier, line_width, node_size):
    p0 = np.array([*bezier[:2], node_size])
    p1 = np.array([*bezier[2:4], node_size])
    p2 = np.array([*bezier[4:6], node_size])
    return {PrimitiveType.PT_POINT: [p0, p1, p2],
            PrimitiveType.PT_LINE: [
                np.array([*bezier[:4], line_width]),
                np.array([*bezier[2:6], line_width])]}


_get_edges_vahe = {PrimitiveType.PT_LINE: get_line_edge_vahe, PrimitiveType.PT_BEZIER: get_bezier_edge_vahe,
                   PrimitiveType.PT_QBEZIER: get_qbezier_edge_vahe}
_get_nodes_vahe = {PrimitiveType.PT_LINE: get_line_nodes_vahe, PrimitiveType.PT_BEZIER: get_bezier_nodes_vahe,
                   PrimitiveType.PT_QBEZIER: get_qbezier_nodes_vahe}
_get_controls_vahe = {PrimitiveType.PT_LINE: get_line_controls_vahe, PrimitiveType.PT_BEZIER: get_bezier_controls_vahe,
                      PrimitiveType.PT_QBEZIER: get_qbezier_controls_vahe}