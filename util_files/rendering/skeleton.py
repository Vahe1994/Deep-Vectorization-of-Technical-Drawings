import numpy as np

from util_files.data.graphics_primitives import PT_CBEZIER, PT_LINE, PT_POINT, PT_QBEZIER, PT_QBEZIER_B


def make_skeleton_vahe(primitive_sets, line_width, node_size, control_line_width, control_node_size):
    edges = {prim_type: [_get_edges_vahe[prim_type](prim, line_width) for prim in prims] for prim_type, prims in
             primitive_sets.items()}
    nodes = {PT_POINT: [node for prim_type, prims in primitive_sets.items() for prim in prims for node in
                                      _get_nodes_vahe[prim_type](prim, node_size)]}

    control_points = []
    control_lines = []
    for prim_type, prims in primitive_sets.items():
        for prim in prims:
            controls = _get_controls_vahe[prim_type](prim, control_line_width, control_node_size)
            control_points += controls[PT_POINT]
            control_lines += controls[PT_LINE]
    controls = {PT_LINE: control_lines, PT_POINT: control_points}
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


def get_qbezier_b_nodes_vahe(bezier, node_size):
    b = np.array([*bezier[7:9], node_size])
    nodes = *get_qbezier_nodes_vahe(bezier, node_size), b
    return nodes


def get_line_controls_vahe(line, line_width, node_size):
    return {PT_POINT: [], PT_LINE: []}


def get_bezier_controls_vahe(bezier, line_width, node_size):
    p0 = np.array([*bezier[:2], node_size])
    p1 = np.array([*bezier[2:4], node_size])
    p2 = np.array([*bezier[4:6], node_size])
    p3 = np.array([*bezier[6:8], node_size])
    return {PT_POINT: [p0, p1, p2, p3],
            PT_LINE: [
                np.array([*bezier[:4], line_width]),
                np.array([*bezier[2:6], line_width]),
                np.array([*bezier[4:8], line_width])]}


def get_qbezier_controls_vahe(bezier, line_width, node_size):
    p0 = np.array([*bezier[:2], node_size])
    p1 = np.array([*bezier[2:4], node_size])
    p2 = np.array([*bezier[4:6], node_size])
    return {PT_POINT: [p0, p1, p2],
            PT_LINE: [np.array([*bezier[:4], line_width]), np.array([*bezier[2:6], line_width])]}


def get_qbezier_b_controls_vahe(bezier, line_width, node_size):
    bezier_controls = get_qbezier_controls_vahe(bezier, line_width, node_size)

    p0 = bezier[:2]
    b = np.array([*bezier[7:9], node_size])
    p2 = bezier[4:6]
    return {PT_POINT: bezier_controls[PT_POINT] + [b],
            PT_LINE: (bezier_controls[PT_LINE] +
                      [np.array([*p0, *b[:2], line_width]), np.array([*b[:2], *p2, line_width])])}


_get_edges_vahe = {PT_LINE: get_line_edge_vahe, PT_CBEZIER: get_bezier_edge_vahe,
                   PT_QBEZIER: get_qbezier_edge_vahe, PT_QBEZIER_B: get_qbezier_edge_vahe}
_get_nodes_vahe = {PT_LINE: get_line_nodes_vahe, PT_CBEZIER: get_bezier_nodes_vahe,
                   PT_QBEZIER: get_qbezier_nodes_vahe, PT_QBEZIER_B: get_qbezier_b_nodes_vahe}
_get_controls_vahe = {PT_LINE: get_line_controls_vahe, PT_CBEZIER: get_bezier_controls_vahe,
                      PT_QBEZIER: get_qbezier_controls_vahe, PT_QBEZIER_B: get_qbezier_b_controls_vahe}
