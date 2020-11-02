from scipy.spatial import ConvexHull
import numpy as np

from .utils import sqlen


def find_longest_flat(points, threshold, fixed_ends=False):
    if fixed_ends:
        return _find_longest_flat_fixed_ends(points, threshold)
    else:
        return _find_longest_flat_free_ends(points, threshold)


def _find_longest_flat_fixed_ends(points, threshold):
    '''Returns `line, N` such that each of the first N points are no further than threshold from the line
       and `line = points[0], points[N-1]`'''
    assert len(points) > 1

    for N in range(len(points), 1, -1):
        line = points[[0,N-1]]
        if points_are_in_line(points[:N], line, threshold):
            return line, N

    assert False, 'Something\'s wrong' # the function should return at least the first two points


def _find_longest_flat_free_ends(points, threshold):
    '''Returns `line, N` such that each of the first N points are no further than threshold from the line.'''
    assert len(points) > 1
    
    N = 2
    line = points[:N].copy()
    
    while N < len(points):
        p = points[N:N+1]
        try:
            hull.add_points(p)
        except NameError:
            if points_are_on_line(points[:N+1], line, threshold):
                N += 1
                line = get_endpoints(np.append(line, p, axis=0))
                continue
            else:
                hull = ConvexHull(points[:N+1], incremental=True)
        vertices = points[hull.vertices]
        
        edge_directions = np.roll(vertices, -1, axis=0) - vertices
        edge_directions = edge_directions / np.linalg.norm(edge_directions, axis=-1, keepdims=True)
        edge_normals = np.roll(edge_directions, 1, axis=-1);  edge_normals[..., 0] *= -1
        
        widths = np.einsum('ijk,ik->ij', vertices[None] - vertices[:, None], edge_normals).max(axis=-1)
        
        best_width_id = widths.argmin()
        best_width = widths[best_width_id]
        if best_width > threshold * 2:
            break
        
        N += 1
        
        p0_best = vertices[best_width_id]
        l_best = edge_directions[best_width_id]
        projections = (vertices - p0_best).dot(l_best)
        line[0] = p0_best + projections.min() * l_best
        line[1] = p0_best + projections.max() * l_best
        line = line + edge_normals[best_width_id] * best_width / 2
        
    return line, N


def get_endpoints(collinear_points):
    if len(collinear_points) == 1: # end points are the same as the only one point
        return collinear_points[:1].repeat(2, axis=0)
    
    points = collinear_points - collinear_points[0]
    
    # find line direction
    for i in range(1, len(points)):
        l = points[i]
        if np.any(l != 0):
            break
            
    if np.all(l == 0): # all points coincide
        return collinear_points[:2]
    
    disps = points.dot(l)
    return collinear_points[[disps.argmin(), disps.argmax()]]


def points_are_in_line(ps, line, threshold):
    if np.all(line[0] == line[1]): # degenerate line
        if len(ps) == 1:
            return sqlen(line[0] - ps[0]) <= threshold ** 2
        else: # make a new line from the degenerate one and the first point and check that the other points lie in that
            return points_are_in_line(ps[1:], (line[0], ps[0]), threshold)
    l = line[1] - line[0]
    length = np.linalg.norm(l)
    l /= length
    n = np.roll(l, 1, axis=-1);  n[0] *= -1
    rs = ps - line[0]
    if np.any(np.abs(rs.dot(n)) > threshold):
        return False
    projections = rs.dot(l)
    return np.all(projections >= 0) & np.all(projections <= length)


def points_are_on_line(ps, line, threshold):
    if np.all(line[0] == line[1]):
        if len(ps) == 1: # any single point lies on a degenerate line
            return True
        else: # make a new line from the degenerate one and the first point and check that the other points lie on that
            return points_are_on_line(ps[1:], (line[0], ps[0]), threshold)
    n = np.roll(line[1] - line[0], 1, axis=-1);  n[0] *= -1
    n /= np.linalg.norm(n)
    rs = ps - line[0]
    return np.all(np.abs(rs.dot(n)) <= threshold)
