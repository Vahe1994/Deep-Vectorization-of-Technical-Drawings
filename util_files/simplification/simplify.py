import numpy as np
import svgpathtools

from . import curve


def bezier(bezier, same_line_threshold, max_segments_n):
    poly = bezier.poly()
    poly_x = np.poly1d(poly.coefficients.real)
    poly_y = np.poly1d(poly.coefficients.imag)
    points_n = curve.bezier_steps(bezier, same_line_threshold)
    all_ts = np.linspace(0, 1, points_n)
    
    segments = []
    point_to_start_next_segment = 0
    for segment_i in range(max_segments_n):
        ts = all_ts[point_to_start_next_segment:]
        
        line, segmented_points_n = curve.find_longest_flat(poly_x, poly_y, ts, same_line_threshold)
        if line is None:
            return bezier,
        segments.append(_line_from_points(line))
        
        total_segmented_points_n = point_to_start_next_segment + segmented_points_n
        if total_segmented_points_n == points_n:
            break
        point_to_start_next_segment = total_segmented_points_n - 1
    
    if total_segmented_points_n < points_n: # failed to segment whole curve into max_segments_n segments
        return bezier,
    
    return segments


def _line_from_points(points):
    return svgpathtools.Line(*(complex(*p) for p in points))
