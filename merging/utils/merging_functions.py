#!/usr/bin/env python3
from __future__ import division

import os, sys

sys.path.append("..")
sys.path.append(os.path.join(os.getcwd(), '..'))

from tqdm import tqdm

import numpy as np
import os
import math
from PIL import Image


from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

import util_files.data.graphics_primitives as graphics_primitives
from util_files.rendering.cairo import render_with_skeleton

from util_files.geometric import liang_barsky_screen
from util_files.rendering.cairo  import render,render_with_skeleton
from util_files.data.graphics_primitives import PT_LINE,  PT_CBEZIER, PT_QBEZIER




def ordered(line):
    min_x = min(line[0], line[2])
    min_y = min(line[1], line[3])
    max_x = max(line[0], line[2])
    max_y = max(line[1], line[3])

    return np.array([min_x, min_y, max_x, max_y])

def clip_to_box(y_pred, box_size=(64, 64)):
    width, height = box_size
    bbox = (0, 0, width, height)
    point1, point2 = y_pred[:2], y_pred[2:4]
    try:
        clipped_point1, clipped_point2, is_drawn = \
            liang_barsky_screen(point1, point2, bbox)
    except:
        return np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    if (clipped_point1 and clipped_point2):
        return np.asarray([clipped_point1, clipped_point2, y_pred[4:]]).ravel()
    else:
        return np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

def assemble_vector_patches_curves(patches_vector, patches_offsets):
    primitives = []
    i = 0
    for patch_vector, patch_offset in zip(patches_vector, patches_offsets):
        i += 1

        patch_vector[:, [0, 2, 4]] += patch_offset[1]


        patch_vector[:, [1, 3, 5]] += patch_offset[0]
        primitives.append(patch_vector)
    return np.array(primitives)

def assemble_vector_patches_lines(patches_vector, patches_offsets):
    primitives = []
    i = 0
    for patch_vector, patch_offset in zip(patches_vector, patches_offsets):
        i += 1

        patch_vector[:, [0, 2]] += patch_offset[1]

        patch_vector[:, [1, 3]] += patch_offset[0]

        primitives.append(patch_vector)
    return np.array(primitives)

def tensor_vector_graph_numpy(y_pred_render, patches_offsets, options):

    nump = np.array(list(map(clip_to_box, y_pred_render.reshape(-1, 6).cpu().detach().numpy())))

    nump = assemble_vector_patches_lines(np.array((nump.reshape(-1, options.model_output_count, 6))),
                                         np.array(patches_offsets))


    nump = nump.reshape(-1, 6)


    nump = nump[~np.isnan(nump).any(axis=1)]

    nump = nump[(nump[:, -2] > 0.3)]
    nump = nump[(nump[:, -1] > 0.5)]

    nump = nump[((nump[:, 0] - nump[:, 2]) ** 2 + (nump[:, 1] - nump[:, 3]) ** 2 >= 3)]

    return nump


def merge_close_lines(lines, threshold=0.5):
    #     min_x = min(lines[:,0].min(), lines[:,2].min())
    #     max_x = max(lines[:,0].max(), lines[:,2].max())
    #     min_y = min(lines[:,1].min(), lines[:,3].min())
    #     max_y = max(lines[:,1].max(), lines[:,3].max())

    lr = LinearRegression()
    ransac = linear_model.RANSACRegressor()

    dt = np.hstack((lines[:, 0], lines[:, 2]))
    y_t = np.hstack((lines[:, 1], lines[:, 3]))
    #     if lines.shape[0] == 1:
    #         return np.array(lines)
    if lines.shape[0] == 2:
        if (lines[0, 0] <= lines[0, 2] and lines[0, 1] <= lines[0, 3]) or (lines[0, 2] <= lines[0, 0] and lines[0, 3] <= lines[0, 1]):
            return np.array([np.min(dt), np.min(y_t), np.max(dt), np.max(y_t)])
        else:
            return np.array([np.min(dt), np.max(y_t), np.max(dt), np.min(y_t)])
    try:
        ransac.fit(dt.reshape(-1, 1), y_t)
        inlier_mask = ransac.inlier_mask_

        lr.fit(dt[inlier_mask].reshape(-1, 1), y_t[inlier_mask])
    except:
        lr.fit(dt.reshape(-1, 1), y_t)

    if abs(lr.coef_) >= threshold:  # vertical line
        lr = LinearRegression()

        dt = np.hstack((lines[:, 1], lines[:, 3]))
        y_t = np.hstack((lines[:, 0], lines[:, 2]))

        lr.fit(dt.reshape(-1, 1), y_t)

        dt = np.sort(dt)

        y_pred = lr.predict(dt.reshape(-1, 1))
        return np.array([y_pred[0], dt[0], y_pred[-1], dt[-1]])

    lr = LinearRegression()
    lr.fit(dt.reshape(-1, 1), y_t)
    dt = np.sort(dt)
    y_pred = lr.predict(dt.reshape(-1, 1))

    return np.array([dt[0], y_pred[0], dt[-1], y_pred[-1]])

def point_to_line_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)
    return np.abs(dy * px - dx * py + x2 * y1 - x1 * y2) / np.sqrt(dy ** 2 + dx ** 2)

def point_segment_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)

    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y

    return math.hypot(dx, dy)

def dist(line0, line1):
    if (point_to_line_distance(line0[:2], line1[:4]) >= 2 or point_to_line_distance(line0[2:4], line1[
                                                                                                :4]) >= 2 or point_to_line_distance(
        line1[:2], line0[:4]) >= 2 or point_to_line_distance(line1[2:4], line0[:4]) >= 2):
        return 9999

    return min(distance.euclidean(line0[:2], line1[:2]), distance.euclidean(line0[2:4], line1[:2]),
               distance.euclidean(line0[:2], line1[2:4]), distance.euclidean(line0[2:4], line1[2:4]),
               point_segment_distance(line0[:2], line1[:4]), point_segment_distance(line0[2:4], line1[:4]),
               point_segment_distance(line1[:2], line0[:4]), point_segment_distance(line1[2:4], line0[:4]),
               )


def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(set(graph[vertex]) - visited)
    return visited


def line_legth(line):
    return np.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2)

def intersect(line0, line1):
    # Solve the system [p1-p0, q1-q0]*[t1, t2]^T = q0 - p0
    # where line0 = (p0, p1) and line1 = (q0, q1)

    denom = ((line0[2] - line0[0]) * (line1[1] - line1[3]) -
             (line0[3] - line0[1]) * (line1[0] - line1[2]))

    if np.isclose(denom, 0):
        return []
    t1 = (line1[0] * (line0[1] - line1[3]) -
          line1[2] * (line0[1] - line1[1]) -
          line0[0] * (line1[1] - line1[3])) / denom
    t2 = -(line0[2] * (line0[1] - line1[1]) -
           line0[0] * (line0[3] - line1[1]) -
           line1[0] * (line0[1] - line0[3])) / denom
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return [(t1, t2)]
    return []

def angle_radians(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product / (len1 * len2))


def normalize(x):
    return x / np.linalg.norm(x)

def compute_angle(line0, line1):
    pt1 = normalize([line0[2] - line0[0], line0[3] - line0[1]])
    pt2 = normalize([line1[2] - line1[0], line1[3] - line1[1]])

    try:
        angle = math.degrees(angle_radians(pt1, pt2))
    except:
        angle = 0
    if (angle >= 90 and angle <= 270):
        angle = np.abs(180 - angle)
    elif (angle > 270 and angle <= 360):
        angle = 360 - angle
    return angle

def merge_close(lines, idx, widths, tol=1e-3, max_dist=5, max_angle=15, window_width=100):
    window = [-window_width, -window_width, window_width, window_width]
    n = len(lines)
    close = [[] for _ in range(n)]

    for i in tqdm(range(n)):
        #         if (line_legth(lines[i, :4]) < 3):
        #             continue
        for j in idx.intersection(lines[i, :4] + window):
            if i == j:
                continue

            if (line_legth(lines[j, :4]) < 3):
                continue
            if ((dist(lines[i], lines[j]) < max_dist or  # lines are close
                 intersect(lines[i], lines[j])) and  # lines intersect
                    compute_angle(lines[i], lines[j]) < max_angle):  # the angle is less than threshold
                close[i].append(j)
    result = []
    merged = set()

    for i in range(n):

        if (line_legth(lines[i, :4]) < 3):
            continue

        elif (close[i]) and (i not in merged):
            path = list(dfs(close, i))
            width = widths[path].mean(keepdims=True)
            new_line = merge_close_lines(lines[path])
            result.append(np.concatenate((new_line, width, np.ones(width.shape))))
            merged.update(path)
        elif i not in merged:
            result.append((lines[i]))
    return result

def draw_with_skeleton(lines, drawing_scale=1, skeleton_line_width=0, skeleton_node_size=0, max_x=64, max_y=64):
    scaled_primitives = lines.copy()
    scaled_primitives[..., :-1] *= drawing_scale
    return render_with_skeleton(
        {graphics_primitives.PrimitiveType.PT_LINE: scaled_primitives},
        (max_x * drawing_scale, max_y * drawing_scale), data_representation='vahe',
        line_width=skeleton_line_width, node_size=skeleton_node_size)


def maximiz_final_iou(nump, input_rgb):
    aa = (draw_with_skeleton(nump, max_x=input_rgb.shape[1], max_y=input_rgb.shape[0]) / 255.)[..., 0]
    k = input_rgb[..., 0] / 255.
    mse_ref = ((np.array(aa) - np.array(k)) ** 2).mean()
    lines = list(nump)
    it = 0
    l = 0
    while it < len(lines):
        poped_line = lines.pop(it)
        tmp_scr = (draw_with_skeleton(np.array(lines), max_x=input_rgb.shape[1], max_y=input_rgb.shape[0]) / 255.)[
            ..., 0]
        tmp_scr = ((np.array(tmp_scr) - np.array(k)) ** 2).mean()
        if (tmp_scr > mse_ref):
            lines.insert(it, poped_line)
            it += 1
        else:
            mse_ref = tmp_scr
        l += 1
    return lines
def two_point_dist(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt((np.square(p1-p2)).sum())

def line(p1):
    A = (p1[1] - p1[3])
    B = (p1[2] - p1[0])
    C = (p1[0]*p1[3] - p1[2]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return []

def lines_matching(lines, frac = 0.01):
    """
    lines: lines
    frac: fraction of line
    """
    line_inter = [[(np.inf,np.inf),(np.inf,np.inf)] for i in range(len(lines))]
    for idx_1 in range(len(lines)):
        for idx_2 in range(idx_1,len(lines)):
            intr = intersection(line(lines[idx_1]), line(lines[idx_2]))
            if intr:
                if(two_point_dist(intr,lines[idx_1][:2]) < two_point_dist(line_inter[idx_1][0],lines[idx_1][:2])):
                    line_inter[idx_1][0] = intr
                if(two_point_dist(intr,lines[idx_1][2:4]) < two_point_dist( line_inter[idx_1][1],lines[idx_1][2:4])):
                    line_inter[idx_1][1] = intr
                if(two_point_dist(intr,lines[idx_2][:2]) < two_point_dist( line_inter[idx_2][0],lines[idx_2][:2])):
                    line_inter[idx_2][0] = intr
                if(two_point_dist(intr,lines[idx_2][2:4]) < two_point_dist( line_inter[idx_2][1],lines[idx_2][2:4])):
                    line_inter[idx_2][1] = intr
    for idx in range(len(lines)):
        if two_point_dist(lines[idx][:2],lines[idx][2:4])* frac >=  two_point_dist(line_inter[idx][0],lines[idx][:2]):
            lines[idx][:2] = list(line_inter[idx][0])

        if two_point_dist(lines[idx][:2],lines[idx][2:4])* frac >=  two_point_dist(line_inter[idx][1],lines[idx][2:4]):
            lines[idx][2:4] = list(line_inter[idx][1])

    return lines

def save_svg(result_vector, size, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if len(size)==2:
        size = (1,size[0],size[1])
    a ={PT_LINE:np.concatenate((result_vector[...,:-1], result_vector[...,-1][...,None]),axis=1)}
    rendered_image = render(a,(size[2],size[1]), data_representation='vahe',linecaps='round')
    Image.fromarray(rendered_image).save(output_dir + name)
    return rendered_image

