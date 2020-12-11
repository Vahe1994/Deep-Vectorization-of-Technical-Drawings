import os

import numpy as np
import svgwrite
# https://anaconda.org/pythonocc/pythonocc-core
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.Bnd import Bnd_Box2d
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C1
from OCC.Core.GeomConvert import GeomConvert_BSplineCurveToBezierCurve
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pnt2d, gp_Ax2
from utils.topology_utils import get_sorted_hlr_edges, discretize_edge

CURVE_TYPES = {1: 'Circle', 2: 'Ellipse', 6: 'BSpline'}


def read_step_file(filename, return_as_shapes=False, verbosity=True):
    """ read the STEP file and returns a compound
    filename: the file path
    return_as_shapes: optional, False by default. If True returns a list of shapes,
                      else returns a single compound
    verbosity: optional, False by default.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError("%s not found." % filename)

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == IFSelect_RetDone:  # check status
        if verbosity:
            failsonly = False
            step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
            step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)

        shapes_to_return = []

        for root_num in range(1, step_reader.NbRootsForTransfer() + 1):
            transfer_result = step_reader.TransferRoot(root_num)

            if not transfer_result:
                raise AssertionError("Transfer failed.")

            shape = step_reader.Shape(root_num)

            if shape.IsNull():
                raise AssertionError("Shape is null.")

            shapes_to_return.append(shape)

        if return_as_shapes:
            return shapes_to_return

        builder = BRep_Builder()
        compound_shape = TopoDS_Compound()
        builder.MakeCompound(compound_shape)

        for shape in shapes_to_return:
            builder.Add(compound_shape, shape)

        return compound_shape

    else:
        raise AssertionError("Error: can't read file.")


# functions for OCC C++ API array creation
# taken from pythonocc-core tests
def _Tcol_dim_1(li, _type):
    """function factory for 1-dimensional TCol* types"""
    pts = _type(0, len(li) - 1)
    for n, i in enumerate(li):
        pts.SetValue(n, i)
    return pts


def point_list_to_TColgp_Array1OfPnt(li):
    return _Tcol_dim_1(li, TColgp_Array1OfPnt)


###


def add_to_bounding_box(points_2d):
    box2d = Bnd_Box2d()
    for p in points_2d:
        box2d.Add(gp_Pnt2d(*p))
    return box2d


def line_to_svg(curve):
    line = curve.Line()
    location = np.array(line.Location().Coord()[:2])
    direction = np.array(line.Direction().Coord()[:2])
    start = location + curve.FirstParameter() * direction
    end = location + curve.LastParameter() * direction
    return svgwrite.shapes.Line(start, end, fill="none"), add_to_bounding_box((start, end))


def approx_points_by_piecewise_bezier(points_3d, degree, tol):
    if degree not in (2, 3):
        raise RuntimeError("SVG files only support Bezier curves of degree 2 or 3")

    points_3d_occ = [gp_Pnt(*p) for p in points_3d]
    approx_spline = GeomAPI_PointsToBSpline(
        point_list_to_TColgp_Array1OfPnt(points_3d_occ), degree, degree, GeomAbs_C1, tol
    )
    # TO DO:
    # why the following string doesn't yield a spline of degree < 3
    # approx_spline = GeomConvert_ApproxCurve(spline, tol, order, max_segments, max_degree)

    if not approx_spline.IsDone():
        raise RuntimeError("Could not approximate points within a given tolerance")

    return GeomConvert_BSplineCurveToBezierCurve(approx_spline.Curve())


def piecewise_bezier_to_svg(points_2d, bezier_curves, degree):
    path_elements = []
    for i in range(1, bezier_curves.NbArcs() + 1):
        if bezier_curves.Arc(i).Degree() != degree:
            raise RuntimeError(f"Approximated degree of Bezier curves is not {degree}")

        curve = bezier_curves.Arc(i)

        if degree == 2:
            start, control, end = [p.Coord()[:2] for p in list(curve.Poles())]
            path_elements.append(f"M {start[0]},{start[1]} Q {control[0]},{control[1]} {end[0]},{end[1]}".split())

        elif degree == 3:
            start, first_control, second_control, end = [p.Coord()[:2] for p in list(curve.Poles())]
            path_elements.append(
                f"M {start[0]},{start[1]} C {first_control[0]},{first_control[1]} \
                 {second_control[0]},{second_control[1]} {end[0]},{end[1]}".split()
            )

    path_elements.append(f"M{end[0]} {end[1]}".split())

    return svgwrite.path.Path(d=path_elements, fill="none"), add_to_bounding_box(points_2d)


def polyline_to_svg(points_2d):
    return svgwrite.shapes.Polyline(points_2d, fill="none"), add_to_bounding_box(points_2d)


def edge_to_svg(topods_edge, bezier_tol=0.01, bezier_degree=2):
    """
    Returns a svgwrite.Path for the edge, and the 2d bounding box
    :param topods_edge:
    :param bezier_tol: tol of piecewise bezier approximation
    :param bezier_degree: degree of bezier curves
    :return:
    """
    curve = BRepAdaptor_Curve(topods_edge)
    if curve.GetType() == 0:  # line
        return line_to_svg(curve)

    else:
        # discretize the curve to process occlusions and transform to Bezier
        points_3d = discretize_edge(topods_edge, deflection=0.01)
        points_2d = [p[:2] for p in points_3d]

        try:
            return piecewise_bezier_to_svg(
                points_2d, approx_points_by_piecewise_bezier(points_3d, bezier_degree, bezier_tol), bezier_degree
            )

        except RuntimeError:
            print(f"Converting {CURVE_TYPES[curve.GetType()]} to polyline")
            return polyline_to_svg(points_2d)


def export_shape_to_svg(shape, filename=None,
                        width=800, height=600, margin_left=30,
                        margin_top=30, export_hidden_edges=True,
                        location=(0, 0, 0), direction=(1, 1, 1),
                        bezier_degree=2, bezier_tol=0.01,
                        color="black", line_width=0.5):
    """
    export a single shape to an svg file and/or string.
    shape: the TopoDS_Shape to export
    filename (optional): if provided, save to an svg file
    width, height (optional): integers, specify the canva size in pixels
    margin_left, margin_top (optional): integers, in pixel
    export_hidden_edges (optional): whether or not draw hidden edges using a dashed line
    location (optional): a gp_Pnt, the lookat
    direction (optional): to set up the projector direction
    bezier_degree (optional): degree of Bezier curves
    bezier_tol (optional): tolerance for Bezier curves approximation
    color (optional), "default to "black".
    line_width (optional, default to 1): an integer
    """

    if shape.IsNull():
        raise AssertionError("shape is Null")

    # find all edges
    location = gp_Pnt(*location)
    direction = gp_Dir(*direction)
    camera_ax = gp_Ax2(location, direction)
    visible_edges, hidden_edges = get_sorted_hlr_edges(shape, position=location, direction=direction,
                                                       export_hidden_edges=export_hidden_edges)

    # compute paths for all edges
    # we compute a global 2d bounding box as well, to be able to compute
    # the scale factor and translation vector to apply to all 2d edges so that
    # they fit the svg canva
    global_2d_bounding_box = Bnd_Box2d()

    paths = []
    for visible_edge in visible_edges:
        visible_svg_line, visible_edge_box2d = edge_to_svg(
            visible_edge, bezier_degree=bezier_degree, bezier_tol=bezier_tol)

        if visible_svg_line:
            paths.append(visible_svg_line)
            global_2d_bounding_box.Add(visible_edge_box2d)

    if export_hidden_edges:
        for hidden_edge in hidden_edges:
            hidden_svg_line, hidden_edge_box2d = edge_to_svg(
                hidden_edge, bezier_degree=bezier_degree, bezier_tol=bezier_tol)

            if hidden_svg_line:
                # hidden lines are dashed style
                hidden_svg_line.dasharray([5, 5])
                paths.append(hidden_svg_line)
                global_2d_bounding_box.Add(hidden_edge_box2d)

    # translate and scale paths
    # first compute shape translation and scale according to size/margins
    x_min, y_min, x_max, y_max = global_2d_bounding_box.Get()
    bb2d_width = x_max - x_min
    bb2d_height = y_max - y_min

    # build the svg drawing
    drawing = svgwrite.Drawing(filename, (width, height), debug=True)
    # adjust the view box so that the lines fit then svg canvas
    drawing.viewbox(x_min - margin_left, y_min - margin_top,
                    bb2d_width + 2 * margin_left, bb2d_height + 2 * margin_top)

    for path in paths:
        path.stroke(color, width=line_width, linecap="round")
        drawing.add(path)

    # export to string or file according to the user choice
    if filename is not None:
        drawing.save()
        if not os.path.isfile(filename):
            raise AssertionError("svg export failed")
        print(f"Shape successfully exported to {filename}")
        return True
    return drawing.tostring()
