def bbox(segments):
    bbs = [seg.bbox() for seg in segments]
    xmins, xmaxs, ymins, ymaxs = list(zip(*bbs))
    xmin = min(xmins)
    xmax = max(xmaxs)
    ymin = min(ymins)
    ymax = max(ymaxs)
    return xmin, xmax, ymin, ymax


def mirror_point_x(p, x):
    return complex(x * 2 - p.real, p.imag)
