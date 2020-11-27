import numpy as np

# rendering parameters
pixel_center_coodinates_are_integer = False
refinement_linecaps = 'butt'
refinement_linejoin = 'miter'

# thresholds
collinearity_beta = 1 / ((np.abs(np.cos(15 * np.pi / 180)) - 1) ** 2)
coordinates_constrain_padding = 1
dwarfness_ratio = 1
elementary_halfwidth = 1 / 2
empty_pixel_tolerance = 0  # max shading value for which the pixel is considered background in neighbourhood finding procedure
foreground_pixel_min_shading = .5  # min shading of pixel that is considered foreground in reinitialization procedure
min_linear_size = 2**-8  # minimal value of length or width of a primitive
min_visible_width = 1 / 8  # primitives with lower width are considered collapsed and reinitialized
neighbourhood_padding = 2  # padding of primitive neighbourhood
qbezier_y_neighbourhood_padding = .1
neighbourhood_pos_weight = 2 * (1. / .5)
qbezier_max_dt_end = 2e-1
qbezier_min_fold_halfangle_radians = 15 * np.pi / 180  # min half angle between B->P1 and B->P3 for quad beziers
visibility_width_threshold = 1 / 4  # primitives with width lower than this do not make it to the final output

reinit_initial_length = 1
reinit_initial_width = 1

# periods
reinit_period = 20

division_epsilon = 1e-12  # used as `a / (b + division_epsilon)`
