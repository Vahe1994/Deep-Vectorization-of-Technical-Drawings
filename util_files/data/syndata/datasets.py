from collections import OrderedDict, defaultdict
from enum import Enum, auto

import numpy as np
from numpy.random import uniform, normal
# from PIL import Image
from util_files.rendering.cairo import render
from util_files.data.line_drawings_dataset import LineDrawingsDataset
from util_files.data.graphics_primitives import Line, PrimitiveType, BezierCurve, Arc
# from vectran.data.transforms.degradation_models import DegradationGenerator, all_degradations
# from vectran.util.color_utils import rgb_to_gray, img_8bit_to_float, gray_float_to_8bit
from util_files.data.transforms.degradation_models import DegradationGenerator
from util_files.geometric import rotation_matrix_2d

from .patch_topology import TOPOLOGY_BY_NAME
from .utils import choose_with_proba


class JunctionType(Enum):
    NO_JUNCTION = auto()    # parallel lines; lines not intersecting within image;
                            # lines too short to intersect e.g. -|
    L_JUNCTION = auto()     # lines intersecting at edges e.g. L, Г
    T_JUNCTION = auto()     # one line intersecting at non-edge, second at edge e.g. T
    X_JUNCTION = auto()     # two lines intersecting at non-edges, e.g. X, +


class SyntheticDataset(LineDrawingsDataset):
    def __init__(self, *, patch_size, border=4, line_count=1, arc_count=1, bezier_count=0, size=10000,
                 **kwargs):
        super().__init__(patch_size=patch_size, **kwargs)
        self.size = size
        self.border = border
        self.line_count = line_count
        self.arc_count = arc_count
        self.bezier_count = bezier_count

    def _random_line(self):
        x, y = [], []
        for it in range(2):
            x.append(np.random.randint(self.border, self.patch_size[0] - self.border))
            y.append(np.random.randint(self.border, self.patch_size[1] - self.border))
        width = np.random.randint(1, 4)
        if (x[0], y[0]) < (x[1], y[1]):
            return x[0], y[0], x[1], y[1], width
        else:
            return x[1], y[1], x[0], y[0], width

    def _random_bezier(self):
        x, y = [], []
        for it in range(4):
            x.append(np.random.randint(self.border, self.patch_size[0] - self.border))
            y.append(np.random.randint(self.border, self.patch_size[1] - self.border))
        width = np.random.randint(1, 4)
        if (x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3]) < (x[3], y[3], x[2], y[2], x[1], y[1], x[0], y[0]):
            return x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3], width
        else:
            return x[3], y[3], x[2], y[2], x[1], y[1], x[0], y[0], width

    def _random_circle(self):
        x, y = [], []
        for it in range(3):
            if it == 1:
                x.append(np.random.randint(2, self.patch_size[0] - x[0] - self.border))
                y.append(np.random.randint(0, self.patch_size[1] - y[0] - self.border))
            else:
                x.append(np.random.randint(self.border, self.patch_size[0] - self.border - 2))
                y.append(np.random.randint(self.border, self.patch_size[1] - self.border - 2))
        width = np.random.randint(1, 4)
        return x[0], y[0], min(x[1], x[0], y[0]), 0., 2 * np.pi, width

    def _get_vector_item(self, idx):
        vector = defaultdict(list)
        for it in range(self.line_count):
            line = self._random_line()
            vector[PrimitiveType.PT_LINE].append(line)
        for it in range(self.bezier_count):
            bezier = self._random_bezier()
            vector[PrimitiveType.PT_BEZIER].append(bezier)
        for it in range(self.arc_count):
            arc = self._random_circle()
            vector[PrimitiveType.PT_ARC].append(arc)
        return vector

    def __len__(self):
        return self.size



class SyntheticStructuredDataset(LineDrawingsDataset):
    def __init__(self, *, patch_size=(64, 64), size=10000, border=8, min_directions=1, max_directions=4,
                 directions_probas=None, offset_directions_probas=None, directions_min_angle=0, directions_max_angle=np.pi,
                 directions_angle_step=np.pi / 4, junction_directions_probas=None, min_primitives=1, max_primitives=4,
                 total_primitives=float('+inf'), min_primitives_gap=1, max_primitives_gap=10, min_stroke_width=1,
                 max_stroke_width=10, min_stroke_length=10, max_stroke_length=100, primitives_endpoint_noise_sigma=5.,
                 primitives_direction_noise_sigma=np.pi / 180., strokes_probas=None, **kwargs):
        super().__init__(patch_size=patch_size, min_primitives=min_primitives, max_primitives=max_primitives, **kwargs)
        self.border = border
        self.size = size

        self.min_directions = min_directions
        self.max_directions = max_directions
        self.total_primitives = total_primitives
        self.min_primitives_gap = min_primitives_gap
        self.max_primitives_gap = max_primitives_gap
        self.min_stroke_length = min_stroke_length
        self.max_stroke_length = max_stroke_length
        self.min_stroke_width = min_stroke_width
        self.max_stroke_width = max_stroke_width
        self.strokes_probas = strokes_probas

        if None is not directions_probas:
            self.directions_probas = OrderedDict(directions_probas)
        else:
            self.directions_min_angle = directions_min_angle
            self.directions_max_angle = directions_max_angle
            self.directions_angle_step = directions_angle_step
            num_angles = (self.directions_max_angle - self.directions_min_angle) / self.directions_angle_step
            possible_angles = np.linspace(self.directions_min_angle,
                                          self.directions_max_angle,
                                          num_angles)
            self.directions_probas = OrderedDict({angle: 1. / num_angles
                                                  for angle in possible_angles})
        if None is not junction_directions_probas:
            self.junction_directions_probas = OrderedDict(junction_directions_probas)
        if offset_directions_probas is not None:
            self.offset_directions_probas = OrderedDict(offset_directions_probas)
        else:
            self.offset_directions_probas = OrderedDict({0: 1})

        self.primitives_endpoint_noise_sigma = primitives_endpoint_noise_sigma
        self.primitives_direction_noise_sigma = primitives_direction_noise_sigma

    def _get_drawing_arrangement(self):
        # 1. Select number of directions in the image (direction can be curved)
        num_directions = np.random.randint(self.min_directions, self.max_directions + 1)

        # 2. Select junction types between each pair of directions
        #       (== overall spatial arrangement of primitives)
        # This is done on a per-direction pair basis
        junction_props = {}
        patch_width, patch_height = self.patch_size
        # TODO @artonson: rewrite this choice using itertools.product, starting from max length
        for i in range(num_directions):
            # single primitives without junctions
            junction_props[(i, )] = {
                'junction_type': JunctionType.NO_JUNCTION,
                'midpoint': np.array([uniform(high=patch_height), uniform(high=patch_width)])
            }
            for j in range(i + 1, num_directions):
                # pairs of primitives
                junction_props[(i, j)] = {
                    'junction_type': np.random.choice(list(JunctionType)),
                    'midpoint': np.array([
                        uniform(self.border, patch_height - self.border),
                        uniform(self.border, patch_width - self.border)
                    ])
                }

        # 3. Select number of primitives realizing each direction and gaps between them
        direction_props = {}
        possible_angles = list(self.directions_probas.keys())
        probas = list(self.directions_probas.values())
        for i in range(num_directions):
            alpha = np.random.choice(possible_angles, p=probas)
            # direction = np.random.uniform(low=0, high=1, size=2)
            # direction /= np.linalg.norm(direction)
            direction = np.array([np.cos(alpha), np.sin(alpha)])
            props = {
                'primitive_type': PrimitiveType.PT_LINE,
                'direction': direction,
                'num_primitives': np.random.randint(self.min_primitives[PrimitiveType.PT_LINE],
                                                    self.max_primitives[PrimitiveType.PT_LINE] + 1),
                'gap': uniform(self.min_primitives_gap, self.max_primitives_gap),
                'stroke_width': uniform(self.min_stroke_width, self.max_stroke_width),
                'length': uniform(self.min_stroke_length, self.max_stroke_length),
            }
            direction_props[i] = props

        # 4. Select angle between primitives
        # This will generally depend on the junction types
        angle_by_pair = {}
        possible_angles = np.linspace(np.pi / 12, np.pi, 12)
        for i in range(num_directions):
            for j in range(i + 1, num_directions):
                # np.random.uniform(np.pi / 6, np.pi * 5 / 6)
                angle_by_pair[(i, j)] = np.random.choice(possible_angles)


        return num_directions, junction_props, angle_by_pair, direction_props

    def _get_primitives(self, num_directions, junction_props, angles, direction_props):
        # print(junction_props)
        # print(direction_props)
        junction_angles = list(self.junction_directions_probas.keys())
        probas = list(self.junction_directions_probas.values())

        def alter_direction(direction):
            angle1 = np.random.choice(junction_angles, p=probas)
            c, s = np.cos(angle1), np.sin(angle1)
            R = np.array(((c, -s), (s, c)))
            direction1 = np.dot(R, direction)
            return direction1

        def mid2end(mp, direction, length, **kwargs):
            return mp - .5 * length * direction


        primitives = {}
        if num_directions == 1:
            endpoint = mid2end(junction_props[(0, )]['midpoint'], **direction_props[0])
            primitives.update(self.generate_direction(
                endpoint, **direction_props[0]))

        if num_directions == 2:
            junction_type = junction_props[(0, 1)]['junction_type']
            if junction_type == JunctionType.NO_JUNCTION:
                # arbitrary stroke not intersecting with other primitives
                for i in range(num_directions):
                    primitives.update(
                        self.generate_direction(
                            junction_props[(i, )]['midpoint'], **direction_props[i])
                    )

            elif junction_type == JunctionType.L_JUNCTION:
                # stroke sharing same endpoint
                endpoint = junction_props[(0, 1)]['midpoint']
                primitives.update(
                    self.generate_direction(endpoint, **direction_props[0])
                )

                direction_props[1]['direction'] = alter_direction(direction_props[0]['direction'])
                primitives.update(
                    self.generate_direction(endpoint, **direction_props[1])
                )

            elif junction_type == JunctionType.T_JUNCTION:
                # stroke sharing same endpoint connecting to middle of another stroke
                junc_point = junction_props[(0, 1)]['midpoint']
                endpoint = mid2end(junc_point, **direction_props[0])
                primitives.update(
                    self.generate_direction(endpoint, **direction_props[0])
                )

                direction_props[1]['direction'] = alter_direction(direction_props[0]['direction'])
                endpoint = junc_point
                primitives.update(
                    self.generate_direction(endpoint, **direction_props[1])
                )

            elif junction_type == JunctionType.X_JUNCTION:
                # strokes intersecting in middle of each other
                junc_point = junction_props[(0, 1)]['midpoint']
                endpoint = mid2end(junc_point, **direction_props[0])
                primitives.update(
                    self.generate_direction(endpoint, **direction_props[0])
                )

                direction_props[1]['direction'] = alter_direction(direction_props[0]['direction'])
                endpoint = mid2end(junc_point, **direction_props[1])
                primitives.update(
                    self.generate_direction(endpoint, **direction_props[1])
                )
        # TODO fix it, DataLoader still fails on that
        elif num_directions == 3:
            pass
        elif num_directions == 4:
            pass
        else:
            pass
        #
        #
        # for direction in range(num_directions):
        #     primitives = self.generate_direction(**direction_props[direction])
        return primitives

    def generate_direction(self, endpoint, direction, primitive_type, num_primitives, gap, stroke_width, length):
        primitives = []
        ortho_sign = +1 if uniform() > .5 else -1
        ortho_dir = ortho_sign * np.array([-direction[1], direction[0]])
        for i in range(num_primitives):
            random_rotation = rotation_matrix_2d(normal(scale=self.primitives_direction_noise_sigma))
            stroke_direction = np.dot(random_rotation, direction)
            shift = i * (gap + stroke_width) * ortho_dir
            end1noise = normal(scale=self.primitives_endpoint_noise_sigma, size=2)
            x1, y1 = endpoint + shift + end1noise
            end2noise = normal(scale=self.primitives_endpoint_noise_sigma, size=2)
            x2, y2 = np.array([x1, y1]) + np.array(stroke_direction) * length + end2noise
            primitive = Line((x1, y1), (x2, y2), stroke_width)
            primitives.append(primitive)
        return {primitive_type: primitives}

    def _get_vector_item(self, idx):
        """Produce a vector image, specified by a configuration file.

        Algorithm:
        1. Select number of "directions" in the image (direction can be curved)
        2. Select junction types between each pair of directions
        3. Select angle between directions -- note not every combination of junction types
            and angles are realizable
        4. Select parameters of primitives realizing directions:
            - number of primitives realizing each direction
            - gaps between them (in pixels)
            - lengths in pixels
            - stroke thickness
        5. Create the drawing
        6. Select overall orientation of the image (rotate by random angle theta)
        7. Select overall offset of the image (shift by random vector delta)

        Returns:
            primitives_params - list of primitive parametrisations
        """

        num_directions, junction_props, angles, direction_props = self._get_drawing_arrangement()

        vector = self._get_primitives(num_directions, junction_props, angles, direction_props)
        for primitive_type, primitives in vector.items():
            for primitive in primitives:
                primitive.clip_to_box(self.patch_size)

        vector_to_draw_list = []
        for primitive_type, primitives in vector.items():
            for primitive in primitives:
                if primitive.is_drawn:
                    vector_to_draw_list.append((primitive_type, primitive))
        if self.total_primitives < float('+inf'):
            primitives_indexes = np.arange(len(vector_to_draw_list))
            if len(vector_to_draw_list) > self.total_primitives:
                primitives_indexes = np.random.choice(
                    primitives_indexes,
                    size=int(self.total_primitives),
                    replace=False
                )
            vector_to_draw_list = [vector_to_draw_list[i] for i in primitives_indexes]

        vector_to_draw = dict([(key, []) for key in vector.keys()])
        for primitive_type, primitive in vector_to_draw_list:
            vector_to_draw[primitive_type].append(primitive.to_repr())
        return vector_to_draw

    def __len__(self):
        return self.size


class SyntheticHandcraftedDataset(SyntheticStructuredDataset):
    def __init__(self, *, topologies_with_probas, **kwargs):
        self.topologies_with_probas = topologies_with_probas
        super().__init__(**kwargs)

    def _get_vector(self):
        topology_class_name = choose_with_proba(self.topologies_with_probas)
        topology_class = TOPOLOGY_BY_NAME[topology_class_name]
        topology = topology_class.from_dataset(self)
        vector = topology.get_vector()
        return vector

    def _get_vector_item(self, idx):
        vector = self._get_vector()
        for primitive_type, primitives in vector.items():
            for primitive in primitives:
                primitive.clip_to_box(self.patch_size)

        vector_to_draw_list = []
        for primitive_type, primitives in vector.items():
            for primitive in primitives:
                if primitive.is_drawn:
                    vector_to_draw_list.append((primitive_type, primitive))
        if self.total_primitives < float('+inf'):
            primitives_indexes = np.arange(len(vector_to_draw_list))
            if len(vector_to_draw_list) > self.total_primitives:
                primitives_indexes = np.random.choice(
                    primitives_indexes,
                    size=int(self.total_primitives),
                    replace=False
                )
            vector_to_draw_list = [vector_to_draw_list[i] for i in primitives_indexes]

        vector_to_draw = dict([(key, []) for key in vector.keys()])
        for primitive_type, primitive in vector_to_draw_list:
            vector_to_draw[primitive_type].append(primitive.to_repr())
        return vector_to_draw

    # def _render(self, primitive_sets):
    #     return render(primitive_sets, self.patch_size, data_representation='alex')


# class ExhaustiveSyntheticHandcraftedDataset(SyntheticHandcraftedDataset):
#     """Generate samples for all specified parameter values."""
#
#     def _get_vector_item(self, idx):
#         vector = self._get_vector(idx)
#         for primitive_type, primitives in vector.items():
#             for primitive in primitives:
#                 primitive.clip_to_box(self.patch_size)
#
#         vector_to_draw_list = []
#         for primitive_type, primitives in vector.items():
#             for primitive in primitives:
#                 if primitive.is_drawn:
#                     vector_to_draw_list.append((primitive_type, primitive))
#         if self.total_primitives < float('+inf'):
#             primitives_indexes = np.arange(len(vector_to_draw_list))
#             if len(vector_to_draw_list) > self.total_primitives:
#                 primitives_indexes = np.random.choice(
#                     primitives_indexes,
#                     size=int(self.total_primitives),
#                     replace=False
#                 )
#             vector_to_draw_list = [vector_to_draw_list[i] for i in primitives_indexes]
#
#         vector_to_draw = dict([(key, []) for key in vector.keys()])
#         for primitive_type, primitive in vector_to_draw_list:
#             vector_to_draw[primitive_type].append(primitive.to_repr())
#         return vector_to_draw
#
#
# class Sampler:
#     """Abstract class."""
#     def __getitem__(self, item): raise NotImplemented
#     def __len__(self): raise NotImplemented
#
#
# class RandomSampler(Sampler):
#     """Yield random values according to some ."""
#     pass
#
#
# class ListSampler(Sampler):
#     """Yield values from list until it is exhausted."""
#     pass


if __name__ == '__main__':
    pass
    # dg = DegradationGenerator(degradations_list=['kanungo'], max_num_degradations=1)
    # config = dict(patch_size=(64, 64), border=4,
    #               line_count=3, arc_count=0, bezier_count=0,
    #               raster_transform=dg)
    # syndata = SyntheticDataset(**config)
    # from torch.utils.data import DataLoader
    #
    # dataloader = DataLoader(syndata, batch_size=4, shuffle=True, num_workers=4)
    #
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['raster'].size(),
    #           sample_batched['vector'])
    #     if i_batch == 10:
    #         break
    #
    # for angle in np.linspace(0, np.pi, 13):
    # directions_probas = {}
    #     if angle in [0, np.pi / 2]:                 # - horizontal, | vertical
    #         directions_probas[angle] = .30
    #     elif angle in [np.pi / 4, np.pi * 3 / 4]:   # /, \
    #         directions_probas[angle] = .15
    #     elif angle < np.pi:
    #         directions_probas[angle] = .1 / 8
    #
    # junction_directions_probas = {}
    # for angle in np.linspace(np.pi / 6, np.pi - np.pi / 6, 9):
    #     if angle in [np.pi / 2]:                    # - horizontal, | vertical
    #         junction_directions_probas[angle] = .60
    #     elif angle in [np.pi / 4, np.pi * 3 / 4]:   # /, \
    #         junction_directions_probas[angle] = .15
    #     else:
    #         junction_directions_probas[angle] = .1 / 6
    #
    # g = SyntheticStructuredDataset(
    #     patch_size=(64, 64), border=8,
    #     min_directions=1,
    #     max_directions=2,
    #     directions_probas=directions_probas,
    #     junction_directions_probas=junction_directions_probas,
    #     # directions_min_angle=0,
    #     # directions_max_angle=np.pi,
    #     # directions_angle_step=np.pi/4,
    #     min_primitives=0,
    #     max_primitives=6,
    #     total_primitives=10,
    #     min_primitives_gap=4,
    #     max_primitives_gap=10,
    #     min_stroke_width=2,
    #     max_stroke_width=10,
    #     min_stroke_length=90,
    #     max_stroke_length=100,
    #     primitives_endpoint_noise_sigma=0.5,
    #     primitives_direction_noise_sigma=np.pi / 270.,
    # )
    #
    # d = DegradationGenerator(degradations_list=all_degradations,
    #                          max_num_degradations=2)
    # for i in range(100):
    #     png_image, primitives_params = g.get_pair()
    #     print(primitives_params)
    #     png_image = img_8bit_to_float(rgb_to_gray(png_image))
    #     png_image = d.do_degrade(png_image)
    #     png_image = gray_float_to_8bit(png_image)
    #     im = Image.fromarray(png_image, 'L')
    #     im.save('{}.png'.format(i))

