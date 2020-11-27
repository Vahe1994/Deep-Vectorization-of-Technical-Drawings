from util_files.loss_functions.utils import *


def DeepParametricShape(predicted_curves, strokes, images, w_surface=1, w_alignment=0.01,w_int_loss=0.001):
    """
    This is loss function modification suggested in DeepParametricShape CVPR paper https://people.csail.mit.edu/smirnov/deep-parametric-shapes/
    :param predicted_curves:   [b,4*n_loops]
    :param strokes:
    :param images:
    :param w_surface:
    :param w_alignment:
    :return:
    """
    if type(images) is dict:
        target_distance_fields = images['distance_fields']
        target_alignment_fields = images['alignment_fields']
        target_occupancy_fields = images['occupancy_fields']
    else:
        target_distance_fields, target_alignment_fields, target_occupancy_fields = compute_distance_fields_for_image(
            images)
    distance_fields, alignment_fields, occupancy_fields,int_loss = compute_distance_fields_from_curves(predicted_curves, strokes)

    surfaceloss =  th.mean(target_occupancy_fields * distance_fields + target_distance_fields * occupancy_fields)
    # try without **2s
    alignmentloss = th.mean(1-th.sum(target_alignment_fields * alignment_fields, dim=-1))
    # rot = th.tensor([[0., -1.], [1., 0.]])
    # alignmentloss = th.mean(
    #     th.sum(th.mm(target_alignment_fields.reshape(-1, 2), rot) * alignment_fields.reshape(-1, 2),
    #            dim=-1) ** 2)
    loss = w_surface * surfaceloss + w_alignment * alignmentloss + w_int_loss*int_loss
    return loss, surfaceloss, alignmentloss,int_loss

# def DeepParametricShape(predicted_curves, strokes, images, w_surface=1, w_alignment=0.01):
#     """
#     This is loss function modification suggested in DeepParametricShape CVPR paper https://people.csail.mit.edu/smirnov/deep-parametric-shapes/
#     :param predicted_curves:   [b,4*n_loops]
#     :param strokes:
#     :param images:
#     :param w_surface:
#     :param w_alignment:
#     :return:
#     """
#     if images.shape[1] == 3:
#         target_distance_fields = images[:, 0, ...]
#         target_alignment_fields = images[:, 1, ...]
#         target_occupancy_fields = images[:, 2, ...]
#     else:
#         target_distance_fields, target_alignment_fields, target_occupancy_fields = compute_distance_fields_for_image(
#             images)
#     distance_fields, alignment_fields, occupancy_fields = compute_distance_fields_from_curves(predicted_curves, strokes)
#
#     # surfaceloss = th.mean(target_occupancy_fields * distance_fields + target_distance_fields * occupancy_fields)
#     # alignmentloss = th.mean(1-th.sum(target_alignment_fields * alignment_fields, dim=-1) ** 2)
#     # rot = th.tensor([[0., -1.], [1., 0.]])
#     # alignmentloss = th.mean(
#     #     th.sum(th.mm(target_alignment_fields.reshape(-1, 2), rot) * alignment_fields.reshape(-1, 2),
#     #            dim=-1) ** 2)
#     # loss = w_surface * surfaceloss + w_alignment * alignmentloss
#     #############################################################################
#     surfaceloss = target_occupancy_fields * distance_fields + target_distance_fields * occupancy_fields
#     rot = th.tensor([[0., -1.], [1., 0.]])
#     return target_alignment_fields,alignment_fields
