from refinement.our_refinement.utils.lines_refinement_functions import *
import argparse
from util_files.metrics.iou import calc_iou__vect_image
def render_optimization_hard(patches_rgb, patches_vector, device, options, name):
    '''

    :param patches_rgb: patches with rgb(from 0 to 255) [b,3,64,64](can change but code should be changed respectevly)
    :param patches_vector: patches with predicted vector [b,line_count,5](check this)
    :param device: cuda,cpu e.t.c
    :param options: dict with information #Todo add example
    :param name:
    :return:  patches with refined vectors [b,line_count,5](check this)
    '''
    patches_rgb_im = np.copy(patches_rgb)

    first_encounter = True

    patches_vector = torch.tensor(patches_vector)
    y_pred_rend = torch.zeros((patches_vector.shape[0], patches_vector.shape[1], patches_vector.shape[2] - 1))
    patches_rgb = 1 - torch.tensor(patches_rgb).squeeze(3).unsqueeze(1) / 255.
    print('init_random', options.init_random)
    if (options.init_random):
        print('init_random', options.init_random)
        patches_vector = torch.rand((patches_vector.shape)) * 64
        patches_vector[..., 4] = 1

    for it_batches in range(300, patches_vector.shape[0] + 299, 300):
        it_start = it_batches - 300
        if it_batches > patches_vector.shape[0]:
            it_batches = patches_vector.shape[0]

        take_batches_n = 300

        rasters_batch = patches_rgb.detach()

        take_batches = []
        for it in np.asarray(np.arange(it_start, it_batches)):
            if torch.mean(rasters_batch[it]) != 0:
                take_batches.append(it)
        if take_batches == []:
            continue
        rasters_batch = rasters_batch[take_batches, 0].type(dtype).to(device)
        raster_np = (1 - rasters_batch[0].cpu().numpy()) * 255

        initial_vector = patches_vector.detach()[take_batches].cpu()

        removed_lines = initial_vector[..., -1] < .5 * h
        rand_x1 = torch.rand_like(initial_vector[removed_lines, [0]]) * w
        rand_y1 = torch.rand_like(initial_vector[removed_lines, [1]]) * h
        initial_vector[removed_lines, [0]] = rand_x1
        initial_vector[removed_lines, [2]] = rand_x1 + 1
        initial_vector[removed_lines, [1]] = rand_y1
        initial_vector[removed_lines, [3]] = rand_y1 + 1
        initial_vector[removed_lines, [4]] = 2 ** -8
        initial_vector = initial_vector[..., :5].numpy()

        rasters_batch = torch.nn.functional.pad(rasters_batch, [padding, padding, padding, padding])
        lines_batch = torch.from_numpy(initial_vector).type(dtype).to(device)
        lines_n = lines_batch.shape[1]

        # get the canonical parameters
        x1 = lines_batch[:, :, 0]
        y1 = lines_batch[:, :, 1]
        x2 = lines_batch[:, :, 2]
        y2 = lines_batch[:, :, 3]
        width = lines_batch[:, :, 4]
        X = x2 - x1
        Y = y2 - y1
        length = torch.sqrt(X ** 2 + Y ** 2)
        theta = torch.atan2(Y, X)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        del x1, x2, y1, y2, X, Y

        for canonical_parameter in cx, cy, theta, length, width:
            canonical_parameter.requires_grad = True

        pos_optimizer = NonanAdam([cx, cy, theta], lr=1e-1)
        size_optimizer = NonanAdam([length, width], lr=1e-1)
        # initialize optimizers
        (cx + cy + theta + length + width).reshape(-1)[0].backward()
        pos_optimizer.zero_grad()
        pos_optimizer.step()
        size_optimizer.zero_grad()
        size_optimizer.step()

        patches_to_optimize = np.full(lines_batch.shape[0], True,
                                      np.bool)  # my_iou_score(vector_rendering, rasters_batch) < .98
        cx_final = torch.empty_like(cx)
        cy_final = torch.empty_like(cy)
        theta_final = torch.empty_like(theta)
        length_final = torch.empty_like(length)
        width_final = torch.empty_like(width)
        lines_batch_final = torch.empty_like(lines_batch)

        """
        Algorithm is:
        1. Reinitialize excess predictions
        2. Optimize mean field energy w.r.t position parameters with fixed size parameters
           Apply constraints to the parameters of the lines after this and other parts of iteration
        3. Optimize mean field energy w.r.t size parameters
           with fixed positions of the left points of the lines and their orientations
           Penalize collinearity of overlapping lines
        4. Optimize mean field energy w.r.t size parameters
           with fixed positions of the right points of the lines and their orientations
           Penalize collinearity of overlapping lines
        5. Snap lines that have coinciding ends, close orientations and close widths
        6. Prepare the results for logging
        7. Find the patches with high enough IOU value and do not optimize them furher
        """

        vector_rendering = render_lines_pt(lines_batch.detach())

        its_time_to_stop = [False]

        def plotting_sigint(*args):
            its_time_to_stop[0] = True

        iou_mass = []
        iters_n = options.diff_render_it
        mass_for_iou_one = []
        for i in tqdm(range(iters_n)):
            try:
                # 1. Reinitialize excess predictions
                #    The meaning of `patches_to_optimize` is explained below, in step 7
                if i % 20 == 0:
                    x2 = cx.data + length.data * torch.cos(theta.data) / 2
                    x1 = cx.data - length.data * torch.cos(theta.data) / 2
                    y2 = cy.data + length.data * torch.sin(theta.data) / 2
                    y1 = cy.data - length.data * torch.sin(theta.data) / 2
                    lines_batch = torch.stack([x1, y1, x2, y2, width.data], -1)
                    lines_batch.data[..., -1][lines_batch[..., -1] < 1 / 4] = 0
                    vector_rendering[patches_to_optimize] = render_lines_pt(lines_batch[patches_to_optimize].detach())
                    im = rasters_batch[patches_to_optimize].clone()
                    # the line below is explained in `reinit_excess_lines`
                    im.masked_fill_(vector_rendering[patches_to_optimize] > 0, 0)
                    reinit_excess_lines(cx, cy, width, length, im.reshape(im.shape[0], -1),
                                        patches_to_consider=patches_to_optimize)

                # 2. Optimize mean field energy w.r.t position parameters with fixed size parameters
                x2 = cx + length.data * torch.cos(theta) / 2
                x1 = cx - length.data * torch.cos(theta) / 2
                y2 = cy + length.data * torch.sin(theta) / 2
                y1 = cy - length.data * torch.sin(theta) / 2
                lines_batch = torch.stack([x1, y1, x2, y2, width.data], -1)
                mean_field_energy = mean_field_energy_lines(lines_batch[patches_to_optimize],
                                                            rasters_batch[patches_to_optimize])

                pos_optimizer.zero_grad()
                mean_field_energy.backward()
                pos_optimizer.step()
                #    Apply constraints to the parameters of the lines after this and other parts of iteration
                constrain_parameters(cx, cy, theta, length, width, canvas_width=w, canvas_height=h,
                                     size_optimizer=size_optimizer)

                # 3. Optimize mean field energy w.r.t size parameters
                #    with fixed positions of the left points of the lines and their orientations
                x1 = cx.data - length.data * torch.cos(theta.data) / 2
                y1 = cy.data - length.data * torch.sin(theta.data) / 2
                x2 = x1 + length * torch.cos(theta.data)
                y2 = y1 + length * torch.sin(theta.data)
                lines_batch = torch.stack([x1, y1, x2, y2, width], -1)

                excess_energy = size_energy(lines_batch[patches_to_optimize], rasters_batch[patches_to_optimize])
                #    Penalize collinearity of overlapping lines
                collinearity_energy = mean_vector_field_energy_lines(lines_batch[patches_to_optimize])
                size_optimizer.zero_grad()
                (excess_energy + collinearity_energy).backward()
                size_optimizer.step()

                cx.data[patches_to_optimize] = x1.data[patches_to_optimize] + length.data[
                    patches_to_optimize] * torch.cos(theta.data[patches_to_optimize]) / 2
                cy.data[patches_to_optimize] = y1.data[patches_to_optimize] + length.data[
                    patches_to_optimize] * torch.sin(theta.data[patches_to_optimize]) / 2
                constrain_parameters(cx, cy, theta, length, width, canvas_width=w, canvas_height=h,
                                     size_optimizer=size_optimizer)

                # 4. Optimize mean field energy w.r.t size parameters
                #    with fixed positions of the right points of the lines and their orientations
                x2 = cx.data + length.data * torch.cos(theta.data) / 2
                y2 = cy.data + length.data * torch.sin(theta.data) / 2
                x1 = x2 - length * torch.cos(theta.data)
                y1 = y2 - length * torch.sin(theta.data)
                lines_batch = torch.stack([x1, y1, x2, y2, width], -1)

                excess_energy = size_energy(lines_batch[patches_to_optimize], rasters_batch[patches_to_optimize])
                #    Penalize collinearity of overlapping lines
                collinearity_energy = mean_vector_field_energy_lines(lines_batch[patches_to_optimize])
                size_optimizer.zero_grad()
                (excess_energy + collinearity_energy).backward()
                size_optimizer.step()

                cx.data[patches_to_optimize] = x2.data[patches_to_optimize] - length.data[
                    patches_to_optimize] * torch.cos(theta.data[patches_to_optimize]) / 2
                cy.data[patches_to_optimize] = y2.data[patches_to_optimize] - length.data[
                    patches_to_optimize] * torch.sin(theta.data[patches_to_optimize]) / 2
                constrain_parameters(cx, cy, theta, length, width, canvas_width=w, canvas_height=h,
                                     size_optimizer=size_optimizer)

                # 5. Snap lines that have coinciding ends, close orientations and close widths
                if (i + 1) % 20 == 0:
                    snap_lines(cx, cy, theta, length, width, pos_optimizer=pos_optimizer, size_optimizer=size_optimizer)

            except KeyboardInterrupt:
                its_time_to_stop[0] = True

            # 6. Prepare the results for logging
            sigint = signal.signal(signal.SIGINT, plotting_sigint)

            if (i % 20 == 0) or its_time_to_stop[0]:
                # 6.1. Record the current values of parameters separately
                #      The following steps are performed with these separate values and do not affect the parameters being optimized
                cx_final[patches_to_optimize] = cx.data[patches_to_optimize]
                cy_final[patches_to_optimize] = cy.data[patches_to_optimize]
                theta_final[patches_to_optimize] = theta.data[patches_to_optimize]
                length_final[patches_to_optimize] = length.data[patches_to_optimize]
                width_final[patches_to_optimize] = width.data[patches_to_optimize]

                # 6.2. Collapse invisible lines completely
                width_final[width_final < 1 / 4] = 0

                # 6.3. Collapse the lines that don't add new information to the rasterization
                collapse_redundant_lines(cx_final, cy_final, theta_final, length_final, width_final,
                                         patches_to_consider=patches_to_optimize)
                x2 = cx_final.data[patches_to_optimize] + length_final.data[patches_to_optimize] * torch.cos(
                    theta_final.data[patches_to_optimize]) / 2
                x1 = cx_final.data[patches_to_optimize] - length_final.data[patches_to_optimize] * torch.cos(
                    theta_final.data[patches_to_optimize]) / 2
                y2 = cy_final.data[patches_to_optimize] + length_final.data[patches_to_optimize] * torch.sin(
                    theta_final.data[patches_to_optimize]) / 2
                y1 = cy_final.data[patches_to_optimize] - length_final.data[patches_to_optimize] * torch.sin(
                    theta_final.data[patches_to_optimize]) / 2
                #                 print( torch.stack([x1, y1, x2, y2, width_final[patches_to_optimize]], -1).shape)
                #                 print(lines_batch_final[patches_to_optimize].shape)
                #                 print(patches_to_optimize.shape)
                lines_batch_final[patches_to_optimize] = torch.stack([x1, y1, x2, y2, width_final[patches_to_optimize]],
                                                                     -1)

                # 6.4. Render the lines and calculate the difference from the raster
                vector_rendering[patches_to_optimize] = render_lines_pt(lines_batch_final[patches_to_optimize])
                im = rasters_batch[patches_to_optimize] - vector_rendering[patches_to_optimize]

            #TODO add IOU calc

            if (i % 20 == 0):
                iou_mass.append(calc_iou__vect_image(lines_batch_final.data / 64, patches_rgb_im[take_batches]))
                mass_for_iou_one.append(lines_batch_final.cpu().data.detach().numpy())

        print(it_start)
        if first_encounter:
            first_encounter = False
            iou_all = np.array(iou_mass)
            mass_for_iou = mass_for_iou_one
        else:
            iou_all = np.concatenate((iou_all, iou_mass), axis=1)
            mass_for_iou = np.concatenate((mass_for_iou, mass_for_iou_one), axis=1)
        y_pred_rend[take_batches] = lines_batch_final.cpu().detach()
    prd = y_pred_rend[:, :, -1].clone()
    prd[prd > 1] = 1
    prd[prd < 0.3] = 0
    y_pred_rend = torch.cat((y_pred_rend, prd.unsqueeze(2)), dim=-1)
    os.makedirs(options.output_dir + 'arrays/', exist_ok=True)
    if options.init_random:
        np.save(options.output_dir + 'arrays/hard_optimization_iou_random_' + name, iou_all)
        np.save(options.output_dir + 'arrays/hard_optimization_iou_mass_random_' + name, mass_for_iou)
    else:
        np.save(options.output_dir + 'arrays/hard_optimization_iou_' + name, iou_all)
        np.save(options.output_dir + 'arrays/hard_optimization_iou_mass_' + name, mass_for_iou)

    return y_pred_rend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="/vage/Download/testing_line/", help='dir to folder for output')
    parser.add_argument('--diff_render_it', type=int, default=90, help='iteration count')
    parser.add_argument('--init_random', action='store_true', default=False, dest='init_random',
                        help='init model with random [default: False].')
    parser.add_argument('--rendering_type', type=str, default='hard', help='hard -oleg,simple Alexey')
    return parser.parse_args()