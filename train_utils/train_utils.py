import math
import torch
from einops import rearrange
import numpy as np
import os

from dataset.datautils import save_as_laz_file, remove_padding_points_from_bubble

import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def update_metrics_for_f1_score_calculation(n_classes_model, gt_semantic_labels, inf_semantic_labels,
                                            n_TP_semantic_labels_per_class, n_FP_semantic_labels_per_class,
                                            n_FN_semantic_labels_per_class):
    """

    Parameters
    ----------
    n_classes_model: int
    gt_semantic_labels: array of integers
    inf_semantic_labels: array of integers
    n_TP_semantic_labels_per_class: 1D integer array of size n_classes_model
    n_FP_semantic_labels_per_class: 1D integer array of size n_classes_model
    n_FN_semantic_labels_per_class: 1D integer array of size n_classes_model

    """

    for n in range(0, n_classes_model):
        gt_indices_of_current_class_and_bubble = np.argwhere(gt_semantic_labels == n).flatten()
        inf_indices_of_current_class_and_bubble = np.argwhere(inf_semantic_labels == n).flatten()
        n_TP_points_current_class_and_bubble = len(np.argwhere(np.in1d(gt_indices_of_current_class_and_bubble,
                                                                       inf_indices_of_current_class_and_bubble) == True).flatten())
        n_TP_semantic_labels_per_class[n] += n_TP_points_current_class_and_bubble
        n_FP_semantic_labels_per_class[n] += len(inf_indices_of_current_class_and_bubble) - \
                                             n_TP_points_current_class_and_bubble
        n_FN_semantic_labels_per_class[n] += (len(gt_indices_of_current_class_and_bubble) -
                                              n_TP_points_current_class_and_bubble)


def calculate_per_class_f1_score(n_classes_model, n_TP_semantic_labels_per_class, n_FP_semantic_labels_per_class,
                                    n_FN_semantic_labels_per_class):

    per_class_f1_score = np.zeros(n_classes_model)
    per_class_semantic_precision = np.zeros(n_classes_model)
    per_class_semantic_recall = np.zeros(n_classes_model)

    for n in range(0, n_classes_model):
        if (n_TP_semantic_labels_per_class[n] + n_FP_semantic_labels_per_class[n]) > 0:
            per_class_semantic_precision[n] = n_TP_semantic_labels_per_class[n] / \
                                              (n_TP_semantic_labels_per_class[n] + n_FP_semantic_labels_per_class[n])
        if (n_TP_semantic_labels_per_class[n] + n_FN_semantic_labels_per_class[n]) > 0:
            per_class_semantic_recall[n] = n_TP_semantic_labels_per_class[n] / \
                                           (n_TP_semantic_labels_per_class[n] + n_FN_semantic_labels_per_class[n])
        if (per_class_semantic_precision[n] + per_class_semantic_recall[n]) > 0:
            per_class_f1_score[n] = (2 * per_class_semantic_precision[n] * per_class_semantic_recall[n]) / \
                                    (per_class_semantic_precision[n] + per_class_semantic_recall[n])

    return per_class_f1_score


def run_test(model, device, test_dataloader, test_data_vis_dir, n_classes_model, rings_per_bubble, points_per_ring,
             ring_padding):
    model.eval()

    # initialise variables to calculate the F1 score for PCSS
    n_TP_semantic_labels_per_class = np.zeros(n_classes_model)
    n_FP_semantic_labels_per_class = np.zeros(n_classes_model)
    n_FN_semantic_labels_per_class = np.zeros(n_classes_model)

    for i, batch in enumerate(test_dataloader):
        point_tokens = batch[0].to(device)
        mask = batch[2].to(device)

        with torch.no_grad():
            y_predicted = model(point_tokens, mask)  # [batch, n_rings, n_points_per_ring, n_classes_model]
            y_predicted = rearrange(y_predicted, 'a b c d -> (a b c) d')
            labels = torch.argmax(y_predicted, dim=1)
            labels = labels.cpu().numpy()

            # save as laz file
            all_points = rearrange(point_tokens, '1 a b c -> (a b) c').cpu().numpy()
            all_points, labels = remove_padding_points_from_bubble(all_points, labels, rings_per_bubble,
                                                                   points_per_ring,
                                                                   ring_padding)
            file_name = os.path.join(test_data_vis_dir, f'view_{i}.laz')
            save_as_laz_file(points=all_points, classification=labels, filename=file_name)

            gt_semantic_labels = rearrange(batch[1], 'a b c -> (a b c)')
            _, gt_semantic_labels = remove_padding_points_from_bubble(None, gt_semantic_labels,
                                                                      rings_per_bubble, points_per_ring, ring_padding)
            update_metrics_for_f1_score_calculation(n_classes_model, gt_semantic_labels, labels,
                                                    n_TP_semantic_labels_per_class, n_FP_semantic_labels_per_class,
                                                    n_FN_semantic_labels_per_class)

    per_class_f1_score = calculate_per_class_f1_score(n_classes_model, n_TP_semantic_labels_per_class,
                                                        n_FP_semantic_labels_per_class,
                                                      n_FN_semantic_labels_per_class)
    LOGGER.info(f"per_class_f1_score: {np.around(per_class_f1_score, 2)}")


def run_validation(model, device, val_dataloader, criterion, val_dataset, batch_size, writer, epoch):

    model.eval()
    total_samples = len(val_dataset)
    n_iterations = math.ceil(total_samples / batch_size)
    val_loss_accumulation = 0.0

    for i, batch in enumerate(val_dataloader):
        data = batch[0].to(device)
        labels = batch[1].type(torch.LongTensor).to(device)
        mask = batch[2].to(device)

        with torch.no_grad():
            # forward pass
            y_predicted = model(data, mask)
            y_predicted = rearrange(y_predicted, 'a b c d -> (a b c) d')

            # calculate loss
            labels = rearrange(labels, 'a b c -> (a b c)')
            loss = criterion(y_predicted, labels)

            val_loss_accumulation += loss.item()

    val_loss_avg = val_loss_accumulation / n_iterations
    writer.add_scalar('val loss', val_loss_avg, epoch)
    writer.flush()