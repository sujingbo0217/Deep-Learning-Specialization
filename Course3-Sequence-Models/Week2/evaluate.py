# -*- coding = utf-8 -*-
# @Author : Jingbo Su
# @File : evaluate.py

import torch

def evaluate(model, iterator, criterion) -> float:
    # set model to evaluation mode
    model.eval()

    """
    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean
    """

    epoch_loss = 0.

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            outputs = model(src, trg, 0.)   # Turn off teacher_forcing_ratio
            output_size = outputs.size(-1)

            y = trg[1:].view(-1)  # get rid of "<sos>" && flatten
            y_pred = outputs[1:].view(-1, output_size)  # get rid of "<sos>" && flatten

            loss = criterion(y_pred, y)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
