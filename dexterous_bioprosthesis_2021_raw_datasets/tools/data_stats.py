import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


def data_stats(signals:RawSignals):

    stats = dict()

    n_objects = len(signals)
    stats["n_objects"] = n_objects
    labels = signals.get_labels()
    ulabels, counters =  np.unique(labels,return_counts=True)
    stats["ulabels"] = {"labels":ulabels.tolist(), "counters":counters.tolist()}
    stats["fs"] = signals.get_sample_rate()

    lengths = []
    for sig in signals:
        lengths.append(len(sig))

    ulengths, u_len_cnt = np.unique(lengths, return_counts=True)
    stats["signal_lengths"] = {"lengths":ulengths.tolist(), "counters":u_len_cnt.tolist()}

    return stats