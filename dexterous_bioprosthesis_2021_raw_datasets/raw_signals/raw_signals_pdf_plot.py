from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def raw_signals_pdf_plot(out_dir_path,raw_signals:RawSignals):

    os.makedirs(out_dir_path, exist_ok=True)

    for rs_idx, rs in tqdm( enumerate(raw_signals), desc="Signals", total=len(raw_signals)):
        rs_file_plot_path = os.path.join(out_dir_path, "sig_{}.pdf".format(rs_idx))

        num_channels = rs.signal.shape[1]

        with PdfPages(rs_file_plot_path) as pdf:
            for ch_idx in range(num_channels):
                plt.plot(rs.signal[:,ch_idx])
                plt.title("Channel: {}, class: {}".format(ch_idx, rs.object_class))
                plt.xlabel("Sample number")
                plt.ylabel("Signal value")
                pdf.savefig()
                plt.close()
