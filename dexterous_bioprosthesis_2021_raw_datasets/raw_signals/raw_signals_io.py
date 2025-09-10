import datetime
import os
import logging

from joblib import delayed
import pandas as pd
import numpy as np
import zipfile
import tarfile
import re

from .raw_signal import RawSignal
from .raw_signals import RawSignals
from ..tools.progressparallel import ProgressParallel

date_format_string = "%Y-%m-%d %H:%M:%S"
from tqdm import tqdm

def str_sort_key(x):
    return str(x)


def read_signals_from_archive(
    archive_path,
    sample_rate=1000,
    dtype=np.double,
    dir_sorting_key=str_sort_key,
    file_sorting_key=lambda x: str(x),
    filter_regex = None
):
    accapted = RawSignals(sample_rate=sample_rate)
    rejected = RawSignals(sample_rate=sample_rate)
    
     # --- ZIP ---
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as z:
            memberlist = z.namelist()
            memberlist.sort(key=str_sort_key)
            n_members = len(memberlist)
            for member in tqdm(memberlist, total=n_members, desc="Zip file iterating archive elements", leave=True):
                match_regex = True if filter_regex is None else re.match(filter_regex,member)
                if member.endswith(".csv") and match_regex:
                    base_filename = os.path.splitext(os.path.basename(member))[0]
                    class_name = os.path.basename(os.path.dirname(member))
                    dat_name = f"{os.path.dirname(member)}/{base_filename}.dat"
                    is_rejected = os.path.basename(os.path.dirname(os.path.dirname(member))) == "rejected"
                    dat_file_present = dat_name in memberlist
                    with z.open(member) as csv_handler:
                        try:
                            data = np.asfortranarray(
                                pd.read_csv(csv_handler, delimiter=";", decimal=",", header=None).to_numpy(
                                    dtype=dtype
                                ),
                                dtype=dtype,
                            )
                        except Exception as exc:
                            logging.debug(
                                "Failed to load {}. Exception: {}. Skipping".format(member, exc)
                            )
                            continue

                        object_timestamp = 0
                        if dat_file_present:
                            try:
                                with z.open(dat_name, "r") as dat_handler:
                                    data_text_bytes = dat_handler.read().strip()
                                    data_text = data_text_bytes.decode('utf-8')
                                    element = datetime.datetime.strptime(data_text, date_format_string)
                                    object_timestamp = datetime.datetime.timestamp(element)
                            except Exception as exc:
                                logging.debug(
                                    "Failed to determine timestamp for {}. Exception {}".format(
                                        member, exc
                                    )
                                )

                        if not is_rejected:
                            accapted.append(RawSignal(data, class_name, timestamp=object_timestamp))
                        else:
                            rejected.append(RawSignal(data, class_name, timestamp=object_timestamp))

    # --- TAR (supports tar, tar.gz, tar.bz2, tar.xz) ---
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            memberlist = tar.getmembers()
            memberlist.sort(key=str_sort_key)
            n_members = len(memberlist)
            members_names_list = [m.name for m in memberlist]
            for member in tqdm( memberlist, leave=True, desc= "Tar file iterating over archive members", total=n_members):
                match_regex = True if filter_regex is None else re.match(filter_regex,member.name)
                if member.isfile() and member.name.endswith(".csv") and match_regex:
                    member_name = member.name
                    base_filename = os.path.splitext(os.path.basename(member_name))[0]
                    class_name = os.path.basename(os.path.dirname(member_name))
                    dat_name = f"{os.path.dirname(member_name)}/{base_filename}.dat"
                    is_rejected = os.path.basename(os.path.dirname(os.path.dirname(member_name))) == "rejected"
                    dat_file_present = dat_name in members_names_list
                    csv_file_handler = tar.extractfile(member)
                    if csv_file_handler is not None:
                        try:
                            data = np.asfortranarray(
                                pd.read_csv(csv_file_handler, delimiter=";", decimal=",", header=None).to_numpy(
                                    dtype=dtype
                                ),
                                dtype=dtype,
                            )
                        except Exception as exc:
                            logging.debug(
                                "Failed to load {}. Exception: {}. Skipping".format(member, exc)
                            )
                            continue

                        object_timestamp = 0
                        if dat_file_present:
                            try:
                                data_file_member = tar.getmember(dat_name)
                                data_handler = tar.extractfile(data_file_member)
                                data_text_bytes = data_handler.read().strip()
                                data_text = data_text_bytes.decode('utf-8')
                                element = datetime.datetime.strptime(data_text, date_format_string)
                                object_timestamp = datetime.datetime.timestamp(element)
                            except Exception as exc:
                                logging.debug(
                                    "Failed to determine timestamp for {}. Exception {}".format(
                                        member, exc
                                    )
                                )

                        if not is_rejected:
                            accapted.append(RawSignal(data, class_name, timestamp=object_timestamp))
                        else:
                            rejected.append(RawSignal(data, class_name, timestamp=object_timestamp))

    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    if len(rejected) == 0:
        rejected = None
    return {"accepted": accapted, "rejected": rejected}


def read_signals_from_dirs(
    input_dir,
    sample_rate=1000,
    n_jobs=-1,
    parallel_options=dict(),
    dir_sorting_key=str_sort_key,
    file_sorting_key=lambda x: str(x),
    dtype=np.double,
):
    """
    Reads raw signals from the directory structure.
    Return tuple of accepted and rejected signals
    """
    accepted = _read_signals_from_dirs_internal(
        input_dir,
        sample_rate,
        n_jobs=n_jobs,
        parallel_options=parallel_options,
        dir_sorting_key=dir_sorting_key,
        dtype=dtype,
    )

    rejected_measurements_path = os.path.join(input_dir, "rejected")
    if os.path.exists(rejected_measurements_path):
        rejected = _read_signals_from_dirs_internal(
            rejected_measurements_path,
            sample_rate,
            n_jobs=n_jobs,
            parallel_options=parallel_options,
            dir_sorting_key=dir_sorting_key,
            dtype=dtype,
        )
    else:
        rejected = None

    return {"accepted": accepted, "rejected": rejected}


def _read_class_dir(class_dir, file_order_key=str_sort_key, dtype=np.double):
    """
    Read objects from class-specific directory
    Arguments:
     class_dir -- class specific directories. It contains csv and dat files
    """
    csv_files_list = [
        file
        for file in sorted(os.listdir(class_dir), key=file_order_key)
        if file.endswith(".csv")
    ]
    class_name = os.path.basename(class_dir)

    signal_objects = RawSignals()

    for file in csv_files_list:
        file_basename = os.path.splitext(file)[0]
        csv_path = os.path.join(class_dir, "{}.csv".format(file_basename))
        dat_path = os.path.join(class_dir, "{}.dat".format(file_basename))

        try:
            data = np.asfortranarray(
                pd.read_csv(csv_path, delimiter=";", decimal=",", header=None).to_numpy(
                    dtype=dtype
                ),
                dtype=dtype,
            )
        except Exception as exc:
            logging.debug(
                "Failed to load {}. Exception: {}. Skipping".format(csv_path, exc)
            )
            continue

        object_timestamp = 0
        try:
            with open(dat_path, "r") as dat_handler:
                data_text = dat_handler.read().strip()
                element = datetime.datetime.strptime(data_text, date_format_string)
                object_timestamp = datetime.datetime.timestamp(element)
        except Exception as exc:
            logging.debug(
                "Failed to determine timestamp for {}. Exception {}".format(
                    csv_path, exc
                )
            )

        signal_objects.append(RawSignal(data, class_name, timestamp=object_timestamp))

    return signal_objects


def _read_signals_from_dirs_internal(
    input_dir,
    sample_rate=1000,
    n_jobs=-1,
    parallel_options=dict(),
    dir_sorting_key=lambda x: str(x),
    file_order_key=str_sort_key,
    dtype=np.double,
):
    """
    Read the raw dataset from the directory structure.
    """
    sorted_class_dirs = sorted(
        [
            d
            for d in os.listdir(os.path.normpath(input_dir))
            if os.path.isdir(os.path.join(input_dir, d)) and d != "rejected"
        ],
        key=dir_sorting_key,
    )

    data_objects = RawSignals(sample_rate=sample_rate)

    if len(sorted_class_dirs) == 0:
        return data_objects

    class_data_objects = ProgressParallel(
        n_jobs=n_jobs,
        use_tqdm=True,
        total=len(sorted_class_dirs),
        desc="Class directories",
        **parallel_options
    )(
        delayed(_read_class_dir)(
            os.path.join(input_dir, directory), file_order_key, dtype=dtype
        )
        for directory in sorted_class_dirs
    )
    for class_data_obj in class_data_objects:
        data_objects += class_data_obj

    return data_objects


def save_signals_to_dirs(raw_signals: RawSignals, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    signal_labels = raw_signals.get_labels()
    unique_labels = set(signal_labels)

    for label in unique_labels:
        label_dir_path = os.path.join(output_directory, "{}".format(label))
        os.makedirs(label_dir_path, exist_ok=True)

        label_indices = [
            i for i in range(len(raw_signals)) if raw_signals[i].object_class == label
        ]
        signal_label_subset = raw_signals[label_indices]

        subset_signal_indices_string = sorted(
            ["{}".format(i) for i in range(1, len(signal_label_subset) + 1)]
        )

        cnt = 0
        for istr in subset_signal_indices_string:

            data_file_path = os.path.join(label_dir_path, "{}.csv".format(istr))

            signal_np = signal_label_subset[cnt].signal
            signal_df = pd.DataFrame(signal_np)
            signal_df.to_csv(
                data_file_path, sep=";", header=False, index=False, decimal=","
            )

            date_file_path = os.path.join(label_dir_path, "{}.dat".format(istr))
            date_object = datetime.datetime.fromtimestamp(
                signal_label_subset[cnt].timestamp
            )
            date_string = date_object.strftime(date_format_string)

            with open(date_file_path, "w") as file:
                print(date_string, file=file)
            cnt += 1
