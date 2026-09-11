"""I/O utilities for reading and writing raw signal datasets.

Supports loading signals from directory structures and compressed
archives (ZIP, TAR), as well as saving signals back to directories.
"""
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
    """Return the string representation of *x* for sorting."""
    return str(x)


def read_signals_from_archive(
    archive_path,
    sample_rate=1000,
    dtype=np.double,
    dir_sorting_key=str_sort_key,
    file_sorting_key=lambda x: str(x),
    filter_regex = None
):
    """Read raw signals from a ZIP or TAR archive."""
    accapted = RawSignals(sample_rate=sample_rate)
    rejected = RawSignals(sample_rate=sample_rate)
    
     # --- ZIP ---
    channel_names = None
    sample_rate_regex = r".*/sample_rate.txt"
    channel_names_regex = r".*/channel_names.txt"

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as z:
            memberlist = z.namelist()
            memberlist.sort(key=str_sort_key)
            n_members = len(memberlist)
            for member in memberlist:
                if re.match(channel_names_regex, member):
                    try:
                        cn_text = z.read(member).decode('utf-8')
                        channel_names = [line.strip() for line in cn_text.strip().splitlines() if line.strip()]
                    except:
                        logging.warning(f"Invalid channel names in file: {member}")
                    finally:
                        break

            for member in tqdm(memberlist, total=n_members, desc="Zip file iterating archive elements", leave=True):
                if re.match(sample_rate_regex, member):
                    try:
                        sample_rate = int(z.read(member).strip())
                        accapted.set_sample_rate(sample_rate)
                        rejected.set_sample_rate(sample_rate)
                    except:
                        logging.warning(f"Invalid sample rate in file: {member}")
                    continue
                
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
                            accapted.append(RawSignal(data, class_name, channel_names=channel_names, timestamp=object_timestamp))
                        else:
                            rejected.append(RawSignal(data, class_name, channel_names=channel_names, timestamp=object_timestamp))

    # --- TAR (supports tar, tar.gz, tar.bz2, tar.xz) ---
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            memberlist = tar.getmembers()
            memberlist.sort(key=str_sort_key)
            n_members = len(memberlist)
            members_names_list = [m.name for m in memberlist]

            for member in memberlist:
                if re.match(channel_names_regex, member.name):
                    try:
                        cn_handler = tar.extractfile(member)
                        if cn_handler is not None:
                            cn_text = cn_handler.read().decode('utf-8')
                            channel_names = [line.strip() for line in cn_text.strip().splitlines() if line.strip()]
                    except:
                        logging.warning(f"Invalid channel names in file: {member}")
                    finally:
                        break

            for member in tqdm( memberlist, leave=True, desc= "Tar file iterating over archive members", total=n_members):
                if re.match(sample_rate_regex, member.name):
                    try:
                        sample_rate = tar.extractfile(member).read().strip()
                        sample_rate = int(sample_rate)

                        accapted.set_sample_rate(sample_rate)
                        rejected.set_sample_rate(sample_rate)
                    except:
                        logging.warning(f"Invalid sample rate in file: {member}")
                    continue

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
                            accapted.append(RawSignal(data, class_name, channel_names=channel_names, timestamp=object_timestamp))
                        else:
                            rejected.append(RawSignal(data, class_name, channel_names=channel_names, timestamp=object_timestamp))

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
    """Reads raw signals from the directory structure.
    Return tuple of accepted and rejected signals
    """
    sample_rate_file_path  = os.path.join(input_dir, "sample_rate.txt")
    if os.path.exists(sample_rate_file_path):
        with open(sample_rate_file_path, "r") as file:
            try:
                sample_rate = int(file.read().strip())
            except:
                logging.warning(f"Invalid sample rate in file: {sample_rate_file_path}")

    channel_names = None
    channel_names_file_path = os.path.join(input_dir, "channel_names.txt")
    if os.path.exists(channel_names_file_path):
        with open(channel_names_file_path, "r") as file:
            channel_names = [line.strip() for line in file.readlines() if line.strip()]

    accepted = _read_signals_from_dirs_internal(
        input_dir,
        sample_rate,
        n_jobs=n_jobs,
        parallel_options=parallel_options,
        dir_sorting_key=dir_sorting_key,
        dtype=dtype,
        channel_names=channel_names,
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
            channel_names=channel_names,
        )
    else:
        rejected = None

    return {"accepted": accepted, "rejected": rejected}


def _read_class_dir(class_dir, file_order_key=str_sort_key, dtype=np.double, channel_names=None):
    """Read objects from class-specific directory
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

        signal_objects.append(RawSignal(data, class_name, channel_names=channel_names, timestamp=object_timestamp))

    return signal_objects


def _read_signals_from_dirs_internal(
    input_dir,
    sample_rate=1000,
    n_jobs=-1,
    parallel_options=dict(),
    dir_sorting_key=lambda x: str(x),
    file_order_key=str_sort_key,
    dtype=np.double,
    channel_names=None,
):
    """Read the raw dataset from the directory structure.
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
            os.path.join(input_dir, directory), file_order_key, dtype=dtype, channel_names=channel_names
        )
        for directory in sorted_class_dirs
    )
    for class_data_obj in class_data_objects:
        data_objects += class_data_obj

    return data_objects


def save_signals_to_dirs(raw_signals: RawSignals, output_directory):
    """Save raw signals to a directory structure."""
    os.makedirs(output_directory, exist_ok=True)

    signal_labels = raw_signals.get_labels()
    unique_labels = set(signal_labels)
    fs_file_path = os.path.join(output_directory, "sample_rate.txt")
    with open(fs_file_path, "w") as file:
        print(raw_signals.get_sample_rate(), file=file)

    if len(raw_signals) > 0:
        channel_names = raw_signals[0].channel_names
        cn_file_path = os.path.join(output_directory, "channel_names.txt")
        with open(cn_file_path, "w") as file:
            for name in channel_names:
                print(name, file=file)


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


def save_signals_to_archive(raw_signals: RawSignals, archive_path):
    """Save raw signals to a ZIP archive."""
    import io

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        # Determine root prefix from archive filename
        archive_basename = os.path.splitext(os.path.basename(archive_path))[0]
        root_prefix = archive_basename + "/"

        # Write sample_rate.txt
        z.writestr(root_prefix + "sample_rate.txt", str(raw_signals.get_sample_rate()) + "\n")

        # Write channel_names.txt
        if len(raw_signals) > 0:
            channel_names = raw_signals[0].channel_names
            cn_text = "\n".join(channel_names) + "\n"
            z.writestr(root_prefix + "channel_names.txt", cn_text)

        signal_labels = raw_signals.get_labels()
        unique_labels = set(signal_labels)

        for label in unique_labels:
            label_str = str(label)
            label_indices = [
                i for i in range(len(raw_signals)) if raw_signals[i].object_class == label
            ]
            signal_label_subset = raw_signals[label_indices]

            subset_signal_indices_string = sorted(
                ["{}".format(i) for i in range(1, len(signal_label_subset) + 1)]
            )

            cnt = 0
            for istr in subset_signal_indices_string:
                csv_path = f"{root_prefix}{label_str}/{istr}.csv"
                signal_np = signal_label_subset[cnt].signal
                signal_df = pd.DataFrame(signal_np)
                buf = io.StringIO()
                signal_df.to_csv(buf, sep=";", header=False, index=False, decimal=",")
                z.writestr(csv_path, buf.getvalue())

                dat_path = f"{root_prefix}{label_str}/{istr}.dat"
                date_object = datetime.datetime.fromtimestamp(
                    signal_label_subset[cnt].timestamp
                )
                date_string = date_object.strftime(date_format_string)
                z.writestr(dat_path, date_string + "\n")
                cnt += 1
