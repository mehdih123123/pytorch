import argparse
import json
import pickle
from os import listdir
from os.path import isdir, isfile, join

import numpy as np
import tqdm

from inductor_autotuner.util import kernel_iter
from torch._inductor.autotuner.model import AutotunerModel, ModelType
from triton import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to data dir, containing kernel meta data and configs",
)
parser.add_argument(
    "--model_type",
    type=int,
    required=True,
    help="Model type, which is consistent with ModelType in pytorch/torch/_inductor/autotuner/model.py",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to output directory for data"
)
parser.add_argument("--train", type=float, default=0.8, help="Train/test split")
parser.add_argument(
    "--pairwise-threshold",
    type=float,
    default=10,
    help="""When generating data for pairwise loss models, we need to sample pairs. """
    """This threshold controls how many pairs we sample for each config at most.""",
)
parser.add_argument("--seed", type=int, default=0)


def get_baseline_config_num(logpath: str) -> int:
    with open(logpath) as file:
        line = file.readlines()[0]
        startpos = line.find("CachingAutotuner gets ")
        assert startpos != -1
        endpos = line.find(" configs", startpos)
        return int(line[startpos + len("CachingAutotuner gets ") : endpos])


def main(args):
    data_dir = args.data_dir
    model_type = ModelType(args.model_type)
    output_dir = args.output_dir
    train_split = args.train
    assert 0 <= train_split <= 1

    autotuner_model = AutotunerModel(model_type)
    kernel_counter = 0
    seen_kernels = set()

    x_all = list()
    y_all = list()
    y_baseline_all = list()
    y_normalized_all = list()
    qid_all = list()

    assert isdir(data_dir)
    # Gather all the data
    print("Gathering data...")
    for model in tqdm.tqdm(sorted(listdir(data_dir))):
        model_path = join(data_dir, model)
        if not isdir(model_path):
            continue

        for kernel, py, kernel_path, py_path in kernel_iter(model_path, verbose=False):
            kernel_name = py[:-3]
            if kernel_name in seen_kernels:
                continue

            with open(py_path) as file:
                src = file.read()

            seen_kernels.add(kernel_name)
            log_path = join(kernel_path, kernel_name + ".log")
            pkl_path = join(kernel_path, py + ".pkl")
            all_config_path = join(kernel_path, kernel_name + ".all_config")

            # Sanith check, make sure log, pkl, and all_config exist
            # assert isfile(log_path), log_path
            # assert isfile(pkl_path), pkl_path
            # assert isfile(all_config_path), all_config_path
            if not isfile(log_path):
                print("Missing log file: " + log_path)
                continue
            if not isfile(pkl_path):
                print("Missing pkl file: " + pkl_path)
                continue
            if not isfile(all_config_path):
                print("Missing all_config file: " + all_config_path)
                continue

            # Read the raw data
            autotuner_raw_data = pickle.load(open(pkl_path, "rb"))
            src_code = autotuner_raw_data.src_code

            # Sanity check, making sure the metadata is correct
            src_code = src_code.replace("KERNEL_NAME", "triton_")
            assert src_code == src

            # Get the baseline timing (best max autotune) from log
            baseline_config_num = get_baseline_config_num(log_path)
            baseline_timing = 1e6
            with open(all_config_path) as file:
                all_configs = json.load(file)
                for config in all_configs[:baseline_config_num]:
                    baseline_timing = min(baseline_timing, config["timing"])

            # Read all the configs
            config_list = list()
            y_list = list()
            with open(all_config_path) as file:
                all_config = json.load(file)
                # use only max autotune configs for NN_PAIRWISE_SMALL
                for config in (
                    all_config[:baseline_config_num]
                    if model_type == ModelType.NN_PAIRWISE_SMALL
                    else all_config
                ):
                    num_warps = config.pop("num_warps")
                    num_stages = config.pop("num_stages")
                    timing = config.pop("timing")
                    # n_regs, shared, n_spills are only available after compilation
                    # while we want to predict before compilation
                    config.pop("n_regs")
                    config.pop("shared")
                    config.pop("n_spills")
                    config_list.append(
                        Config(
                            kwargs=config,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
                    y_list.append(timing)

            # normalize y to (0, 1]
            y_best = min(y_list)
            y_normalized_list = [y_best / y_ for y_ in y_list]

            # Get feature vector
            x_list = autotuner_model.get_feature_vec(config_list, autotuner_raw_data)

            x_all.extend(x_list)
            y_all.extend(y_list)
            y_normalized_all.extend(y_normalized_list)
            y_baseline_all.extend([baseline_timing] * len(y_list))
            qid_all.extend([kernel_counter] * len(y_list))
            kernel_counter += 1

    # Split the data by qid
    # Note that X might not be np.array, so using array as indices might not work
    # which is also why many of the following code is not so numpy style
    assert len(x_all) == len(y_all) == len(qid_all)
    np.random.seed(args.seed)
    qid_unique = np.random.permutation(np.unique(qid_all))
    qid_train_unique = qid_unique[: int(len(qid_unique) * train_split)]
    qid_test_unique = qid_unique[int(len(qid_unique) * train_split) :]

    def get_data_from_qid(qid_set):
        qid_set = set(qid_set)
        x = list()
        y = list()
        y_baseline = list()
        y_normalized = list()
        qid = list()
        for i in tqdm.tqdm(range(len(qid_all))):
            if qid_all[i] in qid_set:
                x.append(x_all[i])
                y.append(y_all[i])
                y_baseline.append(y_baseline_all[i])
                y_normalized.append(y_normalized_all[i])
                qid.append(qid_all[i])

        # Sanity check: qid is in non-descending order
        for i in range(1, len(qid)):
            assert qid[i] >= qid[i - 1]

        return x, y, y_baseline, y_normalized, qid

    print("Generating training data...")
    (
        x_train,
        y_train,
        y_baseline_train,
        y_normalized_train,
        qid_train,
    ) = get_data_from_qid(qid_train_unique)
    print("Generating testing data...")
    x_test, y_test, y_baseline_test, y_normalized_test, qid_test = get_data_from_qid(
        qid_test_unique
    )

    if model_type in [
        ModelType.NN_POINTWISE,
        ModelType.NN_PAIRWISE,
        ModelType.NN_PAIRWISE_SMALL,
    ]:
        print("Getting training feature groups...")
        x_train = autotuner_model.model.get_feature_groups(x_train, show_progress=True)
        x_train = tuple(X_group.to("cpu") for X_group in x_train)
        print("Getting testing feature groups...")
        x_test = autotuner_model.model.get_feature_groups(x_test, show_progress=True)
        x_test = tuple(X_group.to("cpu") for X_group in x_test)

    def save(file_name, obj):
        print("Saving " + file_name + "...", len(obj))
        with open(join(output_dir, file_name), "wb") as file:
            pickle.dump(obj, file)

    # Save the data
    save("X_train.pkl", x_train)
    save("y_train.pkl", y_train)
    save("y_baseline_train.pkl", y_baseline_train)
    save("y_normalized_train.pkl", y_normalized_train)
    save("qid_train.pkl", qid_train)
    save("X_test.pkl", x_test)
    save("y_test.pkl", y_test)
    save("y_baseline_test.pkl", y_baseline_test)
    save("y_normalized_test.pkl", y_normalized_test)
    save("qid_test.pkl", qid_test)

    # if pairwise model, save the pairwise data
    if model_type in [ModelType.NN_PAIRWISE, ModelType.NN_PAIRWISE_SMALL]:
        threshold = args.pairwise_threshold

        def get_pairwise_data(X, y, qid):
            pointer = 0
            x_pairwise = list(), list()
            y_pairwise = list()

            for id in tqdm.tqdm(np.unique(np.array(qid))):
                x_group = list()
                y_group = list()
                while pointer < len(qid) and qid[pointer] == id:
                    x_group.append(pointer)
                    y_group.append(y[pointer] if not np.isinf(y[pointer]) else 1e6)
                    assert np.max(np.array(y_group)) <= 1e6, y_group
                    pointer += 1
                dedup = dict()
                for i in range(len(x_group)):
                    random_indices = np.random.permutation(len(x_group))
                    counter = 0
                    for j in random_indices:
                        counter += 1
                        if counter > threshold:
                            break
                        if (i, j) in dedup or (j, i) in dedup:
                            continue
                        if y_group[i] > y_group[j]:
                            x_pairwise[0].append(x_group[i])
                            x_pairwise[1].append(x_group[j])
                            y_pairwise.append(y_group[i] - y_group[j])
                        elif y_group[i] < y_group[j]:
                            x_pairwise[0].append(x_group[j])
                            x_pairwise[1].append(x_group[i])
                            y_pairwise.append(y_group[j] - y_group[i])
                        dedup[(i, j)] = True
                        dedup[(j, i)] = True

            x_pairwise = tuple(np.array(X_side) for X_side in x_pairwise)
            x_pairwise = (
                tuple(xg[x_pairwise[0]] for xg in X),
                tuple(xg[x_pairwise[1]] for xg in X),
            )
            return x_pairwise, y_pairwise

        print("Generating pairwise training data...")
        X_pairwise_train, y_pairwise_train = get_pairwise_data(
            x_train, y_train, qid_train
        )
        print("Generating pairwise testing data...")
        X_pairwise_test, y_pairwise_test = get_pairwise_data(x_test, y_test, qid_test)
        save("X_pairwise_train.pkl", X_pairwise_train)
        save("y_pairwise_train.pkl", y_pairwise_train)
        save("X_pairwise_test.pkl", X_pairwise_test)
        save("y_pairwise_test.pkl", y_pairwise_test)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)