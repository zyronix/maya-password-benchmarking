import os
import csv
import glob

from script.config.config import *
from script.utils.download_raw_data import main as download_raw_data
from script.utils.file_operations import write_to_csv
from script.test.tester import Tester

RESULTS_PATH = "results"

def download_dataset(name, path):
    print(f"[INFO] Dataset {name} not found. Attempting automatic download...")
    download_raw_data(chosen_datasets=[name], datasets_folder=path)


def _get_full_dataset_path(dataset):
    pattern = os.path.join("datasets", '**', dataset + '.pickle')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        download_dataset(dataset, "datasets")
        matches = glob.glob(pattern, recursive=True)

    path = matches[0] if matches else None
    return path if path and os.path.isfile(path) else None


def _run_tester(test_settings):
    tester = Tester(test_settings)
    tester.prepare_environment()
    tester.run_test()
    csv_rows = tester.written_rows
    return csv_rows

def _prepare_settings(test, model, dataset, setting_string, display_logs):
    others, n_samples = setting_string.split(os.sep)
    char_bag, max_length, train_chunk_percentage, train_split_percentage = others.split("-")
    args_settings = \
            {'models': model,
            'max_length': int(max_length),
            'train_datasets': dataset,
            'n_samples': int(n_samples),
            'test_config': os.path.join("config", "test", f"{test}.yaml"),
            'train_chunk_percentage': int(train_chunk_percentage),
            'train_split_percentage': int(train_split_percentage),
            'char_bag': inverse_char_bag_mapping[char_bag],
            'autoload': 1,
            'overwrite': 1,
            'display_logs': display_logs,
            }

    return build_args_settings(args_settings)

class Evaluator:
    """Base class for evaluators that compute metrics based on generated guesses and real datasets. 
    Subclasses should implement the _get_entries and _compute_metrics methods.
    
    real_data_mode in search_settings can be
    - test
    - train
    - full

    If set to "full", the evaluator will attempt to use the full real dataset for evaluation,
    If set to "test" or "train", it will use the corresponding split of the dataset. 

    
    mode in search_settings can be
    - guesses
    - matches

    """
    def __init__(self, test_settings, search_settings, csv_settings):
        self._prepare_settings(test_settings, search_settings, csv_settings)

        missing_entries = []
        if not self.test_settings["overwrite"]:
            missing_entries = self._get_entries()

        generated_paths, real_paths = self._get_paths(missing_entries)
        self._compute_metrics(generated_paths, real_paths)

    def _prepare_settings(self, test_settings, search_settings, csv_settings):
        self.written_rows = {}
        self.test_settings = test_settings
        self.search_settings = search_settings
        self.csv_settings = csv_settings

    def _get_entries(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _search_entries(self, searching_for):
        """
        This method checks which entries in the searching_for dictionary are already present in the corresponding CSV files.
        It updates the searching_for dictionary to mark found entries and collects missing entries.
        
        Parameters:
        searching_for (dict): A dictionary where keys are tuples of query parameters and values are flags
        
        Example:
        searching_for = {
            (('model', 'passgan'), ('train-dataset', 'rockyou'), ('test-settings', 'all-12-100-80'), ('n_samples', '500000000'))): 0
        }

        returns:
        missing_entries (list): A list of comma-separated strings representing the missing entries that need to
        be computed.

        """
        missing_entries = []

        test_name = self.csv_settings['test_name']
        csv_path = os.path.join(RESULTS_PATH, test_name, f"{test_name}.csv")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.isfile(csv_path):
            open(csv_path, 'w').close()
        else:
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for query in searching_for:
                        if all(row.get(k) == v or row.get(k) == 'NONE' for k, v in dict(query).items()):
                            searching_for[query] = 1
                            if csv_path not in self.written_rows:
                                self.written_rows[csv_path] = []
                            self.written_rows[csv_path].append(",".join(row[k] for k in row))

        for query in searching_for:
            if searching_for[query] == 0:
                query_dict = dict(query)
                row_str = ",".join(str(query_dict[k]) for k in query_dict)
                missing_entries.append(row_str)
        return missing_entries


    def _prepare_settings_strings(self):
        test_settings = [
            "-".join([
                char_bag_mapping[char_bag],
                str(max_length),
                str(train_chunk),
                str(train_split)
            ])
            for char_bag in self.test_settings['char_bag']
            for max_length in self.test_settings['max_length']
            for train_chunk in self.test_settings['train_chunk_percentage']
            for train_split in self.test_settings['train_split_percentage']
        ]

        settings_strings = [os.path.join(string, str(n_samples)) for string in test_settings for n_samples in
                            self.test_settings['n_samples']]
        return settings_strings

    def _get_real_dataset_path(self, dataset, hash=""):
        real_data_mode = self.search_settings["real_data_mode"]
        path = ""
        if real_data_mode in ["test", "test"]:
            path = os.path.join("data", "splitted", f"{real_data_mode}-{hash}.pickle")
        elif real_data_mode == "full":
            path = _get_full_dataset_path(dataset)
        return path

    def _run_missing_entry(self, test, model, dataset, setting_string):
        try:
            display_logs = self.test_settings.get("display_logs", 0)
            test_settings = _prepare_settings(test, model, dataset, setting_string, display_logs)
            _ = _run_tester(test_settings)
            return 1
        except Exception as e:
            return 0

    def _get_paths(self, missing_entries):
        """
        This method retrieves the paths to the generated guesses and real datasets for the entries specified in missing_entries.
        It returns two nested dictionaries: generated_paths and real_paths, organized by model, setting string, and dataset.

        #TODO: check if generated_path example matches

        Example:        generated_paths = {
            'passgan': {
                'all-12-100-80/500000000': {
                    'rockyou': 'results/rq8/passgan/rockyou/all-12-100-80/500000000/hash/guesses/guesses.gz'
                }
            }
        }
        """
        
        real_paths = {}
        generated_paths = {}

        models = [m.replace("-", "") for m in self.test_settings["models"]]
        datasets = self.test_settings["train_datasets"]
        test = self.test_settings["test"]
        mode = self.search_settings["mode"]
        real_data_mode = self.search_settings["real_data_mode"]

        setting_strings = self._prepare_settings_strings()

        for model in models:
            for dataset in datasets:
                for setting_string in setting_strings:
                    others, n_samples = setting_string.split(os.sep)
                    entry = ",".join([model, dataset, others, n_samples])

                    if self.test_settings["overwrite"] == 0 and entry not in missing_entries:
                        continue

                    if model == "real" and real_data_mode == "full":
                        real_path = self._get_real_dataset_path(dataset)
                        real_paths.setdefault(setting_string, {})[dataset] = real_path
                        continue

                    setting_path = os.path.join(RESULTS_PATH, test, model, dataset, setting_string)
                    if not os.path.exists(setting_path):
                        status = self._run_missing_entry(test, model, dataset, setting_string)
                        if not status:
                            continue

                    hash = [d for d in os.listdir(setting_path) if os.path.isdir(os.path.join(setting_path, d))][0]

                    match_file = os.path.join(setting_path, hash, mode, f"{mode}.gz")
                    if not os.path.exists(match_file):
                        status = self._run_missing_entry(test, model, dataset, setting_string)
                        if not status:
                            continue

                    if os.path.isfile(match_file):
                        generated_paths.setdefault(model,{}).setdefault(setting_string, {})[dataset] = match_file

                    if dataset not in real_paths.get(setting_string, {}):
                        real_path = self._get_real_dataset_path(dataset, hash)
                        if not real_path or not os.path.exists(real_path):
                            status = self._run_missing_entry(test, "NULL", dataset, setting_string)
                            if not status:
                                continue

                        real_paths.setdefault(setting_string, {})[dataset] = real_path

        return generated_paths, real_paths

    def _compute_metrics(self, generated_paths, real_paths):
        """This method should be implemented in the subclass to compute the desired metrics based on the generated guesses and real dataset paths.
        
        Args:
            generated_paths (dict): A nested dictionary containing paths to the generated guesses organized by model, setting string, and dataset.
            real_paths (dict): A nested dictionary containing paths to the real datasets organized by setting string and dataset.
        
        This method writes the results to self.written_rows, which is a dictionary mapping CSV file paths to lists of rows that should be written to those files. 
        Each row should be a comma-separated string of values corresponding to the fieldnames specified in self.csv_settings.
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def prepare_to_csv(self, model, dataset, test_settings, n_samples, variable_data):
        test_name = self.csv_settings["test_name"]
        fieldnames = self.csv_settings["fieldnames"]

        fixed_data = [model, dataset, test_settings, n_samples]

        output_path = os.path.join(RESULTS_PATH, test_name)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{test_name}.csv")

        rows = write_to_csv(path=output_file,
                            fieldnames=fieldnames,
                            fixed_data=fixed_data,
                            variable_data=variable_data)
        return output_file, rows