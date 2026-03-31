import os
import re
import pickle
import gzip
# from script.metrics.statistics.evaluator import Evaluator
import argparse
from collections.abc import Iterable, Sized
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
from zxcvbn import zxcvbn

from script.metrics.statistics.evaluator import Evaluator
from script.metrics.statistics.pattern_distribution_worker import (
    chunk_compute_pattern_distribution,
)


regex = {
    'r1': r'^[A-Za-z]+$',
    'r2': r'^[a-z]+$',
    'r3': r'^[A-Z]+$',
    'r4': r'^[0-9]+$',
    'r5': r'^[\W_]+$',
    'r6': r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]+$',
    'r7': r'^(?=.*[A-Za-z])(?=.*[\W_])[A-Za-z\W_]+$',
    'r8': r'^(?=.*\d)(?=.*[\W_])[\d\W_]+$',
    'r9': r'^(?=.*\d)(?=.*[\W_])(?=.*[A-Za-z])[A-Za-z\d\W_]+$',
    'r10': r'^[a-zA-Z][a-zA-Z0-9\W_]+[0-9]$',
    'r11': r'^[A-Za-z][A-Za-z0-9\W_]+[\W_]$',
    'r12': r'^[0-9][A-Za-z]+$',
    'r13': r'^[0-9][A-Za-z0-9\W_]+[\W_]$',
    'r14': r'^[0-9][A-Za-z0-9\W_]+[0-9]$',
    'r15': r'^[\W_][A-Za-z]+$',
    'r16': r'^[\W_][A-Za-z0-9\W_]+[\W_]$',
    'r17': r'^[\W_][A-Za-z0-9\W_]+[0-9]$',
    'r18': r'^[a-zA-Z0-9\W_]+[!]$',
    'r19': r'^[a-zA-Z0-9\W_]+[1]$',
}

compiled_regex = {k: re.compile(v) for k, v in regex.items()}
DEFAULT_CHUNK_SIZE = 1000


def _decode_utf8_line(raw_line):
    try:
        return raw_line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None



def chunk_iterable(iterable: Iterable, chunk_size=DEFAULT_CHUNK_SIZE):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _iter_passwords(file_path):
    if file_path.endswith('.pickle'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            for line in data:
                password = line.strip()
                if password:
                    yield password
    else:
        with open(file_path, 'rb') as f:
            for raw_line in f:
                decoded = _decode_utf8_line(raw_line)
                if decoded:
                    yield decoded

def count_lines(file_path):
    if file_path.endswith('.pickle'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return len(data)
    else:
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f if _decode_utf8_line(_) is not None)


def read_chunk(file_path, chunk_size=DEFAULT_CHUNK_SIZE):
    yield from chunk_iterable(_iter_passwords(file_path), chunk_size=chunk_size)


def compute_pattern_distribution(file_path):
    print(f"Processing {file_path}")

    aggregated_distribution = {pattern: 0 for pattern in regex}
    aggregated_password_length_distribution = {}
    aggregated_zxcvbn_distribution = {}
    total_passwords = 0

    total_passwords_in_file = count_lines(file_path)

    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(desc="Processing passwords", unit="pwd", total=total_passwords_in_file) as pbar:
            for distribution, length_distribution, zxcvbn_distribution, passwords_in_chunk in executor.map(
                chunk_compute_pattern_distribution,
                read_chunk(file_path),
            ):
                # Aggregate pattern distribution
                for pattern, count in distribution.items():
                    aggregated_distribution[pattern] += count

                # Aggregate length distribution
                for length, count in length_distribution.items():
                    if aggregated_password_length_distribution.get(length) is None:
                        aggregated_password_length_distribution[length] = 0
                    aggregated_password_length_distribution[length] += count

                # Aggregate zxcvbn score distribution
                for score, count in zxcvbn_distribution.items():
                    if aggregated_zxcvbn_distribution.get(score) is None:
                        aggregated_zxcvbn_distribution[score] = 0
                    aggregated_zxcvbn_distribution[score] += count

                total_passwords += passwords_in_chunk
                pbar.update(passwords_in_chunk)

    stats = {}
    # Print results
    print(f"Total passwords: {total_passwords}")
    stats['total_passwords'] = [total_passwords, 100.0]
    print("Pattern distribution:")
    for pattern, count in sorted(aggregated_distribution.items()):
        percentage = (count / total_passwords * 100) if total_passwords > 0 else 0
        stats[pattern] = [count, f"{percentage:.2f}"]
        print(f"  {pattern}: {count} ({percentage:.2f}%)")
    print("Password length distribution:")
    for length, count in sorted(aggregated_password_length_distribution.items()):
        percentage = (count / total_passwords * 100) if total_passwords > 0 else 0
        print(f"  Length {length}: {count} ({percentage:.2f}%)")
        stats[f'length_{length}'] = [count, f"{percentage:.2f}"]
    average_password_length = sum(length * count for length, count in aggregated_password_length_distribution.items()) / total_passwords if total_passwords > 0 else 0
    print(f"Average password length: {average_password_length:.2f}")
    print("zxcvbn score distribution:")
    for score, count in sorted(aggregated_zxcvbn_distribution.items()):
        percentage = (count / total_passwords * 100) if total_passwords > 0 else 0
        print(f"  Score {score}: {count} ({percentage:.2f}%)")
        stats[f'zxcvbn_score_{score}'] = [count, f"{percentage:.2f}"]
    average_zxcvbn_score = sum(score * count for score, count in aggregated_zxcvbn_distribution.items()) / total_passwords if total_passwords > 0 else 0
    print(f"Average zxcvbn score: {average_zxcvbn_score:.2f}")
    stats['average_password_length'] = [f"{average_password_length:.2f}", 100.0]
    stats['average_zxcvbn_score'] = [f"{average_zxcvbn_score:.2f}", 100.0]
    return stats


class RQ8_Evaluator(Evaluator):
    def __init__(self, test_settings, search_settings, csv_settings):
        super().__init__(test_settings, search_settings, csv_settings)

    def _get_entries(self):
        searching_for = {}
        setting_strings = self._prepare_settings_strings()
        models = self.test_settings["models"]
        datasets = self.test_settings["train_datasets"]

        if self.search_settings['real_data_mode'] != "":
            models.append('real')

        for model in models:
            for dataset in datasets:
                for setting_string in setting_strings:
                    test_settings, n_samples = setting_string.split(os.sep)
                    query = {
                        'model': model,
                        'train-dataset': dataset,
                        'test-settings': test_settings,
                        'n_samples': n_samples,
                    }
                    key = tuple(query.items())
                    searching_for.setdefault(key, 0)

        return self._search_entries(searching_for)

    def _compute_metrics(self, guesses_paths, real_paths):
        guesses_paths['real'] = real_paths

        for model in guesses_paths:
            for setting_string in guesses_paths[model]:
                for dataset in guesses_paths[model][setting_string]:

                    file_path = guesses_paths[model][setting_string][dataset]

                    stats = compute_pattern_distribution(file_path)

                    test_settings, n_samples = setting_string.split(os.sep)
                    variable_data = []
                    for key in stats:
                        variable_data.append([key, stats[key][0], stats[key][1]])

                    path, rows = self.prepare_to_csv(model, dataset, test_settings, n_samples, variable_data)

                    if path not in self.written_rows:
                        self.written_rows[path] = []
                    for row in rows:
                        self.written_rows[path].append(row)


def main(test_settings):
    search_settings = {
        'mode': "guesses",
        'real_data_mode': "full",
    }

    csv_settings = {
        'test_name': 'rq8',
        'fieldnames' : ["model", "train-dataset", "test-settings", "n_samples", "pattern", "matches", "match_percentage"]
    }
    evaluator = RQ8_Evaluator(test_settings, search_settings, csv_settings)
    return evaluator.written_rows

    parser = argparse.ArgumentParser(description="Password statistics analyzer")
    parser.add_argument("--input", required=True, help="Path to password file")
    args = parser.parse_args()
    
if __name__ == "__main__":
    main()