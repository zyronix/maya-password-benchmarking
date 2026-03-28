import os
import re
import pickle
import gzip
# from script.metrics.statistics.evaluator import Evaluator
import argparse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm
from zxcvbn import zxcvbn


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
DEFAULT_CHUNK_SIZE = 20480


def _decode_utf8_line(raw_line):
    try:
        return raw_line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None


def count_passwords(file):
    if file.endswith('.pickle'):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return len(data)
    else:
        valid_lines = 0
        with open(file, 'rb') as f:
            for raw_line in f:
                if _decode_utf8_line(raw_line) is not None:
                    valid_lines += 1

        return valid_lines


def read_chunk(file, chunk_size=DEFAULT_CHUNK_SIZE):
    print('Reading file')

    chunk = []
    if file.endswith('.pickle'):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            for line in data:
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = [] 
            if chunk:
                print(f"Yielding final chunk of size {len(chunk)}") 
                yield chunk
    else:
        with open(file, "rb") as f:
            for raw_line in f:

                decoded = raw_line.decode("utf-8", errors="ignore")

                # if bytes were dropped → invalid UTF-8 existed
                if len(decoded) != len(raw_line):
                    continue

                chunk.append(decoded.strip())

                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

            if chunk:
                print(f"Yielding final chunk of size {len(chunk)}") 
                yield chunk
        


def _chunk_compute_pattern_distribution(chunk):
    distribution = {}
    length_distribution = {}
    zxcvbn_distribution = {}
    for pattern in regex:
        distribution[pattern] = 0

    total_passwords = 0

    for password in chunk:
        if not password:
            continue

        password = password.rstrip()
        total_passwords += 1

        # Length distribution
        password_length = len(password)
        if length_distribution.get(password_length) is None:
            length_distribution[password_length] = 0
        length_distribution[password_length] += 1

        # zxcvbn score distribution
        # Score ranges from 0 to 4, where 0 is the weakest and 4 is the strongest
        zxcvbn_score = zxcvbn(password)['score']
        if zxcvbn_distribution.get(zxcvbn_score) is None:
            zxcvbn_distribution[zxcvbn_score] = 0
        zxcvbn_distribution[zxcvbn_score] += 1

        # Pattern distribution over regex patterns
        for id, pattern in compiled_regex.items():
            if pattern.fullmatch(password):
                distribution[id] += 1

    return distribution, length_distribution, zxcvbn_distribution, total_passwords
 

def main():
    parser = argparse.ArgumentParser(description="Password statistics analyzer")
    parser.add_argument("--input", required=True, help="Path to password file")
    args = parser.parse_args()
    total_passwords = count_passwords(args.input)

    aggregated_distribution = {pattern: 0 for pattern in regex}
    aggregated_password_length_distribution = {}
    aggregated_zxcvbn_distribution = {}
    # total_passwords = 0


    with ThreadPoolExecutor(max_workers=3) as executor:
        with tqdm(total=total_passwords, desc="Processing passwords") as pbar:
            for chunk in read_chunk(args.input):
                distribution, length_distribution, zxcvbn_distribution, passwords_in_chunk = executor.submit(_chunk_compute_pattern_distribution, chunk).result()
                # Aggregate results
                # (This part can be implemented to combine the distributions and total_password counts from each chunk
                for pattern, count in distribution.items():
                    aggregated_distribution[pattern] += count
                
                # Aggregate length distribution
                for length, count in length_distribution.items():
                    if aggregated_password_length_distribution.get(length) is None:
                        aggregated_password_length_distribution[length] = 0
                    aggregated_password_length_distribution[length] += count
                # Aggregate zxcvbn distribution
                for score, count in zxcvbn_distribution.items():
                    if aggregated_zxcvbn_distribution.get(score) is None:
                        aggregated_zxcvbn_distribution[score] = 0
                    aggregated_zxcvbn_distribution[score] += count
                pbar.update(passwords_in_chunk)

    
    # Print results
    print(f"Total passwords: {total_passwords}")
    print("Pattern distribution:")
    for pattern, count in sorted(aggregated_distribution.items()):
        percentage = (count / total_passwords * 100) if total_passwords > 0 else 0
        print(f"  {pattern}: {count} ({percentage:.2f}%)")
    print("Password length distribution:")
    for length, count in sorted(aggregated_password_length_distribution.items()):
        percentage = (count / total_passwords * 100) if total_passwords > 0 else 0
        print(f"  Length {length}: {count} ({percentage:.2f}%)")
    average_password_length = sum(length * count for length, count in aggregated_password_length_distribution.items()) / total_passwords if total_passwords > 0 else 0
    print(f"Average password length: {average_password_length:.2f}")
    print("zxcvbn score distribution:")
    for score, count in sorted(aggregated_zxcvbn_distribution.items()):
        percentage = (count / total_passwords * 100) if total_passwords > 0 else 0
        print(f"  Score {score}: {count} ({percentage:.2f}%)")
    average_zxcvbn_score = sum(score * count for score, count in aggregated_zxcvbn_distribution.items()) / total_passwords if total_passwords > 0 else 0
    print(f"Average zxcvbn score: {average_zxcvbn_score:.2f}")

if __name__ == "__main__":
    main()