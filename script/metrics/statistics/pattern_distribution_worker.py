import re
import pickle
from zxcvbn import zxcvbn


REGEX = {
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

COMPILED_REGEX = {k: re.compile(v) for k, v in REGEX.items()}


def chunk_compute_pattern_distribution(chunk):
    distribution = {pattern: 0 for pattern in REGEX}
    length_distribution = {}
    zxcvbn_distribution = {}
    total_passwords = 0

    for password in chunk:
        if not password:
            continue

        password = password.rstrip()
        total_passwords += 1

        password_length = len(password)
        length_distribution[password_length] = length_distribution.get(password_length, 0) + 1

        zxcvbn_score = zxcvbn(password)['score']
        zxcvbn_distribution[zxcvbn_score] = zxcvbn_distribution.get(zxcvbn_score, 0) + 1

        for pattern_id, compiled in COMPILED_REGEX.items():
            if compiled.fullmatch(password):
                distribution[pattern_id] += 1

    return distribution, length_distribution, zxcvbn_distribution, total_passwords
