"""Retain native numerical JSON while adding Python provenance fields."""
import json
from pathlib import Path


def write_augmented_result(path, native_text, native_keys, result, added_keys):
    # The caller already parsed native_text with json.loads and added these
    # fields to result. If native output owns any such key, serialize the
    # merged object normally rather than emit ambiguous duplicate keys.
    if set(added_keys) & native_keys:
        Path(path).write_text(json.dumps(result, separators=(',', ':')) + '\n', encoding='utf-8')
        return
    suffix = json.dumps({key: result[key] for key in added_keys if key in result},
                        separators=(',', ':'))
    body = native_text.strip()
    if not body.startswith('{') or not body.endswith('}'):
        raise ValueError('native SCF result must be a JSON object')
    with Path(path).open('w', encoding='utf-8') as stream:
        stream.write(body[:-1])
        if len(native_keys) and suffix != '{}':
            stream.write(',')
        stream.write(suffix[1:-1])
        stream.write('}\n')
