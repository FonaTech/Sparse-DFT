import json
from pathlib import Path
import tempfile
import unittest

from sparse_dft.result_io import write_augmented_result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError('duplicate JSON key: ' + key)
        result[key] = value
    return result


class ResultIOTests(unittest.TestCase):
    def write_and_check(self, text, additions):
        native = json.loads(text)
        merged = {**native, **additions}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scf_result.json'
            write_augmented_result(path, text, set(native), merged, tuple(additions))
            self.assertEqual(json.loads(path.read_text(), object_pairs_hook=unique_object), merged)

    def test_numerical_payload_is_preserved(self):
        self.write_and_check(' {"density_q":[{"h":-3,"value_real":1.234567890123456e-210,'
                             '"value_imag":-1.234567890123456e200}],"onecenter_states":[],"converged":true} ',
                             {'metadata': {'reciprocal_basis': [[1., 0., 0.]]},
                              'provenance': {'path': '轨道/"data"\\file'}, 'structure_hash': 'abc'})

    def test_native_metadata_is_merged_without_duplicates(self):
        self.write_and_check('{"metadata":{"native":true},"converged":false}',
                             {'metadata': {'native': True, 'python': True}})

    def test_empty_native_object(self):
        self.write_and_check('{}', {'provenance': {'engine': 'Sparse-DFT'}})

    def test_no_new_fields(self):
        self.write_and_check('{"density_q":[]}', {})

    def test_empty_object_without_additions(self):
        self.write_and_check('{}', {})
