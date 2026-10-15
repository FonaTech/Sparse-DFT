import copy
import hashlib
import json
import lzma
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sparse_dft.paw import Paw54JsonArchive
from sparse_dft.errors import ArchiveIntegrityError


class IndexedPawSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        for variant,family in [('standard','standard'),('sv','semicore'),('GW','gw')]:
            text = 'PAW_PBE Si '+variant+'\n'
            self.records.append({'id':'Si:'+variant, 'element':{'symbol':'Si'},
                'variant':{'name':variant,'family':family,'tokens':[]},
                'metadata':{'potcar':{'title':text.strip(),'exchange_correlation':'PE',
                    'valence_electrons':4,'enmax_ev':245.345,'enmin_ev':184.,'eaug':322.}},
                'files':{'POTCAR':{'text':text,'sha256':hashlib.sha256(text.encode('ascii')).hexdigest(),
                    'size_bytes':len(text)},'PSCTR':{'text':'','empty':True,'size_bytes':0,
                    'sha256':hashlib.sha256(b'').hexdigest()}}})
        index = copy.deepcopy(self.records)
        for record in index:
            for file in record['files'].values():file.pop('text')
        (self.root/'manifest.json').write_text(json.dumps({'schema_version':'1.0',
            'format':'VASP PAW JSONL','record_count':3}))
        (self.root/'index.json').write_text(json.dumps({'records':index,
            'by_element':{'Si':[r['id'] for r in index]}}))
        self.write_records()

    def write_records(self):
        with lzma.open(self.root/'potentials.jsonl.xz','wt',encoding='utf-8') as stream:
            for record in self.records:stream.write(json.dumps(record)+'\n')

    def test_selection_reads_no_numerical_text_and_respects_family(self):
        archive = Paw54JsonArchive(self.root)
        with patch.object(archive,'iter_records',side_effect=AssertionError('unexpected decompression')):
            self.assertEqual([r.record_id for r in archive.candidates('Si')],['Si:standard'])
            self.assertEqual([r.record_id for r in archive.candidates('Si',family='semicore')],['Si:sv'])
            self.assertEqual([r.record_id for r in archive.candidates('Si',family='gw')],['Si:GW'])
        self.assertEqual(archive.restore('Si:standard','PSCTR',self.root/'empty').read_bytes(),b'')

    def test_actual_record_metadata_must_match_selection_index(self):
        self.records[0]['metadata']['potcar']['valence_electrons']=99
        self.write_records()
        with self.assertRaisesRegex(ArchiveIntegrityError,'metadata mismatch'):
            Paw54JsonArchive(self.root).get_record('Si:standard')

    def test_actual_file_hash_must_match_index_and_restored_bytes(self):
        self.records[0]['files']['POTCAR']['sha256']='0'*64
        self.write_records()
        with self.assertRaisesRegex(ArchiveIntegrityError,'file mismatch'):
            Paw54JsonArchive(self.root).restore('Si:standard','POTCAR',self.root/'bad')
        self.records[0]['files']['POTCAR']['sha256']=hashlib.sha256(
            self.records[0]['files']['POTCAR']['text'].encode('ascii')).hexdigest()
        self.records[0]['files']['POTCAR']['text']='corrupt'
        self.write_records()
        with self.assertRaisesRegex(ArchiveIntegrityError,'hash mismatch'):
            Paw54JsonArchive(self.root).restore('Si:standard','POTCAR',self.root/'bad')

    def test_corrupt_element_lookup_cannot_select_foreign_record(self):
        index=json.loads((self.root/'index.json').read_text())
        index['by_element']['O']=['Si:standard']
        (self.root/'index.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ArchiveIntegrityError,'by_element'):
            Paw54JsonArchive(self.root).candidates('O')


if __name__=='__main__':unittest.main()
