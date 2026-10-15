"""Export original UPF PAW arrays for the explicit native UPF operator branch.

This container preserves source identity and numeric units. It is deliberately
incompatible with SDFTPAW1/POTCAR caches. Export does not certify an SCF.
"""
from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

from .upf import read_upf_paw

MAGIC = b'SDFTUPF1'


def encode_native_upf(source):
    dataset = read_upf_paw(source)
    if dataset.functional != 'PBE' or dataset.relativistic != 'scalar' or dataset.summary()['has_soc']:
        raise ValueError('native UPF operator development currently requires scalar PBE PAW')
    arrays, raw_values, sections = [], bytearray(), bytearray()
    def string(value):
        raw=value.encode()
        return struct.pack('<I',len(raw))+raw
    for name, section in sorted(dataset.arrays.items()):
        values = array('d', section.values)
        if sys.byteorder != 'little': values.byteswap()
        arrays.append(dict(name=name, attributes=section.attributes,
                           offset_values=len(raw_values)//8, count=len(values)))
        payload=values.tobytes()
        raw_values.extend(payload)
        sections.extend(string(name)+struct.pack('<Q',len(values))+
                        hashlib.sha256(payload).digest()+payload)
    tree=ET.fromstring(dataset.raw_xml)
    aug=tree.find('PP_NONLOCAL/PP_AUGMENTATION')
    n=int(dataset.header['number_of_proj'])
    projectors=[]
    for i in range(n):
        attributes=dataset.arrays[f'UPF/PP_NONLOCAL/PP_BETA.{i+1}'].attributes
        projectors.append((int(attributes['angular_momentum']),int(attributes['cutoff_radius_index'])))
    original_core_energy=tree.find('PP_PAW').get('core_energy')
    core_energy=float(original_core_energy.replace('D','E').replace('d','e'))
    metadata = dict(schema_version='sparse-dft-native-upf-1', header=dataset.header,
                    source_sha256=dataset.source_sha256, source_path=dataset.source_path,
                    units=dataset.summary()['units'], arrays=arrays,
                    numeric_payload_sha256=hashlib.sha256(raw_values).hexdigest(),
                    original_xml_bytes=len(dataset.raw_xml), native_scf_validated=False)
    chi = []
    for node in tree.findall('PP_PSWFC/*'):
        if node.tag.startswith('PP_CHI.'):
            chi.append((int(node.tag.rsplit('.', 1)[1]), int(node.get('l')),
                        float(node.get('occupation').replace('D', 'E').replace('d', 'e'))))
    chi.sort()
    if [x[0] for x in chi] != list(range(1, len(chi)+1)):
        raise ValueError('original UPF atomic-wavefunction indices must be consecutive')
    metadata['schema_version'] = 'sparse-dft-native-upf-2'
    metadata['atomic_wavefunctions'] = [{'index': i, 'l': l, 'occupation': o} for i, l, o in chi]
    encoded_metadata = json.dumps(metadata, separators=(',', ':'), sort_keys=True).encode()
    # Original XML remains attached, so future readers can verify that the
    # typed arrays were derived from the declared original dataset.
    return (MAGIC + struct.pack('<I',2)+string(dataset.element)+string(dataset.functional)+
            string(dataset.relativistic)+bytes.fromhex(dataset.source_sha256)+
            struct.pack('<dddII',float(dataset.header['z_valence']),core_energy,
                        float(aug.get('augmentation_epsilon')),int(aug.get('cutoff_r_index')),n)+
            b''.join(struct.pack('<II',l,k) for l,k in projectors)+
            struct.pack('<I',len(chi))+b''.join(struct.pack('<Id',l,o) for _,l,o in chi)+
            struct.pack('<I',len(arrays))+sections+
            struct.pack('<Q',len(encoded_metadata))+encoded_metadata+
            struct.pack('<Q',len(dataset.raw_xml))+dataset.raw_xml)


def export_native_upf(source, destination):
    destination = Path(destination)
    payload = encode_native_upf(source)
    with destination.open('xb') as stream: stream.write(payload)
    return dict(path=str(destination.resolve()), bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), native_scf_validated=False)
