"""Strict UPF 2 PAW data reader with explicit native-unit provenance.

Parsing a dataset does not certify its Hamiltonian or mixed-library physics.
The UPF radial augmentation functions are kept in their original convention,
instead of being replaced by the Bessel-pair compensation shapes used for POTCAR data.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .upf_storage import MAGIC, decode_upf


def _true(value: str | None) -> bool:
    return (value or '').strip().lower() in ('t', '.true.', 'true', '1')


@dataclass
class UpfArray:
    path: str
    attributes: dict[str, str]
    values: array


@dataclass
class UpfPawDataset:
    element: str
    functional: str
    relativistic: str
    header: dict[str, str]
    arrays: dict[str, UpfArray]
    raw_xml: bytes
    source_sha256: str
    source_path: str | None
    schema_version: str = 'sparse-dft-upf-paw-data-1'

    def summary(self):
        return {'schema_version': self.schema_version, 'element': self.element,
                'functional': self.functional, 'relativistic': self.relativistic,
                'source_sha256': self.source_sha256, 'source_path': self.source_path,
                'z_valence': float(self.header['z_valence']),
                'suggested_wfc_cutoff_ry': float(self.header['wfc_cutoff']),
                'suggested_rho_cutoff_ry': float(self.header['rho_cutoff']),
                'mesh_size': int(self.header['mesh_size']),
                'projector_count': int(self.header['number_of_proj']),
                'has_soc': _true(self.header.get('has_so')),
                'array_count': len(self.arrays),
                'numeric_value_count': sum(len(a.values) for a in self.arrays.values()),
                'units': {'r': 'Bohr', 'rab': 'dr/di in Bohr',
                          'local_potential': 'Ry', 'dij': 'Ry',
                          'core_energy': 'Ry', 'radial_wavefunctions': 'r*phi(r)',
                          'augmentation': 'original UPF QijL radial convention; no VASP shape substitution'},
                'engine_physics_validated': False,
                'mixed_library_physics_validated': False}


def read_upf_paw(source: bytes | str | Path) -> UpfPawDataset:
    path = None if isinstance(source, bytes) else str(Path(source).resolve())
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if raw.startswith(MAGIC):
        raw = decode_upf(raw)
    if len(raw) > 64 * 1024**2 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('unsupported UPF size or external XML entities')
    root = ET.fromstring(raw)
    if root.tag != 'UPF' or not root.get('version', '').startswith('2.'):
        raise ValueError('only UPF version 2 is supported')
    header_node = root.find('PP_HEADER')
    if header_node is None:
        raise ValueError('UPF header missing')
    header = dict(header_node.attrib)
    if not _true(header.get('is_paw')) or header.get('pseudo_type') != 'PAW':
        raise ValueError('dataset is not PAW; NC/US cannot be silently treated as PAW')
    for key in ('element', 'functional', 'mesh_size', 'number_of_proj', 'relativistic',
                'z_valence', 'wfc_cutoff', 'rho_cutoff'):
        if key not in header:
            raise ValueError('missing UPF header: ' + key)
    mesh, nproj = int(header['mesh_size']), int(header['number_of_proj'])
    if not 3 <= mesh <= 100000 or not 1 <= nproj <= 100:
        raise ValueError('invalid UPF mesh/projector dimensions')
    arrays = {}
    def visit(node, prefix):
        current = prefix + '/' + node.tag if prefix else node.tag
        if list(node):
            for child in node:
                visit(child, current)
            return
        if (node.tag.startswith('PP_') and node.tag not in ('PP_INFO', 'PP_INPUTFILE', 'PP_HEADER')
                and (node.text or '').strip()):
            try:
                values = array('d', (float(t.replace('D', 'E').replace('d', 'e')) for t in node.text.split()))
            except ValueError as error:
                raise ValueError('non-numeric UPF array: ' + current) from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError('nonfinite UPF values: ' + current)
            if 'size' in node.attrib and int(node.get('size')) != len(values):
                raise ValueError('UPF declared size mismatch: ' + current)
            if current in arrays:
                raise ValueError('duplicate UPF section: ' + current)
            arrays[current] = UpfArray(current, dict(node.attrib), values)
    visit(root, '')
    def require(name, size=None):
        full = 'UPF/' + name
        if full not in arrays or size is not None and len(arrays[full].values) != size:
            raise ValueError('missing or incorrectly sized PAW array: ' + name)
        return arrays[full]
    radius = require('PP_MESH/PP_R', mesh).values
    jacobian = require('PP_MESH/PP_RAB', mesh).values
    if any(r < 0 for r in radius) or any(b <= a for a,b in zip(radius,radius[1:])) or any(x <= 0 for x in jacobian):
        raise ValueError('UPF radial mesh/Jacobian is not positive and monotonic')
    for name in ('PP_LOCAL', 'PP_RHOATOM', 'PP_PAW/PP_AE_NLCC', 'PP_PAW/PP_AE_VLOC'):
        require(name, mesh)
    if _true(header.get('core_correction')):
        require('PP_NLCC', mesh)
    dij = require('PP_NONLOCAL/PP_DIJ', nproj * nproj).values
    overlap = require('PP_NONLOCAL/PP_AUGMENTATION/PP_Q', nproj * nproj).values
    require('PP_PAW/PP_OCCUPATIONS', nproj)
    for matrix in (dij, overlap):
        norm = max(1., max(abs(x) for x in matrix))
        if any(abs(matrix[i*nproj+j]-matrix[j*nproj+i])>1e-12*norm
               for i in range(nproj) for j in range(nproj)):
            raise ValueError('UPF D/Q matrix is not symmetric')
    angular = []
    for i in range(1,nproj+1):
        beta = require('PP_NONLOCAL/PP_BETA.'+str(i),mesh)
        l = int(beta.attributes['angular_momentum'])
        if not 0 <= l <= 6 or int(beta.attributes['index']) != i:
            raise ValueError('UPF projector ordering/angular momentum invalid')
        index = int(beta.attributes['cutoff_radius_index'])
        if not 1 <= index <= mesh:
            raise ValueError('UPF projector cutoff outside mesh')
        angular.append(l)
        for prefix in ('PP_AEWFC.', 'PP_PSWFC.'):
            wave = require('PP_FULL_WFC/'+prefix+str(i),mesh)
            if int(wave.attributes['l']) != l:
                raise ValueError('UPF projector/partial-wave angular momentum mismatch')
    aug = root.find('PP_NONLOCAL/PP_AUGMENTATION')
    if aug is None or not _true(aug.get('q_with_l')):
        raise ValueError('PAW augmentation requires explicit QijL arrays')
    # Every allowed angular coupling must be retained, including zero arrays.
    for i in range(nproj):
        for j in range(i,nproj):
            for l in range(abs(angular[i]-angular[j]),angular[i]+angular[j]+1,2):
                name=f'PP_NONLOCAL/PP_AUGMENTATION/PP_QIJL.{i+1}.{j+1}.{l}'
                q = require(name,mesh)
                if (int(q.attributes['first_index']) != i+1 or
                    int(q.attributes['second_index']) != j+1 or
                    int(q.attributes['angular_momentum']) != l):
                    raise ValueError('UPF augmentation tensor identity mismatch')
    tokens = header['functional'].split()
    functional = {('SLA','PW','PBX','PBC'):'PBE', ('SLA','PW','PSX','PSC'):'PBEsol',
                  ('SLA','PZ','NOGX','NOGC'):'PZ'}.get(tuple(tokens), header['functional'].strip())
    return UpfPawDataset(header['element'].strip(), functional, header['relativistic'],
                         header, arrays, raw, hashlib.sha256(raw).hexdigest(), path)
