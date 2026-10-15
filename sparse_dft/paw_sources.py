"""Per-element PAW source resolution with explicit method compatibility.

UPF data ingestion and a compatible selection plan are distinct from a
validated mixed-source SCF implementation. The execution flag remains false
until the native adapter and independent physical tests are completed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import InputError, UnsupportedError
from .paw import Paw54JsonArchive, PawSelectionPolicy
from .upf import read_upf_paw


@dataclass(frozen=True)
class PawSource:
    format: str
    path: str
    record_id: str | None = None


def resolve_paw_sources(species: list[str], sources: dict[str, PawSource]) -> dict:
    """Resolve one deliberate source per species, without fabricating a cache."""
    if set(species) != set(sources):
        raise InputError('PAW source map must cover exactly the requested species',
                         details={'missing':sorted(set(species)-set(sources)),
                                  'unused':sorted(set(sources)-set(species))})
    records=[]
    for element in dict.fromkeys(species):
        source=sources[element]
        if source.format=='upf-paw':
            dataset=read_upf_paw(Path(source.path))
            if dataset.element!=element:
                raise InputError('UPF element does not match selected species')
            if dataset.relativistic!='scalar' or dataset.summary()['has_soc']:
                raise UnsupportedError('Sparse-DFT mixed PAW path does not yet implement SOC/full-relativistic spinors')
            record=dict(dataset.summary(),format=source.format,source_path=str(Path(source.path).resolve()))
            if dataset.functional!='PBE':
                raise UnsupportedError('Current shared PAW SCF functional is PBE; PZ/PBEsol require separate validated functionals')
        elif source.format=='vasp-paw54':
            archive=Paw54JsonArchive(source.path)
            if source.record_id:
                archive.get_record(source.record_id)
                choices=archive.candidates(element,include_nonstandard=True)
                candidate=next((c for c in choices if c.record_id==source.record_id),None)
                if candidate is None:raise InputError('requested PAW record is not a compatible PBE element')
            else:
                choices,report=PawSelectionPolicy().resolve(archive,[element])
                candidate=choices[element]
            record={'element':element,'format':source.format,'source_path':str(Path(source.path).resolve()),
                    'record_id':candidate.record_id,'source_sha256':candidate.potcar_sha256,
                    'functional':'PBE','z_valence':candidate.zval,
                    'suggested_wfc_cutoff_ev':candidate.enmax_ev,'relativistic':'scalar'}
        else:
            raise UnsupportedError('unsupported PAW source format: '+source.format)
        records.append(record)
    mixed= len({r['format'] for r in records})>1
    return {'schema_version':'sparse-dft-paw-source-plan-1','records':records,
            'shared_functional':'PBE','cross_format_mixed':mixed,
            'native_upf_adapter_validated':False,'mixed_physics_validated':False,
            'execution_ready':all(r['format']=='vasp-paw54' for r in records),
            'energy_reference_rule':'compare energy differences at fixed composition and fixed PAW source map; no absolute cross-library energy gate',
            'hubbard_rule':'define and validate the correlated subspace for each dataset; equal U does not establish equal physics'}
