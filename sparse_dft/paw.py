"""Streaming adapter for the user-provided PAW_PBE.54 JSON archive."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator
import hashlib, json, lzma, re, time, struct, math
from .errors import ArchiveIntegrityError, InputError

@dataclass(frozen=True)
class PawCandidate:
    record_id: str
    element: str
    family: str
    variant: str
    tokens: tuple[str,...]
    zval: float
    enmax_ev: float
    enmin_ev: float
    eaug: float
    title: str
    potcar_sha256: str
    psctr_sha256: str
    score: float = 0.0
    reason: str = ""

@dataclass
class PawDataset:
    identity: dict
    metadata: dict
    potcar_path: str
    psctr_path: str
    source_hashes: dict
    parsed_header: dict
    parser_version: str = "sparse-dft-paw54-header-1"
    numeric_data: PawNumericData | None = None


@dataclass
class PawNonlocalChannel:
    """One VASP POTCAR non-local channel.

    The first line of a ``Non local Part`` block is ``l, nproj, rmax``;
    the following square matrix is stored row-major, followed by the 100
    point reciprocal-space and real-space projector tables.  Keeping these
    fields separate is important: the tables are not interchangeable radial
    arrays and the matrix is not part of either table.
    """
    angular_momentum: int
    nprojectors: int
    cutoff_radius: float
    reciprocal_cutoff: float
    d_matrix: list[list[float]]
    reciprocal_projectors: list[list[float]]
    real_projectors: list[list[float]]


@dataclass
class PawRadialData:
    npoints: int
    rmax: float
    grid: list[float]
    ae_potential: list[float]
    core_charge_density: list[float]
    kinetic_energy_density: list[float]
    kinetic_energy_density_pseudized: list[float]
    local_pseudopotential_core: list[float]
    pseudopotential_valence: list[float]
    core_charge_density_pseudized: list[float]
    pseudo_wavefunctions: list[list[float]]
    ae_wavefunctions: list[list[float]]
    augmentation_coefficients: list[float]
    occupancies: list[float]


@dataclass
class PawNumericData:
    """Strictly decoded numerical payload of one POTCAR dataset."""
    main_grid_size: int
    local_header: float | None
    local_potential: list[float]
    core_charge_partial: list[float]
    kinetic_energy_partial: list[float]
    atomic_pseudo_charge: list[float]
    atomic_pseudo_charge_tail: list[float]
    nonlocal_channels: list[PawNonlocalChannel]
    radial: PawRadialData
    atomic_reference_energy_ev: float = 0.0
    core_xc_outside_ev: float = 0.0
    kinetic_error_cutoff_ev: list[float] = field(default_factory=list)
    kinetic_error_ev: list[float] = field(default_factory=list)
    parser_version: str = "sparse-dft-paw54-numeric-1"

    def as_dict(self) -> dict:
        return {
            "parser_version": self.parser_version,
            "main_grid_size": self.main_grid_size,
            "local_header": self.local_header,
            "local_potential": self.local_potential,
            "core_charge_partial": self.core_charge_partial,
            "kinetic_energy_partial": self.kinetic_energy_partial,
            "atomic_pseudo_charge": self.atomic_pseudo_charge,
            "atomic_pseudo_charge_tail": self.atomic_pseudo_charge_tail,
            "nonlocal_channels": [asdict(x) for x in self.nonlocal_channels],
            "radial": asdict(self.radial),
            "atomic_reference_energy_ev": self.atomic_reference_energy_ev,
            "core_xc_outside_ev": self.core_xc_outside_ev,
            "kinetic_error_cutoff_ev": self.kinetic_error_cutoff_ev,
            "kinetic_error_ev": self.kinetic_error_ev,
        }

class Paw54JsonArchive:
    def __init__(self, root: str | Path):
        self.root=Path(root).expanduser().resolve()
        self.manifest_path=self.root/'manifest.json'; self.index_path=self.root/'index.json'; self.data_path=self.root/'potentials.jsonl.xz'
        if not self.root.is_dir(): raise InputError(f"PAW archive directory does not exist: {self.root}")
        for p in (self.manifest_path,self.index_path,self.data_path):
            if not p.is_file(): raise InputError(f"PAW archive missing {p}")
        self.manifest=json.loads(self.manifest_path.read_text(encoding='utf-8')); self.index=json.loads(self.index_path.read_text(encoding='utf-8'))
        self._validate_index()

    def _validate_index(self):
        if self.manifest.get('schema_version')!='1.0': raise ArchiveIntegrityError('unsupported PAW manifest schema',details={'schema':self.manifest.get('schema_version')})
        if self.manifest.get('format')!='VASP PAW JSONL': raise ArchiveIntegrityError('unexpected PAW archive format')
        n=int(self.manifest.get('record_count',-1)); idx=self.index.get('records',[])
        if n != len(idx): raise ArchiveIntegrityError('manifest/index record count mismatch',details={'manifest':n,'index':len(idx)})
        self._index_by_id={x['id']:x for x in idx}
        if len(self._index_by_id)!=n: raise ArchiveIntegrityError('duplicate PAW record id in index')

    def local_hashes(self):
        out={}
        for name in ('manifest.json','index.json','potentials.jsonl.xz','paw_json.py'):
            p=self.root/name
            if p.exists(): out[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    def iter_records(self) -> Iterator[dict]:
        with lzma.open(self.data_path,'rt',encoding='utf-8') as fh:
            for line_no,line in enumerate(fh,1):
                if line.strip():
                    try: rec=json.loads(line)
                    except json.JSONDecodeError as e: raise ArchiveIntegrityError(f'invalid JSON at line {line_no}',details={'line':line_no}) from e
                    yield rec

    def get_record(self, record_id: str) -> dict:
        if record_id not in self._index_by_id: raise InputError(f'unknown PAW record: {record_id}')
        for rec in self.iter_records():
            if rec.get('id')==record_id:
                self._validate_record(rec)
                return rec
        raise ArchiveIntegrityError('indexed PAW record not found in JSONL',details={'record_id':record_id})

    def _validate_record(self, rec):
        rid=rec.get('id'); idx=self._index_by_id.get(rid)
        if not idx: raise ArchiveIntegrityError('record absent from index',details={'record_id':rid})
        for key in ('element','variant','metadata','files'):
            if key not in rec: raise ArchiveIntegrityError(f'record missing {key}',details={'record_id':rid})
        if rec['element'] != idx.get('element') or rec['variant'] != idx.get('variant'):
            raise ArchiveIntegrityError('record/index metadata mismatch',details={'record_id':rid})
        if rec['metadata'] != idx.get('metadata'):
            raise ArchiveIntegrityError('record/index PAW metadata mismatch', details={'record_id':rid})
        if rec['files']['POTCAR'].get('text') is None: raise ArchiveIntegrityError('POTCAR text missing',details={'record_id':rid})
        for kind in ('POTCAR','PSCTR'):
            f=rec['files'].get(kind,{})
            if f.get('text') is None and not (kind=='PSCTR' and f.get('empty')): raise ArchiveIntegrityError(f'{kind} text missing',details={'record_id':rid})
            indexed=idx.get('files',{}).get(kind,{})
            if any(f.get(key)!=value for key,value in indexed.items()):
                raise ArchiveIntegrityError('record/index PAW file mismatch', details={'record_id':rid,'kind':kind})

    def find_records(self, *, element=None, family=None, tag=None) -> Iterator[dict]:
        ids=None
        if element is not None: ids=set(self.index.get('by_element',{}).get(element,[]))
        elif family is not None: ids=set(self.index.get('by_family',{}).get(family,[]))
        elif tag is not None: ids=set(self.index.get('by_tag',{}).get(tag,[]))
        for rec in self.iter_records():
            if ids is not None and rec.get('id') not in ids: continue
            if element is not None and rec.get('element',{}).get('symbol')!=element: continue
            if family is not None and rec.get('variant',{}).get('family')!=family: continue
            if tag is not None and tag not in rec.get('variant',{}).get('tokens',[]): continue
            self._validate_record(rec); yield rec

    def inventory(self, full=False):
        counts={'records':0,'elements':set(),'families':{},'tags':{},'pbe_records':0,'empty_psctr':[]}
        ids=set()
        for rec in self.iter_records():
            self._validate_record(rec); counts['records']+=1; ids.add(rec['id']); counts['elements'].add(rec['element']['symbol'])
            fam=rec['variant']['family']; counts['families'][fam]=counts['families'].get(fam,0)+1
            for tag in rec['variant'].get('tokens',[]): counts['tags'][tag]=counts['tags'].get(tag,0)+1
            if rec['metadata']['potcar'].get('exchange_correlation')=='PE': counts['pbe_records']+=1
            if rec['files'].get('PSCTR',{}).get('empty'): counts['empty_psctr'].append(rec['id'])
        expected=set(self._index_by_id)
        if ids!=expected: raise ArchiveIntegrityError('JSONL/index record set mismatch',details={'missing':sorted(expected-ids),'extra':sorted(ids)})
        counts['elements']=sorted(counts['elements']); counts['source_hashes']=self.local_hashes(); counts['manifest_record_count']=self.manifest['record_count']
        counts['index_record_count']=len(expected); return counts

    def candidate(self, rec, score=0.0, reason=''):
        p=rec['metadata']['potcar']; v=rec['variant']; files=rec['files']
        return PawCandidate(rec['id'],rec['element']['symbol'],v['family'],v['name'],tuple(v.get('tokens',[])),float(p['valence_electrons']),float(p['enmax_ev']),float(p['enmin_ev']),float(p['eaug']),p['title'],files['POTCAR']['sha256'],files.get('PSCTR',{}).get('sha256',''),score,reason)

    def candidates(self, element, *, family=None, include_nonstandard=False):
        out=[]
        # The index contains the complete selection metadata and source file
        # hashes. Numerical use still streams the selected record and checks
        # it against this index; candidate discovery need not decode 327 texts.
        for record_id in self.index.get('by_element',{}).get(element,[]):
            rec=self._index_by_id.get(record_id)
            if rec is None or rec.get('element',{}).get('symbol')!=element:
                raise ArchiveIntegrityError('invalid by_element PAW index', details={'element':element,'record_id':record_id})
            v=rec['variant']; p=rec['metadata']['potcar']
            if p.get('exchange_correlation')!='PE' or 'PAW_PBE' not in p.get('title',''): continue
            if not include_nonstandard and family is None and v['family']!='standard': continue
            if family and v['family']!=family: continue
            score=(100 if v['family']=='standard' else 50 if v['family']=='semicore' else 10)
            out.append(self.candidate(rec,score,f'PBE; family={v["family"]}'))
        return sorted(out,key=lambda x:(-x.score,x.enmax_ev,x.record_id))

    def restore(self, record_id, kind, destination):
        rec=self.get_record(record_id)
        return self._restore_record_file(rec,kind,destination)

    @staticmethod
    def _restore_record_file(rec, kind, destination):
        record_id=rec['id'];kind=kind.upper()
        if kind not in ('POTCAR','PSCTR'): raise InputError('kind must be POTCAR or PSCTR')
        f=rec['files'][kind]
        text=f.get('text','')
        if f.get('empty') and not text: raw=b''
        else:
            try: raw=text.encode('ascii')
            except UnicodeEncodeError as e: raise ArchiveIntegrityError('PAW text is not ASCII',details={'record_id':record_id,'kind':kind}) from e
        actual=hashlib.sha256(raw).hexdigest(); expected=f.get('sha256')
        if actual!=expected: raise ArchiveIntegrityError('PAW text hash mismatch',details={'record_id':record_id,'kind':kind,'expected':expected,'actual':actual})
        dest=Path(destination); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
        if f.get('size_bytes') != len(raw): raise ArchiveIntegrityError('PAW size mismatch',details={'record_id':record_id,'kind':kind,'expected':f.get('size_bytes'),'actual':len(raw)})
        return dest

    @staticmethod
    def parse_header(text: str) -> dict:
        vals={}
        patterns={'VRHFIN':r'VRHFIN\s*=\s*([^\s]+)','LEXCH':r'LEXCH\s*=\s*([^\s]+)','POMASS':r'POMASS\s*=\s*([\d.Ee+-]+)','ZVAL':r'ZVAL\s*=\s*([\d.Ee+-]+)','RCORE':r'RCORE\s*=\s*([\d.Ee+-]+)','RWIGS':r'RWIGS\s*=\s*([\d.Ee+-]+)','ENMAX':r'ENMAX\s*=\s*([\d.Ee+-]+)','ENMIN':r'ENMIN\s*=\s*([\d.Ee+-]+)','RCLOC':r'RCLOC\s*=\s*([\d.Ee+-]+)','EAUG':r'EAUG\s*=\s*([\d.Ee+-]+)','RMAX':r'RMAX\s*=\s*([\d.Ee+-]+)','RAUG':r'RAUG\s*=\s*([\d.Ee+-]+)','RDEP':r'RDEP\s*=\s*([\d.Ee+-]+)','RDEPT':r'RDEPT\s*=\s*([\d.Ee+-]+)'}
        title=(next((x.strip() for x in text.splitlines() if 'TITEL' in x), '')).split('=',1)[-1].strip()
        vals['TITEL']=title
        for k,pat in patterns.items():
            m=re.search(pat,text)
            if m:
                try: vals[k]=float(m.group(1)) if k not in ('VRHFIN','LEXCH') else m.group(1)
                except ValueError: vals[k]=m.group(1)
        return vals

    @staticmethod
    def parse_numeric_sections(text: str) -> dict:
        """Record complete PAW section boundaries and numeric payload counts."""
        lines=text.splitlines(); sections={}; markers={
            'local_part':' local part', 'nonlocal_part':' Non local Part',
            'augmentation_charges':' augmentation charges', 'core_charge_density':'core charge-density',
            'pseudo_wavefunction':'pseudo wavefunction', 'end_dataset':' End of Dataset'
        }
        for name, marker in markers.items():
            hits=[i for i,l in enumerate(lines) if marker.lower() in l.lower()]
            if hits: sections[name]={'line':hits[0],'count':0}
        ordered=sorted((v['line'],k) for k,v in sections.items())
        for idx,(start,name) in enumerate(ordered):
            end=ordered[idx+1][0] if idx+1<len(ordered) else len(lines)
            nums=[]
            for line in lines[start+1:end]:
                for token in line.replace('D','E').split():
                    try: nums.append(float(token))
                    except ValueError: pass
            sections[name].update({'end_line':end,'count':len(nums),'first_values':nums[:8]})
        if 'local_part' not in sections or 'end_dataset' not in sections:
            raise ArchiveIntegrityError('POTCAR numeric sections are incomplete',details={'sections':sorted(sections)})
        return {'section_count':len(sections),'sections':sections,'complete':True}

    @staticmethod
    def _numbers(lines: list[str], start: int, end: int) -> list[float]:
        values = []
        for line in lines[start:end]:
            for token in line.replace('D', 'E').replace('d', 'e').split():
                try:
                    values.append(float(token))
                except ValueError:
                    # POTCAR footers contain flags such as the trailing T.
                    continue
        return values

    @staticmethod
    def _exact_markers(lines: list[str], start: int = 0) -> list[tuple[int, str]]:
        names = {
            'local part', 'core charge-density (partial)',
            'kinetic energy density (partial)', 'atomic pseudo charge-density',
            'gradient corrections used for xc',
            'non local part', 'reciprocal space part', 'real space part',
            'paw radial sets', 'augmentation charges (non sperical)',
            'uccopancies in atom', 'grid', 'aepotential',
            'core charge-density', 'kinetic energy-density',
            'mkinetic energy-density pseudized', 'local pseudopotential core',
            'pspotential valence only', 'core charge-density (pseudized)',
            'pseudo wavefunction', 'ae wavefunction', 'end of dataset',
        }
        return [(i, line.strip().lower()) for i, line in enumerate(lines[start:], start)
                if line.strip().lower() in names]

    @staticmethod
    def parse_numeric_data(text: str) -> PawNumericData:
        """Decode the PAW_PBE.54 numerical layout into typed arrays.

        VASP writes several unrelated arrays without explicit JSON-like
        lengths.  The format does provide unambiguous section markers,
        radial point counts, and projector matrix dimensions; this parser
        uses all three and rejects truncated or shifted sections.
        """
        lines = text.splitlines()
        markers = Paw54JsonArchive._exact_markers(lines)
        by_name: dict[str, list[int]] = {}
        for line, name in markers:
            by_name.setdefault(name, []).append(line)
        required = ('local part', 'non local part', 'paw radial sets', 'end of dataset')
        missing = [x for x in required if x not in by_name]
        if missing:
            raise ArchiveIntegrityError('POTCAR numeric markers missing', details={'missing': missing})

        # Main local/pseudo-charge section.  The local section has one scalar
        # header followed by the common 1000-point radial reciprocal table.
        local_line = by_name['local part'][0]
        first_nonlocal = by_name['non local part'][0]
        first_main_marker = min(
            [line for name in ('gradient corrections used for xc', 'core charge-density (partial)', 'kinetic energy density (partial)', 'atomic pseudo charge-density')
             for line in by_name.get(name, []) if local_line < line < first_nonlocal]
            + [first_nonlocal]
        )
        local_values = Paw54JsonArchive._numbers(lines, local_line + 1, first_main_marker)
        if len(local_values) < 2:
            raise ArchiveIntegrityError('local part payload is empty')
        main_grid_size = len(local_values) - 1
        local_header = local_values[0]
        local_potential = local_values[1:]
        # EATOM and its tabulated cutoff correction set the PAW atomic energy
        # zero. The kinetic-density arrays below are different quantities.
        header = '\n'.join(lines[:local_line])
        scalar_pattern = r'([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)'
        eatom = re.search(r'\bEATOM\s*=\s*' + scalar_pattern, header)
        dexc = re.search(r'\bDEXC\s*=\s*' + scalar_pattern, header)
        if eatom is None:
            raise ArchiveIntegrityError('POTCAR atomic reference EATOM is missing')
        atomic_reference = float(eatom.group(1).replace('D', 'E').replace('d', 'e'))
        core_outside = float(dexc.group(1).replace('D', 'E').replace('d', 'e')) if dexc else 0.0
        error_cutoff, error_values = [], []
        error_line = next((i for i, line in enumerate(lines[:local_line])
                           if 'Error from kinetic energy argument' in line), None)
        if error_line is not None:
            count_match = re.search(r'NDATA\s*=\s*(\d+)', lines[error_line + 1])
            step_match = re.search(r'STEP\s*=\s*' + scalar_pattern + r'\s+' + scalar_pattern,
                                   lines[error_line + 2])
            if count_match is None or step_match is None:
                raise ArchiveIntegrityError('invalid kinetic error table header')
            count = int(count_match.group(1))
            start, factor = (float(x.replace('D', 'E').replace('d', 'e'))
                             for x in step_match.groups())
            if not 2 <= count <= 10000 or start <= 0.0 or factor <= 1.0:
                raise ArchiveIntegrityError('invalid kinetic error table abscissa')
            error_values = Paw54JsonArchive._numbers(lines, error_line + 3,
                error_line + 3 + (count + 7) // 8)
            if len(error_values) != count or not all(math.isfinite(x) for x in error_values):
                raise ArchiveIntegrityError('kinetic error table length/finite mismatch')
            error_cutoff = [start * factor ** i for i in range(count)]

        main_names = {
            'core charge-density (partial)': 'core_charge_partial',
            'kinetic energy density (partial)': 'kinetic_energy_partial',
            'atomic pseudo charge-density': 'atomic_pseudo_charge',
        }
        main_arrays: dict[str, list[float]] = {}
        for name, field_name in main_names.items():
            starts = by_name.get(name, [])
            if not starts or starts[0] <= local_line or starts[0] >= first_nonlocal:
                continue
            start = starts[0]
            later = [line for line, marker in markers if line > start and line < first_nonlocal]
            end = min(later + [first_nonlocal])
            values = Paw54JsonArchive._numbers(lines, start + 1, end)
            if name == 'atomic pseudo charge-density':
                # The last scalar before Non local Part is a format footer
                # (followed by a boolean flag), not charge density.
                if len(values) not in (main_grid_size, main_grid_size + 1):
                    raise ArchiveIntegrityError('atomic pseudo charge length mismatch', details={'expected': main_grid_size, 'actual': len(values)})
                main_arrays[field_name] = values[:main_grid_size]
                main_arrays['atomic_pseudo_charge_tail'] = values[main_grid_size:]
            else:
                if len(values) != main_grid_size and not (name == 'kinetic energy density (partial)' and len(values) % main_grid_size == 0):
                    raise ArchiveIntegrityError('main PAW array length mismatch', details={'section': name, 'expected': main_grid_size, 'actual': len(values)})
                main_arrays[field_name] = values
        main_arrays.setdefault('core_charge_partial', [])
        main_arrays.setdefault('kinetic_energy_partial', [])
        main_arrays.setdefault('atomic_pseudo_charge', [])
        main_arrays.setdefault('atomic_pseudo_charge_tail', [])

        # Non-local projector channels.  Each channel has a (l,nproj,rmax)
        # header, an nproj-by-nproj D matrix, and two exactly 100-point tables.
        channels: list[PawNonlocalChannel] = []
        radial_start = by_name['paw radial sets'][0]
        reciprocal_cutoff_global = 0.0
        for idx, start in enumerate(by_name['non local part']):
            if start >= radial_start:
                continue
            next_nonlocal = [x for x in by_name['non local part'] if x > start]
            channel_end = min(next_nonlocal + [radial_start])
            recips = [x for x in by_name.get('reciprocal space part', []) if start < x < channel_end]
            reals = [x for x in by_name.get('real space part', []) if start < x < channel_end]
            if len(recips) != len(reals):
                raise ArchiveIntegrityError('non-local channel marker count mismatch', details={'line': start + 1, 'reciprocal': recips, 'real': reals})
            if len(recips) == 0:
                raise ArchiveIntegrityError('non-local channel has no projector tables', details={'line': start + 1})
            first_reciprocal = min(recips)
            header_values = Paw54JsonArchive._numbers(lines, start + 1, first_reciprocal)
            if len(header_values) < 3:
                raise ArchiveIntegrityError('non-local channel header is incomplete', details={'line': start + 1})
            l_value = int(round(header_values[0])); nproj = int(round(header_values[1])); cutoff = header_values[2]
            # The scalar immediately preceding each Non local Part marker is
            # the reciprocal-space cutoff of the projector table.  It is
            # distinct from the real-space projector radius in the channel
            # header and fixes the q spacing of the tabulated projectors.
            prior = Paw54JsonArchive._numbers(lines, max(0, start - 2), start)
            reciprocal_cutoff = prior[-1] if prior else 0.0
            if reciprocal_cutoff > 0.0 and reciprocal_cutoff_global == 0.0:
                reciprocal_cutoff_global = reciprocal_cutoff
            if nproj < 1 or abs(header_values[0] - l_value) > 1e-8 or abs(header_values[1] - nproj) > 1e-8:
                raise ArchiveIntegrityError('invalid non-local channel dimensions', details={'line': start + 1, 'header': header_values[:3]})
            matrix_values = header_values[3:]
            expected_matrix = nproj * nproj
            if len(matrix_values) != expected_matrix:
                raise ArchiveIntegrityError('non-local D matrix length mismatch', details={'line': start + 1, 'expected': expected_matrix, 'actual': len(matrix_values), 'nproj': nproj})
            if len(recips) != nproj or len(reals) != nproj:
                raise ArchiveIntegrityError('non-local projector table count mismatch', details={'line': start + 1, 'nproj': nproj, 'reciprocal': len(recips), 'real': len(reals)})
            reciprocal_tables = []
            real_tables = []
            for reciprocal_line, real_line in zip(sorted(recips), sorted(reals)):
                reciprocal = Paw54JsonArchive._numbers(lines, reciprocal_line + 1, real_line)
                if len(reciprocal) != 100:
                    raise ArchiveIntegrityError('reciprocal projector length mismatch', details={'line': reciprocal_line + 1, 'expected': 100, 'actual': len(reciprocal)})
                next_reciprocal = [x for x in recips if x > reciprocal_line]
                real_end = min(next_reciprocal + [channel_end])
                real = Paw54JsonArchive._numbers(lines, real_line + 1, real_end)
                if len(real) != 100:
                    raise ArchiveIntegrityError('real-space projector length mismatch', details={'line': real_line + 1, 'expected': 100, 'actual': len(real)})
                reciprocal_tables.append(reciprocal)
                real_tables.append(real)
            matrix = [matrix_values[row * nproj:(row + 1) * nproj] for row in range(nproj)]
            channels.append(PawNonlocalChannel(l_value, nproj, cutoff, reciprocal_cutoff, matrix, reciprocal_tables, real_tables))
        if reciprocal_cutoff_global > 0.0:
            channels = [PawNonlocalChannel(x.angular_momentum, x.nprojectors, x.cutoff_radius,
                                            reciprocal_cutoff_global, x.d_matrix,
                                            x.reciprocal_projectors, x.real_projectors)
                        for x in channels]
        if not channels:
            raise ArchiveIntegrityError('no non-local PAW projector channels found')

        # Radial sets.  The first line gives the radial point count and rmax;
        # all following named records carry exactly that many values.
        radial_header_end = by_name['augmentation charges (non sperical)'][0]
        radial_header = Paw54JsonArchive._numbers(lines, radial_start + 1, radial_header_end)
        if len(radial_header) != 2:
            raise ArchiveIntegrityError('PAW radial header must contain npoints and rmax', details={'values': radial_header})
        npoints = int(round(radial_header[0])); rmax = radial_header[1]
        if npoints < 1 or abs(radial_header[0] - npoints) > 1e-8:
            raise ArchiveIntegrityError('invalid PAW radial point count', details={'value': radial_header[0]})
        aug_line = by_name['augmentation charges (non sperical)'][0]
        occ_line = by_name['uccopancies in atom'][0]
        grid_line = by_name['grid'][0]
        total_projectors = sum(c.nprojectors for c in channels)
        expected_aug = total_projectors * total_projectors
        augmentation = Paw54JsonArchive._numbers(lines, aug_line + 1, occ_line)
        occupancies = Paw54JsonArchive._numbers(lines, occ_line + 1, grid_line)
        if len(augmentation) != expected_aug or len(occupancies) != expected_aug:
            raise ArchiveIntegrityError('augmentation/occupancy tensor length mismatch', details={'expected': expected_aug, 'augmentation': len(augmentation), 'occupancies': len(occupancies), 'projectors': total_projectors})

        radial_names = [name for line, name in markers if line >= grid_line and name != 'end of dataset']
        radial_positions = [(line, name) for line, name in markers if line >= grid_line and name != 'end of dataset']
        arrays: dict[str, list[float]] = {}
        pseudo_waves: list[list[float]] = []
        ae_waves: list[list[float]] = []
        for position, (start, name) in enumerate(radial_positions):
            end = radial_positions[position + 1][0] if position + 1 < len(radial_positions) else by_name['end of dataset'][0]
            values = Paw54JsonArchive._numbers(lines, start + 1, end)
            if len(values) != npoints:
                raise ArchiveIntegrityError('radial PAW array length mismatch', details={'section': name, 'expected': npoints, 'actual': len(values), 'line': start + 1})
            if name == 'pseudo wavefunction':
                pseudo_waves.append(values)
            elif name == 'ae wavefunction':
                ae_waves.append(values)
            else:
                if name in arrays:
                    raise ArchiveIntegrityError('duplicate radial PAW section', details={'section': name})
                arrays[name] = values
        for required_name in ('grid', 'aepotential', 'core charge-density', 'local pseudopotential core', 'pspotential valence only', 'core charge-density (pseudized)'):
            if required_name not in arrays:
                raise ArchiveIntegrityError('required radial PAW section missing', details={'section': required_name})
        for optional_name in ('kinetic energy-density', 'mkinetic energy-density pseudized'):
            arrays.setdefault(optional_name, [])
        if len(pseudo_waves) != len(ae_waves) or len(pseudo_waves) != total_projectors:
            raise ArchiveIntegrityError('partial-wave count does not match projector count', details={'pseudo': len(pseudo_waves), 'ae': len(ae_waves), 'projectors': total_projectors})
        radial = PawRadialData(
            npoints=npoints, rmax=rmax, grid=arrays['grid'], ae_potential=arrays['aepotential'],
            core_charge_density=arrays['core charge-density'], kinetic_energy_density=arrays['kinetic energy-density'],
            kinetic_energy_density_pseudized=arrays['mkinetic energy-density pseudized'],
            local_pseudopotential_core=arrays['local pseudopotential core'],
            pseudopotential_valence=arrays['pspotential valence only'],
            core_charge_density_pseudized=arrays['core charge-density (pseudized)'],
            pseudo_wavefunctions=pseudo_waves, ae_wavefunctions=ae_waves,
            augmentation_coefficients=augmentation, occupancies=occupancies,
        )
        return PawNumericData(
            main_grid_size=main_grid_size, local_header=local_header,
            local_potential=local_potential, core_charge_partial=main_arrays['core_charge_partial'],
            kinetic_energy_partial=main_arrays['kinetic_energy_partial'],
            atomic_pseudo_charge=main_arrays['atomic_pseudo_charge'],
            atomic_pseudo_charge_tail=main_arrays['atomic_pseudo_charge_tail'],
            nonlocal_channels=channels, radial=radial,
            atomic_reference_energy_ev=atomic_reference, core_xc_outside_ev=core_outside,
            kinetic_error_cutoff_ev=error_cutoff, kinetic_error_ev=error_values,
        )

    @staticmethod
    def _write_numeric_binary(data: PawNumericData, destination: Path) -> str:
        """Write a portable little-endian cache consumed by the C++ core."""
        payload = bytearray()
        payload += b'SDFTPAW1'
        # v4 adds the atomic energy reference, outside-core XC and cutoff
        # error table, so C++ energy evaluation does not need a text sidecar.
        payload += struct.pack('<IIIII', 4, data.main_grid_size, data.radial.npoints,
                               len(data.nonlocal_channels), sum(c.nprojectors for c in data.nonlocal_channels))
        payload += struct.pack('<d', data.radial.rmax)
        def vec(values):
            payload.extend(struct.pack('<I', len(values)))
            if values:
                payload.extend(struct.pack('<' + 'd' * len(values), *values))
        payload += struct.pack('<dd', data.atomic_reference_energy_ev, data.core_xc_outside_ev)
        vec(data.kinetic_error_cutoff_ev)
        vec(data.kinetic_error_ev)
        vec([data.local_header] if data.local_header is not None else [])
        for values in (data.local_potential, data.core_charge_partial, data.kinetic_energy_partial,
                       data.atomic_pseudo_charge, data.atomic_pseudo_charge_tail):
            vec(values)
        for channel in data.nonlocal_channels:
            payload.extend(struct.pack('<ii dd', channel.angular_momentum, channel.nprojectors, channel.cutoff_radius, channel.reciprocal_cutoff))
            for row in channel.d_matrix:
                vec(row)
            for table in channel.reciprocal_projectors:
                vec(table)
            for table in channel.real_projectors:
                vec(table)
        radial = data.radial
        for values in (radial.grid, radial.ae_potential, radial.core_charge_density,
                       radial.kinetic_energy_density, radial.kinetic_energy_density_pseudized,
                       radial.local_pseudopotential_core, radial.pseudopotential_valence,
                       radial.core_charge_density_pseudized, radial.augmentation_coefficients,
                       radial.occupancies):
            vec(values)
        for values in radial.pseudo_wavefunctions + radial.ae_wavefunctions:
            vec(values)
        destination.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def restore_dataset(self, record_id, destination):
        dest=Path(destination); dest.mkdir(parents=True,exist_ok=True)
        source_hashes=self.local_hashes()
        rec=self.get_record(record_id)
        pot=self._restore_record_file(rec,'POTCAR',dest/'POTCAR')
        ps=self._restore_record_file(rec,'PSCTR',dest/'PSCTR')
        pot_text=pot.read_text(errors='strict', encoding='utf-8'); header=self.parse_header(pot_text); numeric_sections=self.parse_numeric_sections(pot_text); numeric_data=self.parse_numeric_data(pot_text)
        if self.local_hashes()!=source_hashes:
            raise ArchiveIntegrityError('PAW archive changed during dataset preparation', details={'record_id':record_id})
        meta={'record_id':record_id,'element':rec['element'],'variant':rec['variant'],'metadata':rec['metadata'],'parser_version':'sparse-dft-paw54-numeric-1','numeric_sections':numeric_sections,'numeric_summary':{'main_grid_size':numeric_data.main_grid_size,'radial_points':numeric_data.radial.npoints,'projector_channels':len(numeric_data.nonlocal_channels),'projector_count':sum(x.nprojectors for x in numeric_data.nonlocal_channels)},'source_hashes':{'archive':source_hashes,'POTCAR':rec['files']['POTCAR']['sha256'],'PSCTR':rec['files']['PSCTR'].get('sha256','')}}
        numeric_json = dest/'numeric.json'; numeric_bin = dest/'numeric.bin'
        numeric_json.write_text(json.dumps(numeric_data.as_dict(),indent=2,sort_keys=True), encoding='utf-8')
        numeric_sha256 = self._write_numeric_binary(numeric_data, numeric_bin)
        meta['numeric_cache'] = {'json': numeric_json.name, 'binary': numeric_bin.name, 'binary_sha256': numeric_sha256, 'binary_format': 'SDFTPAW1/little-endian/v4'}
        (dest/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True), encoding='utf-8')
        return PawDataset(meta['element'],meta['metadata'],str(pot),str(ps),meta['source_hashes'],header,'sparse-dft-paw54-numeric-1',numeric_data)

class PawSelectionPolicy:
    def __init__(self, family='standard', allow_semicore=True): self.family=family; self.allow_semicore=allow_semicore
    def resolve(self, archive: Paw54JsonArchive, elements):
        choices={}; report=[]
        for el in dict.fromkeys(elements):
            c=archive.candidates(el,family=self.family)
            if not c and self.allow_semicore: c=archive.candidates(el,family='semicore')
            if not c: raise InputError(f'no compatible PBE PAW candidate for {el}',details={'element':el})
            choices[el]=c[0]; report.append({'element':el,'selected':asdict(c[0]),'candidates':[asdict(x) for x in c]})
        return choices,report
