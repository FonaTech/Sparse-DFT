"""Dependency-free structure normalization for POSCAR, CIF and XYZ."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib, json, math, re
from .errors import InputError

Vec3 = tuple[float, float, float]
Matrix3 = tuple[Vec3, Vec3, Vec3]

def _dot(a, b): return sum(x*y for x,y in zip(a,b))
def _cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _norm(a): return math.sqrt(_dot(a,a))
def _det(m): return _dot(m[0], _cross(m[1],m[2]))
def _mat_vec(m,v): return tuple(_dot(row,v) for row in m)
def _transpose(m): return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))
def _inverse(m):
    a,b,c=m
    bc=_cross(b,c); ca=_cross(c,a); ab=_cross(a,b); d=_dot(a,bc)
    if abs(d)<1e-14: raise InputError("singular lattice", details={"lattice":m})
    return tuple(tuple(x/d for x in row) for row in (bc,ca,ab))

@dataclass
class Structure:
    species: list[str]
    positions: list[Vec3]
    lattice: Matrix3 | None = None
    fractional: bool = False
    periodic: tuple[bool,bool,bool] = (False,False,False)
    selective_dynamics: list[tuple[bool,bool,bool]] | None = None
    source_format: str = "unknown"
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.species) != len(self.positions): raise InputError("species/positions length mismatch")
        if self.lattice is not None:
            if abs(_det(self.lattice)) < 1e-14: raise InputError("lattice volume is zero")
            self.periodic = tuple(bool(x) for x in self.periodic)
        if self.fractional and self.lattice is None: raise InputError("fractional coordinates require lattice")

    @property
    def natoms(self): return len(self.species)
    @property
    def volume(self): return abs(_det(self.lattice)) if self.lattice else None
    @property
    def reciprocal_lattice(self):
        if self.lattice is None: return None
        # _inverse returns A^-T: its rows are reciprocal vectors without 2*pi.
        inv=_inverse(self.lattice)
        return tuple(tuple(2*math.pi*x for x in row) for row in inv)

    def cartesian_positions(self):
        if not self.fractional: return list(self.positions)
        return [_mat_vec(_transpose(self.lattice),p) for p in self.positions]

    def normalized(self):
        pos=self.cartesian_positions()
        frac=False
        if self.lattice:
            inv=_inverse(self.lattice)
            pos=[_mat_vec(inv,p) for p in pos]
            pos=[tuple(x-math.floor(x) for x in p) for p in pos]
            frac=True
        return Structure(list(self.species), pos, self.lattice, frac, self.periodic,
                         self.selective_dynamics, self.source_format, self.source_path, list(self.warnings))

    def as_dict(self):
        s=self.normalized()
        payload = {"species":s.species,"positions":s.positions,"lattice":s.lattice,
                "fractional":s.fractional,"periodic":s.periodic,"source_format":s.source_format,
                "source_path":s.source_path,"warnings":s.warnings}
        if s.selective_dynamics is not None:
            payload["selective_dynamics"] = s.selective_dynamics
        return payload

    def content_hash(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",",":")).encode()).hexdigest()

def _floats(line, n=3):
    try: vals=[float(x) for x in line.split()[:n]]
    except ValueError as e: raise InputError(f"invalid numeric line: {line}") from e
    if len(vals)!=n: raise InputError(f"expected {n} numbers: {line}")
    return tuple(vals)

def read_poscar(path):
    lines=Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines)<8: raise InputError("POSCAR is too short")
    scale=float(lines[1].strip()); raw=tuple(_floats(lines[i]) for i in range(2,5))
    if scale<0:
        target=-scale; vol=abs(_det(raw)); scale=(target/vol)**(1/3)
    lattice=tuple(tuple(scale*x for x in row) for row in raw)
    tokens=lines[5].split(); counts_i=6
    if all(re.fullmatch(r"[+-]?\d+",x) for x in tokens):
        species=[f"X{i+1}" for i in range(len(tokens))]; counts=[int(x) for x in tokens]
    else:
        species=tokens; counts=[int(x) for x in lines[6].split()]; counts_i=7
    if len(species)!=len(counts) or sum(counts)<=0: raise InputError("invalid POSCAR species/counts")
    i=counts_i; selective=False
    if i<len(lines) and lines[i].strip().lower().startswith("s"):
        selective=True; i+=1
    if i>=len(lines): raise InputError("missing POSCAR coordinate mode")
    mode=lines[i].strip().lower(); i+=1
    frac=mode.startswith("d") or mode.startswith("f")
    if not (frac or mode.startswith("c") or mode.startswith("k")): raise InputError("unknown POSCAR coordinate mode")
    sp=[]; pos=[]; flags=[]
    for sym,n in zip(species,counts):
        for _ in range(n):
            if i>=len(lines): raise InputError("POSCAR has fewer coordinates than declared")
            parts=lines[i].split(); i+=1
            pos.append(_floats(" ".join(parts[:3]))); sp.append(sym)
            if selective: flags.append(tuple(x.lower().startswith("t") for x in parts[3:6]))
    return Structure(sp,pos,lattice,frac,(True,True,True),flags if selective else None,"POSCAR",str(path))

def poscar_text(structure, title="Sparse-DFT structure"):
    """VASP 5 POSCAR (Direct). Atoms are regrouped by first appearance of each
    species, as POSCAR requires; the order inside a species is kept."""
    s=structure.normalized()
    if s.lattice is None: raise InputError("POSCAR needs a periodic cell")
    order=sorted(range(s.natoms),key=lambda i:(list(dict.fromkeys(s.species)).index(s.species[i]),i))
    names=list(dict.fromkeys(s.species))
    rows=[str(title).replace("\n"," "),"1.0"]+[" ".join(f"{v:.16f}" for v in r) for r in s.lattice]
    rows+=[" ".join(names)," ".join(str(s.species.count(n)) for n in names),"Direct"]
    rows+=[" ".join(f"{v:.16f}" for v in s.positions[i]) for i in order]
    return "\n".join(rows)+"\n"

def write_poscar(structure, path, title="Sparse-DFT structure"):
    Path(path).write_text(poscar_text(structure,title),encoding='utf-8')
    return Path(path)

def read_xyz(path):
    lines=Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines)<2: raise InputError("XYZ is too short")
    try: n=int(lines[0].strip())
    except ValueError as e: raise InputError("first XYZ line must be atom count") from e
    if len(lines)<2+n: raise InputError("XYZ has fewer atoms than declared")
    comment=lines[1]; lattice=None; periodic=(False,False,False); warnings=[]
    # Extended XYZ lattice="a11 a12 ... a33" and pbc="T T T".
    m=re.search(r'Lattice="([^"]+)"',comment, re.I)
    if m:
        vals=[float(x) for x in m.group(1).split()]
        if len(vals)!=9: raise InputError("extended XYZ Lattice must contain 9 numbers")
        lattice=tuple(tuple(vals[3*i+j] for j in range(3)) for i in range(3))
        pm=re.search(r'pbc="([^"]+)"',comment,re.I)
        periodic=tuple(x.lower() in ("t","true","1") for x in (pm.group(1).split() if pm else ["T"]*3))
    else: warnings.append("ordinary XYZ has no lattice; treated as non-periodic")
    sp=[]; pos=[]
    for line in lines[2:2+n]:
        p=line.split();
        if len(p)<4: raise InputError(f"invalid XYZ atom line: {line}")
        sp.append(p[0]); pos.append(_floats(" ".join(p[1:4])))
    return Structure(sp,pos,lattice,False,periodic,None,"XYZ",str(path),warnings)

def read_cif(path):
    lines=Path(path).read_text(encoding='utf-8').splitlines(); vals={}; loop_headers=[]; loop_rows=[]; in_loop=False; i=0
    while i<len(lines):
        line=lines[i].strip(); i+=1
        if not line or line.startswith('#'): continue
        if line.lower()=="loop_":
            headers=[]; rows=[]; in_loop=True
            while i<len(lines) and lines[i].strip().startswith('_'):
                headers.append(lines[i].split()[0]); i+=1
            while i<len(lines):
                t=lines[i].strip()
                if not t or t.startswith('#'): i+=1; continue
                if t.lower()=="loop_" or t.startswith('_'): break
                rows.append(t.split()); i+=1
            if headers: loop_headers,loop_rows=headers,rows
            continue
        if line.startswith('_'):
            p=line.split(None,1); vals[p[0]]=p[1].split()[0] if len(p)>1 else (lines[i].strip() if i<len(lines) else '')
    def num(key):
        if key not in vals: raise InputError(f"CIF missing {key}")
        return float(re.sub(r"\([^)]*\)$","",vals[key]))
    a,b,c=map(num,['_cell_length_a','_cell_length_b','_cell_length_c'])
    alpha=math.radians(num('_cell_angle_alpha')); beta=math.radians(num('_cell_angle_beta')); gamma=math.radians(num('_cell_angle_gamma'))
    va=(a,0,0); vb=(b*math.cos(gamma),b*math.sin(gamma),0)
    cx=c*math.cos(beta); cy=c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)
    cz=math.sqrt(max(0,c*c-cx*cx-cy*cy)); lattice=(va,vb,(cx,cy,cz))
    try:
        hi=[h.lower() for h in loop_headers]
        si=next(j for j,h in enumerate(hi) if h in ('_atom_site_type_symbol','_atom_site_label'))
        xi=next(j for j,h in enumerate(hi) if h=='_atom_site_fract_x'); yi=next(j for j,h in enumerate(hi) if h=='_atom_site_fract_y'); zi=next(j for j,h in enumerate(hi) if h=='_atom_site_fract_z')
    except StopIteration as e: raise InputError("CIF lacks fractional atom loop") from e
    sp=[]; pos=[]
    for row in loop_rows:
        if len(row)<=max(si,xi,yi,zi): continue
        sp.append(re.sub(r"[^A-Za-z]", "", row[si])[:2]); pos.append(tuple(float(re.sub(r"\([^)]*\)$","",row[j])) for j in (xi,yi,zi)))
    if not sp: raise InputError("CIF atom loop is empty")
    return Structure(sp,pos,lattice,True,(True,True,True),None,"CIF",str(path))

def read_structure(path):
    p=Path(path); suf=p.suffix.lower()
    if suf in ('.vasp','.poscar') or p.name.upper()=='POSCAR': return read_poscar(p)
    if suf in ('.xyz','.extxyz'): return read_xyz(p)
    if suf=='.cif': return read_cif(p)
    # POSCAR files commonly have no suffix.
    try: return read_poscar(p)
    except Exception as e: raise InputError(f"cannot detect structure format for {p}: {e}") from e
