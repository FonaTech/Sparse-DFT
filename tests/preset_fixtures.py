"""Idealized structures, one or more per preset category (tests only).

Lattice constants are rounded textbook values; they only need to be close
enough for the bond and vacuum analysis.
"""
import math

from sparse_dft.structure import Structure


def fcc(a):
    return [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]


def bcc(a):
    return [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]


def hexagonal(a, c):
    return [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]


def rocksalt(a, cation, anion):
    return Structure([cation, anion], [(0, 0, 0), (.5, .5, .5)], fcc(a), True)


def zincblende(a, x, y):
    return Structure([x, y], [(0, 0, 0), (.25, .25, .25)], fcc(a), True)


def silicon_supercell(n=2):
    a = 5.43
    base = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    cell = base + [tuple(x + .25 for x in p) for p in base]
    pos = [tuple((p[k] + (i, j, l)[k]) / n for k in range(3))
           for i in range(n) for j in range(n) for l in range(n) for p in cell]
    return Structure(['Si'] * len(pos), pos, [[n * a, 0, 0], [0, n * a, 0], [0, 0, n * a]], True)


def graphene(vacuum=15.0):
    return Structure(['C', 'C'], [(1 / 3, 2 / 3, .5), (2 / 3, 1 / 3, .5)], hexagonal(2.46, vacuum), True)


def graphite():
    # AB stacking, c = 6.71 A.
    return Structure(['C'] * 4, [(0, 0, .25), (0, 0, .75), (1 / 3, 2 / 3, .25), (2 / 3, 1 / 3, .75)],
                     hexagonal(2.46, 6.71), True)


def mos2_monolayer(vacuum=18.0):
    h = 1.565 / vacuum
    return Structure(['Mo', 'S', 'S'], [(1 / 3, 2 / 3, .5), (2 / 3, 1 / 3, .5 + h), (2 / 3, 1 / 3, .5 - h)],
                     hexagonal(3.19, vacuum), True)


def mos2_bulk():
    c = 12.3
    h = 1.565 / c
    return Structure(['Mo', 'Mo', 'S', 'S', 'S', 'S'],
                     [(1 / 3, 2 / 3, .25), (2 / 3, 1 / 3, .75), (2 / 3, 1 / 3, .25 + h), (2 / 3, 1 / 3, .25 - h),
                      (1 / 3, 2 / 3, .75 + h), (1 / 3, 2 / 3, .75 - h)], hexagonal(3.16, c), True)


def al_slab(layers=5, vacuum=12.0):
    a, d = 2.8638, 2.025
    c = (layers - 1) * d + vacuum
    return Structure(['Al'] * layers, [((i % 2) * .5, (i % 2) * .5, i * d / c) for i in range(layers)],
                     [[a, 0, 0], [0, a, 0], [0, 0, c]], True)


def carbon_chain():
    return Structure(['C', 'C'], [(.5, .5, 0), (.5, .5, .5)], [[12, 0, 0], [0, 12, 0], [0, 0, 2.56]], True)


def co_box():
    return Structure(['C', 'O'], [(.5, .5, .5 - .0565), (.5, .5, .5 + .0565)],
                     [[10, 0, 0], [0, 10, 0], [0, 0, 10]], True)


def no2_box():
    # Bent NO2 radical (17 valence electrons with standard PAW).
    return Structure(['N', 'O', 'O'], [(.5, .5, .5), (.5 + .11, .5, .5 + .047), (.5 - .11, .5, .5 + .047)],
                     [[10, 0, 0], [0, 10, 0], [0, 0, 10]], True)


def rutile_tio2():
    a, c, u = 4.594, 2.959, 0.305
    return Structure(['Ti', 'Ti', 'O', 'O', 'O', 'O'],
                     [(0, 0, 0), (.5, .5, .5), (u, u, 0), (1 - u, 1 - u, 0), (.5 + u, .5 - u, .5), (.5 - u, .5 + u, .5)],
                     [[a, 0, 0], [0, a, 0], [0, 0, c]], True)


def fluorite(a, cation, anion):
    return Structure([cation, anion, anion], [(0, 0, 0), (.25, .25, .25), (.75, .75, .75)], fcc(a), True)


CASES = {
    # name: (structure factory, expected category)
    'Si': (lambda: zincblende(5.43, 'Si', 'Si'), 'semiconductor'),
    'GaAs': (lambda: zincblende(5.65, 'Ga', 'As'), 'semiconductor'),
    'MgO': (lambda: rocksalt(4.21, 'Mg', 'O'), 'ionic_insulator'),
    'LiF': (lambda: rocksalt(4.03, 'Li', 'F'), 'ionic_insulator'),
    'NaCl': (lambda: rocksalt(5.64, 'Na', 'Cl'), 'ionic_insulator'),
    'TiO2': (rutile_tio2, 'ionic_insulator'),
    'Al': (lambda: Structure(['Al'], [(0, 0, 0)], fcc(4.05), True), 'metal'),
    'Cu': (lambda: Structure(['Cu'], [(0, 0, 0)], fcc(3.61), True), 'metal'),
    'TiC': (lambda: rocksalt(4.33, 'Ti', 'C'), 'metal'),
    'Fe': (lambda: Structure(['Fe'], [(0, 0, 0)], bcc(2.83), True), 'ferromagnetic_metal'),
    'Ni': (lambda: Structure(['Ni'], [(0, 0, 0)], fcc(3.52), True), 'ferromagnetic_metal'),
    'NiO': (lambda: rocksalt(4.17, 'Ni', 'O'), 'tm_oxide'),
    'MnO': (lambda: rocksalt(4.44, 'Mn', 'O'), 'tm_oxide'),
    'CeO2': (lambda: fluorite(5.41, 'Ce', 'O'), 'f_electron'),
    'graphite': (graphite, 'vdw_bulk'),
    'MoS2_bulk': (mos2_bulk, 'vdw_bulk'),
    'graphene': (graphene, 'monolayer'),
    'MoS2': (mos2_monolayer, 'monolayer'),
    'Al_slab': (al_slab, 'slab'),
    'C_chain': (carbon_chain, 'wire'),
    'CO': (co_box, 'molecule'),
    'NO2': (no2_box, 'molecule'),
    'Si64': (silicon_supercell, 'large_cell'),
}
