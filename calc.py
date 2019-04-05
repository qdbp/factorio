from __future__ import annotations

import typing as t
from collections import Counter
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import Enum
from fractions import Fraction as F
from math import ceil


class attrdict(dict):
    def __getattr__(self, key):
        return self[key]


T = t.TypeVar('T')


def name_map_items(items: t.Iterable) -> attrdict:
    return attrdict({
        item.name.replace('-', '_').upper(): item
        for item in items
    })


@dataclass(frozen=True)
class Module:

    name: str
    speed_gain: F = F('0')
    power_gain: F = F('0')
    prod_gain: F = F('0')
    poll_gain: F = F('0')


Modules = name_map_items([
    Module('P3', F('-0.15'), F('0.8'), F('0.10'), F('0.10')),
    Module('P2', F('-0.15'), F('0.6'), F('0.06'), F('0.07')),
    Module('P1', F('-0.15'), F('0.4'), F('0.04'), F('0.05')),
    Module('S3', F('0.5'), F('0.7')),
    Module('S2', F('0.3'), F('0.6')),
    Module('S1', F('0.2'), F('0.5')),
    Module('E3', power_gain=F('-0.5')),
    Module('E2', power_gain=F('-0.4')),
    Module('E1', power_gain=F('-0.5')),
])


class Belt(Enum):
    HY = F('7.5')
    Y = F('15')
    R = F('30')
    B = F('45')

    def __str__(self):
        return {
            self.Y: 'yellow belt',
            self.R: 'red belt',
            self.B: 'blue belt',
        }[self]

    def shortname(self) -> str:
        return {  # type: ignore
            self.Y: 'Y',
            self.R: 'R',
            self.B: 'B',
        }[self]

    @classmethod
    def from_rate(cls, rate: F):
        out: t.Counter[Belt] = Counter()
        while rate > 0:
            for belt in sorted(cls, key=lambda x: -x.value):  # type: ignore
                if rate >= belt.value:
                    out[belt] += 1
                    rate -= belt.value
                    break
            else:
                out[cls.Y] += 1
                rate -= rate
        return out

    @classmethod
    def pack_inflows(
        cls, inflows: t.List[t.Tuple[F, Item]], constraint: t.Counter[Belt]
    ):

        # not enough belt halves, straight up
        if len(inflows) > sum(constraint.values()) * 2:
            raise ValueError('Too many inflows!')


@dataclass(frozen=True)
class Item:
    name: str
    recipe: t.Optional[Recipe]
    is_fluid: bool = False
    is_final: bool = False

    @property
    def solid_inputs_per_item(self) -> t.List[t.Tuple[F, Item]]:
        if self.recipe is None:
            return []
        else:
            return [(F(ingr[0]) / F(self.recipe.amount), ingr[1])
                    for ingr in self.recipe.inputs]

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Recipe:
    amount: int
    time: F
    inputs: t.List[t.Tuple[int, Item]]


@dataclass(frozen=True)
class BaseManuf:
    '''
    Base manufactory class.

    Supports input/output computations.
    '''

    name: str
    base_speed: F
    base_power: F
    base_poll: F
    module_slots: int
    modules: t.List[Module] = field(default_factory=list)
    item: t.Optional[Item] = None

    @property
    def speed(self) -> F:
        return F(
            self.base_speed
            + sum(self.base_speed * m.speed_gain for m in self.modules)
        )

    @property
    def prod(self) -> F:
        return F(F(1) + sum(F(m.prod_gain) for m in self.modules))

    @property
    def power(self) -> F:
        power = self.base_power * sum(m.power_gain for m in self.modules)
        return F(max(self.base_power * F('0.2'), power))

    @property
    def base_cpsu(self) -> F:
        '''
        Base cycles per second.
        '''
        if self.item is None:
            raise ValueError('Must set produced item first!')
        # XXX generalize check when recipe types are discriminated
        if self.item.recipe is None:
            raise ValueError('Recipe incompatible with this manufactory')

        return F(F(self.item.recipe.amount) / F(self.item.recipe.time))

    def with_modules(self, modules: t.List[Module]) -> BaseManuf:
        if len(modules) > self.module_slots:
            raise ValueError(
                f'Too many modules for a {self.name} manufactory!'
            )

        return dc_replace(self, modules=modules)

    def with_producing(self, item: Item) -> BaseManuf:
        return dc_replace(self, item=item)

    def solve_rates(self, belt: Belt) -> t.Tuple[t.List[t.Tuple[F, Item]], F]:
        '''
        Solves the number of units to fill a full belt
        '''
        # TODO include inputs
        out_rate = F(belt.value)
        # base output per sec per unit
        cpsu = F(self.speed) * self.base_cpsu
        need_mfs = F(out_rate / cpsu) / self.prod

        assert self.item is not None
        need_inputs = [(need_mfs * cpsu * x[0], x[1])
                       for x in self.item.solid_inputs_per_item]

        return need_inputs, need_mfs

    def solve_report(self, belt: Belt, target='output'):
        if target == 'output':
            in_rates, need_mfs = self.solve_rates(belt)
        else:
            raise NotImplementedError

        total_power = self.power * need_mfs

        for ing in in_rates:
            belts = Belt.from_rate(ing[0])

        print(
            f'Producing {self.item} in ({self.name})['
            + ','.join(m.name for m in self.modules) + ']\n'
            f'speed: {self.speed} '
            f'(ops = {self.prod * self.speed * self.base_cpsu}); '
            f'power: {self.power} prod: {self.prod}\n'
            f'Target: {target} of one {belt};\n\n'
            f'Output: you need {need_mfs} manufactories '
            f'({round(float(total_power / 1000), 3)} MW);\n'
            f'Output: rounded, this makes lines of {ceil(need_mfs)};\n'
            f'Input: you need to supply:\n\t'
            + '\n\t'.join(f'{x[0]} {x[1].name} per second' for x in in_rates)
        )


class Manuf:
    A1 = BaseManuf('Assembler 1', F('0.5'), F('75'), F('4'), 0)
    A2 = BaseManuf('Assembler 2', F('0.75'), F('150'), F('3'), 2)
    A3 = BaseManuf('Assembler 3', F('1.25'), F('375'), F('2'), 4)
    C = BaseManuf('Chemical plant', F('1'), F('210'), F('4'), 3)
    R = BaseManuf('Refinery', F('1'), F('420'), F('6'), 3)
    S = BaseManuf('Electric Furnace', F('2'), F('180'), F('1'), 2)


# TODO autofill from game data
Items = name_map_items([
    Item('copper-ore', None),
    Item('copper-cable', None),
    Item('plastic-bar', None),
    Item('electronic-circuit', None),
])
Items.update(
    name_map_items([
        Item('copper-plate', Recipe(1, F(32, 10), [(1, Items.COPPER_ORE)])),
        Item(
            'advanced-circuit',
            Recipe(
                1, F(6), [
                    (8, Items.COPPER_CABLE),
                    (4, Items.PLASTIC_BAR),
                    (2, Items.ELECTRONIC_CIRCUIT),
                ]
            )
        ),
    ])
)

if __name__ == '__main__':
    manuf = (
        Manuf.A3.with_modules(4 * [Modules.P3]
                              ).with_producing(Items.ADVANCED_CIRCUIT)
    )

    manuf.solve_report(Belt.Y, 'output')
