from __future__ import annotations

import sys
import typing as t
from pprint import pprint
from argparse import ArgumentParser
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import Enum
from fractions import Fraction as Frac
from math import ceil

from belt_solver import Belt, BeltAssignment, solve_belts
from item import Item, Recipe


@dataclass(frozen=True)
class BaseManuf:
    '''
    Base manufactory class.

    Supports input/output computations.
    '''

    name: str
    base_speed: Frac
    base_power: Frac
    base_poll: Frac
    module_slots: int
    recipe_cap: t.FrozenSet[Recipe.Category]

    modules: t.List[Item.Module] = field(default_factory=list)
    recipe: t.Optional[Recipe] = None

    @property
    def speed(self) -> Frac:
        return Frac(
            self.base_speed
            + sum(self.base_speed * m.speed for m in self.modules)
        )

    @property
    def prod(self) -> Frac:
        return Frac(Frac(1) + sum(Frac(m.prod) for m in self.modules))

    @property
    def power(self) -> Frac:
        power = self.base_power * sum(m.power for m in self.modules)
        return Frac(max(self.base_power * Frac('0.2'), power))

    @property
    def base_rps(self) -> Frac:
        '''
        Base recipes per second.
        '''
        if self.recipe is None:
            raise ValueError('Must set produced recipe first!')
        # XXX generalize check when recipe types are discriminated

        if not self.recipe.is_simple:
            raise NotImplementedError(
                'Only simple recipes are supported for now'
            )

        return Frac(self.recipe.simple_prod.num / self.recipe.time)

    def with_modules(self, modules: t.List[Item.Module]) -> BaseManuf:
        if len(modules) > self.module_slots:
            raise ValueError(
                f'Too many modules for a {self.name} manufactory!'
            )
        if self.recipe and any(m.prod != Frac(0)
                               for m in modules) and not self.recipe.proddable:
            raise ValueError(
                'Assigning productivity modules to non-proddable recipe!'
            )
        return dc_replace(self, modules=modules)

    def with_recipe(self, recipe: Recipe) -> BaseManuf:
        if self.modules and any(m.prod != Frac(0) for m in self.modules
                                ) and not recipe.proddable:
            raise ValueError(
                'Assigning non-proddable recipe to manuf. with prodmods!'
            )
        if recipe.category not in self.recipe_cap:
            raise ValueError(
                f'Recipe ({recipe.category})incompatible with '
                f'this manufactory ({self.recipe_cap}'
            )
        return dc_replace(self, recipe=recipe)

    def solve_rates(
            self, outflow: Frac, **solver_kwargs
    ) -> t.Tuple[t.List[BeltAssignment], int, t.List[Item.Flow]]:
        '''
        Solves the number of units to fill a full belt
        '''

        # NOTE we divide by 2 since the line solution is for half the flow
        # see: symmetric layout assumed (TODO) by belt_solver
        out_rate = outflow / 2

        # base output per sec per unit
        rps = Frac(self.speed) * self.base_rps

        # NOTE this is the needed manufactories per side
        need_mfs = Frac(out_rate / rps) / self.prod

        assert self.recipe is not None

        need_flows = [
            dc_replace(base_flow, num=base_flow.num * need_mfs * rps)
            for base_flow in self.recipe.inputs
        ]

        belt_assignment = solve_belts(need_flows)

        return belt_assignment, need_mfs, need_flows

    def solve_report(self, outflow: Frac, **solver_kwargs):

        print(
            f'Info/Recipe: {self.recipe}\n'
            f'Info/Manuf: ({self.name})['
            + ','.join(m.name for m in self.modules) + ']' + '<' + ' '.join([
                f'{attr}={sum(getattr(m, attr) for m in self.modules)}'
                for attr in ['prod', 'speed', 'poll', 'power']
            ]) + '>'
        )
        print(
            f'Info/Manuf: speed: {self.speed} '
            f'(rps = {self.prod * self.speed * self.base_rps}); '
            f'power: {self.power} kW; prod: {self.prod}'
        )
        print(f'Info/Target: output of {outflow}/s')

        try:
            assignment, need_mfs, need_flows = self.solve_rates(
                outflow, **solver_kwargs
            )
            total_power = self.power * need_mfs
        except BeltAssignment.Infeasible:
            print('No feasible solution!')
            return

        # TODO support different layouts
        print('Solution/Layout:\n' '\t^^   ^v^   ^^\n' '\t01 M 2P2 M 10')
        print(
            f'Solution/Input: supply:\n\t'
            + '\n\t'.join(f'{inflow} per second' for inflow in need_flows)
        )
        # XXX depends on layout
        print(
            f'Solution/ManufCount: 2 x {ceil(need_mfs/2)} '
            f'({need_mfs}) manufactories'
        )
        print(f'Solution/Power: {round(float(total_power / 1000), 3)} MW')
        print(
            'Solution/Belts:\n\t' + '\n\t'.join(map(str, sorted(assignment)))
        )


class Manuf(Enum):
    Assembler1 = BaseManuf(
        'Assembler 1', Frac('0.5'), Frac('75'), Frac('4'), 0,
        frozenset([Recipe.Category.Crafting])
    )
    Assembler2 = BaseManuf(
        'Assembler 2', Frac('0.75'), Frac('150'), Frac('3'), 2,
        frozenset([
            Recipe.Category.Crafting,
            Recipe.Category.CraftingWithFluid,
            Recipe.Category.AdvancedCrafting,
        ])
    )
    Assembler3 = BaseManuf(
        'Assembler 3', Frac('1.25'), Frac('375'), Frac('2'), 4,
        frozenset([
            Recipe.Category.Crafting,
            Recipe.Category.CraftingWithFluid,
            Recipe.Category.AdvancedCrafting,
        ])
    )
    Chemplant = BaseManuf(
        'Chemical plant', Frac('1'), Frac('210'), Frac('4'), 3,
        frozenset([Recipe.Category.Chemistry])
    )
    # TODO we do not currently handle fluids
    Refinery = BaseManuf(
        'Refinery', Frac('1'), Frac('420'), Frac('6'), 3, frozenset()
    )
    Smelter = BaseManuf(
        'Electric Furnace', Frac('2'), Frac('180'), Frac('1'), 2,
        frozenset([Recipe.Category.Smelting])
    )
    Centrifuge = BaseManuf(
        'Centrifuge', Frac('1'), Frac('350'), Frac('4'), 2,
        frozenset([Recipe.Category.Centrifuging])
    )

    @classmethod
    def by_name(cls, s: str) -> Manuf:
        s = s.lower()
        if s.startswith('a'):
            return {
                '1': cls.Assembler1,
                '2': cls.Assembler2,
                '3': cls.Assembler3,
            }[s[-1]]
        elif s.startswith('s'):
            return cls.Smelter
        elif s.startswith('r'):
            return cls.Refinery
        elif s.startswith('ce'):
            return cls.Centrifuge
        elif s.startswith('ch'):
            return cls.Chemplant
        else:
            raise ValueError(f'Manufactory type "s" not understood')


def main():

    parser = ArgumentParser()
    parser.add_argument(
        'recipe',
        help='name of the recipe to use, as it appears in the factorio files'
    )
    parser.add_argument(
        'output', help='how much output is required. Number or belt.'
    )
    parser.add_argument(
        '--manuf', default='assembler3', help='manufactory to use for recipe'
    )
    parser.add_argument(
        '--modules', default='p3,p3,p3,p3', help='list of modules to use'
    )
    parser.add_argument(
        '--recipe-version',
        default='expensive',
        help='which recipe costs to use. "normal" or "expensive"'
    )

    args = parser.parse_args(sys.argv[1:])

    Recipe.initialize(which=Recipe.Version(args.recipe_version))
    modules = [Item.Module.by_name(mod) for mod in args.modules.split(',')]
    manuf = Manuf.by_name(args.manuf).value

    try:
        num_output = Frac(args.output)
    except TypeError:
        num_output = Belt.by_name(args.output).value

    if num_output > 45:
        raise NotImplementedError(
            'Outputs of greater than one blue belt not supported!'
        )
    elif num_output <= 0:
        raise ValueError('Output must be positive!')

    prod_manuf = (
        manuf.with_recipe(Recipe.by_name(args.recipe)).with_modules(modules)
    )

    print('Info/Params: optimizing with parameters:')
    pprint(vars(args))

    prod_manuf.solve_report(num_output)


if __name__ == '__main__':
    main()
