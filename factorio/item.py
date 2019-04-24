from __future__ import annotations

import typing as t
from abc import ABCMeta
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction as Frac
from functools import lru_cache
from pathlib import Path

import slpp

FACTORIO_SYSDIR = Path("/usr/share/factorio/")
FACTORIO_RECIPES_DIR = FACTORIO_SYSDIR.joinpath("base/prototypes/recipe/")
FACTORIO_ITEMS_DIR = FACTORIO_SYSDIR.joinpath("base/prototypes/item/")


class NoSuchItem(Exception):
    pass


class attrdict(dict):
    def __getattr__(self, key):
        return self[key]


def name_map_objects(items: t.Iterable) -> attrdict:
    return attrdict(
        {item.name.replace("-", "_").upper(): item for item in items}
    )


def safe_frac(f: float) -> Frac:
    return Frac(f).limit_denominator(1000)


def read_lua_table(fn: Path) -> t.Generator[t.List[t.Any], None, None]:
    def _find_matching_paren(s) -> int:
        """
        Finds the first unmatched ")", indicating the closer.
        """
        depth = 0
        for cx, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth < 0:
                return cx
        else:
            raise ValueError("No closer!")

    with fn.open() as f:

        body = f.read().strip()

    # extend_str marks the opening of tables we read
    extend_str = "data:extend("
    while body.find(extend_str) >= 0:
        data_start_ix = body.find(extend_str) + len(extend_str)
        close_ix = _find_matching_paren(body[data_start_ix:]) + data_start_ix
        yield slpp.slpp.decode(body[data_start_ix:close_ix])
        body = body[close_ix:]


@lru_cache()
def get_proddable_items() -> t.FrozenSet[str]:
    fn = FACTORIO_ITEMS_DIR.joinpath("module.lua")
    header = "productivity_module_limitation"
    with fn.open() as f:
        body = f.read()

    head_ix = body.find(header)
    assert head_ix >= 0
    start = head_ix + len(header)
    start += body[start:].find("{")
    end = start + body[start:].find("}") + 1
    return frozenset(slpp.slpp.decode(body[start:end]))


@dataclass(frozen=True)
class Item:
    @dataclass(frozen=True)
    class Flow:
        """
        Represents an input or output rate of an item.
        """

        num: Frac
        item: Item

        def __str__(self) -> str:
            return f"{self.num}×{self.item}"

    @dataclass(frozen=True)
    class Subtype(metaclass=ABCMeta):
        pass

    @dataclass(frozen=True)
    class Plain(Subtype):
        pass

    @dataclass(frozen=True)
    class Module(Subtype):
        name: str
        prod: Frac
        poll: Frac
        speed: Frac
        power: Frac

        @staticmethod
        def simple_name(s: str) -> str:
            return s[0].upper() + s[-1]

        @staticmethod
        def by_name(s: str) -> Item.Module:
            try:
                prefix = {
                    "p": "productivity",
                    "e": "effectivity",
                    "s": "speed",
                }[s[0].lower()]
                suffix = s[1]
                return t.cast(
                    Item.Module, Item.by_name(f"{prefix}-module-{suffix}").meta
                )

            except KeyError:
                raise NoSuchItem

    __items: t.ClassVar[t.Dict[str, Item]] = {}
    __modules: t.ClassVar[t.Dict[str, Module]] = {}

    name: str
    meta: Subtype

    @staticmethod
    def renormalize_name(s: str) -> str:
        return s.replace("-", "_").upper()

    @staticmethod
    def by_name(key: str) -> Item:
        if not Item.__items:
            raise RuntimeError("Items was not constructed!")
        else:
            try:
                return Item.__items[Item.renormalize_name(key)]
            except KeyError:
                raise NoSuchItem

    @classmethod
    def initialize(cls) -> None:
        if cls.__items:
            return

        for fn in FACTORIO_ITEMS_DIR.glob("*.lua"):
            for raw in read_lua_table(fn):
                for item in raw:
                    if item["type"] in ["item-group", "item-subgroup"]:
                        continue
                    if "flags" in item and "hidden" in item["flags"]:
                        continue

                    # XXX why can't mypy handle 'cls.Subtype'
                    meta: Item.Subtype

                    if item["type"] == "module":
                        effect = item["effect"]
                        # declarative programmer's demands:
                        # assignment expressions NOW!
                        meta = cls.Module(
                            name=cls.Module.simple_name(item["name"]),
                            **{
                                k: safe_frac(
                                    effect.get(v, {"bonus": 0}).get("bonus", 0)
                                )
                                for k, v in {
                                    "prod": "productivity",
                                    "poll": "pollution",
                                    "speed": "speed",
                                    "power": "consumption",
                                }.items()
                            },
                        )
                    else:
                        meta = cls.Plain()

                    name = cls.renormalize_name(item["name"])
                    cls.__items[name] = cls(name=name, meta=meta)

        print("Loaded items.")

    @property
    def is_module(self) -> bool:
        return isinstance(self.meta, self.Module)

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name


@dataclass(frozen=True)
class Recipe:
    @dataclass(frozen=True)
    class Outcome:
        prob: Frac
        flows: t.List[Item.Flow]

        @classmethod
        def pure(cls, flows: t.List[Item.Flow]):
            return cls(Frac(1), flows)

    class Category(Enum):
        CraftingWithFluid = "crafting-with-fluid"
        Crafting = "crafting"
        AdvancedCrafting = "advanced-crafting"
        Centrifuging = "centrifuging"
        Smelting = "smelting"
        Chemistry = "chemistry"
        RocketBuilding = "rocket-building"

    class Version(Enum):
        Expensive = "expensive"
        Normal = "normal"

    __recipes: t.ClassVar[t.Dict[str, Recipe]] = {}
    __which: t.ClassVar[Recipe.Version]

    name: str
    time: Frac
    inputs: t.List[Item.Flow]
    outcomes: t.List[Outcome]
    category: Category
    proddable: bool

    @classmethod
    def by_name(cls, s: str) -> Recipe:
        s = Item.renormalize_name(s)
        try:
            return cls.__recipes[s]
        except KeyError:
            print(f"Could not find recipe for s. Partial matches:")
            for word in s.split("_"):
                for key, recipe in cls.__recipes.items():
                    if word in key:
                        print(key, ":", recipe)
            raise

    @classmethod
    def initialize(cls, which: Version = Version.Expensive):

        if cls.__recipes and cls.__which == which:
            return

        Item.initialize()

        def disambiguate_results(
            raw: t.Dict[str, t.Any]
        ) -> t.Optional[t.List[Recipe.Outcome]]:
            """
            Parses raw results entries and coerces them into a common format.

            This function is a disgusting monster. Handle it lovingly but with
            gloves.  """

            # this is a "simple" recipe
            if "result" in raw:
                return [
                    Recipe.Outcome.pure(
                        [
                            Item.Flow(
                                raw.get("result_count", 1),
                                Item.by_name(raw["result"]),
                            )
                        ]
                    )
                ]

            raw_results = raw["results"]
            assert isinstance(raw_results, list)

            # XXX dump fluid producing recipes
            if any(
                isinstance(x, dict) and x.get("type") == "fluid"
                for x in raw_results
            ):
                return None

            # XXX either all instances have a probability or none do
            all_prob = all(
                isinstance(x, dict) and "probability" in x for x in raw_results
            )
            no_prob = not any(
                isinstance(x, dict) and "probability" in x for x in raw_results
            )

            if all_prob:
                return [
                    Recipe.Outcome(
                        prob=safe_frac(prod["probability"]),
                        flows=[
                            Item.Flow(
                                num=prod["amount"],
                                item=Item.by_name(prod["name"]),
                            )
                        ],
                    )
                    for prod in raw_results
                ]
            elif no_prob:
                return [
                    Recipe.Outcome(
                        prob=Frac(1),
                        flows=[
                            (Item.Flow(item=Item.by_name(res[0]), num=res[1]))
                            if isinstance(res, list)
                            else (
                                Item.Flow(
                                    item=Item.by_name(res["name"]),
                                    num=res.get("amount", 1),
                                )
                            )
                            for res in raw_results
                        ],
                    )
                ]
            else:
                raise NotImplementedError("Mixed probability recipes!")

        for fn in FACTORIO_RECIPES_DIR.glob("*.lua"):
            for raw_recipes in read_lua_table(fn):
                for raw_recipe in raw_recipes:
                    if raw_recipe["type"] != "recipe":
                        continue
                    # XXX add fluids to our capabilities?
                    if raw_recipe.get("category") in ["oil-processing"]:
                        continue

                    raw_name = raw_recipe.get("name", raw_recipe.get("result"))
                    name = Item.renormalize_name(raw_name)
                    proddable = raw_name in get_proddable_items()
                    category = Recipe.Category(
                        raw_recipe.get("category", "crafting")
                    )

                    if "expensive" in raw_recipe:
                        try:
                            raw_recipe = raw_recipe[which.value]
                        except KeyError:
                            raise ValueError(
                                '"which" parameter must be one of '
                                '"expensive" or "normal"'
                            )

                    try:
                        results = disambiguate_results(raw_recipe)
                    except NoSuchItem:
                        continue

                    if results is None:
                        continue

                    time = safe_frac(raw_recipe.get("energy_required", "0.5"))

                    # XXX ignore fluid ingredients
                    inputs = [
                        Item.Flow(num=safe_frac(x[1]), item=Item.by_name(x[0]))
                        for x in raw_recipe["ingredients"]
                        if not isinstance(x, dict)
                    ]

                    recipe = Recipe(
                        name=name,
                        time=time,
                        category=category,
                        outcomes=results,
                        inputs=inputs,
                        proddable=proddable,
                    )

                    cls.__recipes[name] = recipe

        print("Loaded recipes.")

    @property
    def is_deterministic(self) -> bool:
        return len(self.outcomes) == 1

    @property
    def is_simple(self) -> bool:
        return self.is_deterministic and len(self.outcomes[0].flows) == 1

    @property
    def simple_prod(self) -> Item.Flow:
        if not self.is_simple:
            raise ValueError("Recipe is not simple!")
        else:
            return self.outcomes[0].flows[0]

    @property
    def products(self) -> t.FrozenSet[Item]:
        """
        Returns a set of all possible products
        """
        return frozenset(
            [flow.item for outcome in self.outcomes for flow in outcome.flows]
        )

    def __post_init__(self):
        assert sum(outcome.prob for outcome in self.outcomes) == Frac(
            1
        ), "Recipe constructed with outcome probabilities not summing to 1!"

    def __str__(self) -> str:
        out = (
            f'Recipe<{self.category.value}> "{self.name}": '
            f'{"+".join(map(str, self.inputs))} -> '
        )

        if self.is_simple:
            out_flow = self.simple_prod
            out += str(out_flow)
        else:
            # TODO prettify
            out += str(self.outcomes)

        return out
