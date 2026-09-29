"""Lazy adapters keep benchmark dependencies isolated until selected."""


def make_environment(name, options):
    if name == "alfworld":
        from .alfworld import ALFWorld
        return ALFWorld(**options)
    if name == "appworld":
        from .appworld import AppWorld
        return AppWorld(**options)
    if name == "scienceworld":
        from .scienceworld import ScienceWorld
        return ScienceWorld(**options)
    raise ValueError("Unknown benchmark: " + name)
