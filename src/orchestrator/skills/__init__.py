"""Skill registry with dynamic discovery.

Skills are loaded at start-up (and on ``POST /api/admin/reload``) from four sources:

1. built-in modules in :mod:`orchestrator.skills.builtin`;
2. extra importable packages listed in ``ORCH_SKILL_PACKAGES`` (comma-separated);
3. installed distributions that declare the ``agent_orchestrator.skills`` entry-point group;
4. ``*.py`` files in the plugin directory ``ORCH_SKILL_PLUGIN_DIR`` (default ``plugins/skills``).

Every concrete :class:`~orchestrator.contracts.Skill` subclass found is registered under its
``name``. Two skills with the same name are a configuration error.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
from collections.abc import Iterable
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any

from orchestrator.contracts import Skill, SkillContract

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "agent_orchestrator.skills"
BUILTIN_PACKAGE = "orchestrator.skills.builtin"


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill[Any, Any]] = {}
        self._sources: dict[str, str] = {}

    @classmethod
    def discover(cls, packages: Iterable[str] = (), plugin_dir: Path | None = None) -> SkillRegistry:
        registry = cls()
        for pkg in [BUILTIN_PACKAGE, *packages]:
            for module in _walk_package(pkg):
                registry._register_module(module, f"package:{pkg}")
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            obj = ep.load()
            if isinstance(obj, ModuleType):
                registry._register_module(obj, f"entry-point:{ep.name}")
            else:
                registry._register_class(obj, f"entry-point:{ep.name}")
        if plugin_dir is not None and plugin_dir.is_dir():
            for path in sorted(plugin_dir.glob("*.py")):
                if not path.name.startswith("_"):
                    registry._register_module(_load_file(path), f"plugin:{path.name}")
        log.info("skills loaded", extra={"skills": registry.names(), "sources": dict(registry._sources)})
        return registry

    def get(self, name: str) -> Skill[Any, Any]:
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"unknown skill {name!r}; registered: {', '.join(self.names())}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        return sorted(self._skills)

    def contracts(self) -> list[SkillContract]:
        return [self._skills[n].contract(self._sources[n]) for n in self.names()]

    def _register_module(self, module: ModuleType, source: str) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                self._register_class(obj, source)

    def _register_class(self, obj: object, source: str) -> None:
        if not (isinstance(obj, type) and issubclass(obj, Skill)) or inspect.isabstract(obj):
            return
        name = getattr(obj, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"skill class {obj.__qualname__} ({source}) has no name")
        if name in self._skills:
            raise ValueError(f"skill {name!r} from {source} clashes with one from {self._sources[name]}")
        self._skills[name] = obj()
        self._sources[name] = source


def _walk_package(name: str) -> list[ModuleType]:
    package = importlib.import_module(name)
    modules = [package]
    for info in pkgutil.iter_modules(getattr(package, "__path__", []), prefix=f"{name}."):
        modules.append(importlib.import_module(info.name))
    return modules


def _load_file(path: Path) -> ModuleType:
    module_name = f"orchestrator_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load skill plugin {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
