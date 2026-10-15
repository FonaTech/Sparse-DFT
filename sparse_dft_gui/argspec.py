"""GUI-neutral description of the ``sparse_dft`` argparse tree (no Qt imports).

Forms are generated from :func:`sparse_dft.cli.build_parser`, so options added to
the CLI later (for example MPI controls) appear in the GUI automatically.
This module turns argparse actions into :class:`OptionSpec` records, turns
widget values back into an argv list, and parses an argv list into values.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import shlex
import subprocess
from dataclasses import dataclass, field

# Delegated subcommands that are not user-facing forms.
SKIP_COMMANDS = {'worker'}
SECRET_FLAGS = {'--api-key'}

# Input files given as positionals or options (dest names / long flags).
_OPEN_POSITIONALS = {'structure', 'vertices', 'reference', 'candidate', 'bands'}
_OPEN_FLAGS = {'--structure', '--path', '--doscar', '--procar', '--dos-scf', '--scf-checkpoint', '--vertices'}
_DIR_FLAGS = {'--paw-root', '--root'}
# Commands whose --out names a file rather than a directory.
_FILE_OUT_COMMANDS = {'make-band-path', 'bands-compare'}

STRUCTURE_FILTER = 'Structures (*POSCAR* *CONTCAR* *.vasp *.poscar *.cif *.xyz);;All files (*)'
JSON_FILTER = 'JSON (*.json *.jsonl);;All files (*)'
ANY_FILTER = 'All files (*)'

# GUI sections (collapsible group boxes) in display order.
SECTION_ORDER = ('inputs', 'basis', 'convergence', 'spin', 'relax', 'parallel', 'group', 'options')
_BASIS = {'cutoff', 'density_cutoff', 'accuracy', 'kpoint', 'scf_grid', 'mesh_mode', 'atomic_units',
          'projector_interpolation', 'bands', 'task', 'spacing', 'family', 'points_per_segment',
          'no_duplicate_vertices'}
_CONVERGENCE = {'max_iterations', 'density_tolerance', 'onecenter_density_tolerance', 'mixing', 'smearing_ev'}
_SPIN = {'spin_channels', 'initial_moments', 'hubbard'}
_RELAX = {'cell', 'max_steps', 'max_displacement', 'max_strain', 'derivative_method', 'force_tolerance',
          'stress_tolerance'}
_PARALLEL_HINTS = ('mpi', 'group', 'worker', 'thread', 'rank', 'omp')


@dataclass
class OptionSpec:
    command: str
    dest: str
    flags: list
    positional: bool
    widget: str              # text | path | float | int | choice | check | count | fixed | multi | append
    value_type: str          # str | float | int | other
    type_fn: object = None
    nargs: object = None
    count: int | None = None
    choices: list | None = None
    default: object = None
    required: bool = False
    help: str = ''
    metavar: object = None
    group: str | None = None  # argparse group title for custom groups
    path_mode: str | None = None  # open | dir | save
    file_filter: str = ANY_FILTER
    secret: bool = False
    action_name: str = ''
    section: str = 'options'

    @property
    def flag(self) -> str | None:
        """Preferred option string (first long form)."""
        if self.positional:
            return None
        longs = [f for f in self.flags if f.startswith('--')]
        return (longs or self.flags)[0]

    @property
    def key(self) -> str:
        return self.dest

    @property
    def metavars(self) -> list:
        if isinstance(self.metavar, (tuple, list)):
            return [str(m) for m in self.metavar]
        if self.count:
            return [str(self.metavar or self.dest.upper())] * self.count
        return [str(self.metavar)] if self.metavar else []

    # ---- conversion helpers
    def convert(self, token: str):
        """Convert one token with the real argparse type (raises ValueError/TypeError)."""
        if self.type_fn is None:
            return token
        return self.type_fn(token)

    def tokens(self, value) -> list[str]:
        """Split a multi/append widget value into tokens."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).replace(',', ' ') if self.widget == 'multi' else str(value)
        return [t for t in (text.split() if self.widget == 'multi' else _split_append(text)) if t]


def _split_append(text: str) -> list[str]:
    # Repeated options: separated by whitespace, newlines or ';'.
    return [t.strip() for t in text.replace(';', ' ').split() if t.strip()]


def format_value(value) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _type_name(fn) -> str:
    if fn is float:
        return 'float'
    if fn is int:
        return 'int'
    if fn is None or fn is str:
        return 'str'
    return 'other'


def _path_mode(command: str, dest: str, flag: str | None, positional: bool) -> tuple[str | None, str]:
    if positional and dest in _OPEN_POSITIONALS:
        return 'open', STRUCTURE_FILTER if dest == 'structure' else JSON_FILTER
    if flag in _DIR_FLAGS:
        return 'dir', ANY_FILTER
    if flag == '--structure':
        return 'open', STRUCTURE_FILTER
    if flag in _OPEN_FLAGS:
        return 'open', JSON_FILTER if flag in ('--path', '--dos-scf', '--scf-checkpoint', '--vertices') else ANY_FILTER
    if flag == '--out':
        return ('save', JSON_FILTER) if command in _FILE_OUT_COMMANDS else ('dir', ANY_FILTER)
    lowered = (flag or dest).lower()
    if any(word in lowered for word in ('-file', '-dir', 'path', 'checkpoint')):
        return ('dir', ANY_FILTER) if 'dir' in lowered else ('open', ANY_FILTER)
    return None, ANY_FILTER


def _section(spec: OptionSpec) -> str:
    if spec.group:
        return 'group'
    if spec.positional or spec.required or spec.path_mode:
        return 'inputs'
    if spec.dest in _BASIS:
        return 'basis'
    if spec.dest in _CONVERGENCE:
        return 'convergence'
    if spec.dest in _SPIN:
        return 'spin'
    if spec.dest in _RELAX:
        return 'relax'
    if any(h in spec.dest for h in _PARALLEL_HINTS):
        return 'parallel'
    return 'options'


def spec_from_action(command: str, action: argparse.Action, group: str | None = None) -> OptionSpec | None:
    if isinstance(action, (argparse._HelpAction, argparse._VersionAction, argparse._SubParsersAction)):
        return None
    positional = not action.option_strings
    flag = None if positional else ([f for f in action.option_strings if f.startswith('--')] or action.option_strings)[0]
    type_fn = action.type
    value_type = _type_name(type_fn)
    nargs = action.nargs
    count = None
    choices = list(action.choices) if action.choices is not None else None
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                           argparse._StoreConstAction)) or (
            hasattr(argparse, 'BooleanOptionalAction') and isinstance(action, argparse.BooleanOptionalAction)):
        widget = 'check'
    elif isinstance(action, argparse._CountAction):
        widget, value_type = 'count', 'int'
    elif isinstance(action, argparse._AppendAction):
        widget = 'append'
    elif isinstance(nargs, int) and nargs > 1:
        widget, count = 'fixed', nargs
    elif nargs in ('+', '*') or (isinstance(nargs, int) and nargs == 1):
        widget = 'multi'
    elif choices is not None:
        widget = 'choice'
    elif value_type == 'int':
        widget = 'int'
    elif value_type == 'float':
        widget = 'float'
    else:
        widget = 'text'
    path_mode, file_filter = _path_mode(command, action.dest, flag, positional)
    if path_mode and widget == 'text':
        widget = 'path'
    default = action.default
    if default is argparse.SUPPRESS:
        default = None
    spec = OptionSpec(command=command, dest=action.dest, flags=list(action.option_strings),
                      positional=positional, widget=widget, value_type=value_type, type_fn=type_fn,
                      nargs=nargs, count=count, choices=choices, default=default,
                      required=bool(action.required) if not positional else nargs not in ('?', '*'),
                      help=(action.help or '') if action.help is not argparse.SUPPRESS else '',
                      metavar=action.metavar, group=group, path_mode=path_mode, file_filter=file_filter,
                      secret=flag in SECRET_FLAGS, action_name=type(action).__name__)
    spec.section = _section(spec)
    return spec


def subparsers(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def command_help(parser: argparse.ArgumentParser) -> dict[str, str]:
    action = subparsers(parser)
    return {choice.dest: (choice.help or '') for choice in action._choices_actions} if action else {}


def parser_for(command: str, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser | None:
    """argparse parser of ``command``; delegated commands expose ``build_parser()`` in their module."""
    root = parser or _root_parser()
    action = subparsers(root)
    if action is None or command not in action.choices:
        return None
    sub = action.choices[command]
    if not sub.add_help and not [a for a in sub._actions if not isinstance(a, argparse._HelpAction)]:
        delegated = _delegated_parser(command)
        if delegated is not None:
            return delegated
    return sub


def _delegated_parser(command: str):
    try:
        from sparse_dft import cli
        module_name = getattr(cli, '_DELEGATED', {}).get(command, (None,))[0]
        if not module_name:
            return None
        module = importlib.import_module(module_name)
        builder = getattr(module, 'build_parser', None)
        return builder() if callable(builder) else None
    except Exception:  # noqa: BLE001 - optional module may be absent or broken
        return None


def _root_parser():
    from sparse_dft.cli import build_parser
    return build_parser()


def command_specs(command: str, parser: argparse.ArgumentParser | None = None) -> list[OptionSpec]:
    """Every option of ``command`` in declaration order, with argument-group titles."""
    sub = parser_for(command, parser)
    if sub is None:
        return []
    specs = []
    for group in sub._action_groups:
        title = group.title if group.title not in ('positional arguments', 'options', 'optional arguments') else None
        for action in group._group_actions:
            spec = spec_from_action(command, action, title)
            if spec is not None:
                specs.append(spec)
    # Mutually exclusive groups share actions with their container; keep first.
    seen, unique = set(), []
    for spec in specs:
        if spec.dest not in seen:
            seen.add(spec.dest); unique.append(spec)
    order = {action.dest: i for i, action in enumerate(sub._actions)}
    unique.sort(key=lambda s: order.get(s.dest, 0))
    return unique


def all_commands(parser: argparse.ArgumentParser | None = None) -> list[str]:
    root = parser or _root_parser()
    action = subparsers(root)
    return [name for name in action.choices if name not in SKIP_COMMANDS] if action else []


# --------------------------------------------------------------------------- values <-> argv

class FormError(ValueError):
    """One or more fields are invalid; ``errors`` maps dest -> message."""

    def __init__(self, errors: dict):
        super().__init__('; '.join(f'{k}: {v}' for k, v in errors.items()))
        self.errors = errors


def default_value(spec: OptionSpec):
    """The widget value that represents "not given"."""
    if spec.widget == 'check':
        return bool(spec.default) if isinstance(spec.default, bool) else False
    if spec.widget == 'count':
        return int(spec.default or 0)
    if spec.widget == 'fixed':
        if isinstance(spec.default, (list, tuple)):
            return [format_value(v) for v in spec.default]
        return [''] * (spec.count or 0)
    if spec.default is None:
        return ''
    if spec.widget in ('multi', 'append') and isinstance(spec.default, (list, tuple)):
        return ' '.join(format_value(v) for v in spec.default)
    return format_value(spec.default)


def widget_value(spec: OptionSpec, value):
    """Widget value for a typed value (e.g. from ``sparse_dft.presets``): fixed -> list of
    strings, multi/append -> space-joined, check -> bool, others -> formatted string;
    ``None`` -> the widget default (option not given)."""
    if value is None:
        return default_value(spec)
    if spec.widget == 'check':
        return bool(value)
    if spec.widget == 'count':
        return int(value or 0)
    items = list(value) if isinstance(value, (list, tuple)) else None
    if spec.widget == 'fixed':
        return [format_value(v) for v in (items if items is not None else [value])]
    if spec.widget in ('multi', 'append'):
        return ' '.join(format_value(v) for v in (items if items is not None else [value]))
    return '' if value is None else format_value(value)


def canonical_value(spec: OptionSpec, value):
    """Comparable form of a widget value: converted tokens, the default when unset."""
    if spec.widget in ('check', 'count'):
        return value
    if spec.widget == 'fixed':
        items = [str(v).strip() for v in (value or [])]
        if not any(items):
            items = default_value(spec)
        tokens = items
    elif spec.widget in ('multi', 'append'):
        tokens = spec.tokens(value) or spec.tokens(default_value(spec))
    else:
        text = '' if value is None else str(value).strip()
        tokens = [text or default_value(spec)]
    out = []
    for t in tokens:
        try:
            out.append(spec.convert(t) if t != '' else None)
        except (TypeError, ValueError, argparse.ArgumentTypeError):
            out.append(t)
    return out


def _same(spec: OptionSpec, converted, default) -> bool:
    if default is None:
        return False
    try:
        if isinstance(default, (list, tuple)):
            return list(converted) == [spec.convert(format_value(v)) if not isinstance(v, type(converted[0])) else v
                                       for v in default]
        return converted == default
    except Exception:  # noqa: BLE001
        return False


def _check_choice(spec, converted, errors):
    if spec.choices is not None and converted not in spec.choices:
        errors[spec.dest] = f'must be one of {spec.choices}'
        return False
    return True


def build_argv(command: str, specs: list[OptionSpec], values: dict, *, include_defaults: bool = False) -> list[str]:
    """argv for ``sparse_dft <command>`` from widget values; raises :class:`FormError`."""
    errors, positionals, options = {}, [], []
    for spec in specs:
        value = values.get(spec.dest, default_value(spec))
        try:
            if spec.widget == 'check':
                if bool(value) and not (isinstance(spec.default, bool) and spec.default is True
                                        and spec.action_name == '_StoreTrueAction'):
                    options.append(spec.flag)
                continue
            if spec.widget == 'count':
                options.extend([spec.flag] * int(value or 0))
                continue
            if spec.widget == 'fixed':
                items = [str(v).strip() for v in (value or [])]
                if not any(items):
                    if spec.required:
                        errors[spec.dest] = 'required'
                    continue
                if len(items) != spec.count or not all(items):
                    errors[spec.dest] = f'needs {spec.count} values'
                    continue
                converted = [spec.convert(t) for t in items]
                if not include_defaults and _same(spec, converted, spec.default):
                    continue
                options.extend([spec.flag, *items]) if spec.flag else positionals.extend(items)
                continue
            if spec.widget in ('multi', 'append'):
                tokens = spec.tokens(value)
                if not tokens:
                    if spec.required:
                        errors[spec.dest] = 'required'
                    continue
                converted = [spec.convert(t) for t in tokens]
                for c in converted:
                    if not _check_choice(spec, c, errors):
                        break
                else:
                    if not include_defaults and _same(spec, converted, spec.default):
                        continue
                    if spec.widget == 'append':
                        for t in tokens:
                            options.extend([spec.flag, t])
                    elif spec.flag:
                        options.extend([spec.flag, *tokens])
                    else:
                        positionals.extend(tokens)
                continue
            text = '' if value is None else str(value).strip()
            if not text:
                if spec.required:
                    errors[spec.dest] = 'required'
                continue
            converted = spec.convert(text)
            if not _check_choice(spec, converted, errors):
                continue
            if not spec.positional and not include_defaults and not spec.required and _same(spec, converted, spec.default):
                continue
            if spec.positional:
                positionals.append(text)
            else:
                options.extend([spec.flag, text])
        except (TypeError, ValueError, argparse.ArgumentTypeError) as error:
            errors[spec.dest] = f'invalid value ({error})'
    if errors:
        raise FormError(errors)
    return [command, *positionals, *options]


def parse_argv(argv: list[str], parser: argparse.ArgumentParser | None = None) -> tuple[str, dict]:
    """Inverse of :func:`build_argv`: ``(command, widget values)`` for a sparse_dft argv."""
    root = parser or _root_parser()
    command = argv[0]
    sub = parser_for(command, root)
    if sub is None:
        raise FormError({'command': f'unknown command {command!r}'})
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            namespace = sub.parse_args(list(argv[1:]))
        except SystemExit as exit_:
            raise FormError({'argv': 'cannot parse arguments'}) from exit_
    values = {}
    for spec in command_specs(command, root):
        raw = getattr(namespace, spec.dest, None)
        if spec.widget == 'check':
            values[spec.dest] = bool(raw)
        elif spec.widget == 'count':
            values[spec.dest] = int(raw or 0)
        elif spec.widget == 'fixed':
            values[spec.dest] = [format_value(v) for v in raw] if raw is not None else [''] * (spec.count or 0)
        elif spec.widget in ('multi', 'append'):
            values[spec.dest] = ' '.join(format_value(v) for v in raw) if raw else ''
        else:
            values[spec.dest] = '' if raw is None else format_value(raw)
    return command, values


def display_command(argv: list[str], *, program: str = 'sparse-dft', windows: bool | None = None) -> str:
    """Copyable shell command with secrets masked and paths quoted for the target shell."""
    from sparse_dft.runner import mask_argv
    parts = [program, *mask_argv(argv)]
    windows = (os.name == 'nt') if windows is None else windows
    if windows:
        return subprocess.list2cmdline(parts)
    return ' '.join(shlex.quote(p) for p in parts)
