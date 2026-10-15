"""Forms generated from the sparse_dft argparse tree (see :mod:`sparse_dft_gui.argspec`)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from . import argspec, icons
from .i18n import translator, tr
from .widgets import CollapsibleSection, PathEdit, float_edit, int_edit, no_wheel, repolish

_INT_MIN, _INT_MAX = -2_000_000_000, 2_000_000_000


class Field(QWidget):
    """One argparse option: label + editor + hint; exposes :meth:`value`/:meth:`set_value`."""

    changed = pyqtSignal()

    def __init__(self, spec: argspec.OptionSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        # True once the user typed into the editor (or loaded a command line);
        # recommended values never overwrite such a field unless forced.
        self.user_edited = False
        self.label = QLabel(self)
        self.label.setWordWrap(False)
        self.flag_label = QLabel(self)
        self.flag_label.setProperty('role', 'flag')
        self.editor = self._make_editor()
        self.set_value(argspec.default_value(spec))

    # ---- construction
    def _make_editor(self):
        spec = self.spec
        w = spec.widget
        if w == 'check':
            editor = QCheckBox(self)
            editor.toggled.connect(self._user_changed)
        elif w == 'count':
            editor = no_wheel(QSpinBox(self))
            editor.setRange(0, 10)
            editor.valueChanged.connect(self._user_changed)
        elif w == 'choice':
            editor = no_wheel(QComboBox(self))
            if spec.default is None and not spec.required:
                editor.addItem(tr('label.inherit'), '')
            for choice in spec.choices:
                editor.addItem(argspec.format_value(choice), argspec.format_value(choice))
            editor.currentIndexChanged.connect(self._user_changed)
        elif w == 'int' and spec.default is not None and not spec.required:
            editor = no_wheel(QSpinBox(self))
            editor.setRange(_INT_MIN, _INT_MAX)
            editor.setAccelerated(True)
            editor.valueChanged.connect(self._user_changed)
        elif w == 'int':
            editor = int_edit(self)
            editor.textChanged.connect(self._user_changed)
        elif w == 'float':
            editor = float_edit(self)
            editor.textChanged.connect(self._user_changed)
        elif w == 'fixed':
            editor = QWidget(self)
            row = QHBoxLayout(editor)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            editor.parts = []
            for meta in spec.metavars:
                part = float_edit(editor) if spec.value_type == 'float' else (
                    int_edit(editor) if spec.value_type == 'int' else QLineEdit(editor))
                part.setPlaceholderText(meta)
                part.textChanged.connect(self._user_changed)
                row.addWidget(part)
                editor.parts.append(part)
        elif w == 'path':
            editor = PathEdit(spec.path_mode or 'open', spec.file_filter, self)
            editor.textChanged.connect(self._user_changed)
        else:  # text, multi, append, other
            editor = QLineEdit(self)
            if spec.secret:
                editor.setEchoMode(QLineEdit.EchoMode.Password)
            editor.textChanged.connect(self._user_changed)
        editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return editor

    # ---- values
    def _user_changed(self, *_):
        # Editor signals are blocked inside set_value, so this fires only on
        # interactive edits.
        self.user_edited = True
        self.changed.emit()

    def value(self):
        w, e = self.spec.widget, self.editor
        if w == 'check':
            return e.isChecked()
        if w == 'count':
            return e.value()
        if w == 'choice':
            return e.currentData() or ''
        if isinstance(e, QSpinBox):
            return str(e.value())
        if w == 'fixed':
            return [p.text().strip() for p in e.parts]
        return e.text()

    def set_value(self, value, *, user: bool = False):
        """Programmatic update; ``user=True`` marks it as the user's choice."""
        if user:
            self.user_edited = True
        w, e = self.spec.widget, self.editor
        blockers = [e] + list(getattr(e, 'parts', []))
        for b in blockers:
            b.blockSignals(True)
        try:
            if w == 'check':
                e.setChecked(bool(value))
            elif w == 'count':
                e.setValue(int(value or 0))
            elif w == 'choice':
                index = e.findData('' if value is None else str(value))
                e.setCurrentIndex(index if index >= 0 else 0)
            elif isinstance(e, QSpinBox):
                try:
                    e.setValue(int(float(value)))
                except (TypeError, ValueError):
                    e.setValue(int(self.spec.default or 0))
            elif w == 'fixed':
                items = list(value or [])
                for i, part in enumerate(e.parts):
                    part.setText(str(items[i]) if i < len(items) else '')
            else:
                e.setText('' if value is None else str(value))
        finally:
            for b in blockers:
                b.blockSignals(False)
        self.changed.emit()

    def reset(self):
        self.user_edited = False
        self.set_value(argspec.default_value(self.spec))

    def set_invalid(self, invalid: bool, message: str = ''):
        targets = list(getattr(self.editor, 'parts', [])) or [getattr(self.editor, 'edit', self.editor)]
        for t in targets:
            t.setProperty('invalid', bool(invalid))
            repolish(t)
        self.editor.setToolTip(message if invalid else self._tooltip)

    # ---- text
    def retranslate(self):
        t = translator()
        spec = self.spec
        text = t.option_label(spec)
        self.label.setText(text + (' *' if spec.required else ''))
        self.flag_label.setText(spec.flag or f'<{spec.dest}>')
        help_text = t.option_help(spec)
        hint = []
        if help_text:
            hint.append(help_text)
        if spec.default not in (None, '', False) and spec.widget not in ('check',):
            hint.append(tr('label.default_value', value=_fmt_default(spec.default)))
        if spec.widget == 'multi':
            hint.append(tr('label.tokens_hint'))
        if spec.widget == 'append':
            hint.append(tr('label.append_hint'))
        self._tooltip = '\n'.join(hint)
        for widget in (self.label, self.editor, self.flag_label):
            widget.setToolTip(self._tooltip)
        if spec.widget == 'choice' and spec.default is None and not spec.required:
            self.editor.setItemText(0, tr('label.inherit'))
        placeholder = ''
        if spec.widget in ('float', 'int', 'text', 'multi', 'append', 'path'):
            if spec.default not in (None, ''):
                placeholder = _fmt_default(spec.default)
            elif spec.required:
                placeholder = tr('label.required')
            elif spec.widget == 'append' and spec.metavar:
                placeholder = str(spec.metavar)
            else:
                placeholder = tr('label.unset')
            if hasattr(self.editor, 'setPlaceholderText') and not isinstance(self.editor, QSpinBox):
                self.editor.setPlaceholderText(placeholder)
        if isinstance(self.editor, QCheckBox):
            self.editor.setText('')


def _fmt_default(value):
    if isinstance(value, (list, tuple)):
        return ' '.join(argspec.format_value(v) for v in value)
    return argspec.format_value(value)


def group_key(title: str | None) -> str | None:
    if not title:
        return None
    return 'group.' + title.split('(')[0].strip().split()[0].lower()


class CommandForm(QWidget):
    """All options of one ``sparse_dft`` subcommand, a live command preview and a Run button."""

    submitted = pyqtSignal(dict)
    message = pyqtSignal(str, str)  # text, level

    def __init__(self, command: str, parser=None, parent=None, *, show_run=True):
        super().__init__(parent)
        self.command = command
        self.specs = argspec.command_specs(command, parser)
        self.fields: dict[str, Field] = {}
        self.sections: dict[str, CollapsibleSection] = {}
        self._show_required = False
        self.defaults_provider = None  # callable returning settings-derived defaults
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.sections_box = QVBoxLayout()
        self.sections_box.setSpacing(10)
        layout.addLayout(self.sections_box)
        self.empty_label = QLabel(self)
        self.empty_label.setProperty('role', 'muted')
        layout.addWidget(self.empty_label)
        self._build_sections()
        self.empty_label.setVisible(not self.specs)
        # command bar
        bar = QWidget(self)
        bar.setObjectName('CommandBar')
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.preview_label = QLabel(bar)
        self.preview_label.setProperty('role', 'muted')
        self.preview = QLineEdit(bar)
        self.preview.setObjectName('CommandPreview')
        self.preview.setReadOnly(True)
        self.copy_button = QPushButton(bar)
        self.copy_button.setProperty('kind', 'ghost')
        self.copy_button.clicked.connect(self.copy_command)
        self.reset_button = QPushButton(bar)
        self.reset_button.setProperty('kind', 'ghost')
        self.reset_button.clicked.connect(self.reset)
        self.run_button = QPushButton(bar)
        self.run_button.setProperty('kind', 'primary')
        self.run_button.clicked.connect(self.submit)
        self.run_button.setVisible(show_run)
        row.addWidget(self.preview_label)
        row.addWidget(self.preview, 1)
        row.addWidget(self.copy_button)
        row.addWidget(self.reset_button)
        row.addWidget(self.run_button)
        layout.addWidget(bar)
        self.retranslate()
        self.update_preview()

    def _build_sections(self):
        by_section: dict[str, list] = {}
        for spec in self.specs:
            name = spec.section if spec.section != 'group' else f'group:{spec.group}'
            by_section.setdefault(name, []).append(spec)
        def order(name):
            base = name.split(':')[0]
            return argspec.SECTION_ORDER.index(base) if base in argspec.SECTION_ORDER else 99
        small = len(self.specs) <= 8
        for name in sorted(by_section, key=order):
            base = name.split(':')[0]
            expanded = small or base in ('inputs', 'basis', 'convergence')
            style = 'parallel' if base == 'group' and 'parallel' in name.lower() else base
            section = CollapsibleSection('', style, expanded, self)
            section.group_title = name.split(':', 1)[1] if ':' in name else None
            grid = QGridLayout(section.body)
            grid.setContentsMargins(16, 4, 16, 14)
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(8)
            grid.setColumnStretch(1, 1)
            section.grid = grid
            for row_index, spec in enumerate(by_section[name]):
                field = Field(spec, section.body)
                field.changed.connect(self._on_changed)
                grid.addWidget(field.label, row_index, 0, Qt.AlignmentFlag.AlignVCenter)
                grid.addWidget(field.editor, row_index, 1)
                grid.addWidget(field.flag_label, row_index, 2, Qt.AlignmentFlag.AlignVCenter)
                self.fields[spec.dest] = field
            self.sections[name] = section
            self.sections_box.addWidget(section)

    # ---- values
    def values(self) -> dict:
        return {dest: field.value() for dest, field in self.fields.items()}

    def set_values(self, values: dict, *, force: bool = True, user: bool = False) -> list[str]:
        """Set widget values; ``force=False`` skips fields the user edited. Returns the dests set."""
        done = []
        for dest, value in values.items():
            field = self.fields.get(dest)
            if field is None or (field.user_edited and not force):
                continue
            field.set_value(value, user=user)
            done.append(dest)
        self.update_preview()
        return done

    def recommended_changes(self, values: dict, *, force: bool = False, always=()) -> list[dict]:
        """Preview of :meth:`apply_recommended`: one row per recommended option of this form.

        ``values`` holds typed values keyed by dest (``sparse_dft.presets`` options);
        ``action`` is ``set`` (will change), ``same`` (already equal) or ``kept``
        (user-edited, not forced). Dests in ``always`` count as forced.
        """
        rows = []
        for dest, typed in values.items():
            field = self.fields.get(dest)
            if field is None:
                continue
            new = argspec.widget_value(field.spec, typed)
            current = field.value()
            if argspec.canonical_value(field.spec, current) == argspec.canonical_value(field.spec, new):
                action = 'same'
            elif field.user_edited and not (force or dest in always):
                action = 'kept'
            else:
                action = 'set'
            rows.append({'command': self.command, 'dest': dest, 'current': current, 'new': new,
                         'user_edited': field.user_edited, 'action': action})
        return rows

    def apply_recommended(self, values: dict, force: bool = False, *, always=()) -> list[dict]:
        """Write recommended values into fields the user has not edited (all with ``force``)."""
        rows = self.recommended_changes(values, force=force, always=always)
        for row in rows:
            if row['action'] == 'set':
                self.fields[row['dest']].set_value(row['new'])
        self.update_preview()
        return rows

    def reset(self):
        """Back to argparse and settings defaults; clears the user-edited marks."""
        self._show_required = False
        for field in self.fields.values():
            field.reset()
        self.apply_defaults()
        self.update_preview()

    def argv(self) -> list[str]:
        return argspec.build_argv(self.command, self.specs, self.values())

    def load_argv(self, argv: list[str]):
        command, values = argspec.parse_argv(argv)
        if command != self.command:
            raise argspec.FormError({'command': f'{command} != {self.command}'})
        self.set_values(values, user=True)

    def apply_defaults(self, defaults: dict | None = None):
        """Fill empty fields from settings (PAW root, MPI ranks ...); never overwrite user input."""
        if defaults is None:
            defaults = self.defaults_provider() if self.defaults_provider else {}
        for dest, value in (defaults or {}).items():
            field = self.fields.get(dest)
            if field is None or value in (None, ''):
                continue
            current = field.value()
            if current in ('', None, [], argspec.default_value(field.spec)) or (
                    isinstance(current, list) and not any(current)):
                field.set_value(value)
        self.update_preview()

    # ---- preview and validation
    def _on_changed(self):
        self.update_preview()

    def validate(self, *, show_required=None) -> tuple[list | None, dict]:
        show_required = self._show_required if show_required is None else show_required
        try:
            argv = self.argv()
            errors = {}
        except argspec.FormError as error:
            argv, errors = None, error.errors
        for dest, field in self.fields.items():
            message = errors.get(dest)
            shown = message is not None and (show_required or message != 'required')
            field.set_invalid(shown, message or '')
        return argv, errors

    def update_preview(self):
        argv, errors = self.validate()
        if argv is None:
            partial = [self.command] + [f'<{d}>' for d in errors]
            self.preview.setText(argspec.display_command(partial))
        else:
            self.preview.setText(argspec.display_command(argv))
        self.preview.setCursorPosition(0)

    def copy_command(self):
        QApplication.clipboard().setText(self.preview.text())
        self.message.emit(tr('msg.copied'), 'ok')

    def submit(self):
        self._show_required = True
        argv, errors = self.validate(show_required=True)
        if argv is None:
            names = ', '.join(translator().option_label(self.fields[d].spec) if d in self.fields else d for d in errors)
            self.message.emit(tr('msg.invalid_form', fields=names), 'error')
            return None
        secret_env = {}
        for spec in self.specs:
            if spec.secret and spec.flag in argv:
                index = argv.index(spec.flag)
                if index + 1 < len(argv):
                    secret_env['MP_API_KEY'] = argv[index + 1]
                del argv[index:index + 2]
        payload = {'command': self.command, 'argv': argv, 'secret_env': secret_env,
                   'label': translator().command_title(self.command)}
        self.submitted.emit(payload)
        return payload

    # ---- text
    def retranslate(self):
        for name, section in self.sections.items():
            if section.group_title:
                key = group_key(section.group_title)
                section.setTitle(tr(key, section.group_title.split('(')[0].strip()))
            else:
                section.setTitle(tr(f'section.{section.section}'))
        for field in self.fields.values():
            field.retranslate()
            if isinstance(field.editor, PathEdit):
                field.editor.retranslate()
        self._align_columns()
        self.empty_label.setText(tr('label.no_options'))
        self.preview_label.setText(tr('label.command'))
        self.copy_button.setText(tr('btn.copy'))
        self.reset_button.setText(tr('btn.reset'))
        self.run_button.setText(tr('btn.run'))

    def _align_columns(self):
        """One label and one flag column width for every section, so editors line up form-wide."""
        if not self.fields:
            return
        label_width = max(150, *(f.label.sizeHint().width() for f in self.fields.values()))
        flag_width = max(f.flag_label.sizeHint().width() for f in self.fields.values())
        for section in self.sections.values():
            section.grid.setColumnMinimumWidth(0, min(label_width, 260))
            section.grid.setColumnMinimumWidth(2, flag_width)

    def set_icon_colour(self, colour: str, run_colour: str = '#1C2A18'):
        self.copy_button.setIcon(icons.icon('copy', colour, 16))
        self.run_button.setIcon(icons.icon('play', run_colour, 16))
        for field in self.fields.values():
            if isinstance(field.editor, PathEdit):
                field.editor.set_icon_colour(colour)
