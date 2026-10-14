"""Widgets in the shared style: checkbox rows, radio groups, and a slider over
channel names. Ported from meshmash-benchmarks `viewing.py` and
meshmash-deployment `scripts/view_root.py`."""

from collections.abc import Callable, Sequence
from functools import partial

import pyvista as pv

from .style import INK

BUTTON_SIZE = 16
# Room for the text of a row, so rows do not overlap at a large display scale.
BUTTON_PITCH = BUTTON_SIZE + 14
WIDGET_FONT_SIZE = 9
BUTTON_ON = INK
BUTTON_OFF = "#d4d4d8"


def add_checkbox_row(
    plotter: pv.Plotter,
    callback: Callable[[bool], None],
    *,
    value: bool,
    x: int,
    y: int,
    label: str,
    name: str,
):
    """A checkbox at `(x, y)` (pixels from the bottom left) with its text label."""
    button = plotter.add_checkbox_button_widget(
        callback,
        value=value,
        position=(x, y),
        size=BUTTON_SIZE,
        color_on=BUTTON_ON,
        color_off=BUTTON_OFF,
    )
    plotter.add_text(
        label,
        position=(x + BUTTON_SIZE + 6, y),
        font_size=WIDGET_FONT_SIZE,
        color=INK,
        name=name,
    )
    return button


class RadioGroup:
    """A titled column of checkboxes where exactly one is on.

    `on_change(option)` runs when the selected option changes. The group starts at
    `top` (pixels from the bottom) and grows down; `bottom` is its lowest row.
    """

    def __init__(
        self,
        plotter: pv.Plotter,
        title: str,
        options: Sequence,
        on_change: Callable[[object], None],
        *,
        x: int,
        top: int,
        value=None,
    ):
        self.options = list(options)
        self.value = self.options[0] if value is None else value
        self._on_change = on_change
        plotter.add_text(
            title, position=(x, top), font_size=WIDGET_FONT_SIZE, color=INK, name=f"radio:{title}"
        )
        self._buttons = []
        y = top
        for option in self.options:
            y -= BUTTON_PITCH
            self._buttons.append(
                add_checkbox_row(
                    plotter,
                    partial(self._clicked, option),
                    value=option == self.value,
                    x=x,
                    y=y,
                    label=str(option),
                    name=f"radio:{title}:{option}",
                )
            )
        self.bottom = y

    def _clicked(self, option, state: bool) -> None:
        if state and option != self.value:
            self.value = option
            self._sync()
            self._on_change(option)
        else:
            # NOTE: a click on the selected option unchecks it; check it again.
            self._sync()

    def _sync(self) -> None:
        for option, button in zip(self.options, self._buttons):
            button.GetRepresentation().SetState(int(option == self.value))


class ChannelSlider:
    """A slider that picks one of `names`; `on_change(name)` runs on a new pick.
    Its title is the picked name."""

    def __init__(
        self,
        plotter: pv.Plotter,
        names: Sequence[str],
        on_change: Callable[[str], None],
        *,
        pointa=(0.55, 0.12),
        pointb=(0.95, 0.12),
        label=lambda name: name,
    ):
        self.names = list(names)
        self.value = self.names[0]
        self._on_change = on_change
        self._label = label

        def on_slide(value: float, widget) -> None:
            index = round(value)
            representation = widget.GetRepresentation()
            representation.SetValue(index)
            representation.SetTitleText(self._label(self.names[index]))
            if self.names[index] != self.value:
                self.value = self.names[index]
                self._on_change(self.value)

        self.widget = plotter.add_slider_widget(
            on_slide,
            rng=(0, len(self.names) - 1),
            value=0,
            title=self._label(self.value),
            pointa=pointa,
            pointb=pointb,
            style="modern",
            pass_widget=True,
            title_height=0.02,
            title_color=INK,
        )
        # The value is a position in `names`, which says nothing; the title names it.
        self.widget.GetRepresentation().ShowSliderLabelOff()

    def set_visible(self, on: bool) -> None:
        self.widget.SetEnabled(on)
