import os
import random
import wx
import wx.lib.agw.floatspin as fs
from wx.lib.intctrl import IntCtrl
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.figure import Figure
from matplotlib import cm as colormap
import matplotlib.ticker as ticker

from collections import Counter
from enum import Enum
from math import exp
from math import log
import numpy as np

from algorithms import Laplace as Lap
from algorithms import factor


def laplace(epsilon, monotonicity, sensitivity, x, y):
    scale = monotonicity * sensitivity
    A = Lap(scale/epsilon, 0)
    B = Lap(A.scale, A.loc + x)
    C = Lap(B.scale, B.loc + y)

    return A.larger(B), A.larger(C), B.larger(A), C.larger(A)


def exponential(epsilon, monotonicity, sensitivity, x, y):
    scale = monotonicity * sensitivity
    A = 1  # == exp(epsilon * 0 / 2), since we fix one query at 0
    norm_x = 1 / (1 + exp(epsilon * x / scale))
    B = norm_x * exp(epsilon * x / scale)
    A_b = norm_x * A

    norm_xy = 1 / (1 + exp(epsilon * (x + y) / scale))
    C = norm_xy * exp(epsilon * (x + y) / scale)
    A_c = norm_xy * A

    return A_b, A_c, B, C


class FunctionProxy(object):
    """Mask a function as an Object."""
    # source: https://stackoverflow.com/a/40486992/5147619
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Mode(Enum):
    Laplace = FunctionProxy(laplace)
    Exponential = FunctionProxy(exponential)

    @classmethod
    def names(cls):
        return list(cls._member_map_)


class Model(object):
    def __init__(
        self, epsilon, sensitivity=1, monotonic=True, offset=1, mode=Mode.Laplace):
        self.mode = mode
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.monotonic = monotonic
        self.offset = 1
        self.privacy_loss = 1

    def compute(self, x, y):
        A_b, A_c, B, C = self.mode.value(self.epsilon, factor(self.monotonic), self.sensitivity, x, y)
        return max(abs(log(B/C)), abs(log(A_b/A_c)))


class StaticBox(wx.StaticBox):
    def SetSizer(self, sizer):
        super(wx.StaticBox, self).SetSizer(sizer)
        # the label's height is always included in the total size, so compensate
        _, label_height = self.GetSize()
        self.SetMinSize(sizer.GetMinSize() + (0, label_height))


class Graph3D(wx.Panel):
    def __init__(self, parent, model, x=100, y=5, z=1, step=1):
        super().__init__(parent)
        self.figure = Figure()
        # order is important: https://stackoverflow.com/a/28399322/5147619
        self.canvas = FigCanvas(self, wx.ID_ANY, self.figure)
        self.axes = self.figure.gca(projection="3d")
        self.model = model

        self.x = wx.SpinCtrl(
            self, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=0, max=1000, initial=x)
        self.y = wx.SpinCtrl(
            self, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=0, max=100, initial=y)
        self.z = fs.FloatSpin(
            self, agwStyle=fs.FS_RIGHT, size=(60, -1),
            min_val=0.01, max_val=100, digits=2, increment=0.01, value=z)
        self.step = wx.SpinCtrl(
            self, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=1, max=2048, initial=step)

        self.sizer = self.create_sizer()

    def plot(self):
        raise NotImplementedError

    @property
    def domain(self):
        X = np.arange(-self.x.GetValue(), self.x.GetValue()+1, 1)
        Y = np.arange(-self.y.GetValue(), self.y.GetValue()+1, 1)
        return X, Y

    def create_sizer(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, proportion=1, flag=wx.LEFT | wx.TOP | wx.EXPAND)
        bounds = wx.BoxSizer(wx.HORIZONTAL)
        bounds.Add(wx.StaticText(self, label="X range"))
        bounds.Add(self.x)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self, label="Y range"))
        bounds.Add(self.y)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self, label="Z range"))
        bounds.Add(self.z)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self, label="Step"))
        bounds.Add(self.step)
        bounds.AddStretchSpacer()
        vbox.Add(bounds, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        for widget in (self.x, self.y, self.step):
            self.Bind(wx.EVT_SPINCTRL, self.plot, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, widget)
        self.Bind(fs.EVT_FLOATSPIN, self.plot, self.z)

        self.SetSizer(vbox)
        return vbox


class PrivacyLoss(Graph3D):
    def plot(self, event):
        ax = self.axes
        ax.clear()

        X, Y = self.domain
        xs, ys = np.meshgrid(X, Y)
        zs = np.vectorize(self.model.compute)(xs, ys)
        ax.plot_surface(xs, ys, zs, cmap=colormap.viridis, linewidth=0, antialiased=False)

        def slice(x):
            return self.model.compute(x, self.model.offset)

        ys = np.full(len(X), self.model.offset)
        zs = np.vectorize(slice)(X)
        ax.plot(X, ys, zs=zs, color="red")

        # this is not nice, plotting should be read-only, but anyways
        # update privacy privacy loss here, since we're at it
        self.model.privacy_loss = max(zs)

        ax.set_xlim(min(X), max(X))
        ax.set_ylim(min(Y), max(Y))
        ax.set_zlim(0, self.z.GetValue())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(r'$\mathcal{L}^{x}_{A \| B}$')
        ax.azim = 45
        ax.elev = 15
        mode = self.model.mode.name
        self.figure.suptitle("Privacy Loss for Report Noisy Max")
        self.canvas.draw()


class Frame(wx.Frame):
    title = 'Differential Privacy of Report Noisy Max'

    head_size = (80, -1)
    element_size = (30, -1)
    spinctrl_size = (80, -1)

    def __init__(self):
        wx.Frame.__init__(self, None, title=self.title)

        self.menubar = self.create_menu()
        self.model = Model(epsilon=0.1)
        self.create_view()
        self.draw()

    def create_menu(self):
        menubar = wx.MenuBar()

        menu_file = wx.Menu()
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(wx.ID_ANY, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        menu_help = wx.Menu()
        m_about = menu_help.Append(wx.ID_ANY, "&About\tF1", "About the demo")
        self.Bind(wx.EVT_MENU, self.on_about, m_about)

        menubar.Append(menu_file, "&File")
        menubar.Append(menu_help, "&Help")
        self.SetMenuBar(menubar)

        return menubar

    def create_view(self):
        self.main_panel = wx.Panel(self)

        self.mode = self.create_mode_control(self.main_panel)
        self.parameter_control = self.create_parameter_control(self.main_panel)
        self.graphs = self.create_graphs(self.main_panel)
        self.stats = self.create_stats(self.main_panel)

        main = wx.BoxSizer(wx.VERTICAL)
        lower = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)

        left.Add(self.mode, flag=wx.TOP | wx.BOTTOM | wx.EXPAND, border=10)
        left.Add(self.parameter_control, flag=wx.BOTTOM | wx.EXPAND, border=10)
        left.Add(self.stats, flag=wx.BOTTOM | wx.EXPAND, border=10)

        lower.Add(left, flag=wx.RIGHT | wx.LEFT, border=10)
        lower.Add(self.graphs, proportion=1)

        main.Add(lower, flag=wx.EXPAND)

        self.main_panel.SetSizer(main)

        # set the first column of independent boxes to the same width
        # and accomodate the panel if it got wider in the process
        left_panels = [self.parameter_control, self.stats]
        label_width = max(i.Sizer.GetChildren()[0].Size[0] for i in left_panels)
        for panel in left_panels:
            sizer = panel.Sizer
            sizer.SetItemMinSize(0, label_width, -1)
            min_size = sizer.GetMinSize()
            sizer.SetMinSize(min_size)
            sizer.Layout()
            min_width, _ = min_size
        left.SetMinSize((min_width, -1))

        main.Fit(self)

    def create_mode_control(self, parent):
        mode = wx.Choice(parent, choices=Mode.names())
        self.Bind(wx.EVT_CHOICE, self.on_mode, mode)
        return mode

    def create_parameter_control(self, parent):

        panel = StaticBox(parent, label="Algorithm parameters")

        epsilon_label = wx.StaticText(
            panel, label="ε", style=wx.ALIGN_RIGHT)
        self.epsilon = fs.FloatSpin(
            panel, agwStyle=fs.FS_RIGHT,
            min_val=0.001, max_val=1, value=self.model.epsilon,
            increment=0.01, digits=3, size=self.spinctrl_size)

        sensitivity_label = wx.StaticText(
            panel, label="Δ", style=wx.ALIGN_RIGHT)
        self.sensitivity = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=self.spinctrl_size,
            min=0, max=100, initial=self.model.sensitivity)

        monotonic_label = wx.StaticText(
            panel, label="Monotonic", style=wx.ALIGN_RIGHT)
        self.monotonic = wx.CheckBox(panel)
        self.monotonic.SetValue(self.model.monotonic)


        offset_label = wx.StaticText(
            panel, label="offset", style=wx.ALIGN_RIGHT)
        self.offset = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=self.spinctrl_size,
            min=0, max=100, initial=self.model.sensitivity)

        grid = [
            [epsilon_label, self.epsilon],
            [sensitivity_label, self.sensitivity],
            [monotonic_label, self.monotonic],
            [offset_label, self.offset],
        ]
        sizer = wx.FlexGridSizer(rows=len(grid), cols=len(grid[0]), gap=(5, 5))
        for line in grid:
            for item in line:
                sizer.Add(item, flag=wx.EXPAND)

        self.Bind(fs.EVT_FLOATSPIN, self.on_epsilon, self.epsilon)
        self.Bind(wx.EVT_SPINCTRL, self.on_sensitivity, self.sensitivity)
        self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, self.sensitivity)
        self.Bind(wx.EVT_CHECKBOX, self.on_monotonic, self.monotonic)
        self.Bind(wx.EVT_SPINCTRL, self.on_offset, self.offset)

        panel.SetSizer(sizer)
        return panel

    def create_graphs(self, parent):
        graphs = wx.Panel(parent)

        privacy_loss = PrivacyLoss(graphs, self.model)

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(privacy_loss, proportion=0, flag=wx.EXPAND)

        graphs.SetSizer(box)
        return graphs

    def create_stats(self, parent):
        panel = StaticBox(parent, label="Statistics")

        privacy_loss_label = wx.StaticText(
            panel, label="privacy loss", style=wx.ALIGN_RIGHT)

        self.privacy_loss = wx.StaticText(panel)

        grid = [
            [privacy_loss_label, self.privacy_loss],
        ]
        sizer = wx.FlexGridSizer(rows=len(grid), cols=len(grid[0]), gap=(5, 5))
        for line in grid:
            for item in line:
                sizer.Add(item, flag=wx.EXPAND)

        panel.SetSizer(sizer)
        return panel

    def update_stats(self):
        self.privacy_loss.SetLabel("{:.3f}".format(self.model.privacy_loss))


    def draw(self):
        for g in self.graphs.Children:
            g.plot(None)
        # this is shitty design, but order is important here
        # since plotting updates privacy loss
        self.update_stats()
        self.main_panel.Layout()

    def on_mode(self, event):
        self.model.mode = Mode[event.GetEventObject().GetString(self.mode.CurrentSelection)]
        self.on_parameter_change()

    def on_epsilon(self, event):
        self.model.epsilon = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_sensitivity(self, event):
        self.model.sensitivity = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_monotonic(self, event):
        self.model.monotonic = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_offset(self, event):
        self.model.offset = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_parameter_change(self):
        self.draw()

    def on_exit(self, event):
        self.Destroy()

    def on_about(self, event):
        msg = """Dynamically parametrize Report Noisy Max Experiment

        * Adjust the algorithm parameters epsilon, monotonicity, sensitivity
        * Set the y position of a cut through x

        The program displays the maximum privacy loss for Report Noisy Max and
        two queries with their distance on the x-axis. The red line shows
        privacy loss for a given offset between neighboring databases on the y-axis.
        """
        dlg = wx.MessageDialog(self, msg, "About", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()


def on_spin_enter(event):
    # workaround for annoying behavior of wxPython.
    # > if the user modifies the text in the edit part of the spin control directly,
    #   the EVT_TEXT is generated, like for the wx.TextCtrl. When the use enters text
    #   into the text area, the text is not validated until the control loses focus
    #   (e.g. by using the TAB key).
    # <https://wxpython.org/Phoenix/docs/html/wx.SpinCtrl.html#styles-window-styles>
    # solution: cycle focus
    spinctrl = event.GetEventObject()
    textctrl, spinbutton = spinctrl.GetChildren()
    spinbutton.SetFocus()
    spinctrl.SetFocus()

if __name__ == '__main__':
    app = wx.App()
    app.frame = Frame()
    app.frame.Show()
    app.MainLoop()
