import os
import pprint
import random
import wx
import wx.lib.agw.floatspin as fs

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.figure import Figure

from math import log

from algorithms import *


class LineGraph(FigCanvas):
    def __init__(self, parent, drawfunc, lower=0, upper=100, step=1):
        self.parent = parent
        self.figure = Figure()
        super(FigCanvas, self).__init__(parent, wx.ID_ANY, self.figure)
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.lower = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=lower)
        self.upper = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=upper)
        self.step = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=0, max=2048, initial=step)
        self.sizer = self.create_sizer()
        self.plot = drawfunc

    @property
    def abscissa(self):
        return np.arange(self.lower.GetValue(), self.upper.GetValue(), self.step.GetValue())

    def create_sizer(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(fig, proportion=1, flag=wx.LEFT | wx.TOP | wx.EXPAND)
        bounds = wx.BoxSizer(wx.HORIZONTAL)
        bounds.Add(wx.StaticText(self.parent, label="Lower bound"))
        bounds.Add(fig.lower)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.parent, label="Step"))
        bounds.Add(fig.step)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.parent, label="Upper bound"))
        bounds.Add(fig.upper)
        vbox.Add(bounds, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        for widget in (fig.lower, fig.upper, fig.step):
            self.Bind(wx.EVT_SPINCTRL, fig.plot, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_bounds_enter, widget)

        return vbox


class BarGraph(FigCanvas):
    def __init__(self, parent, drawfunc):
        self.parent = parent
        self.plot = drawfunc


class Frame(wx.Frame):
    title = 'Differential Privacy of the Above Threshold Mechanism'

    def __init__(self):
        wx.Frame.__init__(self, None, -1, self.title)

        self.create_menu()
        self.create_model()
        self.create_view()

        self.draw()

    def create_menu(self):
        self.menubar = wx.MenuBar()

        menu_file = wx.Menu()
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(wx.ID_ANY, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        menu_help = wx.Menu()
        m_about = menu_help.Append(wx.ID_ANY, "&About\tF1", "About the demo")
        self.Bind(wx.EVT_MENU, self.on_about, m_about)

        self.menubar.Append(menu_file, "&File")
        self.menubar.Append(menu_help, "&Help")
        self.SetMenuBar(self.menubar)

    def create_model(self):
        # TODO: init input vectors
        pass

    def create_view(self):
        self.main_panel = wx.Panel(self)

        self.vector_control = self.create_vector_control(self.main_panel)
        self.parameter_control = self.create_parameter_control(self.main_panel)
        self.accuracy_control = self.create_accuracy_control(self.main_panel)
        self.graphs = self.create_graphs(self.main_panel)
        self.stats = self.create_stats(self.main_panel)

        main = wx.BoxSizer(wx.VERTICAL)
        lower = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)

        main.Add(self.vector_control, proportion=0, flag=wx.ALL, border=10)
        lower.Add(left)
        lower.Add(self.graphs, proportion=1)
        left.Add(self.parameter_control)
        left.Add(self.stats)
        left.Add(self.accuracy_control)

        self.main_panel.SetSizer(main)
        main.Fit(self)

    def create_control(self):
        self.query_a = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=-1000, max=1000, initial=105)
        self.query_b = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=-1000, max=1000, initial=100)

        self.label_epsilon = wx.StaticText(self.panel, -1,
            "Epsilon (1/1000)")
        self.slider_epsilon = wx.Slider(self.panel, -1,
            minValue=1, maxValue=1000, value=100,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider_epsilon.SetTickFreq(1)
        self.label_interval = wx.StaticText(self.panel, -1,
            "Divergence interval")
        self.slider_interval = fs.FloatSpin(self.panel, -1,
            min_val=0.01, max_val=1000, value=10, digits=2,
            agwStyle=fs.FS_RIGHT)
        self.a_greater_b = wx.StaticText(self.panel, -1, "")
        self.calculate_a_greater_b()

        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_parameter_change, self.slider_epsilon)
        self.Bind(fs.EVT_FLOATSPIN, self.on_slider_interval, self.slider_interval)
        for widget in [self.query_a, self.query_b]:
            self.Bind(wx.EVT_SPINCTRL, self.on_parameter_change, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_bounds_enter, widget)

        controls = wx.BoxSizer(wx.VERTICAL)
        flags = wx.EXPAND | wx.TOP | wx.BOTTOM
        controls.Add(wx.StaticText(self.panel, -1, "Query A"))
        controls.Add(self.query_a, 0, border=3, flag=flags)
        controls.Add(wx.StaticText(self.panel, -1, "Query B"))
        controls.Add(self.query_b, 0, border=3, flag=flags)
        controls.Add(self.label_epsilon, 0, flag=flags)
        controls.Add(self.slider_epsilon, 0, border=3, flag=flags)
        controls.Add(self.label_interval, 0, flag=flags)
        controls.Add(self.slider_interval, 0, border=3, flag=flags)
        controls.Add(self.a_greater_b, 0, border=5, flag=flags)
        return controls

    def create_vector_control(self, parent):
        pass

    def create_parameter_control(self, parent):
        pass

    def create_accuracy_control(self, parent):
        pass

    def create_graphs(self, parent):
        graphs = wx.Panel(parent)

        bars_original = BarGraph(self.graphs, self.draw_original)
        bars_shifted = BarGraph(self.graphs, self.draw_shifted)
        accuracy = Graph(self.graphs, self.draw_accuracy, lower=80, upper=120)

        box = wx.BoxSizer(wx.VERTICAL, flag=wx.EXPAND)
        box.Add(bars_original.sizer, proportion=0, flag=wx.EXPAND)
        box.Add(bars_shifted.sizer, proportion=0, flag=wx.EXPAND)
        box.Add(accuracy.sizer, proportion=0, flag=wx.EXPAND)

        graphs.SetSizer(box)

        return graphs

    def create_stats(self, parent):
        pass

    def draw(self):
        pass

    def on_vector_change(self, event):
        pass

    def on_parameter_change(self, event):
        self.draw()

    def on_exit(self, event):
        self.Destroy()

    def on_about(self, event):
        msg = """Dynamically parametrize the Above Threshold Algorithm

        * Set a result vector
        * Set a query vector
        * Set a query vector for a neighboring database
        * Adjust the algorithm parameters T, e1, e2, sensitivity, count

        The program displays the queries' individual probabilities to produce
        the given result vector enties, as well as the probability of the whole
        query vector producing the given result vector.

        Query values below threshold are highlighted, as well as incorrect
        result vector entries.
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
