from enum import Enum
import os
import wx
import wx.lib.agw.floatspin as fs

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.figure import Figure

from math import log

from algorithms import *

class Mode(Enum):
    Laplace = Laplace
    Gaussian = Gaussian

    @classmethod
    def names(cls):
        return list(cls._member_map_)

class Graph(FigCanvas):
    def __init__(self, parent, drawfunc, lower=-50, upper=50, steps=256):
        self.figure = Figure()
        super(FigCanvas, self).__init__(parent, -1, self.figure)
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.lower = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=lower)
        self.upper = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=upper)
        self.steps = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=0, max=2048, initial=steps)
        self.plot = drawfunc

    @property
    def abscissa(self):
        return np.linspace(self.lower.GetValue(), self.upper.GetValue(), self.steps.GetValue())


class Frame(wx.Frame):
    title = 'Differential Privacy of the Laplace mechanism'

    def __init__(self):
        wx.Frame.__init__(self, None, -1, self.title)

        self.create_menu()
        self.create_main_panel()

        self.draw_figure()

    def create_menu(self):
        self.menubar = wx.MenuBar()

        menu_file = wx.Menu()
        m_expt = menu_file.Append(-1, "&Save plot\tCtrl-S", "Save plot to file")
        self.Bind(wx.EVT_MENU, self.on_save_plot, m_expt)
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(-1, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        menu_help = wx.Menu()
        m_about = menu_help.Append(-1, "&About\tF1", "About the demo")
        self.Bind(wx.EVT_MENU, self.on_about, m_about)

        self.menubar.Append(menu_file, "&File")
        self.menubar.Append(menu_help, "&Help")
        self.SetMenuBar(self.menubar)

    def create_main_panel(self):
        self.panel = wx.Panel(self)

        self.queries = Graph(self.panel, self.draw_queries, steps=512)
        self.difference = Graph(self.panel, self.draw_difference)
        self.differenceCDF = Graph(self.panel, self.draw_differenceCDF)
        self.divergence = Graph(self.panel, self.draw_divergence)

        self.controls = self.create_controls()
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.controls, 0, wx.ALL, border=10)
        graphs = wx.GridSizer(rows=2, cols=2, hgap=10, vgap=0)
        graphs.Add(self.create_figure(self.queries), 0, wx.EXPAND)
        graphs.Add(self.create_figure(self.difference), 0, wx.EXPAND)
        graphs.Add(self.create_figure(self.divergence), 0, wx.EXPAND)
        graphs.Add(self.create_figure(self.differenceCDF), 0, wx.EXPAND)

        hbox.Add(graphs, 1)

        self.panel.SetSizer(hbox)
        hbox.Fit(self)

    def create_figure(self, fig):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(fig, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        bounds = wx.BoxSizer(wx.HORIZONTAL)
        bounds.Add(wx.StaticText(self.panel, -1, "Lower bound"))
        bounds.Add(fig.lower)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.panel, -1, "Steps"))
        bounds.Add(fig.steps)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.panel, -1, "Upper bound"))
        bounds.Add(fig.upper)
        vbox.Add(bounds, 0, wx.ALL | wx.EXPAND, border=10)

        for widget in [fig.lower, fig.upper, fig.steps]:
            self.Bind(wx.EVT_SPINCTRL, fig.plot, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_bounds_enter, widget)

        return vbox

    def create_controls(self):
        self.mode = wx.Choice(self.panel, choices=Mode.names())
        self.query_a = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=-1000, max=1000, initial=0)
        self.query_b = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=-1000, max=1000, initial=5)
        self.sensitivity = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=1, max=100, initial=1)

        self.label_epsilon = wx.StaticText(self.panel, label="ε (1/1000)")
        self.slider_epsilon = wx.Slider(
            self.panel, minValue=1, maxValue=1000, value=100,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider_epsilon.SetTickFreq(1)
        self.label_delta = wx.StaticText(self.panel, label="δ (1/1000)")
        self.slider_delta = wx.Slider(
            self.panel, minValue=1, maxValue=1000, value=10,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider_delta.SetTickFreq(1)
        self.a_greater_b = wx.StaticText(self.panel)

        self.Bind(wx.EVT_CHOICE, self.on_parameter_change, self.mode)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_parameter_change, self.slider_epsilon)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_parameter_change, self.slider_delta)
        for widget in [self.query_a, self.query_b, self.sensitivity]:
            self.Bind(wx.EVT_SPINCTRL, self.on_parameter_change, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_bounds_enter, widget)

        controls = wx.BoxSizer(wx.VERTICAL)
        flags = wx.EXPAND | wx.TOP | wx.BOTTOM
        controls.Add(self.mode)
        controls.Add(wx.StaticText(self.panel, label="q(D)"))
        controls.Add(self.query_a, 0, border=3, flag=flags)
        controls.Add(wx.StaticText(self.panel, label="q(D')"))
        controls.Add(self.query_b, 0, border=3, flag=flags)
        controls.Add(wx.StaticText(self.panel, label="Δq"))
        controls.Add(self.sensitivity, 0, border=3, flag=flags)
        controls.Add(self.label_epsilon, 0, flag=flags)
        controls.Add(self.slider_epsilon, 0, border=3, flag=flags)
        controls.Add(self.label_delta, 0, flag=flags)
        controls.Add(self.slider_delta, 0, border=3, flag=flags)
        controls.Add(self.a_greater_b, 0, border=5, flag=flags)
        return controls

    def draw_figure(self):
        self.update_model()
        for a in [self.queries, self.difference, self.differenceCDF, self.divergence]:
            a.plot()
        self.calculate_a_greater_b()

    def draw_queries(self, event=None):
        ax = self.queries.axes
        ax.clear()

        a, b = self.get_distributions()
        xs = self.queries.abscissa
        ys = [a.pdf(x) for x in xs]
        ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-", label="Pr(A)")
        ys = [b.pdf(x) for x in xs]
        ax.plot(xs, ys, color="green", linewidth=2.0, linestyle="-", label="Pr(B)")
        ax.legend(loc='upper right')

        self.queries.figure.suptitle("Query distributions")
        self.queries.draw()

    def draw_difference(self, event=None):
        ax = self.difference.axes
        ax.clear()

        a, b = self.get_distributions()
        xs = self.difference.abscissa
        f = a.difference(b)
        ys = [f(x) for x in xs]
        ax.plot(xs, ys, color="red", linewidth=2.0, linestyle="-", label="Pr(A-B)")
        ax.legend(loc='upper right')

        self.difference.figure.suptitle("PDF of difference between queries")
        self.difference.draw()

    def draw_differenceCDF(self, event=None):
        ax = self.differenceCDF.axes
        ax.clear()

        a, b = self.get_distributions()
        xs = self.differenceCDF.abscissa
        f = a.differenceCDF(b)
        ys = [f(x) for x in xs]
        ax.plot(xs, ys, color="red", linewidth=2.0, linestyle="-", label="CDF Pr(A-B)")
        ax.legend(loc='upper right')

        self.differenceCDF.figure.suptitle("CDF of difference between queries")
        self.differenceCDF.draw()

    def draw_divergence(self, event=None):
        ax = self.divergence.axes
        ax.clear()

        a, b = self.get_distributions()

        def divergence(x):
            one = log(a.pdf(x)/b.pdf(x))
            two = log((1-a.pdf(x))/(1-b.pdf(x)))
            return max(abs(one), abs(two))

        xs = self.divergence.abscissa
        ys = [divergence(x) for x in xs]
        ax.plot(xs, ys, color="red", linewidth=2.0, linestyle="-")
        epsilon = self.slider_epsilon.GetValue() / 1000
        ax.axhline(y=epsilon, color="green")

        self.divergence.figure.suptitle("Pointwise divergence of queries")
        self.divergence.draw()

    def get_distributions(self):
        return self.a, self.b

    def update_model(self):
        mode = Mode[self.mode.GetString(self.mode.CurrentSelection)]
        sensitivity = self.sensitivity.GetValue()
        epsilon = self.slider_epsilon.GetValue() / 1000

        if mode == Mode.Laplace:
            self.a = Laplace(sensitivity/epsilon, self.query_a.GetValue())
            self.b = Laplace(sensitivity/epsilon, self.query_b.GetValue())
            self.slider_delta.Enable(False)
            self.label_delta.Enable(False)
        else:
            delta = self.slider_delta.GetValue() / 1000
            c = sqrt(2*log(1.25/delta))
            self.a = Gaussian(c*sensitivity/epsilon, self.query_a.GetValue())
            self.b = Gaussian(c*sensitivity/epsilon, self.query_b.GetValue())
            self.slider_delta.Enable(True)
            self.label_delta.Enable(True)

    def calculate_a_greater_b(self):
        a, b = self.get_distributions()
        self.a_greater_b.SetLabel("Pr(A > B) = {:.3f}".format(a.larger(b)))

    def on_parameter_change(self, event):
        self.draw_figure()

    def on_save_plot(self, event):
        file_choices = "PNG (*.png)|*.png"

        dlg = wx.FileDialog(
            self,
            message="Save plot as...",
            defaultDir=os.getcwd(),
            defaultFile="plot.png",
            wildcard=file_choices,
            style=wx.SAVE)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)

    def on_exit(self, event):
        self.Destroy()

    def on_about(self, event):
        msg = """Dynamically parametrize the Report Noisy Max Algorithm

        * Enter values for query A and B
        * Adjust the value of epsilon
        * Adjust the bounds of the graph
        """
        dlg = wx.MessageDialog(self, msg, "About", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()


def on_bounds_enter(event):
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
