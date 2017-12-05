"""
This demo demonstrates how to embed a matplotlib (mpl) plot
into a wxPython GUI application, including:

* Using the navigation toolbar
* Adding data to the plot
* Dynamically modifying the plot's properties
* Processing mpl events
* Saving the plot to a file from a menu

The main goal is to serve as a basis for developing rich wx GUI
applications featuring mpl plots (using the mpl OO API).

Eli Bendersky (eliben@gmail.com)
License: this code is in the public domain
Last modified: 30.07.2008
"""
import os
import pprint
import random
import wx

# The recommended way to use wx with mpl is with the WXAgg
# backend.
#
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar

from math import inf

from algorithms import *

class Canvas(FigCanvas):
    def __init__(self, parent, lower=0, upper=100):
        self.figure = Figure()
        super(FigCanvas, self).__init__(parent, -1, self.figure)
        self.axes = self.figure.add_subplot(1,1,1)
        self.lower = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=lower)
        self.upper = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=upper)


class BarsFrame(wx.Frame):
    """ The main frame of the application
    """
    title = 'Report Noisy Max'

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
        """ Creates the main panel with all the controls on it:
             * mpl canvas
             * mpl navigation toolbar
             * Control panel for interaction
        """
        self.panel = wx.Panel(self)
        self.queries = Canvas(self.panel, lower=80, upper=120)
        self.difference = Canvas(self.panel, lower=-100, upper=100)



        #
        # Layout with box sizers
        #
        self.controls = self.create_controls()
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.controls, 0, wx.ALL, border=10)
        hbox.Add(self.create_figure(self.queries), 1, wx.EXPAND)
        hbox.Add(self.create_figure(self.difference), 1, wx.EXPAND)

        self.spincontrols = [
            self.queries.lower,
            self.queries.upper,
            self.query_a,
            self.query_b,
        ]

        for widget in self.spincontrols:
            self.Bind(wx.EVT_SPINCTRL, self.on_text_enter, widget)
            self.Bind(wx.EVT_TEXT_ENTER, self.on_bounds_enter, widget)

        self.panel.SetSizer(hbox)
        hbox.Fit(self)


    def create_figure(self, fig):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(fig, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        bounds = wx.BoxSizer(wx.HORIZONTAL)
        bounds.Add(wx.StaticText(self.panel, -1, "Lower bound"))
        bounds.Add(fig.lower)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.panel, -1, "Upper bound"))
        bounds.Add(fig.upper)
        vbox.Add(bounds, 0, wx.ALL | wx.EXPAND, border=10)
        return vbox

    def create_controls(self):
        self.query_a = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=-1000, max=1000, initial=100)
        self.query_b = wx.SpinCtrl(
            self.panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=-1000, max=1000, initial=105)

        self.slider_label = wx.StaticText(self.panel, -1,
            "Epsilon (1/1000): ")
        self.slider_epsilon = wx.Slider(self.panel, -1,
            minValue=1, maxValue=1000, value=100,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider_epsilon.SetTickFreq(1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_slider_epsilon, self.slider_epsilon)

        controls = wx.BoxSizer(wx.VERTICAL)
        flags = wx.EXPAND | wx.ALL | wx.ALIGN_CENTER_VERTICAL
        controls.Add(wx.StaticText(self.panel, -1, "Query A"))
        controls.Add(self.query_a, 0, border=3, flag=flags)
        controls.Add(wx.StaticText(self.panel, -1, "Query B"))
        controls.Add(self.query_b, 0, border=3, flag=flags)
        controls.Add(self.slider_label, 0, flag=flags)
        controls.Add(self.slider_epsilon, 0, border=3, flag=flags)
        return controls

    def draw_figure(self):
        self.draw_queries()
        self.draw_difference()

    def draw_queries(self):
        ax = self.queries.axes
        ax.clear()

        epsilon = self.slider_epsilon.GetValue() / 1000
        a = Laplace(1/epsilon, self.query_a.GetValue())
        b = Laplace(1/epsilon, self.query_b.GetValue())

        lower = self.queries.lower.GetValue()
        upper = self.queries.upper.GetValue()
        xs = np.linspace(lower, upper, 512)
        for f in [a,b]:
            ys = [f(x) for x in xs]
            ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-")
        ax.set_xlabel("Query distributions")

        self.queries.figure.suptitle("PDF of query difference")
        self.queries.draw()

    def draw_difference(self):
        ax = self.difference.axes
        ax.clear()

        epsilon = self.slider_epsilon.GetValue() / 1000
        a = Laplace(1/epsilon, self.query_a.GetValue())
        b = Laplace(1/epsilon, self.query_b.GetValue())

        lower = int(self.difference.lower.GetValue())
        upper = int(self.difference.upper.GetValue())
        xs = np.linspace(lower, upper, 512)
        ys = [a.difference(b)(x) for x in xs]
        ax.plot(xs, ys, color="red", linewidth=2.0, linestyle="-")
        ax.set_xlabel("PDF of difference between queries")

        self.difference.figure.suptitle("PDF of query difference")
        self.difference.draw()

    def on_slider_epsilon(self, event):
        self.draw_figure()

    def on_text_enter(self, event):
        self.draw_figure()

    def on_bounds_enter(self, event):
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


if __name__ == '__main__':
    app = wx.App()
    app.frame = BarsFrame()
    app.frame.Show()
    app.MainLoop()

