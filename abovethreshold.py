import os
import pprint
import random
import wx
import wx.lib.agw.floatspin as fs
from wx.lib.intctrl import IntCtrl

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.figure import Figure

from math import log

from algorithms import *


class Model(object):
    def __init__(
        self, threshold, e1, e2, sensitivity=1, monotonic=True,
            length=5, shift=1, maxint=1000):

        self.maxint = maxint
        self.shift = shift
        self.length = length

        self.random_response()
        self.random_queries()
        self.set_shift_vector()

    def random_response(self):
        self.response = [self.randbool() for _ in range(self.length)]

    def random_queries(self):
        self.queries = [self.randint() for _ in range(self.length)]

    def set_shift_vector(self):
        self.shift_vector = [self.shift] * self.length

    def randbool(self):
        return random.choice([True, False])

    def randint(self):
        return random.randint(0, self.maxint)

    def push(self):
        self.response.append(self.randbool())
        self.queries.append(self.randint())
        self.shift_vector.append(self.shift)
        self.length += 1

    def pop(self):
        if self.length > 1:
            self.response.pop()
            self.queries.pop()
            self.shift_vector.pop()
            self.length -= 1
            return True
        else:
            return False


class StaticBox(wx.StaticBox):
    def SetSizer(self, sizer):
        super(wx.StaticBox, self).SetSizer(sizer)
        # the label's height is always included in the total size, so compensate
        _, label_height = self.GetSize()
        self.SetMinSize(sizer.GetMinSize() + (0, label_height))


class LineGraph(FigCanvas):
    def __init__(self, parent, drawfunc, lower=0, upper=100, step=1):
        self.parent = parent
        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        super(FigCanvas, self).__init__(parent, wx.ID_ANY, self.figure)

        self.lower = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=lower)
        self.upper = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=upper)
        self.step = wx.SpinCtrl(
            parent, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=1, max=2048, initial=step)
        self.plot = drawfunc
        self.sizer = self.create_sizer()

    @property
    def abscissa(self):
        return np.arange(self.lower.GetValue(), self.upper.GetValue(), self.step.GetValue())

    def create_sizer(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self, proportion=1, flag=wx.LEFT | wx.TOP | wx.EXPAND)
        bounds = wx.BoxSizer(wx.HORIZONTAL)
        bounds.Add(wx.StaticText(self.parent, label="Lower bound"))
        bounds.Add(self.lower)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.parent, label="Step"))
        bounds.Add(self.step)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self.parent, label="Upper bound"))
        bounds.Add(self.upper)
        vbox.Add(bounds, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        for widget in (self.lower, self.upper, self.step):
            self.Bind(wx.EVT_SPINCTRL, self.plot, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, widget)

        return vbox


class BarGraph(FigCanvas):
    def __init__(self, parent, drawfunc):
        self.parent = parent
        self.figure = Figure()
        super(FigCanvas, self).__init__(parent, wx.ID_ANY, self.figure)
        self.plot = drawfunc


class Frame(wx.Frame):
    title = 'Differential Privacy of the Above Threshold Mechanism'

    head_size = (80, -1)
    element_size = (30, -1)
    spinctrl_size = (60, -1)

    def __init__(self):
        wx.Frame.__init__(self, None, title=self.title)

        self.menubar = self.create_menu()
        self.model = Model(100, 0.2, 0.1)
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

        self.vector_control = self.create_vector_control(self.main_panel)
        self.parameter_control = self.create_parameter_control(self.main_panel)
        self.accuracy_control = self.create_accuracy_control(self.main_panel)
        self.graphs = self.create_graphs(self.main_panel)
        self.stats = self.create_stats(self.main_panel)

        main = wx.BoxSizer(wx.VERTICAL)
        lower = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)

        left.Add(self.parameter_control, flag=wx.BOTTOM | wx.EXPAND, border=10)
        left.Add(self.stats, flag=wx.BOTTOM | wx.EXPAND, border=10)
        left.Add(self.accuracy_control, flag=wx.BOTTOM | wx.EXPAND, border=10)

        lower.Add(left, flag=wx.RIGHT | wx.LEFT, border=10)
        lower.Add(self.graphs, proportion=1)

        main.Add(self.vector_control, flag=wx.ALL | wx.EXPAND, border=10)
        main.Add(lower, flag=wx.EXPAND)

        self.main_panel.SetSizer(main)

        # set the first column of independent boxes to the same width
        # and accomodate the panel if it got wider in the process
        left_panels = [self.parameter_control, self.stats, self.accuracy_control]
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

        self.Bind(fs.EVT_FLOATSPIN, self.on_slider_interval, self.slider_interval)
        for widget in [self.query_a, self.query_b]:
            self.Bind(wx.EVT_SPINCTRL, self.on_parameter_change, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_bounds_enter, widget)

        return controls

    def create_vector_control(self, parent):
        panel = wx.Panel(parent)

        response_label = wx.StaticText(
            panel, label="Response", style=wx.ALIGN_RIGHT)
        response_button = wx.Button(panel, label="Random", size=self.head_size)
        self.response_vector = wx.BoxSizer(wx.HORIZONTAL)
        for i in self.model.response:
            self.create_response_element(panel, i)

        queries_label = wx.StaticText(
            panel, label="Queries", style=wx.ALIGN_RIGHT)
        queries_button = wx.Button(panel, label="Random", size=self.head_size)
        self.queries_vector = wx.BoxSizer(wx.HORIZONTAL)
        for i in self.model.queries:
            self.create_queries_element(panel, i)

        shift_label = wx.StaticText(
            panel, label="Shift", style=wx.ALIGN_RIGHT)
        shift_control = wx.SpinCtrl(
            panel, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT,
            min=0, max=1000, initial=1, size=self.head_size)
        self.shift_vector = wx.BoxSizer(wx.HORIZONTAL)
        for i in self.model.shift_vector:
            self.create_shift_element(panel, i)

        plus = wx.Button(panel, label="+", size=self.element_size)
        minus = wx.Button(panel, label="-", size=self.element_size)

        self.Bind(wx.EVT_BUTTON, self.random_response, response_button)
        self.Bind(wx.EVT_BUTTON, self.random_queries, queries_button)
        self.Bind(wx.EVT_SPINCTRL, self.set_shift_vector, shift_control)
        self.Bind(wx.EVT_BUTTON, self.push, plus)
        self.Bind(wx.EVT_BUTTON, self.pop, minus)

        sizer = wx.FlexGridSizer(rows=3, cols=4, gap=(5, 5))
        sizer.AddGrowableCol(2)
        sizer.Add(response_label, flag=wx.EXPAND)
        sizer.Add(response_button)
        sizer.Add(self.response_vector, flag=wx.EXPAND)
        sizer.Add(plus)

        sizer.Add(queries_label, flag=wx.EXPAND)
        sizer.Add(queries_button)
        sizer.Add(self.queries_vector, flag=wx.EXPAND)
        sizer.Add(minus)

        sizer.Add(shift_label, flag=wx.EXPAND)
        sizer.Add(shift_control)
        sizer.Add(self.shift_vector, flag=wx.EXPAND)

        panel.SetSizer(sizer)
        sizer.Fit(panel)
        return panel

    def create_parameter_control(self, parent):
        panel = StaticBox(parent, label="Algorithm parameters")

        threshold_label = wx.StaticText(
            panel, label="T", style=wx.ALIGN_RIGHT)
        self.threshold = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=self.spinctrl_size,
            min=0, max=1000, initial=100)

        epsilon1_label = wx.StaticText(
            panel, label="ε₁ (¹⁄₁₀₀₀)", style=wx.ALIGN_RIGHT)
        self.epsilon1 = fs.FloatSpin(
            panel, agwStyle=fs.FS_RIGHT,
            min_val=0.001, max_val=1, value=0.1,
            increment=0.01, digits=3, size=self.spinctrl_size)

        epsilon2_label = wx.StaticText(
            panel, label="ε₂ (¹⁄₁₀₀₀)", style=wx.ALIGN_RIGHT)
        self.epsilon2 = fs.FloatSpin(
            panel, agwStyle=fs.FS_RIGHT,
            min_val=0.001, max_val=1, value=0.1,
            increment=0.01, digits=3, size=self.spinctrl_size)

        sensitivity_label = wx.StaticText(
            panel, label="Δf", style=wx.ALIGN_RIGHT)
        self.sensitivity = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=self.spinctrl_size,
            min=0, max=100, initial=1)

        monotonic_label = wx.StaticText(
            panel, label="Monotonic", style=wx.ALIGN_RIGHT)
        self.monotonic = wx.CheckBox(panel)
        self.monotonic.SetValue(True)

        sizer = wx.FlexGridSizer(rows=5, cols=2, gap=(5, 5))
        sizer.AddGrowableCol(1)
        grid = [
            threshold_label, self.threshold,
            epsilon1_label, self.epsilon1,
            epsilon2_label, self.epsilon2,
            sensitivity_label, self.sensitivity,
            monotonic_label, self.monotonic,
        ]
        for i in grid:
            sizer.Add(i, flag=wx.EXPAND)

        panel.SetSizer(sizer)
        return panel

    def create_response_element(self, parent, value):
        button = wx.Button(
            parent, label=("T" if value else "F"),
            size=self.element_size)
        button.index = self.response_vector.GetItemCount()
        self.response_vector.Add(button, flag=wx.EXPAND | wx.RIGHT, border=5)
        self.Bind(wx.EVT_BUTTON, self.on_response_button, button)

    def create_queries_element(self, parent, value):
        field = IntCtrl(
            parent, value=value, min=0,
            style=wx.TE_PROCESS_ENTER | wx.TE_RIGHT,
            size=self.element_size)
        field.index = self.queries_vector.GetItemCount()
        self.queries_vector.Add(field, flag=wx.EXPAND | wx.RIGHT, border=5)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_query_field, field)

    def create_shift_element(self, parent, value):
        field = IntCtrl(
            parent, value=value, min=0,
            style=wx.TE_PROCESS_ENTER | wx.TE_RIGHT,
            size=self.element_size)
        field.index = self.shift_vector.GetItemCount()
        self.shift_vector.Add(field, flag=wx.EXPAND | wx.RIGHT, border=5)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_shift_field, field)

    def create_accuracy_control(self, parent):
        panel = StaticBox(parent, label="Compute accuracy")
        spinctrl_size = (60, -1)

        alpha_label = wx.StaticText(
            panel, label="α", style=wx.ALIGN_RIGHT)
        beta_label = wx.StaticText(
            panel, label="β", style=wx.ALIGN_RIGHT)

        self.alpha = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=spinctrl_size,
            min=0, max=1000, initial=10)
        beta = wx.StaticText(panel, label="0")

        sizer = wx.FlexGridSizer(rows=2, cols=2, gap=(5, 5))
        grid = [
            alpha_label, self.alpha,
            beta_label, beta,
        ]
        for i in grid:
            sizer.Add(i, flag=wx.EXPAND)

        panel.SetSizer(sizer)
        return panel

    def create_graphs(self, parent):
        graphs = wx.Panel(parent)

        bars_original = BarGraph(graphs, self.draw_original)
        bars_shifted = BarGraph(graphs, self.draw_shifted)
        accuracy = LineGraph(graphs, self.draw_accuracy, lower=80, upper=120)

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(bars_original, proportion=0, flag=wx.EXPAND)
        box.Add(bars_shifted, proportion=0, flag=wx.EXPAND)
        box.Add(accuracy.sizer, proportion=0, flag=wx.EXPAND)

        graphs.SetSizer(box)
        return graphs

    def create_stats(self, parent):
        panel = StaticBox(parent, label="Vector properties")

        vector_label = wx.StaticText(
            panel, label="ℙ(Response)", style=wx.ALIGN_RIGHT)
        error_label = wx.StaticText(
            panel, label="ℙ(Error)", style=wx.ALIGN_RIGHT)
        alpha_min_label = wx.StaticText(
            panel, label="αₘᵢₙ", style=wx.ALIGN_RIGHT)
        beta_label = wx.StaticText(
            panel, label="β(αₘᵢₙ)", style=wx.ALIGN_RIGHT)

        vector = wx.StaticText(panel, label="0")
        error = wx.StaticText(panel, label="0")
        alpha_min = wx.StaticText(panel, label="0")
        beta = wx.StaticText(panel, label="0")

        sizer = wx.FlexGridSizer(rows=4, cols=2, gap=(5, 5))
        grid = [
            vector_label, vector,
            error_label, error,
            alpha_min_label, alpha_min,
            beta_label, beta,
        ]
        for i in grid:
            sizer.Add(i, flag=wx.EXPAND)

        panel.SetSizer(sizer)
        return panel

    def draw_original(self):
        pass

    def draw_shifted(self):
        pass

    def draw_accuracy(self):
        pass

    def draw(self):
        pass

    def push(self, event):
        self.model.push()
        parent = self.vector_control
        self.create_response_element(parent, self.model.response[-1])

        self.create_queries_element(parent, self.model.queries[-1])

        self.create_shift_element(parent, self.model.shift_vector[-1])

        self.layout()

    def pop(self, event):
        if self.model.pop():
            for v in [self.response_vector, self.queries_vector, self.shift_vector]:
                idx = len(v.GetChildren()) - 1
                v.GetChildren()[idx].DeleteWindows()
                v.Remove(idx)

        self.layout()

    def random_response(self, event):
        self.model.random_response()
        for i, v in enumerate(self.response_vector.GetChildren()):
            v.Window.SetLabel("T" if self.model.response[i] else "F")

    def random_queries(self, event):
        self.model.random_queries()
        for i, v in enumerate(self.queries_vector.GetChildren()):
            v.Window.SetValue(self.model.queries[i])

    def set_shift_vector(self, event):
        self.model.shift = event.GetEventObject().GetValue()
        self.model.set_shift_vector()
        for i, v in enumerate(self.shift_vector.GetChildren()):
            v.Window.SetValue(self.model.shift_vector[i])

    def on_response_button(self, event):
        button = event.GetEventObject()
        idx = button.index
        self.model.response[idx] = not self.model.response[idx]
        button.SetLabel("T" if self.model.response[idx] else "F")

    def on_query_field(self, event):
        field = event.GetEventObject()
        idx = field.index
        self.model.queries[idx] = field.GetValue()

    def on_shift_field(self, event):
        field = event.GetEventObject()
        idx = field.index
        self.model.shift_vector[idx] = field.GetValue()

    def layout(self):
        self.main_panel.Layout()

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
