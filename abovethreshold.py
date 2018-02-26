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
from matplotlib.ticker import MaxNLocator

from collections import Counter
from math import log
from numpy import product
from scipy.integrate import quad

from algorithms import *
from accuracy import accuracy_overestimate
from accuracy import probability_overestimate
from accuracy import probability_baseline
from accuracy import probability_optimized
from accuracy import probability_precise
from experiments import precise as probability_data
from experiments import compute_alphas


class Model(object):
    def __init__(
        self, threshold, e1, e2, sensitivity=1, monotonic=True, compute=False,
            length=5, shift=1):

        self.threshold = threshold
        self.epsilon1 = e1
        self.epsilon2 = e2
        self.sensitivity = sensitivity
        self.monotonic = monotonic
        self.compute = compute

        self.length = length
        self.shift = shift
        self.maxint = 2*threshold

        self.response = self.random_response()
        self.queries = self.random_queries()
        self.shift_vector = self.new_shift_vector()
        self.count = self.get_count()

        """probability of getting `response`, given `queries` and `threshold`"""
        self.pr_response = 1
        """probability of getting `response`, given `queries` + `shift_vector` and `threshold"""
        self.pr_shifted = 1
        """probability of getting a correct response,
        given `queries` and `threshold`"""
        self.pr_correct = 1
        """probability of getting an alpha-accurate response,
        given `queries` and `threshold`"""
        self.pr_accurate = 1
        """probabilities of each response item with respect to queries and threshold"""
        self.pr_items = []


    def random_response(self):
        # prevent responses with zero count
        while True:
            response = [self.randbool() for _ in range(self.length)]
            if any(response):
                break
        return response

    def random_queries(self):
        return [self.randint() for _ in range(self.length)]

    def new_shift_vector(self):
        return [self.shift] * self.length

    def set_random_response(self):
        self.response = self.random_response()

    def set_random_queries(self):
        self.queries = self.random_queries()

    def set_shift_vector(self, value):
        self.shift = value
        self.shift_vector = self.new_shift_vector()

    def randbool(self):
        return random.choice([True, False])

    def randint(self):
        return random.randint(0, self.maxint)

    def push(self):
        self.response.append(self.randbool())
        self.queries.append(self.randint())
        self.shift_vector.append(self.shift)

    def pop(self):
        if self.length > 1:
            self.response.pop()
            self.queries.pop()
            self.shift_vector.pop()
            return True
        else:
            return False

    def update(self):
        # this does long computation once
        self.update_length()
        self.pr_response = self.get_probability(self.response, self.queries)
        self.pr_shifted = self.get_probability(self.response, self.shifted_queries)
        self.pr_correct = self.get_probability(self.correct_response, self.queries)
        self.pr_items = self.get_pr_items(self.response, self.queries)
        self.pr_shifted_items = self.get_pr_items(self.response, self.shifted_queries)

    def update_length(self):
        self.length = len(self.response)
        assert len(self.queries) == self.length
        assert len(self.shift_vector) == self.length

    def get_count(self):
        return len([x for x in self.response if x])

    def get_probability(self, response, queries):

        def pred(x):
            return product([self.pr_single_response(r, q, x)
                            for (r, q) in zip(response, queries)])

        def state(x):
            return self.threshold_state(x) * pred(x)

        error = 1/1e12
        T_bound = self.threshold_scale * log(1/error)

        return quad(state, self.threshold-T_bound, self.threshold+T_bound, points=[self.threshold])[0]

    def pr_single_response(self, is_above, query, threshold):
        """Pr(query => is_above | threshold_value )"""
        pr_above = 1 - self.query_dist(query).cdf(threshold)
        if is_above:
            return pr_above
        else:
            return 1 - pr_above

    @property
    def pr_diff(self):
        """differential probability of original and shifted query vector"""
        return abs(log(self.pr_response/self.pr_shifted))


    @property
    def alphas(self):
        c = self.count
        T = self. threshold
        k = self.length
        counts = self.counts
        return compute_alphas(c, T, k, counts)

    @property
    def counts(self):
        return dict(Counter(self.queries))

    @property
    def correct_response(self):
        return [q >= self.threshold for q in self.queries]

    @property
    def shifted_queries(self):
        return [a + b for (a, b) in zip(self.queries, self.shift_vector)]

    @property
    def threshold_state(self):
        return self.threshold_dist.state

    @property
    def threshold_dist(self):
        return Laplace(self.threshold_scale, loc=self.threshold)


    def query_dist(self, value):
        return Laplace(self.query_scale, loc=value)

    @property
    def threshold_scale(self):
        return self.sensitivity / self.epsilon1

    @property
    def query_scale(self):
        return (self.factor*self.count*self.sensitivity) / self.epsilon2

    @property
    def factor(self):
        return 1 if self.monotonic else 2

    def get_pr_items(self, response, queries):
        items = zip(response, queries)
        return [self.pr_single_item(r, q) for (r, q) in items]

    def pr_single_item(self, is_above, query):
        pr_above = self.query_dist(query).larger(self.threshold_dist)
        if is_above:
            return pr_above
        else:
            return 1 - pr_above


class StaticBox(wx.StaticBox):
    def SetSizer(self, sizer):
        super(wx.StaticBox, self).SetSizer(sizer)
        # the label's height is always included in the total size, so compensate
        _, label_height = self.GetSize()
        self.SetMinSize(sizer.GetMinSize() + (0, label_height))


class LineGraph(FigCanvas):
    def __init__(self, parent, model, lower=0, upper=100, step=1):
        self.parent = parent
        self.figure = Figure(figsize=(5,3))
        self.axes = self.figure.add_subplot(1, 1, 1)
        super(FigCanvas, self).__init__(parent, wx.ID_ANY, self.figure)
        self.model = model

        self.lower = wx.SpinCtrl(
            self, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=lower)
        self.upper = wx.SpinCtrl(
            self, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=-1000, max=1000, initial=upper)
        self.step = wx.SpinCtrl(
            self, style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=(60, -1),
            min=1, max=2048, initial=step)
        self.sizer = self.create_sizer()

    def plot(self):
        raise NotImplementedError

    @property
    def abscissa(self):
        return np.arange(self.lower.GetValue(), self.upper.GetValue(), self.step.GetValue())

    def create_sizer(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self, proportion=1, flag=wx.LEFT | wx.TOP | wx.EXPAND)
        bounds = wx.BoxSizer(wx.HORIZONTAL)
        bounds.Add(wx.StaticText(self, label="Lower bound"))
        bounds.Add(self.lower)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self, label="Step"))
        bounds.Add(self.step)
        bounds.AddStretchSpacer()
        bounds.Add(wx.StaticText(self, label="Upper bound"))
        bounds.Add(self.upper)
        vbox.Add(bounds, proportion=0, flag=wx.ALL | wx.EXPAND, border=10)

        for widget in (self.lower, self.upper, self.step):
            self.Bind(wx.EVT_SPINCTRL, self.plot, widget)
            self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, widget)

        return vbox


class BarGraph(FigCanvas):
    def __init__(self, parent, model):
        self.parent = parent
        self.figure = Figure(figsize=(5,2))
        self.axes = self.figure.add_subplot(1, 1, 1)
        super(FigCanvas, self).__init__(parent, wx.ID_ANY, self.figure)
        self.model = model

    def plot(self):
        raise NotImplementedError


class Probabilities(BarGraph):
    def plot(self):
        ax = self.axes
        ax.clear()

        xs = np.arange(self.model.length)
        ys = self.model.pr_items
        zs = self.model.pr_shifted_items
        for x, y, z in zip(xs, ys, zs):
            if y > z:
                original = ax.bar(x, y, color="blue")
                shifted = ax.bar(x, z, color="red")
            else:
                shifted = ax.bar(x, z, color="red")
                original = ax.bar(x, y, color="blue")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0,1)
        ax.legend((original[0], shifted[0]), ("original", "shifted"), loc='upper right')
        self.figure.suptitle("Probabilities of individual responses")
        self.draw()


class Accuracy(LineGraph):
    def plot(self):
        ax = self.axes
        ax.clear()

        T = self.model.threshold
        k = self.model.length
        s1 = self.model.threshold_scale
        s2 = self.model.query_scale

        MAX = accuracy_overestimate(0.01, k, s1, s2)

        xs = np.arange(MAX)
        ax.plot(xs, [probability_overestimate(x, k, s1, s2) for x in xs], color="red", linewidth=2.0, label="overestimate")
        ax.plot(xs, [probability_baseline(x, k, s1, s2) for x in xs], color="green", linewidth=2.0, label="baseline")
        ax.plot(xs, [probability_optimized(x, k, s1, s2) for x in xs], color="blue", linewidth=2.0, label="optimized")
        if self.model.compute:
            ax.plot(xs, [probability_precise(x, k, s1, s2) for x in xs], color="black", linewidth=2.0, label="precise")
            queries = self.model.queries
            alphas = self.model.alphas
            xs = [0] + list(alphas.keys())
            ys = [probability_data(x, k, s1, s2, queries, alphas, T) for x in alphas.keys()] + [0]
            ax.step(xs, ys, where='post',
                color="magenta", linewidth=2.0, label="data-bound")
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, MAX)
        self.figure.suptitle("Accuracy estimation")
        self.draw


class Frame(wx.Frame):
    title = 'Differential Privacy of the Above Threshold Mechanism'

    head_size = (80, -1)
    element_size = (30, -1)
    spinctrl_size = (80, -1)

    def __init__(self):
        wx.Frame.__init__(self, None, title=self.title)

        self.menubar = self.create_menu()
        self.model = Model(100, e1=0.1, e2=0.2)
        self.create_view()
        self.model.update()
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
        self.graphs = self.create_graphs(self.main_panel)
        self.stats = self.create_stats(self.main_panel)

        main = wx.BoxSizer(wx.VERTICAL)
        lower = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)

        left.Add(self.parameter_control, flag=wx.BOTTOM | wx.EXPAND, border=10)
        left.Add(self.stats, flag=wx.BOTTOM | wx.EXPAND, border=10)

        lower.Add(left, flag=wx.RIGHT | wx.LEFT, border=10)
        lower.Add(self.graphs, proportion=1)

        main.Add(self.vector_control, flag=wx.ALL | wx.EXPAND, border=10)
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
            min=-1000, max=1000, initial=1, size=self.head_size)
        self.shift_vector = wx.BoxSizer(wx.HORIZONTAL)
        for i in self.model.shift_vector:
            self.create_shift_element(panel, i)

        self.plus = wx.Button(panel, label="+", size=self.element_size)
        self.minus = wx.Button(panel, label="-", size=self.element_size)

        self.Bind(wx.EVT_BUTTON, self.on_random_response, response_button)
        self.Bind(wx.EVT_BUTTON, self.on_random_queries, queries_button)
        self.Bind(wx.EVT_SPINCTRL, self.on_set_shift_vector, shift_control)
        self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, shift_control)
        self.Bind(wx.EVT_BUTTON, self.on_plus, self.plus)
        self.Bind(wx.EVT_BUTTON, self.on_minus, self.minus)

        sizer = wx.FlexGridSizer(rows=3, cols=4, gap=(5, 5))
        sizer.AddGrowableCol(2)
        sizer.Add(response_label, flag=wx.EXPAND)
        sizer.Add(response_button)
        sizer.Add(self.response_vector, flag=wx.EXPAND)
        sizer.Add(self.plus)

        sizer.Add(queries_label, flag=wx.EXPAND)
        sizer.Add(queries_button)
        sizer.Add(self.queries_vector, flag=wx.EXPAND)
        sizer.Add(self.minus)

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
            min=0, max=1000, initial=self.model.threshold)

        epsilon1_label = wx.StaticText(
            panel, label="ε₁", style=wx.ALIGN_RIGHT)
        self.epsilon1 = fs.FloatSpin(
            panel, agwStyle=fs.FS_RIGHT,
            min_val=0.001, max_val=1, value=self.model.epsilon1,
            increment=0.01, digits=3, size=self.spinctrl_size)

        epsilon2_label = wx.StaticText(
            panel, label="ε₂", style=wx.ALIGN_RIGHT)
        self.epsilon2 = fs.FloatSpin(
            panel, agwStyle=fs.FS_RIGHT,
            min_val=0.001, max_val=1, value=self.model.epsilon2,
            increment=0.01, digits=3, size=self.spinctrl_size)

        sensitivity_label = wx.StaticText(
            panel, label="Δ", style=wx.ALIGN_RIGHT)
        self.sensitivity = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=self.spinctrl_size,
            min=0, max=100, initial=self.model.sensitivity)

        count_label = wx.StaticText(
            panel, label="c", style=wx.ALIGN_RIGHT)
        self.count = wx.SpinCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER | wx.ALIGN_RIGHT, size=self.spinctrl_size,
            min=1, max=100, initial=self.model.count)

        monotonic_label = wx.StaticText(
            panel, label="Monotonic", style=wx.ALIGN_RIGHT)
        self.monotonic = wx.CheckBox(panel)
        self.monotonic.SetValue(self.model.monotonic)
        compute_label = wx.StaticText(
            panel, label="Slow graphs", style=wx.ALIGN_RIGHT)
        self.compute = wx.CheckBox(panel)
        self.compute.SetValue(self.model.compute)

        grid = [
            [threshold_label, self.threshold],
            [epsilon1_label, self.epsilon1],
            [epsilon2_label, self.epsilon2],
            [sensitivity_label, self.sensitivity],
            [count_label, self.count],
            [monotonic_label, self.monotonic],
            [compute_label, self.compute],
        ]
        sizer = wx.FlexGridSizer(rows=len(grid), cols=len(grid[0]), gap=(5, 5))
        for line in grid:
            for item in line:
                sizer.Add(item, flag=wx.EXPAND)

        self.Bind(wx.EVT_SPINCTRL, self.on_threshold, self.threshold)
        self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, self.threshold)
        self.Bind(fs.EVT_FLOATSPIN, self.on_epsilon1, self.epsilon1)
        self.Bind(fs.EVT_FLOATSPIN, self.on_epsilon2, self.epsilon2)
        self.Bind(wx.EVT_SPINCTRL, self.on_sensitivity, self.sensitivity)
        self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, self.sensitivity)
        self.Bind(wx.EVT_SPINCTRL, self.on_count, self.count)
        self.Bind(wx.EVT_TEXT_ENTER, on_spin_enter, self.count)
        self.Bind(wx.EVT_CHECKBOX, self.on_monotonic, self.monotonic)
        self.Bind(wx.EVT_CHECKBOX, self.on_compute, self.compute)

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
            parent, value=value,
            style=wx.TE_PROCESS_ENTER | wx.TE_RIGHT,
            size=self.element_size)
        field.index = self.shift_vector.GetItemCount()
        self.shift_vector.Add(field, flag=wx.EXPAND | wx.RIGHT, border=5)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_shift_field, field)

    def create_graphs(self, parent):
        graphs = wx.Panel(parent)

        bars_original = Probabilities(graphs, self.model)
        accuracy = Accuracy(graphs, self.model, lower=80, upper=120)

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(bars_original, proportion=0, flag=wx.EXPAND)
        box.Add(accuracy.sizer, proportion=0, flag=wx.EXPAND)

        graphs.SetSizer(box)
        return graphs

    def create_stats(self, parent):
        panel = StaticBox(parent, label="Vector properties")

        pr_response_label = wx.StaticText(
            panel, label="ℙ(response)", style=wx.ALIGN_RIGHT)
        pr_shifted_label = wx.StaticText(
            panel, label="ℙ(response')", style=wx.ALIGN_RIGHT)
        pr_diff_label = wx.StaticText(
            panel, label="privacy loss", style=wx.ALIGN_RIGHT)
        pr_correct_label = wx.StaticText(
            panel, label="ℙ(correct)", style=wx.ALIGN_RIGHT)

        self.pr_response = wx.StaticText(panel)
        self.pr_shifted = wx.StaticText(panel)
        self.pr_diff = wx.StaticText(panel)
        self.pr_correct = wx.StaticText(panel)

        grid = [
            [pr_response_label, self.pr_response],
            [pr_shifted_label, self.pr_shifted],
            [pr_correct_label, self.pr_correct],
            [pr_diff_label, self.pr_diff],
        ]
        sizer = wx.FlexGridSizer(rows=len(grid), cols=len(grid[0]), gap=(5, 5))
        for line in grid:
            for item in line:
                sizer.Add(item, flag=wx.EXPAND)

        panel.SetSizer(sizer)
        return panel

    def update_stats(self):
        self.pr_response.SetLabel("{:.3f}".format(self.model.pr_response))
        self.pr_shifted.SetLabel("{:.3f}".format(self.model.pr_shifted))
        self.pr_diff.SetLabel("{:.3f}".format(self.model.pr_diff))
        self.pr_correct.SetLabel("{:.3f}".format(self.model.pr_correct))


    def draw(self):
        self.update_stats()
        self.main_panel.Layout()
        for g in self.graphs.Children:
            g.plot()

    def on_threshold(self, event):
        self.model.threshold = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_epsilon1(self, event):
        self.model.epsilon1 = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_epsilon2(self, event):
        self.model.epsilon2 = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_sensitivity(self, event):
        self.model.sensitivity = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_count(self, event):
        self.model.count = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_monotonic(self, event):
        self.model.monotonic = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_compute(self, event):
        self.model.compute = event.GetEventObject().GetValue()
        self.on_parameter_change()

    def on_plus(self, event):
        self.model.push()
        parent = self.vector_control
        self.create_response_element(parent, self.model.response[-1])
        self.create_queries_element(parent, self.model.queries[-1])
        self.create_shift_element(parent, self.model.shift_vector[-1])

        self.on_parameter_change()

    def on_minus(self, event):
        if self.model.pop():
            vectors = [self.response_vector,self.queries_vector, self.shift_vector]
            for v in vectors:
                idx = len(v.GetChildren()) - 1
                v.GetChildren()[idx].DeleteWindows()
                v.Remove(idx)

        self.on_parameter_change()

    def on_random_response(self, event):
        self.model.set_random_response()
        for i, v in enumerate(self.response_vector.GetChildren()):
            v.Window.SetLabel("T" if self.model.response[i] else "F")
        self.on_parameter_change()

    def on_random_queries(self, event):
        self.model.set_random_queries()
        for i, v in enumerate(self.queries_vector.GetChildren()):
            v.Window.SetValue(self.model.queries[i])
        self.on_parameter_change()

    def on_set_shift_vector(self, event):
        shift = event.GetEventObject().GetValue()
        self.model.set_shift_vector(shift)
        for i, v in enumerate(self.shift_vector.GetChildren()):
            v.Window.SetValue(self.model.shift_vector[i])
        self.on_parameter_change()

    def on_response_button(self, event):
        button = event.GetEventObject()
        idx = button.index
        self.model.response[idx] = not self.model.response[idx]
        button.SetLabel("T" if self.model.response[idx] else "F")
        self.on_parameter_change()

    def on_query_field(self, event):
        field = event.GetEventObject()
        idx = field.index
        self.model.queries[idx] = field.GetValue()
        self.on_parameter_change()

    def on_shift_field(self, event):
        field = event.GetEventObject()
        idx = field.index
        self.model.shift_vector[idx] = field.GetValue()
        self.on_parameter_change()

    def on_parameter_change(self):
        self.model.update()
        self.draw()

    def on_exit(self, event):
        self.Destroy()

    def on_about(self, event):
        msg = """Dynamically parametrize the Above Threshold Algorithm

        * Set a response vector
        * Set a query vector
        * Set a query vector for a neighboring database
        * Adjust the algorithm parameters T, e1, e2, sensitivity, count

        The program displays the queries' individual probabilities to produce
        the given response vector enties, as well as the probability of the whole
        query vector producing the given response vector.

        Query values below threshold are highlighted, as well as incorrect
        response vector entries.
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
