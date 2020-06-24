import os
import matplotlib
from kivy.app import App
from kivy.lang import Builder
from plyer import filechooser
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from objects import DE, read_csv
import matplotlib.pyplot as plt
# from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas

matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
Config.set('kivy', 'exit_on_escape', '1')


class ParamWindow(Screen):
    rplower = ObjectProperty(None)
    rpupper = ObjectProperty(None)
    alower = ObjectProperty(None)
    aupper = ObjectProperty(None)
    i0lower = ObjectProperty(None)
    i0upper = ObjectProperty(None)
    ipvlower = ObjectProperty(None)
    ipvupper = ObjectProperty(None)
    rslower = ObjectProperty(None)
    rsupper = ObjectProperty(None)
    tempc = ObjectProperty(None)
    series = ObjectProperty(None)
    parallel = ObjectProperty(None)
    path = ""

    def browse_files(self):
        try:
            self.path = filechooser.open_file(path=os.getcwd(), title="Pick a CSV file..",
                                              filters=[("Comma-separated Values", "*.csv")])[0]
        except TypeError:
            self.path = ""

    def calculate(self, root_manager):
        # Construct bounds dict
        bounds = {'rp': [float(self.rplower.text), float(self.rpupper.text)],
                  'rs': [float(self.rslower.text), float(self.rsupper.text)],
                  'a':   [float(self.alower.text), float(self.aupper.text)],
                  'i0':  [float(self.i0lower.text), float(self.i0upper.text)],
                  'ipv': [float(self.ipvlower.text), float(self.ipvupper.text)]}
        Ns, Np = int(self.series.text), int(self.parallel.text)

        temperature = float(self.tempc.text) + 275.15
        algo = DE(bounds, [*read_csv(self.path)], Ns, Np, temperature)
        algo.solve()
        f1, ax1 = algo.plot_fit_hist()
        canvas1 = f1.canvas
        f2, ax2 = algo.plot_result()
        canvas2 = f2.canvas
        root_manager.results_scrn.graphid1.add_widget(canvas1)
        root_manager.results_scrn.graphid2.add_widget(canvas2)

        sol_vec, fitness = algo.final_res
        root_manager.results_scrn.set_results(*sol_vec, fitness)


class ResultsWindow(Screen):

    rp_label = ObjectProperty(None)
    rs_label = ObjectProperty(None)
    a_label = ObjectProperty(None)
    i0_label = ObjectProperty(None)
    ipv_label = ObjectProperty(None)
    rmse_label = ObjectProperty(None)
    graphid1 = ObjectProperty(None)
    graphid2 = ObjectProperty(None)

    def set_results(self, rp, rs, a, i0, ipv, rmse):
        self.rp_label.text = str(rp)
        self.rs_label.text = str(rs)
        self.a_label.text = str(a)
        self.i0_label.text = str(i0)
        self.ipv_label.text = str(ipv)
        self.rmse_label.text = str(rmse)

    def go_back(self):
        # clear box layouts
        self.graphid1.clear_widgets()
        self.graphid2.clear_widgets()


class WindowManager(ScreenManager):
    # results_scrn = ObjectProperty(None)
    pass


kv = Builder.load_file("main.kv")


class MainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MainApp().run()
