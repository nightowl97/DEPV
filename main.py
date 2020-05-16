import os
import matplotlib
from kivy.app import App
from kivy.lang import Builder
from plyer import filechooser
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from objects import DE, read_csv
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas

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
        algo = DE(bounds, [*read_csv(self.path)], Ns, Np)
        algo.solve(temperature)
        f1, ax1 = algo.plot_fit_hist()
        canvas1 = f1.canvas
        f2, ax2 = algo.plot_result(temperature)
        canvas2 = f2.canvas
        root_manager.results_scrn.graphid.add_widget(canvas1)
        root_manager.results_scrn.graphid.add_widget(canvas2)
        print(algo.final_res)


class ResultsWindow(Screen):
    graphid = ObjectProperty(None)


class WindowManager(ScreenManager):
    results_scrn = ObjectProperty(None)


kv = Builder.load_file("main.kv")


class MainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MainApp().run()
