# main_windows.py — Main application window for the Probabilistic CIM Simulator
# Author:      Emilie Ye
# Date:        2025-06-27
# Version:     0.1
# Description: Defines the main application window that coordinates mode selection, 
#              displays the appropriate input panels, and shows simulation outputs.
# Copyright (c) 2025

'''
main_windows.py
│
├── main() → ModeSelectDialog.exec()
│                    ↓
│        User selects "Memory"/"Demo".
│                    ↓
├── MainSimulatorWindow(core, choice)
│        ↓
│   switchMode(choice) → add InputPage + OutputPanel
│        ↓
│   [User clicks "Start Simulation"]
│        ↓
├── InputPage.start_simulation()
│        ↓
├── SimulatorCore.run_demo() / run_memory() / ...(run simulation, save results to the folder)
│        ↓
├── simulationRequested.emit()
│        ↓
├── OutputPanel.run_simulation()(display results)
│        ↓
├── _show_xxx_outputs() → create_plot + reload_images


'''
import os, sys
from PySide6.QtWidgets import QApplication
from mode_select import ModeSelectDialog
from core_function import SimulatorCore
from memory_page import MemoryInputPage
from compute_page import ComputeInputPage
from demo_page import DemoInputPage
from output_panel import OutputPanel
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QDialog, QMainWindow, QStackedWidget
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(out_dir, exist_ok=True)

class MainSimulatorWindow(QWidget):
    def __init__(self, core, mode):
        super().__init__()
        self.core = core
        self.setWindowTitle(f"Simulator – {mode}")
        self.setMinimumSize(1450, 800)

        core.plots_dir = out_dir    #./plots

        self.output_panel = OutputPanel(core)

        self.current_input = None

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(10)

        self.main_layout.addWidget(self.output_panel, 3)

        self.setLayout(self.main_layout)

        self.switchMode(mode)

    #Switches the diaplayed mode and updates the input panel accordingly.
    def switchMode(self, mode: str):
        # delete previous input page
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        self.core.mode = mode
    
        if mode == "Memory":
            inp = MemoryInputPage(self.core)
        elif mode == "Compute in Memory":
            inp = ComputeInputPage(self.core)
        else:
            inp = DemoInputPage(self.core)
            
        inp.simulationRequested.connect(self.output_panel.run_simulation)

        inp.backRequested.connect(self.onBack)
        
        self.output_panel.clear(show_placeholder=True)

        self.main_layout.addWidget(inp, 1)
        self.main_layout.addWidget(self.output_panel, 3)

        self.current_input = inp

        self.core.reset()
        self.setWindowTitle(f"Simulator – {mode}")


    def onBack(self):
        dlg = ModeSelectDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.choice:
            self.switchMode(dlg.choice)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    dlg = ModeSelectDialog()    # Create the mode selection dialog
    if dlg.exec() == QDialog.Accepted and dlg.choice:   # show the dialog and wait for user input
        core = SimulatorCore()
        win = MainSimulatorWindow(core, dlg.choice)
        win.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
