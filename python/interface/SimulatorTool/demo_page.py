# demo_page.py 
# Author:      Emilie Ye
# Date:        2025-06-27
# Version:     0.1
# Description: Provides controls for choosing the RNG source, setting μ, σ, sample times, and bins, 
#              and running the demo simulation mode.
# Copyright (c) 2025
import sys, os, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../python')))

from PySide6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QComboBox, QLineEdit, QGroupBox,
    QFormLayout, QFileDialog, QFormLayout, QSizePolicy
)
from PySide6.QtCore import Signal,Qt
from PySide6.QtGui import QFontMetrics

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

from simulator.memory import *

from interface.load_parameter import *


class EmittingStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        if text.strip():
            self.text_edit.append(text.strip())

    def flush(self):
        pass

class DemoInputPage(QWidget):
    #：updated_nn, updated_bnn, updated_sampled_fc1
    simulationRequested = Signal()
    backRequested      = Signal()

    def __init__(self, core, parent=None):
        super().__init__(parent)
        self.simulator_core = core
        self.file_path = ""

        self.setWindowTitle("Probabilistic CIM Simulator Tool")
        
        # Left (input), Right (output)
        main_layout = QHBoxLayout(self)

        # Left Input Panel
        input_panel = QGroupBox("Demo Simulation Settings")
        input_panel.setStyleSheet("""
            QGroupBox {
                background-color: #F8F8F8;
                border: 1px solid #CCC;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
        """)

        self.input_layout = QFormLayout()
        input_panel.setLayout(self.input_layout)

        self.rng_mode = QComboBox()
        self.rng_mode.addItems(["ideal", "Choose file"])
        self.rng_mode.currentIndexChanged.connect(self._toggle_file_btn)
        self.input_layout.addRow("RNG Source:", self.rng_mode)


        self.data_button = QPushButton("Choose RNG File")
        self.data_button.clicked.connect(self.choose_device_data)
        self.input_layout.addRow("RNG File:", self.data_button)
        self.data_button.setFixedHeight(24)

        self.weight_number = QComboBox()
        self.weight_number.addItems(["1", "2", "3"])
        self.weight_number.currentIndexChanged.connect(self._update_weight_inputs)
        self.input_layout.addRow("Weight Number:", self.weight_number)


        #self.mu_input = QLineEdit("1")
        #self.input_layout.addRow("mu:", self.mu_input)
        self.mu_container = QWidget()
        self.mu_fields_layout = QHBoxLayout(self.mu_container)
        self.mu_fields_layout.setSpacing(0)
        self.input_layout.addRow("μ:", self.mu_container)


        #self.sigma_input = QLineEdit("1")
        #self.input_layout.addRow("sigma:", self.sigma_input)
        self.sigma_container = QWidget()
        self.sigma_fields_layout = QHBoxLayout(self.sigma_container)
        self.sigma_fields_layout.setSpacing(0)
        self.input_layout.addRow("σ:", self.sigma_container)


        self.sample_input = QLineEdit("100")
        self.input_layout.addRow("Sampling Times:", self.sample_input)


        self.bins_input = QLineEdit("50")
        self.input_layout.addRow("Bins:", self.bins_input)


        # Buttons
        self.back_button = QPushButton("Back") 
        self.back_button.clicked.connect(lambda: self.backRequested.emit()) #send signal to the main window to switch back to the mode selection
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 10pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2893F3;
            }
            QPushButton:pressed {
                background-color: #005EA6;
            }
        """)
        self.back_button.setFixedWidth(100)


        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 10pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2893F3;
            }
            QPushButton:pressed {
                background-color: #005EA6;
            }
        """)
        self.start_button.setFixedWidth(150)

        self.reset_button = QPushButton("Reset Inputs")
        self.reset_button.clicked.connect(self.reset_inputs)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 10pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2893F3;
            }
            QPushButton:pressed {
                background-color: #005EA6;
            }
        """)
        self.reset_button.setFixedWidth(125)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)
        self.input_layout.addRow(button_layout)

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(input_panel, 1)


        main_layout.setStretch(0, 1)  # Input panel
        
        self._update_weight_inputs()
        self._toggle_file_btn()

        self.setLayout(main_layout)

    def _update_weight_inputs(self):
        number = int(self.weight_number.currentText())

        # Clear μ
        while self.mu_fields_layout.count():
            widget = self.mu_fields_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.mu_lineedits = []
        for idx in range(number):
            mu_edit = QLineEdit("1")
            mu_edit.setPlaceholderText(f"μ{idx+1}")
            mu_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.mu_fields_layout.addWidget(mu_edit, 1)
            self.mu_lineedits.append(mu_edit)

        # Clear σ
        while self.sigma_fields_layout.count():
            widget = self.sigma_fields_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.sigma_lineedits = []
        for idx in range(number):
            sigma_edit = QLineEdit("1")
            sigma_edit.setPlaceholderText(f"σ{idx+1}")
            sigma_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.sigma_fields_layout.addWidget(sigma_edit, 1)
            self.sigma_lineedits.append(sigma_edit)

    # Input Actions
    def choose_device_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select RNG File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.file_path = file_path

            fm = QFontMetrics(self.data_button.font())
            file_path_text = fm.elidedText(file_path, Qt.ElideLeft, self.data_button.width()-16)

            self.data_button.setText(file_path_text)
            self.data_button.setToolTip(file_path)
            self.data_button.setStyleSheet("""
                QPushButton {
                    text-align: right;
                    padding-right: 8px;
                }
            """)
            print("Selected RNG File:", file_path)

    def _toggle_file_btn(self):
        need_file = self.rng_mode.currentText() == "Choose file"
        self.data_button.setEnabled(need_file)
        if not need_file:
            self.file_path = ""

    # When click start simulation
    def start_simulation(self):
        updated_nn = False
        updated_bnn = False
        updated_sampled_fc1 = False


        self.simulator_core.mode   = "Demo"

        plots = self.simulator_core.plots_dir
        #clean the plots folder
        for f in glob.glob(os.path.join(plots, "*.svg")):
            try:
                os.remove(f)
            except OSError:
                pass

        print("Starting simulation...")

        self.simulator_core.demo_mode    = self.rng_mode.currentText()
        self.simulator_core.demo_file    = self.file_path

        demo_mus    = [float(edit.text()) for edit in self.mu_lineedits]
        demo_sigmas = [float(edit.text()) for edit in self.sigma_lineedits]
        self.simulator_core.demo_mus    = demo_mus
        self.simulator_core.demo_sigmas = demo_sigmas

        self.simulator_core.demo_samples = int(self.sample_input.text())
        self.simulator_core.demo_bins    = int(self.bins_input.text())

        self.simulator_core.export_demo_config_to_yaml()    # save the config to yaml file

        self.simulator_core.run_demo()  # run the demo simulation--generate the gaussian distribution and plot the results

        self.simulationRequested.emit() # emit the signal to run the simulation in the output panel file

    def reset_inputs(self):
        self.rng_mode.setCurrentIndex(0)

        self.weight_number.setCurrentIndex(0)
        self._update_weight_inputs()

        self.sample_input.setText("100")
        self.bins_input.setText("50")

        # Reset folder paths
        self.file_path  = None
        self._toggle_file_btn()
        self.data_button.setText("Choose RNG File")
        self.data_button.setToolTip("")
        self.data_button.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
            }}
        """)

        if hasattr(self.parent(), "output_panel"):
            self.parent().output_panel.clear(show_placeholder=True)

        print("Inputs have been reset to default values.")

