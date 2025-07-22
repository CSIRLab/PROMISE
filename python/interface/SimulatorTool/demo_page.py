# =============================================================================
# demo_page.py — Demo mode input panel for the PROMISE Simulator
# Author:      Emilie Ye
# Date:        2025-06-27
#
# Description:
#   This file defines the user interface for the Demo Mode of the PROMISE Simulator.
#   It provides controls for setting parameters for two configurations:
#       Single Gaussian mode: users can select the RNG source (ideal generator or imported data),
#           specify the number of weights, provide mean (μ) and standard deviation (σ) for each weight,
#           configure sampling time for iterations, and set the number of bins for detailed histograms.
#       Gaussian Mixture Model (GMM) mode: users can configure multiple Gaussian components,
#           select the RNG source, define their means, variances, mixture weights (α),
#           where α specifies the relative contribution of each component distribution in the mixture,
#           and set sampling times and number of histogram bins.
#   The panel links user-configured parameters to the simulation core for running Demo Mode experiments.
#
# Copyright (c) 2025
# 
# =============================================================================

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../python')))
from PySide6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QComboBox, QLineEdit, QGroupBox,
    QFormLayout, QFileDialog, QFormLayout, QSizePolicy, QLabel, QPlainTextEdit,QTextBrowser,QVBoxLayout
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

        self.gaussian_mode = QComboBox()
        self.gaussian_mode.addItems(["Single Gaussian", "Gaussian Mixture Model", "Multiple GMM"])
        self.gaussian_mode.currentIndexChanged.connect(self._on_gaussian_mode_changed)
        self.input_layout.addRow("Mode:", self.gaussian_mode)


        self.rng_mode = QComboBox()
        self.rng_mode.addItems(["ideal", "Choose file"])
        self.rng_mode.currentIndexChanged.connect(self._toggle_file_btn)
        self.input_layout.addRow("RNG Source:", self.rng_mode)


        self.data_button = QPushButton("Choose RNG File")
        self.data_button.clicked.connect(self.choose_device_data)
        self.input_layout.addRow("RNG File:", self.data_button)
        self.data_button.setFixedHeight(24)

        self.gmm_number_label = QLabel("GMM Number:")
        self.gmm_number = QComboBox()
        self.gmm_number.addItems(["2", "3"])
        self.gmm_number.currentIndexChanged.connect(self._update_weight_inputs)
        self.input_layout.addRow(self.gmm_number_label, self.gmm_number)


        self.weight_number_label = QLabel("Weight Number:")  
        self.weight_number = QComboBox()
        self.weight_number.addItems(["1", "2", "3"])
        self.weight_number.currentIndexChanged.connect(self._update_weight_inputs)
        self.input_layout.addRow(self.weight_number_label, self.weight_number) 



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

        self.alpha_label = QLabel("α:")
        self.alpha_container = QWidget()
        self.alpha_fields_layout = QVBoxLayout()
        self.alpha_fields_layout.setSpacing(2)
        self.alpha_container.setLayout(self.alpha_fields_layout)
        self.input_layout.addRow(self.alpha_label, self.alpha_container)
        # Hide by default
        self.alpha_label.hide()
        self.alpha_container.hide()
        self.alpha_lineedits = []


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

        self.help_box = QTextBrowser()
        self.help_box.setReadOnly(True)
        self.help_box.setPlainText(self._get_help_text())
        self.help_box.setFixedHeight(400)
        self.help_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.help_box.setStyleSheet("""
            QTextBrowser {
            background-color: #FFFFFF;
            font-family: Consolas, monospace;
            font-size: 11pt;
            border: none;
            }
        """)

        self.input_layout.addRow(self.help_box)

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(input_panel, 1)


        main_layout.setStretch(0, 1)  # Input panel
        
        self._update_weight_inputs()
        self._toggle_file_btn()

        self.setLayout(main_layout)
        self._on_gaussian_mode_changed()


    def _get_help_text(self):
        if self.gaussian_mode.currentText() in ["Gaussian Mixture Model", "Multiple GMM"]:
            return """
            <div style="border: 2px solid #0078D7; border-radius: 6px; padding: 8px; background-color: #F9F9F9;">
            <b>Algorithm 2: Gaussian Mixture Sampling Procedure</b><br><br>
            <b>Parameters:</b> μ = {μ₁, ..., μ<sub>K</sub>}, σ = {σ₁, ..., σ<sub>K</sub>},<br>
            α = {α₁, ..., α<sub>K</sub>} (mixture weights), S = sample count, Mode ∈ {ideal, file}<br>
            <b>Inputs:</b> n = number of weights<br>
            <b>Output:</b> Sampled values X = {X₁, ..., X<sub>n</sub>}<br><br>

            <b>Function:</b> <code>GMM_Sample(μ, σ, α, S, Mode)</code><br>
            &nbsp;&nbsp;for i = 1 to n do<br>
            &nbsp;&nbsp;&nbsp;&nbsp;begin<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Draw k<sub>i</sub> ∼ Categorical(α)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if Mode == "ideal" then<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ε<sub>i</sub> ← torch.randn(S)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ε<sub>i</sub> ← sample_from_file(S)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;X<sub>i</sub> ← μ<sub>k<sub>i</sub></sub> + σ<sub>k<sub>i</sub></sub> ⊙ ε<sub>i</sub><br>
            &nbsp;&nbsp;&nbsp;&nbsp;end<br>
            <b>return</b> X<br>
            <b>end Function</b>
            </div>
            """
        else:
            return """
            <div style="border: 2px solid #0078D7; border-radius: 6px; padding: 8px; background-color: #F9F9F9;">
            <b>Algorithm 1: Single Gaussian Sampling Procedure</b><br><br>
            <b>Parameters:</b> μ = {μ₁, μ₂, ..., μₙ}, σ = {σ₁, σ₂, ..., σₙ},<br>
            S = number of samples per weight, Mode ∈ {ideal, file}<br>
            <b>Inputs:</b> n = number of weights<br>
            <b>Output:</b> Sampled values X = {X₁, ..., Xₙ}<br><br>
            <b>Function:</b> <code>Sample(μ, σ, S, Mode)</code><br>
            &nbsp;&nbsp;for i = 1 to n do<br>
            &nbsp;&nbsp;&nbsp;&nbsp;begin<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if Mode == "ideal" then<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;εᵢ ← torch.randn(S)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;εᵢ ← sample_from_file(S)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Xᵢ ← μᵢ + σᵢ ⊙ εᵢ<br>
            &nbsp;&nbsp;&nbsp;&nbsp;end<br>
            <b>return</b> X<br>
            <b>end Function</b>
            </div>
            """



    def _on_gaussian_mode_changed(self):
        mode = self.gaussian_mode.currentText()
        self.weight_number.blockSignals(True)
        self.weight_number.clear()
        if mode == "Gaussian Mixture Model":
            self.weight_number_label.setText("Distribution Number:")
            self.weight_number.addItems(["2", "3"])
            self.gmm_number_label.hide()
            self.gmm_number.hide()

        elif mode == "Multiple GMM":
            self.weight_number_label.setText("Distribution Number:")
            self.weight_number.addItems(["2", "3"])
            self.weight_number.setCurrentIndex(0)
            self.gmm_number_label.show()
            self.gmm_number.show()

        else:  # Single Gaussian
            self.weight_number_label.setText("Weight Number:")
            self.weight_number.addItems(["1", "2", "3"])
            self.gmm_number_label.hide()
            self.gmm_number.hide()

        self.weight_number.setCurrentIndex(0)
        self.weight_number.blockSignals(False)
        self._update_weight_inputs()
        self.help_box.setHtml(self._get_help_text())



    def _update_weight_inputs(self):
        txt = self.weight_number.currentText()
        if not txt.isdigit():
            return
        number = int(txt)
        num_gmms = int(self.gmm_number.currentText()) if self.gaussian_mode.currentText() == "Multiple GMM" else 1
        mode = self.gaussian_mode.currentText()

        # Clear μ
        while self.mu_fields_layout.count():
            widget = self.mu_fields_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.mu_lineedits = []

        for g in range(num_gmms):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setSpacing(2)
            mu_group = []
            for idx in range(number):
                mu_edit = QLineEdit("1")
                mu_edit.setPlaceholderText(f"μ{idx+1} (GMM{g+1})" if num_gmms > 1 else f"μ{idx+1}")
                mu_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                row_layout.addWidget(mu_edit)
                mu_group.append(mu_edit)
            self.mu_fields_layout.addWidget(row_widget)
            self.mu_lineedits.append(mu_group)

        # Clear σ
        while self.sigma_fields_layout.count():
            widget = self.sigma_fields_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.sigma_lineedits = []

        for g in range(num_gmms):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setSpacing(2)
            sigma_group = []
            for idx in range(number):
                sigma_edit = QLineEdit("1")
                sigma_edit.setPlaceholderText(f"σ{idx+1} (GMM{g+1})" if num_gmms > 1 else f"σ{idx+1}")
                sigma_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                row_layout.addWidget(sigma_edit)
                sigma_group.append(sigma_edit)
            self.sigma_fields_layout.addWidget(row_widget)
            self.sigma_lineedits.append(sigma_group)

        is_mixture_mode = self.gaussian_mode.currentText() in ["Gaussian Mixture Model", "Multiple GMM"]

        if is_mixture_mode:
            self.alpha_label.show()
            self.alpha_container.show()

            # Clear existing
            while self.alpha_fields_layout.count():
                w = self.alpha_fields_layout.takeAt(0).widget()
                if w:
                    w.deleteLater()

            self.alpha_lineedits = []
            for g in range(num_gmms):
                row_widget = QWidget()
                row_layout = QHBoxLayout()
                alpha_group = []
                for idx in range(number):
                    alpha_edit = QLineEdit(f"{1.0 / number:.4f}")
                    alpha_edit.setPlaceholderText(f"α{idx+1} (GMM{g+1})" if num_gmms > 1 else f"α{idx+1}")
                    alpha_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                    row_layout.addWidget(alpha_edit)
                    alpha_group.append(alpha_edit)
                row_widget.setLayout(row_layout)
                self.alpha_fields_layout.addWidget(row_widget)
                self.alpha_lineedits.append(alpha_group)

        else:
            self.alpha_label.hide()
            self.alpha_container.hide()
            self.alpha_lineedits = []


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
        print(">>> start_simulation CALLED")
        updated_nn = False
        updated_bnn = False
        updated_sampled_fc1 = False
        mode = self.gaussian_mode.currentText()

        self.simulator_core.mode   = "Demo"

        plots = self.simulator_core.plots_dir
        #clean the plots folder
        for f in glob.glob(os.path.join(plots, "*.svg")):
            try:
                os.remove(f)
            except OSError:
                pass

        print("Starting simulation...")

        if mode == "Multiple GMM":
            mus = [torch.tensor([float(e.text()) for e in g]) for g in self.mu_lineedits]
            sigmas = [torch.tensor([float(e.text()) for e in g]) for g in self.sigma_lineedits]
            alphas = [torch.tensor([float(e.text()) for e in g]) for g in self.alpha_lineedits]
            self.simulator_core.demo_mus = mus
            self.simulator_core.demo_sigmas = sigmas
            self.simulator_core.demo_alphas = alphas

        elif mode == "Gaussian Mixture Model":
            self.simulator_core.demo_mus = torch.tensor([float(e.text()) for e in self.mu_lineedits[0]])
            self.simulator_core.demo_sigmas = torch.tensor([float(e.text()) for e in self.sigma_lineedits[0]])
            self.simulator_core.demo_alphas = torch.tensor([float(e.text()) for e in self.alpha_lineedits[0]])

        elif mode == "Single Gaussian":
            mus = torch.tensor([float(e.text()) for e in self.mu_lineedits[0]])
            sigmas = torch.tensor([float(e.text()) for e in self.sigma_lineedits[0]])

            self.simulator_core.demo_mus = mus  
            self.simulator_core.demo_sigmas = sigmas
            self.simulator_core.demo_alphas = None

        self.simulator_core.demo_mode    = self.rng_mode.currentText()

        self.simulator_core.gaussian_mode = self.gaussian_mode.currentText()

        self.simulator_core.demo_file    = self.file_path


        self.simulator_core.demo_samples = int(self.sample_input.text())
        self.simulator_core.demo_bins    = int(self.bins_input.text())

        self.simulator_core.export_demo_config_to_yaml()    # save the config to yaml file

        self.simulator_core.run_demo()  # run the demo simulation--generate the gaussian distribution and plot the results

        self.simulationRequested.emit() # emit the signal to run the simulation in the output panel file

    def reset_inputs(self):
        self.rng_mode.setCurrentIndex(0)

        self.gaussian_mode.setCurrentIndex(0)  # Single Gaussian

        self.weight_number.setCurrentIndex(0)

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

