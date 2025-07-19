# =============================================================================
# memory_page.py — Memory mode input panel for the PROMISE Simulator
# Author:      Emilie Ye
# Date:        2025-06-27
#
# Description: 
#   This file defines the user interface for Memory Mode in the PROMISE Simulator.
#   It provides controls for selecting device data, loading NN/BNN weights,
#   configuring key simulation parameters, including sampling times, bins, technology,
#   and uploading configuration options via YAML files.
#   The panel links user-configured parameters to the simulation core for running Memory Mode experiments.
#
# Copyright (c) 2025
# 
# =============================================================================

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../python')))
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QComboBox, QLineEdit, QGroupBox, QFormLayout,
    QFileDialog, QDialog, QSizePolicy, QFormLayout, QDialog, QInputDialog
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QFontMetrics
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
icon_path = os.path.join(os.path.dirname(__file__), "icons", "config.jpg")
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

class MemoryInputPage(QWidget):
    #：updated_nn, updated_bnn, updated_sampled_fc1
    simulationRequested = Signal(bool, bool, bool)
    backRequested      = Signal()

    def __init__(self, core, parent=None):
        super().__init__(parent)
        self.simulator_core = core

        self.setWindowTitle("Probabilistic CIM Simulator Tool")
        
        # Left (input), Right (output)
        main_layout = QHBoxLayout(self)

        # Left Input Panel
        input_panel = QGroupBox("Memory Simulation Settings")
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

        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(5)

        self.config_button = QPushButton()
        self.config_button.setIcon(QIcon(icon_path))
        self.config_button.setIconSize(QSize(24, 24))
        self.config_button.setFixedSize(30, 30)
        self.config_button.setToolTip("Open advanced config…")
        self.config_button.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self.config_button.clicked.connect(self.open_config_dialog)

        config_label = QLabel("Config")
        config_label.setStyleSheet("font-weight: normal;")

        config_layout.addWidget(self.config_button)
        config_layout.addWidget(config_label)
        config_container = QWidget()
        config_container.setLayout(config_layout)

        self.input_layout.addRow(config_container)


        self.data_button = QPushButton("Choose File")
        self.data_button.clicked.connect(self.choose_device_data)
        self.data_button.setFixedHeight(24)
        self.input_layout.addRow("Device Data:", self.data_button)

        self.weight_combo = QComboBox()
        self.weight_combo.addItems(["A", "B", "C"])
        self.input_layout.addRow("Weight:", self.weight_combo)

        self.technology_input = QLineEdit("0")
        self.input_layout.addRow("Technology:", self.technology_input)

        self.frequency_input = QLineEdit("1e9")
        self.input_layout.addRow("Frequency:", self.frequency_input)

        self.precision_mu_input = QLineEdit("1")
        self.input_layout.addRow("Precision mu:", self.precision_mu_input)

        self.precision_sigma_input = QLineEdit("1")
        self.input_layout.addRow("Precision sigma:", self.precision_sigma_input)

        self.sample_input = QLineEdit("100")
        self.input_layout.addRow("Sampling Times:", self.sample_input)

        self.upload_folder_button = QPushButton("Upload Weights Folder")
        self.upload_folder_button.clicked.connect(self.choose_folder)
        self.upload_folder_button.setFixedHeight(24)
        self.input_layout.addRow("NN Weights Folder:", self.upload_folder_button)

        self.upload_bnn_folder_button = QPushButton("Upload BNN Weights Folder")
        self.upload_bnn_folder_button.clicked.connect(self.choose_bnn_folder)
        self.upload_bnn_folder_button.setFixedHeight(24)
        self.input_layout.addRow("BNN Weights Folder:", self.upload_bnn_folder_button)

        self.fc1_flat_input = QLineEdit("0")
        self.input_layout.addRow("BNN FC1 Index (0~47999):", self.fc1_flat_input)

        self.fc2_flat_input = QLineEdit("0")
        self.input_layout.addRow("BNN FC2 Index (0~10079):", self.fc2_flat_input)

        self.fc3_flat_input = QLineEdit("0")
        self.input_layout.addRow("BNN FC3 Index (0~839):", self.fc3_flat_input)

        self.bins_input = QLineEdit("50")
        self.input_layout.addRow("Bins:", self.bins_input)
        
        self.pie_config_button = QPushButton("Upload Pie Config YAML")
        self.pie_config_button.clicked.connect(self.choose_pie_config)
        self.pie_config_button.setFixedHeight(24)
        self.input_layout.addRow("Pie Config File:", self.pie_config_button)
        self.pie_config_path = None

        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems([
            "Near-memory Digital",
            "Near-memory Analog",
            "In-memory Aggregation",
            "In-memory Computation"
        ])
        self.architecture_combo.setCurrentIndex(0)
        self.architecture_combo.currentIndexChanged.connect(self.update_architecture_image)
        self.input_layout.addRow("CIM Architecture:", self.architecture_combo)

        # Architecture Image Preview
        self.architecture_image_label = QLabel()
        self.architecture_image_label.setAlignment(Qt.AlignCenter)
        #set the height to a fixed value
        #self.architecture_image_label.setMinimumHeight(180)
        self.architecture_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  #set the label width to the maximum available width
        self.current_architecture_pixmap_path = None

        self.architecture_image_label.setStyleSheet("""
            border: 1px solid #DDD;
            background-color: #FAFAFA;
            border-radius: 8px;
        """)
        self.input_layout.addRow("", self.architecture_image_label)


        # Buttons
        self.back_button = QPushButton("Back") 
        self.back_button.clicked.connect(lambda: self.backRequested.emit())
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
        
        self.setLayout(main_layout)
        self.update_architecture_image()
        QTimer.singleShot(0, self._scale_architecture_pixmap)

    # Input Actions
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing .csv Files")
        if folder:
            self.simulator_core.weights_folder = folder

            fm = QFontMetrics(self.upload_folder_button.font())
            file_path_text = fm.elidedText(folder, Qt.ElideLeft, self.upload_folder_button.width()-16)

            self.upload_folder_button.setText(file_path_text)
            self.upload_folder_button.setToolTip(folder)
            self.upload_folder_button.setStyleSheet("""
                QPushButton {
                    text-align: right;
                    padding-right: 8px;
                }
            """)

            print("Selected NN folder:", folder)

    def choose_bnn_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing BNN .csv Files")
        if folder:
            self.simulator_core.bnn_weights_folder = folder
            fm = QFontMetrics(self.upload_bnn_folder_button.font())
            file_path_text = fm.elidedText(folder, Qt.ElideLeft, self.upload_bnn_folder_button.width()-16)

            self.upload_bnn_folder_button.setText(file_path_text)
            self.upload_bnn_folder_button.setToolTip(folder)
            self.upload_bnn_folder_button.setStyleSheet("""
                QPushButton {
                    text-align: right;
                    padding-right: 8px;
                }
            """)

            print("Selected BNN folder:", folder)

    def choose_device_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Device Data File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            
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

            self.simulator_core.device_data_path = file_path

            print("Selected Device Data:", file_path)

    def choose_pie_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Pie Config YAML", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if file_path:
            self.pie_config_path = file_path
            self.simulator_core.pie_config_path = file_path
            fm = QFontMetrics(self.pie_config_button.font())
            file_path_text = fm.elidedText(file_path, Qt.ElideLeft, self.pie_config_button.width()-16)

            self.pie_config_button.setText(file_path_text)
            self.pie_config_button.setToolTip(file_path)
            self.pie_config_button.setStyleSheet("""
                QPushButton {
                    text-align: right;
                    padding-right: 8px;
                }
            """)
            print("Selected Pie Config File:", file_path)

    def _choose_yaml(self, label, callback):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {label.capitalize()} YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            print(f"[{label}] YAML selected:", path)
            callback(path)

    def open_config_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Configuration")
        dlg.setMinimumWidth(400)

        form = QFormLayout(dlg)


        # ADC: first SAR/Flash, then Custom/Default
        self.adc_type_btn = QPushButton("ADC:")
        self.adc_type_value = "SAR"

        def choose_adc_type():
            val, ok = QInputDialog.getItem(dlg, "ADC Configuration", "Choose ADC Type:", ["SAR", "Flash"], 0, False)
            if ok:
                self.adc_type_value = val
                self.adc_type_btn.setText(f"ADC: {val}")

        
        self.adc_type_btn.clicked.connect(choose_adc_type)

        self.memory_adc = QComboBox()
        self.memory_adc.addItems(["Default", "Custom"])

        row_adc = QHBoxLayout()
        row_adc.addWidget(self.adc_type_btn)
        row_adc.addWidget(self.memory_adc)
        form.addRow(row_adc)


        self.adc_file_button = QPushButton("Choose ADC YAML…")
        self.adc_file_button.setFixedHeight(24) 
        self.adc_file_button.hide()
        row_adc_file = QHBoxLayout()
        row_adc_file.addWidget(QLabel())  # left
        row_adc_file.addWidget(self.adc_file_button)
        form.addRow(row_adc_file)

        self.memory_adc.currentTextChanged.connect(
            lambda t: self.adc_file_button.setVisible(t == "Custom")
        )
        #self.adc_file_button.clicked.connect(
        #    lambda: self._choose_yaml("adc", lambda val: setattr(self, "memory_adc_config", val))
        #)
        self.adc_file_button.clicked.connect(self.choose_adc_file)
        

        # Buffer: Custom/Default + file
        self.memory_buffer = QComboBox()
        self.memory_buffer.addItems(["Default", "Custom"])

        label_buffer = QLabel("Buffer:")
        label_buffer.setMinimumWidth(100)
        label_buffer.setAlignment(Qt.AlignCenter)

        row_buffer = QHBoxLayout()
        row_buffer.addWidget(label_buffer)
        row_buffer.addWidget(self.memory_buffer)
        form.addRow(row_buffer)

        self.buffer_file_button = QPushButton("Choose Buffer YAML…")
        self.buffer_file_button.hide()
        self.buffer_file_button.setFixedHeight(24) 
        row_buffer_file = QHBoxLayout()
        row_buffer_file.addWidget(QLabel())  # left
        row_buffer_file.addWidget(self.buffer_file_button)
        form.addRow(row_buffer_file)

        self.memory_buffer.currentTextChanged.connect(
            lambda t: self.buffer_file_button.setVisible(t == "Custom")
        )
        #self.buffer_file_button.clicked.connect(
        #    lambda: self._choose_yaml("buffer", lambda val: setattr(self, "memory_buffer_config", val))
        #)
        self.buffer_file_button.clicked.connect(self.choose_buffer_file)
      
        #SenseAMP
        self.memory_SenseAMP = QComboBox()
        self.memory_SenseAMP.addItems(["Default", "Custom"])

        label_senseamp = QLabel("SenseAMP:")
        label_senseamp.setMinimumWidth(100)
        label_senseamp.setAlignment(Qt.AlignCenter)
        row_senseamp = QHBoxLayout()
        row_senseamp.addWidget(label_senseamp)
        row_senseamp.addWidget(self.memory_SenseAMP)
        form.addRow(row_senseamp)
    
        self.senseamp_file_button = QPushButton("Choose SenseAmp YAML…")
        self.senseamp_file_button.hide()
        self.senseamp_file_button.setFixedHeight(24) 
        row_senseamp_file = QHBoxLayout()
        row_senseamp_file.addWidget(QLabel())  # left
        row_senseamp_file.addWidget(self.senseamp_file_button)
        form.addRow(row_senseamp_file)

        self.memory_SenseAMP.currentTextChanged.connect(
            lambda t: self.senseamp_file_button.setVisible(t == "Custom")
        )
        #self.senseamp_file_button.clicked.connect(
        #    lambda: self._choose_yaml("senseamp", lambda val: setattr(self, "memory_senseamp_config", val))
        #)
        self.senseamp_file_button.clicked.connect(self.choose_senseamp_file)
      
         #Decoder
        self.memory_Decoder = QComboBox()
        self.memory_Decoder.addItems(["Default", "Custom"])

        label_decoder = QLabel("Decoder:")
        label_decoder.setMinimumWidth(100)
        label_decoder.setAlignment(Qt.AlignCenter)

        row_decoder = QHBoxLayout()
        row_decoder.addWidget(label_decoder)
        row_decoder.addWidget(self.memory_Decoder)
        form.addRow(row_decoder)
        
        self.decoder_file_button = QPushButton("Choose Decoder YAML…")
        self.decoder_file_button.hide()
        self.decoder_file_button.setFixedHeight(24) 
        row_decoder_file = QHBoxLayout()
        row_decoder_file.addWidget(QLabel())  # left
        row_decoder_file.addWidget(self.decoder_file_button)
        form.addRow(row_decoder_file)

        self.memory_Decoder.currentTextChanged.connect(
            lambda t: self.decoder_file_button.setVisible(t == "Custom")
        )
        #self.decoder_file_button.clicked.connect(
        #    lambda: self._choose_yaml("decoder", lambda val: setattr(self, "memory_decoder_config", val))
        #)
        self.decoder_file_button.clicked.connect(self.choose_decoder_file)
                

        #Memory: SRAM/RRAM/FeFET + Custom/Default + file
        self.memory_tech_btn = QPushButton("Memory:")
        self.memory_tech_value = "SRAM"

        def choose_memory_tech():
            val, ok = QInputDialog.getItem(dlg, "Memory Configuration", "Choose Memory Tech:", ["SRAM", "RRAM", "FeFET"], 0, False)
            if ok:
                self.memory_tech_value = val
                self.memory_tech_btn.setText(f"Memory: {val}")

        self.memory_tech_btn.clicked.connect(choose_memory_tech)

        self.memory_MemoryType = QComboBox()
        self.memory_MemoryType.addItems(["Default", "Custom"])

        row_mem = QHBoxLayout()
        row_mem.addWidget(self.memory_tech_btn)
        row_mem.addWidget(self.memory_MemoryType)
        form.addRow(row_mem)


        self.memory_file_button = QPushButton("Choose Memory YAML…")
        self.memory_file_button.hide()
        self.memory_file_button.setFixedHeight(24) 
        row_memory_file = QHBoxLayout()
        row_memory_file.addWidget(QLabel())  # left
        row_memory_file.addWidget(self.memory_file_button)
        form.addRow(row_memory_file)

        self.memory_MemoryType.currentTextChanged.connect(
            lambda t: self.memory_file_button.setVisible(t == "Custom")
        )
        #self.memory_file_button.clicked.connect(
        #    lambda: self._choose_yaml("memory", lambda val: setattr(self, "memory_memory_config", val))
        #)
        self.memory_file_button.clicked.connect(self.choose_memory_file)
       
        # Technology: 65nm / 45nm / 28nm / 14nm
        label_tech = QLabel("Technology:")
        label_tech.setMinimumWidth(100)
        label_tech.setAlignment(Qt.AlignCenter)

        self.technology_combo = QComboBox()
        self.technology_combo.addItems(["65nm", "45nm", "28nm", "14nm"])

        row_tech = QHBoxLayout()
        row_tech.addWidget(label_tech)
        row_tech.addWidget(self.technology_combo)
        form.addRow(row_tech)

        # OK / Cancel
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")

        btn_style = """
        QPushButton {
            background-color: #0078D7;
            color: white;
            border: none;
            padding: 6px 16px;
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
        """
        btn_ok.setStyleSheet(btn_style)
        btn_cancel.setStyleSheet(btn_style)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()

        form.addRow(btn_row)

        if dlg.exec() == QDialog.Accepted:
            self.simulator_core.config_adc_type           = self.adc_type_value
            self.simulator_core.config_adc_config_mode    = self.memory_adc.currentText()
            self.simulator_core.config_adc_yaml           = getattr(self, "memory_adc_config", "")
            self.simulator_core.config_buffer_config_mode = self.memory_buffer.currentText()
            self.simulator_core.config_buffer_yaml        = getattr(self, "memory_buffer_config", "")
            self.simulator_core.config_senseamp_mode      = self.memory_SenseAMP.currentText()
            self.simulator_core.config_senseamp_yaml      = getattr(self, "memory_senseamp_config", "")
            self.simulator_core.config_decoder_mode       = self.memory_Decoder.currentText()
            self.simulator_core.config_decoder_yaml       = getattr(self, "memory_decoder_config", "")
            self.simulator_core.config_memory_tech        = self.memory_tech_value
            self.simulator_core.config_memory_config_mode = self.memory_MemoryType.currentText()
            self.simulator_core.config_memory_yaml        = getattr(self, "memory_memory_config", "")
            self.simulator_core.config_technology_node    = self.technology_combo.currentText()
            print("Config saved")
        else:
            print("Configuration canceled")

    def choose_adc_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ADC YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.memory_adc_config = path
            fm = QFontMetrics(self.adc_file_button.font())
            txt = fm.elidedText(path, Qt.ElideLeft, self.adc_file_button.width()-16)
            self.adc_file_button.setText(txt)
            self.adc_file_button.setToolTip(path)
            self.adc_file_button.setStyleSheet("QPushButton { text-align: right; padding-right: 8px; }")

    def choose_buffer_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Buffer YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.memory_buffer_config = path
            fm = QFontMetrics(self.buffer_file_button.font())
            txt = fm.elidedText(path, Qt.ElideLeft, self.buffer_file_button.width()-16)
            self.buffer_file_button.setText(txt)
            self.buffer_file_button.setToolTip(path)
            self.buffer_file_button.setStyleSheet("QPushButton { text-align: right; padding-right: 8px; }")

    def choose_senseamp_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SenseAmp YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.memory_senseamp_config = path
            fm = QFontMetrics(self.senseamp_file_button.font())
            txt = fm.elidedText(path, Qt.ElideLeft, self.senseamp_file_button.width()-16)
            self.senseamp_file_button.setText(txt)
            self.senseamp_file_button.setToolTip(path)
            self.senseamp_file_button.setStyleSheet("QPushButton { text-align: right; padding-right: 8px; }")

    def choose_decoder_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Decoder YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.memory_decoder_config = path
            fm = QFontMetrics(self.decoder_file_button.font())
            txt = fm.elidedText(path, Qt.ElideLeft, self.decoder_file_button.width()-16)
            self.decoder_file_button.setText(txt)
            self.decoder_file_button.setToolTip(path)
            self.decoder_file_button.setStyleSheet("QPushButton { text-align: right; padding-right: 8px; }")

    def choose_memory_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Memory YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.memory_memory_config = path
            fm = QFontMetrics(self.memory_file_button.font())
            txt = fm.elidedText(path, Qt.ElideLeft, self.memory_file_button.width()-16)
            self.memory_file_button.setText(txt)
            self.memory_file_button.setToolTip(path)
            self.memory_file_button.setStyleSheet("QPushButton { text-align: right; padding-right: 8px; }")

    def update_architecture_image(self):

        architecture = self.architecture_combo.currentText()
        image_map = {
            "Near-memory Digital": "architecture1.svg",
            "Near-memory Analog": "architecture2.svg",
            "In-memory Aggregation": "architecture3.svg",
            "In-memory Computation": "architecture4.svg",
        }

        # Go up 3 levels from current file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        image_filename = os.path.join(base_dir, "fig", image_map.get(architecture, ""))

        if os.path.exists(image_filename):
            self.current_architecture_pixmap_path = image_filename
            self._scale_architecture_pixmap()

        else:
            self.architecture_image_label.clear()
            self.architecture_image_label.setText("No image found.")

    def _scale_architecture_pixmap(self):
        if not self.current_architecture_pixmap_path:
            return

        pixmap = QPixmap(self.current_architecture_pixmap_path)
        if pixmap.isNull():
            self.architecture_image_label.setText("Failed to load image.")
            return

        label_size = self.architecture_image_label.size()

        scaled = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.architecture_image_label.setPixmap(scaled)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_architecture_pixmap_path:
            self._scale_architecture_pixmap()


    # When click start simulation
    def start_simulation(self):
        updated_nn = False
        updated_bnn = False
        updated_sampled_fc1 = False

        self.simulator_core.mode   = "Memory"

        plots = self.simulator_core.plots_dir
        for f in glob.glob(os.path.join(plots, "*.svg")):
            try:
                os.remove(f)
            except OSError:
                pass

        #if not self.simulator_core.weights_folder and not self.simulator_core.bnn_weights_folder and not self.simulator_core.device_data_path:
        if (not self.simulator_core.weights_folder
            and not self.simulator_core.bnn_weights_folder
            and not self.simulator_core.device_data_path
            and not self.pie_config_path
        ):
            print("No input provided. Skipping simulation.")
            return

        print("Starting simulation...")

        self.simulator_core.weight = self.weight_combo.currentText()
        self.simulator_core.technology = self.technology_input.text()
        self.simulator_core.frequency = self.frequency_input.text()
        self.simulator_core.precision_mu = self.precision_mu_input.text()
        self.simulator_core.precision_sigma = self.precision_sigma_input.text()
        self.simulator_core.sampling_times = self.sample_input.text()
        self.simulator_core.fc1_index = self.fc1_flat_input.text()
        self.simulator_core.fc2_index = self.fc2_flat_input.text()
        self.simulator_core.fc3_index = self.fc3_flat_input.text()
        self.simulator_core.bins = self.bins_input.text()
        self.simulator_core.architecture = self.architecture_combo.currentText()


        updated_nn = self.simulator_core.generate_nn_weight_plots()
        updated_bnn = self.simulator_core.generate_bnn_weight_plots()
        updated_sampled_fc1 = self.simulator_core.generate_sampled_fc1_weight_plot()

        self.simulator_core.generate_all_pie_charts(yaml_path=self.pie_config_path)

        self.simulator_core.export_memory_config_to_yaml()

        self.simulationRequested.emit(
         updated_nn,
         updated_bnn,
         updated_sampled_fc1
         )
        


    def reset_inputs(self):

        self.weight_combo.setCurrentIndex(0)

        # Reset LineEdits to initial values
        self.technology_input.setText("0")
        self.frequency_input.setText("1e9")
        self.precision_mu_input.setText("1")
        self.precision_sigma_input.setText("1")
        self.sample_input.setText("100")

        self.fc1_flat_input.setText("0")
        self.fc2_flat_input.setText("0")
        self.fc3_flat_input.setText("0")
        self.bins_input.setText("50")

        # Reset folder paths
        self.simulator_core.weights_folder = None
        self.simulator_core.bnn_weights_folder = None
        self.simulator_core.device_data_path = None

        self.architecture_combo.setCurrentIndex(0)
        self.pie_config_path = None
        self.simulator_core.pie_config_path = None

        self.fc1_image = None
        self.fc2_image = None
        self.fc3_image = None

        self.bnn_weight_fc1_plot = None
        self.bnn_weight_fc2_plot = None
        self.bnn_weight_fc3_plot = None

        self.bnn_weight_fc1_qq_plot = None
        self.bnn_weight_fc2_qq_plot = None
        self.bnn_weight_fc3_qq_plot = None

        self.bnn_fc1_miu_plot = None
        self.bnn_fc2_miu_plot = None
        self.bnn_fc3_miu_plot = None

        self.bnn_fc1_sigma_plot = None
        self.bnn_fc2_sigma_plot = None
        self.bnn_fc3_sigma_plot = None

        self.tabs = None
        self.current_tab_index = 0

        self.data_button.setText("Choose File")
        self.data_button.setToolTip("")
        self.data_button.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
            }}
        """)

        self.upload_folder_button.setText("Upload Weights Folder")
        self.upload_folder_button.setToolTip("")
        self.upload_folder_button.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
            }}
        """)

        self.upload_bnn_folder_button.setText("Upload BNN Weights Folder")
        self.upload_bnn_folder_button.setToolTip("")
        self.upload_bnn_folder_button.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
            }}
        """)

        self.pie_config_button.setText("Upload Pie Config YAML")
        self.pie_config_button.setToolTip("")
        self.pie_config_button.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
            }}
        """)

        self.memory_adc_config    = None
        self.memory_buffer_config = None
        self.memory_senseamp_config = None
        self.memory_decoder_config  = None
        self.memory_memory_config   = None

        self.simulationRequested.emit(False, False, False)

        print("Inputs have been reset to default values.")