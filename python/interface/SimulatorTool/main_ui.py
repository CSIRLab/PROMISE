import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QGroupBox, QFormLayout, QTableWidget, QTabWidget,
    QTableWidgetItem, QScrollArea, QFileDialog, QDialog, QSizePolicy
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QSize

from core_function import SimulatorCore
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probabilistic CIM Simulator Tool")
        self.setMinimumSize(1700, 1100)

        self.simulator_core = SimulatorCore()

        # Left (input), Right (output)
        main_layout = QHBoxLayout()

        # Left Input Panel
        input_panel = QGroupBox("Simulation Settings")
        input_panel.setStyleSheet("""
            QGroupBox {
                background-color: #F8F8F8;
                border: 1px solid #CCC;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
        """)
        input_layout = QFormLayout()

        self.data_button = QPushButton("Choose File")
        self.data_button.clicked.connect(self.choose_device_data)
        input_layout.addRow("Device Data:", self.data_button)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Memory", "Compute in Memory", "C"])
        input_layout.addRow("Mode:", self.mode_combo)

        self.weight_combo = QComboBox()
        self.weight_combo.addItems(["A", "B", "C"])
        input_layout.addRow("Weight:", self.weight_combo)

        self.technology_input = QLineEdit("0")
        input_layout.addRow("Technology:", self.technology_input)

        self.frequency_input = QLineEdit("1e9")
        input_layout.addRow("Frequency:", self.frequency_input)

        self.precision_mu_input = QLineEdit("1")
        input_layout.addRow("Precision mu:", self.precision_mu_input)

        self.precision_sigma_input = QLineEdit("1")
        input_layout.addRow("Precision sigma:", self.precision_sigma_input)

        self.sample_input = QLineEdit("100")
        input_layout.addRow("Sampling Times:", self.sample_input)

        self.upload_folder_button = QPushButton("Upload Weights Folder")
        self.upload_folder_button.clicked.connect(self.choose_folder)
        input_layout.addRow("NNWeights Folder:", self.upload_folder_button)

        self.upload_bnn_folder_button = QPushButton("Upload BNN Weights Folder")
        self.upload_bnn_folder_button.clicked.connect(self.choose_bnn_folder)
        input_layout.addRow("BNN Weights Folder:", self.upload_bnn_folder_button)

        self.fc1_flat_input = QLineEdit("0")
        input_layout.addRow("BNN FC1 Index (0~47999):", self.fc1_flat_input)

        self.fc2_flat_input = QLineEdit("0")
        input_layout.addRow("BNN FC2 Index (0~10079):", self.fc2_flat_input)

        self.fc3_flat_input = QLineEdit("0")
        input_layout.addRow("BNN FC3 Index (0~839):", self.fc3_flat_input)

        self.bins_input = QLineEdit("50")
        input_layout.addRow("Bins:", self.bins_input)

        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems([
            "Near-memory Digital",
            "Near-memory Analog",
            "In-memory Aggregation",
            "In-memory Computation"
        ])
        self.architecture_combo.setCurrentIndex(0)
        self.architecture_combo.currentIndexChanged.connect(self.update_architecture_image)
        input_layout.addRow("CIM Architecture:", self.architecture_combo)
        # Architecture Image Preview
        self.architecture_image_label = QLabel()
        self.architecture_image_label.setAlignment(Qt.AlignCenter)
        #set the height to a fixed value
        self.architecture_image_label.setMinimumHeight(180)
        self.architecture_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)   #set the label width to the maximum available width
        self.architecture_image_label.setStyleSheet("""
            border: 1px solid #DDD;
            background-color: #FAFAFA;
            border-radius: 8px;
        """)
        input_layout.addRow("", self.architecture_image_label)


        # Buttons
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
        self.reset_button.setFixedWidth(150)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)
        input_layout.addRow(button_layout)

        input_panel.setLayout(input_layout)
        # Add input panel to main layout and set stretch factors
        main_layout.addWidget(input_panel, 1)

        # Right Output Panel
        output_scroll = QScrollArea()
        output_scroll.setWidgetResizable(True)

        self.output_container = QWidget()
        self.output_layout = QVBoxLayout(self.output_container)

        output_scroll.setWidget(self.output_container)
        # Add output scroll area to main layout and set stretch factors
        main_layout.addWidget(output_scroll, 3)

        # Initial placeholder
        placeholder_label = QLabel("Output will appear here after running simulation.")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("""
            border: 1px dashed #AAA;
            background-color: #FAFAFA;
            color: #888;
            font-size: 14pt;
            padding: 100px;
        """)
        self.output_layout.addWidget(placeholder_label)
        self.setLayout(main_layout)
        self.update_architecture_image()


    # Input Actions
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing .csv Files")
        if folder:
            self.simulator_core.weights_folder = folder
            print("Selected NN folder:", folder)

    def choose_bnn_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing BNN .csv Files")
        if folder:
            self.simulator_core.bnn_weights_folder = folder
            print("Selected BNN folder:", folder)

    def choose_device_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Device Data File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.simulator_core.device_data_path = file_path
            print("Selected Device Data:", file_path)

    def reset_inputs(self):
        while self.output_layout.count() > 0:
            child = self.output_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add placeholder back
        placeholder_label = QLabel("Output will appear here after running simulation.")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("""
            border: 1px dashed #AAA;
            background-color: #FAFAFA;
            color: #888;
            font-size: 14pt;
            padding: 100px;
        """)
        self.output_layout.addWidget(placeholder_label)
        
        # Reset ComboBox
        self.mode_combo.setCurrentIndex(0)
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

        print("Inputs have been reset to default values.")

    def start_simulation(self):
        updated_nn = False
        updated_bnn = False
        updated_sampled_fc1 = False

        mode = self.mode_combo.currentText()

        if mode == "Memory":
            #if not self.simulator_core.weights_folder and not self.simulator_core.bnn_weights_folder and not self.simulator_core.device_data_path:
            if not self.simulator_core.weights_folder and not self.simulator_core.bnn_weights_folder:

                print("No input provided. Skipping simulation.")
                return

        if hasattr(self, "tabs") and mode == "Memory" and self.tabs:
            try:
                current_tab_index = self.tabs.currentIndex()
            except RuntimeError:
                current_tab_index = 0
        else:
            current_tab_index = 0

        # Clear output first
        while self.output_layout.count() > 0:
            child = self.output_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        print("Starting simulation...")

        # Update config
        self.simulator_core.mode = self.mode_combo.currentText()
        self.simulator_core.weight_mode = self.weight_combo.currentText()
        self.simulator_core.technology = self.technology_input.text()
        self.simulator_core.frequency = self.frequency_input.text()
        self.simulator_core.precision_mu = self.precision_mu_input.text()
        self.simulator_core.precision_sigma = self.precision_sigma_input.text()
        self.simulator_core.sampling_times = self.sample_input.text()
        self.simulator_core.fc1_index = self.fc1_flat_input.text()
        self.simulator_core.fc2_index = self.fc2_flat_input.text()
        self.simulator_core.fc3_index = self.fc3_flat_input.text()
        self.simulator_core.bins = self.bins_input.text()

        # Run plots
        if self.simulator_core.mode == "Memory":
            updated_nn = self.simulator_core.generate_nn_weight_plots()
            updated_bnn = self.simulator_core.generate_bnn_weight_plots()
            #updated_sampled_fc1 = self.simulator_core.generate_sampled_fc1_plots()
            updated_sampled_fc1 = self.simulator_core.generate_sampled_fc1_weight_plot()

        else:
            updated_nn = False
            updated_bnn = False
            updated_sampled_fc1 = False
        
        self.simulator_core.export_config_to_yaml()

        # Show output widgets
        self.add_output_widgets()
        # Generate energy/latency pie chart
        self.simulator_core.generate_all_pie_charts()

        if self.mode_combo.currentText() == "Memory" and hasattr(self, "tabs") and self.tabs:
            self.tabs.setCurrentIndex(current_tab_index)


        # Reload images
        self.reload_images(updated_nn=updated_nn, updated_bnn=updated_bnn, updated_sampled_fc1=updated_sampled_fc1)

    def add_output_widgets(self):
        mode = self.mode_combo.currentText()

        if mode == "Compute in Memory":
            # Accuracy Table
            self.table = QTableWidget(2, 2)
            self.table.setHorizontalHeaderLabels(["Metric", "Value"])
            self.table.setItem(0, 0, QTableWidgetItem("Accuracy"))
            self.table.setItem(1, 0, QTableWidgetItem("ECE"))
            self.table.setItem(0, 1, QTableWidgetItem("--"))
            self.table.setItem(1, 1, QTableWidgetItem("--"))
            self.table.setStyleSheet("""
                    QTableWidget {
                        background-color: white;
                        gridline-color: #EEE;
                    }
                    QHeaderView::section {
                        background-color: #F1F1F1;
                        font-weight: bold;
                        border: 1px solid #DDD;
                    }
                """)
            

            accuracy_title = QLabel("Accuracy / ECE Table")
            accuracy_title.setAlignment(Qt.AlignCenter)
            accuracy_title.setStyleSheet("""
                font-weight: bold;
                font-size: 11pt;
                margin: 10px;
            """)
            self.output_layout.addWidget(accuracy_title)

            self.output_layout.addWidget(self.table)

        if mode == "Memory":

            # === Tabs ===
            self.tabs = QTabWidget()

            # Tab 1
            #tab1_widget = QWidget()
            #tab1_layout = QHBoxLayout(tab1_widget)
            #tab1_layout.addWidget(self.create_plot("Distribution Plot", "Normal Distribution.png"))
            #tab1_layout.addWidget(self.create_plot("QQ Plot", "QQ plot.png"))
            #tab1_layout.addWidget(self.create_plot("Autocorrelation", "Autocorrelation.png"))
            #self.tabs.addTab(tab1_widget, "Plots")

            # Tab 2
            tab2_widget = QWidget()
            tab2_layout = QHBoxLayout(tab2_widget)
            self.fc1_image = self.create_plot("fc1 Weights", "fc1_weights_plot.svg", show_initial_image=False)
            self.fc2_image = self.create_plot("fc2 Weights", "fc2_weights_plot.svg", show_initial_image=False)
            self.fc3_image = self.create_plot("fc3 Weights", "fc3_weights_plot.svg", show_initial_image=False)

            tab2_layout.addWidget(self.fc1_image)
            tab2_layout.addWidget(self.fc2_image)
            tab2_layout.addWidget(self.fc3_image)
            self.tabs.addTab(tab2_widget, "NN Weights")

            # Tab 3 for BNN single weight plot
            tab3_widget = QWidget()
            tab3_full_layout = QVBoxLayout(tab3_widget)

            # Row 1 Distribution
            tab3_layout1 = QHBoxLayout()
            self.bnn_weight_fc1_plot = self.create_plot("BNN FC1 Weight Distribution", "bnn_fc1_weight_plot.svg", show_initial_image=False)
            self.bnn_weight_fc2_plot = self.create_plot("BNN FC2 Weight Distribution", "bnn_fc2_weight_plot.svg", show_initial_image=False)
            self.bnn_weight_fc3_plot = self.create_plot("BNN FC3 Weight Distribution", "bnn_fc3_weight_plot.svg", show_initial_image=False)

            tab3_layout1.addWidget(self.bnn_weight_fc1_plot)
            tab3_layout1.addWidget(self.bnn_weight_fc2_plot)
            tab3_layout1.addWidget(self.bnn_weight_fc3_plot)

            # Row 2 QQ plot
            tab3_layout2 = QHBoxLayout()
            self.bnn_weight_fc1_qq_plot = self.create_plot("BNN FC1 QQ Plot", "bnn_fc1_qq_plot.svg", show_initial_image=False)
            self.bnn_weight_fc2_qq_plot = self.create_plot("BNN FC2 QQ Plot", "bnn_fc2_qq_plot.svg", show_initial_image=False)
            self.bnn_weight_fc3_qq_plot = self.create_plot("BNN FC3 QQ Plot", "bnn_fc3_qq_plot.svg", show_initial_image=False)

            tab3_layout2.addWidget(self.bnn_weight_fc1_qq_plot)
            tab3_layout2.addWidget(self.bnn_weight_fc2_qq_plot)
            tab3_layout2.addWidget(self.bnn_weight_fc3_qq_plot)

            # Combine both rows into full tab layout
            tab3_full_layout.addLayout(tab3_layout1)
            tab3_full_layout.addLayout(tab3_layout2)

            
            # Tab 4 for BNN mu / sigma distributions
            tab4_widget = QWidget()
            tab4_full_layout = QVBoxLayout(tab4_widget)

            # Row 1 mu distributions
            tab4_layout1 = QHBoxLayout()
            self.bnn_fc1_miu_plot = self.create_plot("BNN FC1 μ Distribution", "bnn_fc1_miu_dist.svg", show_initial_image=False)
            self.bnn_fc2_miu_plot = self.create_plot("BNN FC2 μ Distribution", "bnn_fc2_miu_dist.svg", show_initial_image=False)
            self.bnn_fc3_miu_plot = self.create_plot("BNN FC3 μ Distribution", "bnn_fc3_miu_dist.svg", show_initial_image=False)

            tab4_layout1.addWidget(self.bnn_fc1_miu_plot)
            tab4_layout1.addWidget(self.bnn_fc2_miu_plot)
            tab4_layout1.addWidget(self.bnn_fc3_miu_plot)

            # Row 2 sigma distributions
            tab4_layout2 = QHBoxLayout()
            self.bnn_fc1_sigma_plot = self.create_plot("BNN FC1 σ Distribution", "bnn_fc1_sigma_dist.svg", show_initial_image=False)
            self.bnn_fc2_sigma_plot = self.create_plot("BNN FC2 σ Distribution", "bnn_fc2_sigma_dist.svg", show_initial_image=False)
            self.bnn_fc3_sigma_plot = self.create_plot("BNN FC3 σ Distribution", "bnn_fc3_sigma_dist.svg", show_initial_image=False)

            tab4_layout2.addWidget(self.bnn_fc1_sigma_plot)
            tab4_layout2.addWidget(self.bnn_fc2_sigma_plot)
            tab4_layout2.addWidget(self.bnn_fc3_sigma_plot)

            # Combine
            tab4_full_layout.addLayout(tab4_layout1)
            tab4_full_layout.addLayout(tab4_layout2)

            self.tabs.addTab(tab4_widget, "BNN μ / σ Distributions")

            self.tabs.addTab(tab3_widget, "BNN Weights")


            # Tab 5 for BNN FC1 sampled weight vs target
            tab5_widget = QWidget()
            tab5_layout = QHBoxLayout(tab5_widget)


            self.bnn_sampled_fc1_N1_plot = self.create_plot("N1 Distribution", "bnn_fc1_sampled_N1_distribution.svg", show_initial_image=False)
            self.bnn_sampled_fc1_plot = self.create_plot("BNN FC1 mu + sigma * N1 Distribution", "bnn_fc1_sampled_weight_plot.svg", show_initial_image=False)
            self.bnn_sampled_fc1_qq_plot = self.create_plot("BNN FC1 Sampled QQ Plot", "bnn_fc1_sampled_weight_qq_plot.svg", show_initial_image=False)

            tab5_layout.addWidget(self.bnn_sampled_fc1_N1_plot)
            tab5_layout.addWidget(self.bnn_sampled_fc1_plot)
            tab5_layout.addWidget(self.bnn_sampled_fc1_qq_plot)

            self.tabs.addTab(tab5_widget, "BNN Sampled FC1")

            # Add tabs to output layout and set fixed height
            self.tabs.setMinimumHeight(600)
            self.tabs.setMaximumHeight(600)
            self.output_layout.addWidget(self.tabs)
            
        # Pie chart
        # Pie Charts Section
        pie_group = QGroupBox("System-Level Pie Charts")
        pie_group.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 15px;")
        pie_layout = QHBoxLayout(pie_group)

        # 3 pie chart plots
        pie1 = self.create_plot("Energy Breakdown", "Pie_energy.svg", show_initial_image=True)
        pie2 = self.create_plot("Latency Breakdown", "Pie_latency.svg", show_initial_image=True)
        pie3 = self.create_plot("Area Breakdown", "Pie_area.svg", show_initial_image=True)

        # Ensure proper scaling
        pie1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pie2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pie3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        pie_layout.addWidget(pie1)
        pie_layout.addWidget(pie2)
        pie_layout.addWidget(pie3)

        #Add to overall output layout
        self.output_layout.addWidget(pie_group)


    def reload_images(self, updated_nn=False, updated_bnn=False, updated_sampled_fc1=False):
        if not hasattr(self, "tabs") or not self.tabs:
            return
        
        #if not self.simulator_core.weights_folder and not self.simulator_core.bnn_weights_folder:
         #   print("No weights. Skipping reload.")
          #  return

        
        def replace_widget_image(parent_widget, image_filename, show_image):
            image_filename = os.path.join(out_dir, image_filename)
            if parent_widget is None:
                return
    
            image_widget = parent_widget.findChild(QWidget, "plot_image")
            if image_widget is None:
                return
            
            parent_layout = image_widget.parentWidget().layout()

            # Remove old widget
            parent_layout.removeWidget(image_widget)
            image_widget.deleteLater()

            # Decide new widget type
            if show_image and image_filename.endswith(".svg") and os.path.exists(image_filename):
                new_widget = QSvgWidget(image_filename)
                new_widget.setFixedSize(280, 180)
            else:
                new_widget = QLabel()  # no text
                new_widget.setFixedSize(280, 180)
                new_widget.setAlignment(Qt.AlignCenter)
                new_widget.setStyleSheet("""
                    border: 1px solid #DDD;
                    background-color: #FAFAFA;
                    border-radius: 8px;
                """)
            new_widget.setObjectName("plot_image")
            parent_layout.addWidget(new_widget)

        # --- Reload Tab2 NN Weights ---
        if hasattr(self, "fc1_image"):
            replace_widget_image(self.fc1_image, "fc1_weights_plot.svg", updated_nn)
            replace_widget_image(self.fc2_image, "fc2_weights_plot.svg", updated_nn)
            replace_widget_image(self.fc3_image, "fc3_weights_plot.svg", updated_nn)

        # --- Reload Tab3 BNN Weights ---
        if hasattr(self, "bnn_weight_fc1_plot"):
            replace_widget_image(self.bnn_weight_fc1_plot, "bnn_fc1_weight_plot.svg", updated_bnn)
            replace_widget_image(self.bnn_weight_fc2_plot, "bnn_fc2_weight_plot.svg", updated_bnn)
            replace_widget_image(self.bnn_weight_fc3_plot, "bnn_fc3_weight_plot.svg", updated_bnn)

        # --- Reload Tab3 BNN QQ Plots ---
        if hasattr(self, "bnn_weight_fc1_qq_plot"):
            replace_widget_image(self.bnn_weight_fc1_qq_plot, "bnn_fc1_qq_plot.svg", updated_bnn)
            replace_widget_image(self.bnn_weight_fc2_qq_plot, "bnn_fc2_qq_plot.svg", updated_bnn)
            replace_widget_image(self.bnn_weight_fc3_qq_plot, "bnn_fc3_qq_plot.svg", updated_bnn)

        # --- Reload Tab4 BNN μ / σ ---
        if hasattr(self, "bnn_fc1_miu_plot"): 
            replace_widget_image(self.bnn_fc1_miu_plot, "bnn_fc1_miu_dist.svg", updated_bnn)
            replace_widget_image(self.bnn_fc2_miu_plot, "bnn_fc2_miu_dist.svg", updated_bnn)
            replace_widget_image(self.bnn_fc3_miu_plot, "bnn_fc3_miu_dist.svg", updated_bnn)

            replace_widget_image(self.bnn_fc1_sigma_plot, "bnn_fc1_sigma_dist.svg", updated_bnn)
            replace_widget_image(self.bnn_fc2_sigma_plot, "bnn_fc2_sigma_dist.svg", updated_bnn)
            replace_widget_image(self.bnn_fc3_sigma_plot, "bnn_fc3_sigma_dist.svg", updated_bnn)

        # --- Reload Tab5 BNN Sampled FC1 ---
        if hasattr(self, "bnn_sampled_fc1_N1_plot"):
            replace_widget_image(self.bnn_sampled_fc1_N1_plot, "bnn_fc1_sampled_N1_distribution.svg", updated_sampled_fc1)
        if hasattr(self, "bnn_sampled_fc1_plot"):
            replace_widget_image(self.bnn_sampled_fc1_plot, "bnn_fc1_sampled_weight_plot.svg", updated_sampled_fc1)
        if hasattr(self, "bnn_sampled_fc1_qq_plot"):
            replace_widget_image(self.bnn_sampled_fc1_qq_plot, "bnn_fc1_sampled_weight_qq_plot.svg", updated_sampled_fc1)

    def show_full_image(self, filename):
        popup = QDialog()
        popup.setWindowTitle("Full Size Image")

        layout = QVBoxLayout()

        target_width = 800  # your fixed width for popup image

        if filename.endswith(".svg"):
            # Use QSvgWidget to show vector image
            svg_widget = QSvgWidget(filename)

            # Get original size
            svg_renderer = svg_widget.renderer()
            orig_width = svg_renderer.defaultSize().width()
            orig_height = svg_renderer.defaultSize().height()

            # Scale factor to make width = target_width
            scale_factor = target_width / orig_width
            target_height = int(orig_height * scale_factor)

            svg_widget.setFixedSize(target_width, target_height)

            layout.addWidget(svg_widget)
            popup.setLayout(layout)
            popup.resize(target_width + 40, target_height + 40)

        else:
            # fallback to normal image (png/jpg)
            pixmap = QPixmap(filename)

            scale_factor = target_width / pixmap.width()
            target_height = int(pixmap.height() * scale_factor)

            scaled_pixmap = pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = QLabel()
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)

            layout.addWidget(label)
            popup.setLayout(layout)

            popup.resize(target_width + 40, target_height + 40)  # add a bit margin

        popup.exec()


    def create_plot(self, title, filename, show_initial_image=False):
        layout = QVBoxLayout()
        filename = os.path.join(out_dir, filename)
        # Create a horizontal layout for the title + zoom button
        title_bar_layout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) 

        zoom_button = QPushButton()
        icon_dir = os.path.dirname(out_dir)
        zoom_in_dir = os.path.join(icon_dir, "icons", "zoom-in.svg")
        zoom_button.setIcon(QIcon(zoom_in_dir))
        zoom_button.setIconSize(QSize(16, 16))
        zoom_button.setFixedSize(24, 24)
        zoom_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0,0,0,0.05);
                border-radius: 12px;
            }
        """)
        zoom_button.clicked.connect(lambda checked, f=filename: self.show_full_image(f))

        # Add title and zoom button to title_bar_layout
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addWidget(zoom_button, alignment=Qt.AlignRight)

        # Now add the title bar to main layout
        layout.addLayout(title_bar_layout)

        if show_initial_image and os.path.exists(filename):
            # if file exists and we want to show it initially
            if filename.endswith(".svg"):
                image = QSvgWidget(filename)
                image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                # image.setMinimumSize(280, 180)
                image.setObjectName("plot_image")
            else:
                image = QLabel()
                image.setPixmap(QPixmap(filename).scaled(280, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image.setAlignment(Qt.AlignCenter)
                image.setStyleSheet("border: 1px solid #CCC; background-color: white; padding: 5px;")
        else:
            # Show placeholder
            image = QLabel()
            image.setFixedSize(280, 180)
            image.setAlignment(Qt.AlignCenter)
            image.setStyleSheet("""
                border: 1px solid #DDD;
                background-color: #FAFAFA;
                border-radius: 8px;
            """)

        image.setObjectName("plot_image")   # Important! keep this name so reload_images can find it

        # Add image
        layout.addWidget(image)

        # Wrap everything into a QWidget
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper
    
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
            pixmap = QPixmap(image_filename)
            scaled_pixmap = pixmap.scaled(
                #get the width and height of the architecture_image_label
                self.architecture_image_label.width(),  
                self.architecture_image_label.height(),
                Qt.KeepAspectRatio,     #keep the aspect ratio
                Qt.SmoothTransformation # to ensure smooth scaling
            )
            self.architecture_image_label.setPixmap(scaled_pixmap)
        else:
            self.architecture_image_label.clear()
            self.architecture_image_label.setText("No image found.")

# === Main entry point ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
