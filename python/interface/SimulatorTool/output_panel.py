# =============================================================================
# output_panel.py — Output panel for PROMISE Simulator
# Author:      Emilie Ye
# Date:        2025-06-27
#
# Description:
#   This file defines the OutputPanel, which displays simulation results
#   and console logs for PROMISE Simulator. It organizes result plots for
#   "Memory", "Compute in Memory", and "Demo" modes into tabs and manages
#   console output display.
#
# Copyright (c) 2025
# =============================================================================

import os
import sys
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plots_dir, exist_ok=True)
from PySide6.QtWidgets import QTextEdit, QWidget, QVBoxLayout, QLabel, QScrollArea,QTabWidget, QHBoxLayout, QGroupBox, QPushButton, QSizePolicy, QDialog
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QIcon, QPixmap,QPainter
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtSvg import QSvgRenderer
from memory_page import EmittingStream

class ScaledSvgWidget(QWidget):
    def __init__(self, svg_path, parent=None):
        super().__init__(parent)
        self.renderer = QSvgRenderer(svg_path)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect() 
        vb = self.renderer.viewBoxF()
        scale = min(rect.width()/vb.width(), rect.height()/vb.height())
        w = vb.width() * scale
        h = vb.height() * scale
        x = (rect.width() - w) / 2
        y = (rect.height() - h) / 2
        target = QRectF(x, y, w, h)
        self.renderer.render(painter, target)
        
class OutputPanel(QWidget):
    def __init__(self, core, parent=None):
        super().__init__(parent)
        self.core = core

        self.plots_dir = plots_dir

        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                color: #000000;
                font-family: Consolas;
                font-size: 9pt;
                border: 1px solid #888;
                border-radius: 5px;
            }
        """)
        sys.stdout = EmittingStream(self.console_text)
        sys.stderr = EmittingStream(self.console_text)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.container = QWidget()
        self.layout_inside = QVBoxLayout(self.container)
        self.scroll.setWidget(self.container)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.scroll, stretch=3)



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
        self.layout_inside.addWidget(placeholder_label)



    def run_simulation(self, updated_nn: bool = False, updated_bnn: bool = False, updated_sampled_fc1: bool = False):
        
        if self.core.mode in ["Memory", "Compute in Memory"]:
            if (not self.core.weights_folder
                and not self.core.bnn_weights_folder
                and not self.core.device_data_path
                and not getattr(self.core, 'pie_config_path', None)):
                self.clear()
                return
        
        self.clear(show_placeholder=False)

        if self.core.mode == "Memory":
            self._show_memory_outputs()
            self.reload_images(
                updated_nn=updated_nn,
                updated_bnn=updated_bnn,
                updated_sampled_fc1=updated_sampled_fc1
            )
        elif self.core.mode == "Compute in Memory":
            self._show_compute_outputs()
            
        elif self.core.mode == "Demo":
            self._show_demo_outputs()
        
        

    def clear(self, show_placeholder=True):
        old_parent = self.console_text.parentWidget()
        if old_parent is not None:
            old_parent.layout().removeWidget(self.console_text)
            self.console_text.setParent(None)
        
        while self.layout_inside.count():
            item = self.layout_inside.takeAt(0)
            w = item.widget()
            if w is None:
                continue
            w.setParent(None)
            w.deleteLater()

        self.demo_hist_plot = None
        self.demo_qq_plot = None

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

        self.bnn_sampled_fc1_N1_plot = None
        self.bnn_sampled_fc1_plot = None
        self.bnn_sampled_fc1_qq_plot = None


        if show_placeholder:
            placeholder = QLabel("Output will appear here after running simulation.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                border: 1px dashed #AAA;
                background-color: #FAFAFA;
                color: #888;
                font-size: 14pt;
                padding: 100px;
            """)
            self.layout_inside.addWidget(placeholder)

    def _show_demo_outputs(self):
        self.tabs = QTabWidget()

        title_label = QLabel("Demo Result")
        title_label.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 15px;")
        title_label.setAlignment(Qt.AlignLeft)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        content_layout.addWidget(title_label)

        count = len(getattr(self.core, "demo_mus", [1.0]))
        distribution_mode = getattr(self.core, "gaussian_mode", "Single Gaussian")


        # Histogram + QQ Plot
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(10)
        
        if distribution_mode != "Multiple GMM":
            # Histogram (top)
            hist_row = QHBoxLayout()
            hist_row.setContentsMargins(0, 0, 0, 0)
            hist_row.setSpacing(20)
            for i in range(count):
                fname = f"demo_histogram_{i+1}.svg"
                plot = self.create_plot(f"Histogram #{i+1}", fname, show_initial_image=True)
                plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                hist_row.addWidget(plot, 1)
            results_layout.addLayout(hist_row)

            # QQ Plot (bottom)
            qq_row = QHBoxLayout()
            qq_row.setContentsMargins(0, 0, 0, 0)
            qq_row.setSpacing(20)
            for i in range(count):
                fname = f"demo_qqplot_{i+1}.svg"
                plot = self.create_plot(f"QQ Plot #{i+1}", fname, show_initial_image=True)
                plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                qq_row.addWidget(plot, 1)
            results_layout.addLayout(qq_row)


        # Add multi dimensional distribution plot if available
        if count >= 2:
            summary_row = QHBoxLayout()
            summary_row.setContentsMargins(0, 0, 0, 0)
            summary_row.setSpacing(20)
            if distribution_mode == "Single Gaussian":
                # choose distribution plot based on count
                if count <= 3:
                    summary_fname = f"demo_nd_histogram.svg"  # for instance summary_view_1.svg / summary_view_2.svg / summary_view_3.svg
                else:
                    summary_fname = "demo_nd_histogram.svg"  # use default for more than 3 dimensions
        
                summary_plot = self.create_plot("Summary View", summary_fname, show_initial_image=True)
                summary_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                summary_row.addWidget(summary_plot, 1)
        
                results_layout.addLayout(summary_row, stretch=2)

            elif distribution_mode == "Gaussian Mixture Model":
                summary_plot = self.create_plot(
                    "Gaussian Mixture Model Histogram",
                    "demo_mixed_histogram.svg",
                    show_initial_image=True
                )
                summary_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

                summary_row.addWidget(summary_plot, 1)
                results_layout.addLayout(summary_row, stretch=2)

            elif distribution_mode == "Multiple GMM":
                summary_row = QHBoxLayout()
                summary_row.setContentsMargins(0, 0, 0, 0)
                summary_row.setSpacing(20)
                for i in range(count):
                    fname = f"demo_multiple_gmm_{i+1}_histogram.svg"
                    plot = self.create_plot(f"GMM #{i+1}", fname, show_initial_image=True)
                    plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                    summary_row.addWidget(plot, 1)
                results_layout.addLayout(summary_row)

                summary_fname = "demo_multiple_gmm_nd_histogram.svg"
                title = "2D Summary" if count == 2 else "3D Summary"
                summary_plot = self.create_plot(title, summary_fname, show_initial_image=True)
                summary_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                results_layout.addWidget(summary_plot, stretch=2)

        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(0)
        console_container.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 1px solid #DDD;
                border-radius: 8px;
            }
        """)
        console_layout.addWidget(self.console_text)

        #console_container.setLayout(console_layout)


        self.tabs.clear()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.addTab(results_widget, "Results")
        self.tabs.addTab(console_container, "Console")

        content_layout.addWidget(self.tabs, stretch=1)

        self.layout_inside.addWidget(content)
        self.reload_images()


    def _show_memory_outputs(self):
            # === Tabs ===
            self.tabs = QTabWidget()

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
            #self.output_layout.addWidget(pie_group)

            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)

            content_layout.addWidget(self.tabs, stretch=2)
            content_layout.addWidget(pie_group, stretch=1)

            console_tab = QWidget()
            console_layout = QVBoxLayout(console_tab)
            console_layout.setContentsMargins(0,0,0,0)
            console_layout.addWidget(self.console_text)
        
            self.tabs.addTab(console_tab, "Console")

            # Make it expand vertically
            content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Replace placeholder
            self.layout_inside.addWidget(content_widget)

    def _show_compute_outputs(self):
            # === Tabs ===
            self.tabs = QTabWidget()

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

            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)

            content_layout.addWidget(self.tabs, stretch=2)
            content_layout.addWidget(pie_group, stretch=1)

            console_tab = QWidget()
            console_layout = QVBoxLayout(console_tab)
            console_layout.setContentsMargins(0,0,0,0)
            console_layout.addWidget(self.console_text)
        
            self.tabs.addTab(console_tab, "Console")

            content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Replace placeholder
            self.layout_inside.addWidget(content_widget)

    def reload_images(self, updated_nn=False, updated_bnn=False, updated_sampled_fc1=False):
        if not hasattr(self, "tabs") or not self.tabs:
            return

        def replace_widget_image(parent_widget, image_filename, show_image):
            image_filename = os.path.join(self.plots_dir, image_filename)
            if parent_widget is None:
                return
    
            image_widget = parent_widget.findChild(QWidget, "plot_image")
            if image_widget:
                image_widget.hide()
                image_widget.setParent(None) 
                image_widget.deleteLater() 

            if show_image and image_filename.endswith(".svg") and os.path.exists(image_filename):
                new_widget = QSvgWidget(image_filename)
                new_widget.setFixedSize(280, 180)
            else:
                new_widget = QLabel()  # no text
                # Size
                new_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                new_widget.setFixedSize(280, 180)
                new_widget.setAlignment(Qt.AlignCenter)
                new_widget.setStyleSheet("""
                    border: 1px solid #DDD;
                    background-color: #FAFAFA;
                    border-radius: 8px;
                """)
            new_widget.setObjectName("plot_image")

            layout = parent_widget.layout()
            if layout:
                layout.addWidget(new_widget)

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

        # --- Reload Demo Plots ---
        if hasattr(self, "demo_hist_plot") and self.demo_hist_plot is not None:
            replace_widget_image(self.demo_hist_plot, "demo_histogram.svg", True)
        if hasattr(self, "demo_qq_plot") and self.demo_qq_plot is not None:
            replace_widget_image(self.demo_qq_plot, "demo_qqplot.svg", True)
            
    def show_full_image(self, filename):    # Show full size image in a popup dialog
        if os.path.basename(filename) == "demo_nd_histogram.svg":
            demo_data = getattr(self.core, "demo_data", None)
            if demo_data is not None and demo_data.shape[1] == 3:
                self.core.generate_nd_histogram(demo_data, 3, filename=None, interactive=True)
                return
        elif os.path.basename(filename) == "demo_multiple_gmm_nd_histogram.svg":
            demo_data = getattr(self.core, "demo_data_multiple_gmm", None)
            if demo_data is not None and demo_data.shape[1] in (2, 3):
                self.core.generate_nd_histogram(
                    demo_data,
                    demo_data.shape[1],
                    filename=None,
                    interactive=True
                )
                return
        popup = QDialog()
        popup.setWindowTitle("Full Size Image")

        layout = QVBoxLayout()

        target_width = 800 
        if filename.endswith(".svg"):
            svg_widget = QSvgWidget(filename)

            # Get original size
            svg_renderer = svg_widget.renderer()
            orig_width = svg_renderer.defaultSize().width()
            orig_height = svg_renderer.defaultSize().height()

            # Scale to target_width
            scale_factor = target_width / orig_width
            target_height = int(orig_height * scale_factor)

            svg_widget.setFixedSize(target_width, target_height)

            layout.addWidget(svg_widget)
            popup.setLayout(layout)
            popup.resize(target_width + 40, target_height + 40)

        else:
            pixmap = QPixmap(filename)

            scale_factor = target_width / pixmap.width()
            target_height = int(pixmap.height() * scale_factor)

            scaled_pixmap = pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = QLabel()
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)

            layout.addWidget(label)
            popup.setLayout(layout)

            popup.resize(target_width + 40, target_height + 40) 
        popup.exec()


    def create_plot(self, title, filename, show_initial_image=False):
        layout = QVBoxLayout()
        filename = os.path.join(self.plots_dir, filename)

        # Create a horizontal layout for the title + zoom button
        title_bar_layout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) 

        zoom_button = QPushButton()
        icon_dir = os.path.dirname(self.plots_dir)
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
        zoom_button.clicked.connect(lambda checked, f=filename: self.show_full_image(f))    # Show full size image in a popup dialog

        # Add title and zoom button to title_bar_layout
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addWidget(zoom_button, alignment=Qt.AlignRight)

        # Now add the title bar to main layout
        layout.addLayout(title_bar_layout)


        if show_initial_image and os.path.exists(filename):
            if filename.endswith(".svg"):
                image = ScaledSvgWidget(filename)
                image.setMinimumSize(280, 180)
                image.setObjectName("plot_image")
            else:
                image = QLabel()
                image.setPixmap(QPixmap(filename).scaled(280, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image.setAlignment(Qt.AlignCenter)
                image.setStyleSheet("border: 1px solid #CCC; background-color: white; padding: 5px;")
        else:
            # Show placeholder
            image = QLabel()
            image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            image.setMinimumSize(280, 180)
            image.setAlignment(Qt.AlignCenter)
            image.setStyleSheet("""
                border: 1px solid #DDD;
                background-color: #FAFAFA;
                border-radius: 8px;
            """)

        image.setObjectName("plot_image")  

        # Add image
        layout.addWidget(image)

        wrapper = QWidget()
        wrapper.setLayout(layout)

        # Size
        wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return wrapper
    
    