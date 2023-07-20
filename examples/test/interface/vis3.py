import random
import sys
import time
from PyQt5.QtWidgets import QApplication, QTabWidget, QMainWindow, QProgressBar, QVBoxLayout, QPushButton, QWidget, QFileDialog, QSplitter, QHBoxLayout, QTextEdit, QMessageBox, QLabel, QSpinBox
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontMetrics, QFont

import os
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisMPL
from geomdl import utilities
from geomdl import NURBS
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from supervised import supervised
from unsupervised import unsupervised

# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

class DemoApp(QMainWindow):
    def __init__(self, parent=None):
        super(DemoApp, self).__init__(parent)

        # Create the main window
        self.setWindowTitle('Surface Parameterization Demo')
        self.setGeometry(0, 0, 800, 600)
        self.ctrlpts = []
        # Set control point values
        self.num_control_points_u = 10
        self.num_control_points_v = 10
        self.vertex_positions = []
        # Set degree value
        self.degree = 3
        self.sampled_resolution = 10
        # TODO: set delta value
        self.delta = 0.025
        self.out_dim_u = 32
        self.out_dim_v = 32
        self.fig = plt.figure(figsize=(15, 4))
        self.tab_widget = QTabWidget()
        
        # Create the central widget
        supervised_central_widget = QWidget()
        geomdl_central_widget = QWidget()
        unsuperivsed_central_widget = QWidget()
        subd_central_widget = QWidget()

        # Create a horizontal layout to hold the input and output widgets
        input_output_layout = QHBoxLayout()
        input_layout, output_layout = QVBoxLayout(), QVBoxLayout()
        # Create the PyVistaQt interactor widget and add it to the left half of the layout
        self.unsupervised_input_plotter = QtInteractor()
        input_label = QLabel("Input")
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.unsupervised_input_plotter, stretch=10)
        
        # Create the output widget and add it to the right half of the layout
        self.unsupervised_output_widget = QtInteractor()
        output_label = QLabel("Output")
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.unsupervised_output_widget)
        self.unsupervised_output_widget.set_background('#EEEEEE')
        self.unsupervised_input_plotter.set_background('#EEEEEE')

        input_output_layout.addLayout(input_layout)
        input_output_layout.addLayout(output_layout)
        
        # Add spin boxes to set the number of control points and degree
        spin_box_layout = QVBoxLayout()

        sampled_resolution_spin_box_layout = QHBoxLayout()
        sampled_resolution_spin_box_label = QLabel("sample resolution: ")
        # sampled_resolution_spin_box_label.setContentsMargins(200, 0, 0, 0)
        sampled_resolution_spin_box_layout.addWidget(sampled_resolution_spin_box_label)
        self.sampled_resolution_spin_box = QSpinBox()
        self.sampled_resolution_spin_box.setMinimum(5)
        self.sampled_resolution_spin_box.setMaximum(128)
        self.sampled_resolution_spin_box.setMaximumSize(35, 20)
        self.sampled_resolution_spin_box.setMinimumSize(35, 20)
        self.sampled_resolution_spin_box.setValue(self.sampled_resolution)
        sampled_resolution_spin_box_layout.addWidget(self.sampled_resolution_spin_box)
        spin_box_layout.addLayout(sampled_resolution_spin_box_layout)

        degree_layout = QHBoxLayout()
        degree_label = QLabel("Degree: ")
        degree_layout.addWidget(degree_label)
        self.degree_spin_box_1 = QSpinBox()
        self.degree_spin_box_1.setMinimum(2)
        self.degree_spin_box_1.setMaximum(4)
        self.degree_spin_box_1.setValue(self.degree)
        self.degree_spin_box_1.setMaximumSize(35, 20)
        self.degree_spin_box_1.setMinimumSize(35, 20)
        degree_layout.addWidget(self.degree_spin_box_1)
        degree_layout.setContentsMargins(0, 30, 0, 15)
        spin_box_layout.addLayout(degree_layout)

        control_points_u_layout = QHBoxLayout()
        control_points_u_label = QLabel("Number of Control Points on u: ")
        # control_points_u_label.setContentsMargins(200, 0, 0, 0)
        control_points_u_layout.addWidget(control_points_u_label)
        self.control_points_u_spin_box_1 = QSpinBox()
        self.control_points_u_spin_box_1.setMinimum(4)
        self.control_points_u_spin_box_1.setMaximum(50)
        self.control_points_u_spin_box_1.setMaximumSize(35, 20)
        self.control_points_u_spin_box_1.setMinimumSize(35, 20)
        self.control_points_u_spin_box_1.setValue(self.num_control_points_u)
        control_points_u_layout.addWidget(self.control_points_u_spin_box_1)
        control_points_u_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(control_points_u_layout)

        control_points_v_layout = QHBoxLayout()
        control_points_v_label = QLabel("Number of Control Points on v: ")
        # control_points_v_label.setContentsMargins(200, 0, 0, 0)
        control_points_v_layout.addWidget(control_points_v_label)
        self.control_points_v_spin_box_1 = QSpinBox()
        self.control_points_v_spin_box_1.setMinimum(4)
        self.control_points_v_spin_box_1.setMaximum(50)
        self.control_points_v_spin_box_1.setMaximumSize(35, 20)
        self.control_points_v_spin_box_1.setMinimumSize(35, 20)
        self.control_points_v_spin_box_1.setValue(self.num_control_points_v)
        control_points_v_layout.addWidget(self.control_points_v_spin_box_1)
        control_points_v_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(control_points_v_layout)

        out_dim_u_layout = QHBoxLayout()
        out_dim_u_label = QLabel("Out Dim u: ")
        # out_dim_u_label.setContentsMargins(200, 0, 0, 0)
        out_dim_u_layout.addWidget(out_dim_u_label)
        self.out_dim_u_spin_box_1 = QSpinBox()
        self.out_dim_u_spin_box_1.setMinimum(32)
        self.out_dim_u_spin_box_1.setMaximum(128)
        self.out_dim_u_spin_box_1.setMaximumSize(35, 20)
        self.out_dim_u_spin_box_1.setMinimumSize(35, 20)
        self.out_dim_u_spin_box_1.setValue(self.out_dim_u)
        out_dim_u_layout.addWidget(self.out_dim_u_spin_box_1)
        out_dim_u_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(out_dim_u_layout)
        
        out_dim_v_layout = QHBoxLayout()
        out_dim_v_label = QLabel("Out Dim v: ")
        # out_dim_v_label.setContentsMargins(200, 0, 0, 0)
        out_dim_v_layout.addWidget(out_dim_v_label)
        self.out_dim_v_spin_box_1 = QSpinBox()
        self.out_dim_v_spin_box_1.setMinimum(8)
        self.out_dim_v_spin_box_1.setMaximum(128)
        self.out_dim_v_spin_box_1.setMaximumSize(35, 20)
        self.out_dim_v_spin_box_1.setMinimumSize(35, 20)
        self.out_dim_v_spin_box_1.setValue(self.out_dim_v)
        out_dim_v_layout.addWidget(self.out_dim_v_spin_box_1)
        out_dim_v_layout.setContentsMargins(0, 15, 0, 30)
        spin_box_layout.addLayout(out_dim_v_layout)
        
        # Connect the valueChanged signal to a slot function
        self.control_points_u_spin_box_1.valueChanged.connect(self.update_control_points_u)
        self.control_points_v_spin_box_1.valueChanged.connect(self.update_control_points_v)
        self.degree_spin_box_1.valueChanged.connect(self.update_degree)
        self.out_dim_u_spin_box_1.valueChanged.connect(self.update_out_dim_u)
        self.out_dim_v_spin_box_1.valueChanged.connect(self.update_out_dim_v)
        self.sampled_resolution_spin_box.valueChanged.connect(self.update_sampled_resolution)
        # Set spacing between the label and the spin box
        # spin_box_layout.setSpacing(10)
        
        # Create a vertical layout to hold the buttons widget
        buttons_layout = QVBoxLayout()

        # Create a widget to hold the buttons
        # buttons_widget = QWidget()
        # buttons_widget.setLayout(buttons_layout)

        # Create the "Load" button
        self.load_button = QPushButton('Load Input Point Cloud')
        self.load_button.setContentsMargins(0, 10, 0, 10)
        self.load_button.clicked.connect(self.load_point_cloud)
        buttons_layout.addWidget(self.load_button)

        
                
        # Create the "Run Algorithm" button
        self.run_button = QPushButton('Run Unsupervised Algorithm')
        self.run_button.setContentsMargins(0, 10, 0, 10)
        self.run_button.clicked.connect(self.run_algorithm_unsupervised)
        buttons_layout.addWidget(self.run_button)

        # Create the "Cancel" button
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setContentsMargins(0, 10, 0, 10)
        self.cancel_button.clicked.connect(self.remove_point_clouds)
        buttons_layout.addWidget(self.cancel_button)

        # Add the input/output and buttons layouts to the central widget
        main_layout1 = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.addLayout(spin_box_layout)
        left_panel.addLayout(buttons_layout)
        main_layout1.addLayout(left_panel)
        main_layout1.addLayout(input_output_layout)

        layout = QVBoxLayout()
        layout.addLayout(main_layout1)
  
        # layout.addWidget(buttons_widget)
        unsuperivsed_central_widget.setLayout(layout)
        
        # Create the output widget and add it to the right half of the layout
        # self.supervised_output_widget = QtInteractor()
        # output_label = QLabel("Output")
        # output_layout = QVBoxLayout()
        # output_layout.addWidget(output_label)
        # output_layout.addWidget(self.supervised_output_widget)
        # self.supervised_output_widget.set_background('#EEEEEE')
        # self.supervised_output_widget.set_background('#EEEEEE')
        self.supervised_output_widget = FigureCanvas(self.fig)
        ##################################################################################
        ##################################################################################
        spin_box_layout = QVBoxLayout()

        control_points_u_layout = QHBoxLayout()
        control_points_u_label = QLabel("Number of Control Points on u: ")
        # control_points_u_label.setContentsMargins(200, 0, 0, 0)
        control_points_u_layout.addWidget(control_points_u_label)
        self.control_points_u_spin_box_2 = QSpinBox()
        self.control_points_u_spin_box_2.setMinimum(4)
        self.control_points_u_spin_box_2.setMaximum(50)
        self.control_points_u_spin_box_2.setMaximumSize(35, 20)
        self.control_points_u_spin_box_2.setMinimumSize(35, 20)
        self.control_points_u_spin_box_2.setValue(self.num_control_points_u)
        control_points_u_layout.addWidget(self.control_points_u_spin_box_2)
        control_points_u_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(control_points_u_layout)

        control_points_v_layout = QHBoxLayout()
        control_points_v_label = QLabel("Number of Control Points on v: ")
        # control_points_v_label.setContentsMargins(200, 0, 0, 0)
        control_points_v_layout.addWidget(control_points_v_label)
        self.control_points_v_spin_box_2 = QSpinBox()
        self.control_points_v_spin_box_2.setMinimum(4)
        self.control_points_v_spin_box_2.setMaximum(50)
        self.control_points_v_spin_box_2.setMaximumSize(35, 20)
        self.control_points_v_spin_box_2.setMinimumSize(35, 20)
        self.control_points_v_spin_box_2.setValue(self.num_control_points_v)
        control_points_v_layout.addWidget(self.control_points_v_spin_box_2)
        control_points_v_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(control_points_v_layout)

        degree_layout = QHBoxLayout()
        degree_label = QLabel("Degree: ")
        # degree_label.setContentsMargins(200, 0, 0, 0)
        degree_layout.addWidget(degree_label)
        self.degree_spin_box_2 = QSpinBox()
        self.degree_spin_box_2.setMinimum(2)
        self.degree_spin_box_2.setMaximum(4)
        self.degree_spin_box_2.setMaximumSize(35, 20)
        self.degree_spin_box_2.setMinimumSize(35, 20)
        self.degree_spin_box_2.setValue(self.degree)
        degree_layout.addWidget(self.degree_spin_box_2)
        degree_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(degree_layout)
        
        out_dim_u_layout = QHBoxLayout()
        out_dim_u_label = QLabel("Out Dim u: ")
        # out_dim_u_label.setContentsMargins(200, 0, 0, 0)
        out_dim_u_layout.addWidget(out_dim_u_label)
        self.out_dim_u_spin_box_2 = QSpinBox()
        self.out_dim_u_spin_box_2.setMinimum(32)
        self.out_dim_u_spin_box_2.setMaximum(128)
        self.out_dim_u_spin_box_2.setMaximumSize(35, 20)
        self.out_dim_u_spin_box_2.setMinimumSize(35, 20)
        self.out_dim_u_spin_box_2.setValue(self.out_dim_u)
        out_dim_u_layout.addWidget(self.out_dim_u_spin_box_2)
        out_dim_u_layout.setContentsMargins(0, 15, 0, 15)
        spin_box_layout.addLayout(out_dim_u_layout)
        
        out_dim_v_layout = QHBoxLayout()
        out_dim_v_label = QLabel("Out Dim v: ")
        # out_dim_v_label.setContentsMargins(200, 0, 0, 0)
        out_dim_v_layout.addWidget(out_dim_v_label)
        self.out_dim_v_spin_box_2 = QSpinBox()
        self.out_dim_v_spin_box_2.setMinimum(32)
        self.out_dim_v_spin_box_2.setMaximum(128)
        self.out_dim_v_spin_box_2.setMaximumSize(35, 20)
        self.out_dim_v_spin_box_2.setMinimumSize(35, 20)
        self.out_dim_v_spin_box_2.setValue(self.out_dim_v)
        out_dim_v_layout.addWidget(self.out_dim_v_spin_box_2)
        out_dim_v_layout.setContentsMargins(0, 15, 0, 30)
        spin_box_layout.addLayout(out_dim_v_layout)
        
        # Add the buttons and label to a layout
        
        buttons_layout2 = QVBoxLayout()
        # Create a button to trigger the file dialog
        self.btn_load = QPushButton('Load File', self)
        self.btn_load.clicked.connect(self.load_control_points_file)

        # Create a label to display the filename
        self.filename_label_1 = QLabel('No file loaded', self)
        combined_layout = QHBoxLayout()
        combined_layout.addWidget(self.filename_label_1)
        combined_layout.addWidget(self.btn_load)
        # self.filename_label_1.setContentsMargins(0, 0, 0, 0)
        # Create the "Run Algorithm" button
        self.run_button = QPushButton('Run Supervised Algorithm')
        self.run_button.clicked.connect(self.run_algorithm_supervised)
 
        # Create a button to reset the label
        self.btn_reset = QPushButton('Reset', self)
        self.btn_reset.clicked.connect(self.remove_control_points_file)
        buttons_layout2.addLayout(combined_layout)
        buttons_layout2.addWidget(self.run_button)
        buttons_layout2.addWidget(self.btn_reset)
        
        parameter_layout = QVBoxLayout()
        parameter_layout.addLayout(spin_box_layout)
        parameter_layout.addLayout(buttons_layout2)
        # main_layout2.maximumSize((200, 30))
        # main_layout2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # main_layout2.addStretch(0)
        self.input_point_cloud_widget = QtInteractor()
        self.output_point_cloud_widget = QtInteractor()
        self.input_point_cloud_widget.set_background('#EEEEEE')
        self.output_point_cloud_widget.set_background('#EEEEEE')
        
        main_layout2 = QHBoxLayout()
        main_layout2.addLayout(parameter_layout)
        main_layout2.addWidget(self.input_point_cloud_widget)
        main_layout2.addWidget(self.output_point_cloud_widget)
        
        layout = QVBoxLayout()
        layout.addLayout(main_layout2)
        layout.addWidget(self.supervised_output_widget)

        supervised_central_widget.setLayout(layout)
        
        # Connect the valueChanged signal to a slot function
        self.control_points_u_spin_box_2.valueChanged.connect(self.update_control_points_u)
        self.control_points_v_spin_box_2.valueChanged.connect(self.update_control_points_v)
        self.degree_spin_box_2.valueChanged.connect(self.update_degree)
        self.out_dim_u_spin_box_2.valueChanged.connect(self.update_out_dim_u)
        self.out_dim_v_spin_box_2.valueChanged.connect(self.update_out_dim_v)
        
        ######################################################################################################
        ######################################################################################################
        # Create the output widget and add it to the right half of the layout
        self.geomdl_output_widget = QtInteractor()
        output_label = QLabel("Output")
        output_layout = QVBoxLayout()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.geomdl_output_widget)
        self.geomdl_output_widget.set_background('#EEEEEE')
        self.geomdl_output_widget.set_background('#EEEEEE')
 
        
        # Create a button to trigger the file dialog
        self.btn_load = QPushButton('Load File', self)
        self.btn_load.clicked.connect(self.load_control_points_file)

        # Create a label to display the filename
        self.filename_label_2 = QLabel('No file loaded', self)
        
        combined_layout = QHBoxLayout()
        combined_layout.addWidget(self.filename_label_2)
        combined_layout.addWidget(self.btn_load)

        
        # Create the "Run Algorithm" button
        self.run_button = QPushButton('Run NURBS Algorithm')
        self.run_button.clicked.connect(self.run_algorithm_nurbs)
 
        # Create a button to reset the label
        self.btn_reset = QPushButton('Reset', self)
        self.btn_reset.clicked.connect(self.remove_control_points_file)

        spin_box_layout = QVBoxLayout()

        control_points_u_layout = QHBoxLayout()
        control_points_u_label = QLabel("Number of Control Points on u: ")
        # control_points_u_label.setContentsMargins(200, 0, 0, 0)
        control_points_u_layout.addWidget(control_points_u_label)
        self.control_points_u_spin_box_3 = QSpinBox()
        self.control_points_u_spin_box_3.setMinimum(4)
        self.control_points_u_spin_box_3.setMaximum(50)
        self.control_points_u_spin_box_3.setMinimumSize(35, 20)
        self.control_points_u_spin_box_3.setMaximumSize(35, 20)
        self.control_points_u_spin_box_3.setValue(self.num_control_points_u)
        control_points_u_layout.addWidget(self.control_points_u_spin_box_3)
        spin_box_layout.addLayout(control_points_u_layout)

        control_points_v_layout = QHBoxLayout()
        control_points_v_label = QLabel("Number of Control Points on v: ")
        # control_points_v_label.setContentsMargins(200, 0, 0, 0)
        control_points_v_layout.addWidget(control_points_v_label)
        self.control_points_v_spin_box_3 = QSpinBox()
        self.control_points_v_spin_box_3.setMinimum(4)
        self.control_points_v_spin_box_3.setMaximum(50)
        self.control_points_v_spin_box_3.setMinimumSize(35, 20)
        self.control_points_v_spin_box_3.setMaximumSize(35, 20)
        self.control_points_v_spin_box_3.setValue(self.num_control_points_v)
        control_points_v_layout.addWidget(self.control_points_v_spin_box_3)
        spin_box_layout.addLayout(control_points_v_layout)

        degree_layout = QHBoxLayout()
        degree_label = QLabel("Degree: ")
        # degree_label.setMaximumSize(100, 20)
        # degree_label.setMinimumSize(100, 20)
        # degree_label.setContentsMargins(0, 0, 0, 0)
        # degree_label.setContentsMargins(200, 0, 0, 0)
        degree_layout.addWidget(degree_label)
        self.degree_spin_box_3 = QSpinBox()
        self.degree_spin_box_3.setMinimum(2)
        self.degree_spin_box_3.setMaximum(4)
        self.degree_spin_box_3.setMinimumSize(35, 20)
        self.degree_spin_box_3.setMaximumSize(35, 20)
        self.degree_spin_box_3.setValue(self.degree)
        degree_layout.addWidget(self.degree_spin_box_3)
        spin_box_layout.addLayout(degree_layout)
        
        buttons_layout3 = QVBoxLayout()
        buttons_layout3.addLayout(combined_layout)
        buttons_layout3.addWidget(self.run_button)
        buttons_layout3.addWidget(self.btn_reset)
        
        left_panel = QVBoxLayout()
        left_panel.addLayout(spin_box_layout)
        left_panel.addLayout(buttons_layout3)
        
        # Add the buttons and label to a layout
        layout = QHBoxLayout()
        layout.addLayout(left_panel)
        layout.addWidget(self.geomdl_output_widget)

        geomdl_central_widget.setLayout(layout)
        
        # Connect the valueChanged signal to a slot function
        self.control_points_u_spin_box_3.valueChanged.connect(self.update_control_points_u)
        self.control_points_v_spin_box_3.valueChanged.connect(self.update_control_points_v)
        self.degree_spin_box_3.valueChanged.connect(self.update_degree)
        
        # Create a button to trigger the file dialog
        self.btn_load = QPushButton('Load Files', self)
        self.btn_load.clicked.connect(self.load_control_points_files)
        layout = QVBoxLayout()
        layout.addWidget(self.btn_load)
        subd_central_widget.setLayout(layout)
        
        self.tab_widget.addTab(unsuperivsed_central_widget, "Unsupervised")
        self.tab_widget.addTab(supervised_central_widget, "Supervised")
        self.tab_widget.addTab(geomdl_central_widget, "NURBS Generator")
        # self.tab_widget.addTab(subd_central_widget, "Subdivision Reconstruction")
        
        self.setCentralWidget(self.tab_widget)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        
    def update_sampled_resolution(self, value):
        self.sampled_resolution = value
        self.sampled_resolution_spin_box.setValue(self.sampled_resolution)
    
    def update_degree(self, value):
        # Update the degree of the NURBS curve
        self.degree = value
        # Update the value of the degree spin box
        self.degree_spin_box_1.setValue(self.degree)
        self.degree_spin_box_2.setValue(self.degree)
        self.degree_spin_box_3.setValue(self.degree)
    
    def update_control_points_u(self, value):
        # Update the number of control points
        self.num_control_points_u = value
        self.control_points_u_spin_box_1.setValue(self.num_control_points_u)
        self.control_points_u_spin_box_2.setValue(self.num_control_points_u)
        self.control_points_u_spin_box_3.setValue(self.num_control_points_u)
        
    def update_control_points_v(self, value):
        # Update the number of control points
        self.num_control_points_v = value
        self.control_points_v_spin_box_1.setValue(self.num_control_points_v)
        self.control_points_v_spin_box_2.setValue(self.num_control_points_v)
        self.control_points_v_spin_box_3.setValue(self.num_control_points_v)
    
    def update_out_dim_u(self, value):
        self.out_dim_u = value
        self.out_dim_u_spin_box_1.setValue(self.out_dim_u)
        self.out_dim_u_spin_box_2.setValue(self.out_dim_u)
        
    
    def update_out_dim_v(self, value):
        self.out_dim_v = value
        self.out_dim_v_spin_box_1.setValue(self.out_dim_v)
        self.out_dim_v_spin_box_2.setValue(self.out_dim_v)
    
    def load_file(self):
        message_box = QMessageBox(QMessageBox.Information, "Success", "The algorithm completed successfully.", QMessageBox.NoButton, self, Qt.WindowFlags(Qt.NonModal))
        message_box.show()
        QTimer.singleShot(3000, message_box.hide)
    
    def load_point_cloud(self):

        # Open a file dialog for loading the point cloud
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Point Cloud', '', 'Point Cloud Files (*.pcd);;NOFF Files (*.noff);;OFF Files (*.off);;All Files (*)', options=options)

        if file_name:
            self.remove_point_clouds()
            # Load the point cloud from file 
            with open(file_name, 'r') as f:
                lines = f.readlines()

                # skip the first line
                lines = lines[2:]
                lines = random.sample(lines, k=self.sampled_resolution * self.sampled_resolution)
                # extract vertex positions
                vertex_positions = []
                min_coord = max_coord = 0
                for line in lines:
                    x, y, z = map(float, line.split()[:3])
                    min_coord = min(min_coord, x, y, z)
                    max_coord = max(max_coord, x, y, z)
                    vertex_positions.append((x, y, z))
                range_coord = max(abs(min_coord), abs(max_coord)) / 1
                # range_coord = 1
                self.vertex_positions = [(x/range_coord, y/range_coord, z/range_coord) for x, y, z in vertex_positions]
            target = np.array(self.vertex_positions).reshape(-1, 3)
            input_point_cloud = pv.PolyData(target)
            # Add the point cloud to the plotter
            self.unsupervised_input_plotter.add_mesh(input_point_cloud, render_points_as_spheres=True, point_size=5, color='cyan', name='input_point_cloud')
            
    def load_control_points_file(self):

        # Show a file dialog to allow the user to select a file
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            # Get the selected file path
            selected_file = file_dialog.selectedFiles()[0]
            # Open the file and read its contents into a string
            if selected_file is not None:
                self.ctrlpts = exchange.import_txt(selected_file, separator=" ")
            else:
                print("No file selected")
            # with open(selected_file, 'r') as f:
            #     contents = f.read()
            # Display the contents of the file
            # Set the font for the label
            label_font = QFont()
            label_font.setPointSize(8)
            self.filename_label_1.setFont(label_font)
            self.filename_label_2.setFont(label_font)
            # Truncate the label text to a fixed length of 20 characters
            font_metrics = QFontMetrics(label_font)
            truncated_text = font_metrics.elidedText(selected_file, Qt.ElideRight, 100)
            self.filename_label_1.setText(truncated_text)
            # self.filename_label_1.setText(selected_file)
            self.filename_label_2.setText(truncated_text)
        else:
            # The user cancelled the file dialog
            self.filename_label_1.setText('No file loaded')
            self.filename_label_2.setText('No file loaded')
            print('File load cancelled')
    
    def load_control_points_files(self):
        # Ask the user to select one or more files to load
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Open Files',)

        # Create a label for each loaded file
        for filename in filenames:
            print(exchange.import_txt(filename, separator=" "))
           
    def run_algorithm_unsupervised(self):
        
        # TODO: Call your algorithm here
        # Replace the following line with the actual call to your algorithm
        # try:
            # # show the progress bar
            # self.progress_bar.setVisible(True)

            # # set the minimum and maximum values for the progress bar
            # self.progress_bar.setMinimum(0)
            # self.progress_bar.setMaximum(100)
            # self.progress_bar.setFormat("Running algorithm... %p%")
            # self.progress_bar.setWindowTitle("Running algorithm")
            # self.progress_bar.setAlignment(Qt.AlignCenter)        
            # self.progress_bar.setStyleSheet("""
            #     QProgressBar {
            #         border: 2px solid grey;
            #         border-radius: 5px;
            #         background-color: #FFFFFF;
            #         max-height: 30px;
            #         min-height: 30px;
            #         min-width: 400px;
            #         max-width: 400px;
            #         font-size: 24px;
            #         color: #00FF00;
            #     }

            #     QProgressBar::chunk {
            #         background-color: #2196F3;
            #         width: 30px;
                    
            #     }
            # """)
            
            # # run the algorithm
            # for i in range(0, 101):
            #     # perform the algorithm's task
            #     time.sleep(0.1)

            #     # update the progress bar
            #     self.progress_bar.setValue(i)
            #     QApplication.processEvents()

            # # hide the progress bar when the algorithm is finished
            # self.progress_bar.setVisible(False)
            # output_point_cloud = pv.Sphere()  # Dummy output

        try:
            unsupervised(self.progress_bar, 
                        vertex_positions=self.vertex_positions, resolution=self.sampled_resolution,
                        num_epochs=1000,
                        degree=self.degree, ctrl_pts_1=self.num_control_points_u, ctrl_pts_2=self.num_control_points_v, out_dim_u=self.out_dim_u, out_dim_v=self.out_dim_v,)
            self.progress_bar.setVisible(False)
            filename = f'generated/unsupervised_ctrpts_{self.num_control_points_u}x{self.num_control_points_v}_eval_{self.sampled_resolution}_reconstruct_{self.out_dim_u}x{self.out_dim_v}.OFF'
            with open(filename, 'r') as f:
                lines = f.readlines()

                # skip the first line
                lines = lines[2:]
                
                vertex_positions = []
                for line in lines:
                    x, y, z = map(float, line.split()[:3])
                    vertex_positions.append((x, y, z))

            target = np.array(vertex_positions).reshape(-1, 3)
            output_point_cloud = pv.PolyData(target)
            # Add the output point cloud to the plotter
            filename = f'generated/unsupervised_predicted_ctrpts_ctrpts_{self.num_control_points_u}x{self.num_control_points_v}_eval_{self.sampled_resolution}_reconstruct_{self.out_dim_u}x{self.out_dim_v}.OFF'
            with open(filename, 'r') as f:
                lines = f.readlines()

                # skip the first line
                lines = lines[2:]
                
                vertex_positions = []
                for line in lines:
                    x, y, z = map(float, line.split()[:3])
                    vertex_positions.append((x, y, z))

            target = np.array(vertex_positions).reshape(-1, 3)
            output_point_cloud_ctrlpts = pv.PolyData(target)
            self.unsupervised_output_widget.add_mesh(output_point_cloud, render_points_as_spheres=True, point_size=5, color='lightgreen', name='output_point_cloud')
            self.unsupervised_output_widget.add_mesh(output_point_cloud_ctrlpts, render_points_as_spheres=True, point_size=7, color='red', name='output_point_cloud_ctrlpts')
            # Display a success message box
            # message_box = QMessageBox(QMessageBox.Information, "Success", "The algorithm completed successfully.", QMessageBox.NoButton, self, Qt.WindowFlags(Qt.NonModal))
            # message_box.show()
            # QTimer.singleShot(3000, message_box.hide)
            
        except Exception as e:
            # Display an error message box
            self.progress_bar.setVisible(False)
            message_box = QMessageBox(QMessageBox.Critical, "Error", str(e), QMessageBox.NoButton, self, Qt.WindowFlags(Qt.NonModal))
            message_box.show()
        #     # QTimer.singleShot(3000, message_box.hide)
            
    def run_algorithm_nurbs(self):
        
        try:
            # Create a BSpline surface instance
            surf = NURBS.Surface()

            # Set degrees
            # Defined as order = degree + 1
            surf.order_u = self.degree + 1
            surf.order_v = self.degree + 1

            # Set number of control points
            surf.ctrlpts_size_u = self.num_control_points_u
            surf.ctrlpts_size_v = self.num_control_points_v

            # Set control points
            surf.ctrlpts = self.ctrlpts

            # Set knot vectors to be uniform
            # knot_u = np.array([-1.5708, -1.5708, -1.5708, -1.5708, -1.0472, -0.523599, 0, 0.523599, 0.808217,
            #             1.04015, 1.0472, 1.24824, 1.29714, 1.46148, 1.5708, 1.5708, 1.5708, 1.5708])
            # surf.knotvector_u = (knot_u - knot_u.min())/(knot_u.max()-knot_u.min())
            # knot_v = np.array([-3.14159, -3.14159, -3.14159, -3.14159, -2.61799, -2.0944, -1.0472, -0.523599,
            #             6.66134e-016, 0.523599, 1.0472, 2.0944, 2.61799, 3.14159, 3.14159, 3.14159, 3.14159])
            # surf.knotvector_v = (knot_v - knot_v.min())/(knot_v.max()-knot_v.min())
            # surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
            # surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
            surf.knotvector_u = [0, 0, 0, 0] + [0.99 for i in range(surf.ctrlpts_size_u - surf.degree_u - 1)] + [1, 1, 1, 1]
            surf.knotvector_v = [0, 0, 0, 0] + [0.99 for i in range(surf.ctrlpts_size_v - surf.degree_v - 1)] + [1, 1, 1, 1]
            # Set evaluation delta
            surf.delta = 0.025

            surf.evaluate()
            
            # Import and use Matplotlib's colormaps
            from matplotlib import cm

            # Plot the control point grid and the evaluated surface
            vis_comp = VisMPL.VisSurface(ctrlpts=True, legend=False, 
                                        axes=True, evalpts=True,
                                        bbox=False, trims=False,
                                        figure_size=[10, 10], 
                                        figure_dpi=100)
            surf.vis = vis_comp
            # Render the surface with selected colormap
            surf.render(colormap=cm.cool, plot=False)
            
            exchange.export_obj(surf, "nurbs.obj")
            mesh = pv.read('nurbs.obj')
            self.geomdl_output_widget.add_mesh(mesh, color='lightgreen', name='geomdl_mesh')
        except Exception as e:
            message_box = QMessageBox(QMessageBox.Critical, "Error", str(e), QMessageBox.NoButton, self, Qt.WindowFlags(Qt.NonModal))
            message_box.show()
            # QTimer.singleShot(3000, message_box.hide)
    def run_algorithm_supervised(self):
        try:
            fig = supervised(self.progress_bar ,self.fig, epoch=1000,
                   degree=self.degree, num_ctrl_pts1=self.num_control_points_u, num_ctrl_pts2=self.num_control_points_v, num_eval_pts_u=self.out_dim_u, num_eval_pts_v=self.out_dim_v,
                   ctrlpts=self.ctrlpts)
            self.progress_bar.setVisible(False)
            with open('supervised.off', 'r') as f:
                lines = f.readlines()

                # skip the first line
                lines = lines[2:]
                
                vertex_positions = []
                for line in lines:
                    x, y, z = map(float, line.split()[:3])
                    vertex_positions.append((x, y, z))

            target = np.array(vertex_positions).reshape(-1, 3)
            output_point_cloud = pv.PolyData(target)
            self.input_point_cloud_widget.add_mesh(output_point_cloud, render_points_as_spheres=True, point_size=5, color='cyan', name='input_point_cloud_supervised')
            
            with open('supervised_trained.off', 'r') as f:
                lines = f.readlines()

                # skip the first line
                lines = lines[2:]
                
                vertex_positions = []
                for line in lines:
                    x, y, z = map(float, line.split()[:3])
                    vertex_positions.append((x, y, z))

            target = np.array(vertex_positions).reshape(-1, 3)
            output_point_cloud = pv.PolyData(target)
            
            self.output_point_cloud_widget.add_mesh(output_point_cloud, render_points_as_spheres=True, point_size=5, color='lightgreen', name='output_point_cloud_supervised')
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            message_box = QMessageBox(QMessageBox.Critical, "Error", str(e), QMessageBox.NoButton, self, Qt.WindowFlags(Qt.NonModal))
            message_box.show()
            # QTimer.singleShot(3000, message_box.hide) 
    def remove_control_points_file(self):
        self.filename_label_1.setText('No file loaded')
        self.filename_label_2.setText('No file loaded')
        self.loaded_file = None
        self.remove_nurbs_mesh()
        self.remove_fig()
    def remove_knot_vector_file(self):
        # optional
        pass
    def remove_nurbs_mesh(self):
        # Remove the NURBS mesh from the plotter
        self.geomdl_output_widget.remove_actor('geomdl_mesh')   
    def remove_point_clouds(self):
        # Remove input and output point clouds from the plotter
        self.unsupervised_input_plotter.remove_actor('input_point_cloud')
        self.unsupervised_output_widget.remove_actor('output_point_cloud')
    # def remove_fig(self):
    #     self.fig.clear()
    #     self.fig.
    #     self.supervised_output_widget.destroy()
    def remove_fig(self):
        self.input_point_cloud_widget.remove_actor('input_point_cloud_supervised')
        self.output_point_cloud_widget.remove_actor('output_point_cloud_supervised')
        self.fig = plt.figure(figsize=(15, 4))
        self.supervised_output_widget = FigureCanvas(self.fig)
    # def remove_fig(self):
    #     self.fig.clear()
    #     self.supervised_output_widget.destroy()
    #     self.fig = plt.figure(figsize=(15, 4))
    #     self.supervised_output_widget = FigureCanvas(self.fig)
    #     self.supervised_output_widget.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo_app = DemoApp()
    demo_app.show()
    sys.exit(app.exec_())