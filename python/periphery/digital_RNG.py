import math
import sys
import yaml
import numpy as np
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology
from periphery.sramWriteDriver import SRAMWriteDriver
from periphery.precharger import Precharger
from periphery.WLdecoder import RowDecoder
from periphery.SenseAmp import SenseAmp
from periphery.DFF import DFF
from periphery.MUX import Mux
from periphery.levelShifter import LevelShifter
from periphery.WLDecoderDriver import WLNewDecoderDriver
from periphery.adder import Adder
from periphery.ADC import SarADC


class digital_RNG:
    def __init__(self, num_block, clt_number,clk_freq, tech, param, mapping, config):
        self.tech = tech
        self.param = param
        self.clt_number = clt_number
        self.num_block = num_block
        self.mapping = mapping
        self.clk_freq = clk_freq
        self.config = config

        self.vdd = tech.get_param('vdd')
        self.feature_size = tech.get_param('featureSize')
        self.temp = config['temperature']
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.readVoltage = self.param['readVoltage']

        self.precision_sigma = self.config['precision_sigma']
        self.clk_freq = self.config['frequency']

        self.RNG_65nm_area = 0
        if self.clt_number == 4:
            self.RNG_65nm_area = 2090.879954e-12
        elif self.clt_number == 6:
            self.RNG_65nm_area = 2657.159938e-12
        elif self.clt_number == 8:
            self.RNG_65nm_area = 3213.359921e-12
        elif self.clt_number == 10:
            self.RNG_65nm_area = 3783.959905e-12
        elif self.clt_number == 12:
            self.RNG_65nm_area = 4342.679888e-12
        elif self.clt_number == 14:
            self.RNG_65nm_area = 4908.959871e-12
        elif self.clt_number == 16:
            self.RNG_65nm_area = 5461.919853e-12
        else:
            print("[RNG] Error: Unsupported CLT number!")
            return
        self.RNG_height = math.sqrt(self.RNG_65nm_area)
        self.RNG_width = self.RNG_height

        #RNG
        self.RNG_height = self.RNG_height/65e-9*self.feature_size
        self.RNG_width = self.RNG_width/65e-9*self.feature_size
        self.area = self.RNG_height * self.RNG_width

        self.initialized = True

    def calculate_area(self,new_height = None, new_width = None):
        # if not self.initialized:
        #     raise ValueError("MUX must be initialized before area calculation.")
        
        if new_height is not None:
            self.height = new_height
            self.width = self.area / self.height * self.num_block
        elif new_width is not None:
            self.width = new_width
            self.height = self.area / self.width * self.num_block
        else:
            self.height = self.RNG_height * self.num_block
            self.width = self.RNG_width

        self.area = self.width * self.height

        return self.area, self.height, self.width
    
    def calculate_latency(self,num_read):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
            return 0

        read_latency = 1 / self.clk_freq * num_read
        return read_latency
    
    def calculate_power(self, num_read):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
            return 0, 0

        if self.clt_number == 4:
            read_dynamic_energy = 140.9678 / self.clk_freq * 1e-6
            leakage = 1.1485e+03 * 1e-9
        elif self.clt_number == 6:
            read_dynamic_energy = 190.4046 / self.clk_freq * 1e-6
            leakage = 1.3616e+03 * 1e-9
        elif self.clt_number == 8:
            read_dynamic_energy = 179.6840 / self.clk_freq * 1e-6
            leakage = 1.5706e+03 * 1e-9
        elif self.clt_number == 10:
            read_dynamic_energy = 228.2045 / self.clk_freq * 1e-6
            leakage = 1.7883e+03 * 1e-9
        elif self.clt_number == 12:
            read_dynamic_energy = 248.7693 / self.clk_freq * 1e-6
            leakage = 1.9971e+03 * 1e-9
        elif self.clt_number == 14:
            read_dynamic_energy = 271.5370 / self.clk_freq * 1e-6
            leakage = 2.2119e+03 * 1e-9
        elif self.clt_number == 16:
            read_dynamic_energy = 258.4441 / self.clk_freq * 1e-6
            leakage = 2.4200e+03 * 1e-9
        else:
            print("[RNG] Error: Unsupported CLT number!")
        
        read_dynamic_energy = read_dynamic_energy * (self.feature_size/65e-9)**3
        read_dynamic_energy = read_dynamic_energy * num_read * self.num_block

        self.read_dynamic_energy = read_dynamic_energy
        self.leakage = leakage

        return read_dynamic_energy, leakage