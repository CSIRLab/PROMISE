import math
import sys
import yaml
sys.path.append('../../python/')
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology
from periphery.adder import Adder
from periphery.HalfAdder import HalfAdder

class ArrayMultiplier:
    """
    """
    def __init__(self, num_bitA, num_bitB, num_mul, clk_freq, tech, config, mapping):
        self.tech = tech
        self.config = config
        self.mapping = mapping
        self.num_bitA = num_bitA
        self.num_bitB = num_bitB
        self.num_mul = num_mul
        self.clk_freq = clk_freq
        self.feature_size = self.tech.get_param('featureSize')
        self.vdd = self.tech.get_param('vdd')
        
        self.width_nand_n = 2 * constant.MIN_NMOS_SIZE * self.feature_size
        self.width_nand_p = self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.feature_size
        
        self.Adder = Adder(num_bit=1,num_adder=1,clk_freq = None,tech=self.tech,config=self.config,mapping=self.mapping)
        self.HalfAdder = HalfAdder(clk_freq = None,tech=self.tech,config=self.config, mapping=self.mapping)

    def calculate_area(self, new_height, new_width, option='NONE'):
        
        w_nand, h_nand, _ = logicGate.calculate_logicgate_area(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p,
            self.feature_size * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        Adder_area,Adder_height,Adder_width = self.Adder.calculate_area(new_height=None,new_width=None,option='NONE')
        HalfAdder_area,HalfAdder_height,HalfAdder_width = self.HalfAdder.calculate_area(new_height=None,new_width=None,option='NONE')
        w_adder = (self.num_bitA-2) * self.num_bitB * Adder_width 
        h_adder = HalfAdder_height + Adder_height + h_nand

        if new_height and option == 'NONE':
            if h_adder > new_height:
                raise ValueError("Adder height exceeds assigned height.")
            height = new_height
            width = w_adder * h_adder * self.num_mul / new_height
        elif new_width and option == 'NONE':
            if w_adder > new_width:
                raise ValueError("Adder width exceeds assigned width.")
            width = new_width
            height = w_adder * h_adder * self.num_mul / new_width
        else:
            width = w_adder
            height = h_adder * self.num_mul

        self.area = width * height
        self.height = height
        self.width = width

        # Capacitance
        self.cap_nand_input, self.cap_nand_output = logicGate.calculate_logicgate_cap(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p,
            self.feature_size * constant.MAX_TRANSISTOR_HEIGHT, self.tech)

        
        return self.area, self.height, self.width

    

    def calculate_latency(self, cap_load, num_read):
        ramp = [1e20] * 10  # initialize with very fast input ramp
        ramp_input = ramp[0]
        self.read_latency = 0
        cap_in = self.cap_nand_input
        cap_out = self.cap_nand_output
        Adder_read_latency = self.Adder.calculate_latency(cap_load=cap_load,num_read=1)
        HalfAdder_read_latency = self.HalfAdder.calculate_latency(cap_load=cap_load,num_read=1)


        self.read_latency += (Adder_read_latency + HalfAdder_read_latency) * self.num_bitB + Adder_read_latency

        self.read_latency *= num_read
        return self.read_latency

    def calculate_power(self, num_read, num_adder_per_op):
        vdd = self.vdd
        Adder_read_energy,Adder_leakage = self.Adder.calculate_power(num_read=1,num_adder_per_op=1)
        HalfAdder_read_energy,HalfAdder_leakage = self.HalfAdder.calculate_power(num_read=1,num_adder_per_op=1)
        num_adder = (self.num_bitA -2) * self.num_bitB
        num_half_adder = self.num_bitB
        total_nand = self.num_bitB * self.num_bitA


        # Leakage
        self.leakage = logicGate.calculate_logicgate_leakage(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p,
            self.config['temperature'], self.tech) * vdd * total_nand
        self.leakage += Adder_leakage * num_adder + HalfAdder_leakage * num_half_adder

        # Dynamic Energy 
        # Calibration data pattern of critical path is A=1111111..., B=1000000... and Cin=1
		# Only count 0 to 1 transition for energy
        

        self.read_energy = Adder_read_energy * num_adder + HalfAdder_read_energy * num_half_adder
        self.read_energy *= num_read
        self.read_energy *= self.num_mul

        return self.read_energy, self.leakage


