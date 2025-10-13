import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology



class Bus:
    def __init__(self, mode, num_row, num_col, delay_tolerance, bus_width,unit_height, unit_width, clk_freq, tech,config, param):
        self.mode = mode
        self.num_row = num_row
        self.num_col = num_col
        self.delay_tolerance = delay_tolerance
        self.bus_width = bus_width
        self.unit_height = unit_height
        self.unit_width = unit_width
        self.clk_freq = clk_freq

        self.tech = tech
        self.param = param
        self.config = config
        self.initialized = False

        self.feature_size = self.tech.get_param('featureSize')
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.temp = self.config['temperature']
        self.vdd = self.tech.get_param('vdd')

        self.unitWireRes = self.param['unitLengthWireResistance']
        self.wireWidth = self.param['wireWidth']
        self.unitWireCap = 0.2e-15 / 1e-6  # 0.2 fF/um = 0.2e-15 F/micron

        # // define min INV resistance and capacitance to calculate repeater size
        self.width_min_inv_n = constant.MIN_NMOS_SIZE * self.feature_size
        self.width_min_inv_p = self.pnSizeRatio * self.width_min_inv_n

        res_on_rep = (logicGate.calculate_on_resistance(self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.width_min_inv_p, constant.PMOS, self.temp, self.tech))/2

        cap_min_inv_input,cap_min_inv_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_min_inv_n, self.width_min_inv_p, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)

        # // optimal repeater design to achieve highest speed

        # Repeater sizing
        self.repeater_size = math.floor(math.sqrt(
            res_on_rep * self.unitWireCap /
            (cap_min_inv_input * self.unitWireRes)
        ))

        self.min_dist = math.sqrt(
            2 * res_on_rep * (cap_min_inv_input + cap_min_inv_output) /
            (self.unitWireRes * self.unitWireCap)
        )

        w_rep, h_rep, rep_area = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_min_inv_n * self.repeater_size, self.width_min_inv_p * self.repeater_size, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
        cap_rep_input, cap_rep_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.repeater_size * self.width_min_inv_n, self.repeater_size * self.width_min_inv_p, h_rep, self.tech)
        res_on_rep = logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_p, constant.PMOS, self.temp, self.tech)
        min_unit_length_delay = 0.7 * (res_on_rep * (cap_rep_input + cap_rep_output + self.unitWireCap * self.min_dist)+ 
                                       0.54 * self.unitWireRes * self.min_dist * self.unitWireCap* self.min_dist + 
                                       self.unitWireRes * self.min_dist * cap_rep_input)/ self.min_dist
        max_unit_length_Energy = (cap_rep_input + cap_rep_output + self.unitWireCap * self.min_dist) * self.vdd * self.vdd / self.min_dist

        # // tradeoff: increase delay to decrease energy
        if self.delay_tolerance:
            delay = 0
            energy = 100
            while delay < min_unit_length_delay*(1+self.delay_tolerance) and self.repeater_size >= 1:  # Simplified condition
                self.repeater_size -= 1
                self.min_dist *= 0.9
                w_rep, h_rep, rep_area = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_min_inv_n * self.repeater_size, self.width_min_inv_p * self.repeater_size, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
                cap_rep_input, cap_rep_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.repeater_size * self.width_min_inv_n, self.repeater_size * self.width_min_inv_p, h_rep, self.tech)
                res_on_rep = (logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_p, constant.PMOS, self.temp, self.tech))/2
                delay = 0.7 * (res_on_rep * (cap_rep_input + cap_rep_output + self.unitWireCap * self.min_dist)+ 
                                       0.5 * self.unitWireRes * self.min_dist * self.unitWireCap* self.min_dist + 
                                       self.unitWireRes * self.min_dist * cap_rep_input)/ self.min_dist
                energy = (cap_rep_input + cap_rep_output + self.unitWireCap * self.min_dist) * self.vdd * self.vdd / self.min_dist


        self.width_inv_n = max(1, self.repeater_size) * self.feature_size * constant.MIN_NMOS_SIZE
        self.width_inv_p = self.width_inv_n * self.pnSizeRatio

        # Calculate total bus wire length
        if self.mode == 'HORIZONTAL':
            self.wire_length = self.unit_width * (self.num_col - 1)
        else:
            self.wire_length = self.unit_height * (self.num_row - 1)

        self.initialized = True

    def calculate_area(self, folded_ratio=1.0, overlap=False):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating area.")

        w_inv,h_inv,area_inv = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_inv_n, self.width_inv_p, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
        self.num_repeater = math.ceil(self.wire_length / self.min_dist)

        if self.num_repeater > 0:
            self.wire_width = self.bus_width * w_inv / folded_ratio
        else:
            self.wire_width = self.bus_width * self.wireWidth / folded_ratio

        if not overlap:
            self.area = self.num_row * self.wire_length * self.wire_width
        else:
            self.area = 0
        self.cap_inv_input, self.cap_inv_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_inv_n, self.width_inv_p, h_inv, self.tech)

        return self.area

    def calculate_latency(self, num_read):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating latency.")
        
        res_on_rep = logicGate.calculate_on_resistance(self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.width_min_inv_p, constant.PMOS, self.temp, self.tech)

        unit_latency_rep = 0.7 * (res_on_rep * (self.cap_inv_input + self.cap_inv_output + self.unitWireCap * self.min_dist) + 0.5 * self.unitWireRes * self.min_dist * self.unitWireCap * self.min_dist + self.unitWireRes * self.min_dist * self.cap_inv_input) / self.min_dist
        unit_latency_wire = 0.7 * self.unitWireRes * self.min_dist * self.unitWireCap * self.min_dist / self.min_dist

        if self.num_repeater > 0:
            read_latency = self.wire_length * unit_latency_rep
        else:
            read_latency = self.wire_length * unit_latency_wire

        # if self.param.get("synchronous"):
        #     read_latency = math.ceil(read_latency * self.clk_freq)

        return read_latency * num_read

    def calculate_power(self, num_bit_access, num_read):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating power.")

        unit_length_leakage = logicGate.calculate_logicgate_leakage(constant.INV, 1, self.width_inv_n, self.width_inv_p, self.temp, self.tech) * self.vdd / self.min_dist
        leakage = self.wire_length * unit_length_leakage * (self.num_row + self.num_col)

        unit_length_energy_rep = (self.cap_inv_input + self.cap_inv_output + self.unitWireCap * self.min_dist) * self.vdd**2 / self.min_dist * 0.25
        unit_length_energy_wire = self.unitWireCap * self.min_dist * self.vdd**2 / self.min_dist * 0.25

        if self.num_repeater > 0:
            read_dynamic_energy = self.wire_length * unit_length_energy_rep
        else:
            read_dynamic_energy = self.wire_length * unit_length_energy_wire

        return read_dynamic_energy * num_bit_access * num_read, leakage
