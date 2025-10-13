import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class DFF:
    def __init__(self, tech, config, param, clk_freq, num_dff):
        self.tech = tech
        self.config = config
        self.param = param

        self.featureSize = tech.get_param('featureSize')
        self.vdd = tech.get_param('vdd')
        self.temp = config['temperature']
        self.latency_mode = config['latency_mode']
        self.gamma = self.param['gamma']


        self.clk_freq = clk_freq
        self.num_dff = num_dff

        self.width_tg_n = constant.MIN_NMOS_SIZE * self.featureSize
        self.width_tg_p = tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.featureSize
        self.width_inv_n = constant.MIN_NMOS_SIZE * self.featureSize
        self.width_inv_p = tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.featureSize

        self.cap_inv_input = 0
        self.cap_inv_output = 0
        self.cap_tg_gate_n = 0
        self.cap_tg_gate_p = 0
        self.cap_tg_drain = 0

    def calculate_area(self, new_height=None, new_width=None, option='NONE'):
        w_inv, h_inv, _ = logicGate.calculate_logicgate_area(
            constant.INV, 1,
            constant.MIN_NMOS_SIZE * self.featureSize,
            constant.MIN_NMOS_SIZE * self.featureSize * self.tech.get_param('pnSizeRatio'),
            self.featureSize * constant.MAX_TRANSISTOR_HEIGHT*1.1,
            self.tech
        )
        h_dff = h_inv
        w_dff = w_inv * 13

        width = w_dff * self.num_dff
        height = h_dff

        if new_height and option == 'NONE':
            num_per_col = int(new_height // h_dff)
            num_per_col = min(num_per_col, self.num_dff)
            num_col = math.ceil(self.num_dff/num_per_col)
            height = new_height
            width = w_dff * num_col
        if new_width and option == 'NONE':
            num_per_row = int(new_width // w_dff)
            num_per_row = min(num_per_row, self.num_dff)
            num_col = math.ceil(self.num_dff/num_per_row)
            width = new_width
            height = h_dff * num_col
        
            

        area = width * height
        # to be done: GAA cap

        self.cap_inv_input, self.cap_inv_output = logicGate.calculate_logicgate_cap(
            constant.INV, 1,
            self.width_inv_n, self.width_inv_p,
            h_inv, self.tech
        )
        self.cap_tg_gate_n = logicGate.calculate_mos_gate_cap(self.width_tg_n, self.tech)
        self.cap_tg_gate_p = logicGate.calculate_mos_gate_cap(self.width_tg_p, self.tech)
        _, self.cap_tg_drain = logicGate.calculate_logicgate_cap(
            constant.INV, 1,
            self.width_tg_n, self.width_tg_p,
            h_inv, self.tech
        )

        self.area = area
        self.height = height
        self.width = width
        return area, height, width

    def calculate_latency(self, num_read):
        if self.latency_mode == 'synchronous':
            read_latency = num_read
        else:
            read_latency = 1 / self.clk_freq / 2 * num_read
        write_latency = read_latency
        return read_latency, write_latency

    def calculate_power(self, num_read, num_dff_per_op, validated):
        # Leakage
        leakage = logicGate.calculate_logicgate_leakage(
            constant.INV, 1,
            self.width_inv_n, self.width_inv_p,
            self.temp, self.tech
        ) * self.vdd * 8 * self.num_dff

        # CLK INV & TG
        # Assume input D=1 and the energy of CLK INV and CLK TG are for 1 clock cycles
		# CLK INV (all DFFs have energy consumption)
        read_energy = (self.cap_inv_input + self.cap_inv_output) * self.vdd**2 * 4 * self.num_dff
        # CLK TG (all DFFs have energy consumption)
        read_energy += self.cap_tg_gate_n * self.vdd**2 * 2 * self.num_dff
        read_energy += self.cap_tg_gate_p * self.vdd**2 * 2 * self.num_dff

        # D to Q path (selected)
        min_dff = min(num_dff_per_op, self.num_dff)
        read_energy += (self.cap_tg_drain * 3 + self.cap_inv_input) * self.vdd**2 * min_dff
        read_energy += (self.cap_tg_drain + self.cap_inv_output) * self.vdd**2 * min_dff
        read_energy += (self.cap_inv_input + self.cap_inv_output) * self.vdd**2 * min_dff

        read_energy *= num_read

        if validated:
            read_energy *= self.gamma

        write_energy = read_energy  # DFF write = read

        return read_energy, write_energy, leakage


