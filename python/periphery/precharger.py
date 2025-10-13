import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class Precharger:
    """
    num_col: number of columns in the array
    res_load: resistance load of the bitline
    activity_col_write: activity factor for write operation
    num_read_cell_op: number of working read cell per operation
    num_write_cell_op: number of working write cell per operation
    tech: Technology object containing technology parameters
    config: dictionary containing additional parameters for the precharger(from interface config.yaml)

    """
    def __init__(self, num_col, res_load, tech, config, mapping):
        self.tech = tech
        self.config = config
        self.mapping = mapping
        self.cap_output_BL = None
        self.num_col = num_col
        self.res_load = res_load
        self.activity_col_write = self.mapping['activity_col_write']
        self.num_read_cell_op =  self.mapping['num_read_cell_op']
        self.num_write_cell_op =  self.mapping['num_write_cell_op']
        self.featureSize = self.tech.get_param('featureSize')
        self.vdd = self.tech.get_param('vdd')
        self.temp = self.config['temperature']
        
        # usually the pmos is large to pull up the bitline quickly
        self.width_pmos_precharger = 24 * self.featureSize
        self.width_pmos_equalizer = 12 * self.featureSize

    def calculate_area(self, num_col, new_height, new_width, option):
    # Constants representing layout modification strategies
        MAGIC = 'MAGIC'
        OVERRIDE = 'OVERRIDE'
        NONE = 'NONE'

        # Step 1: Compute single gate areas
        w_pre, h_pre, _ = logicGate.calculate_logicgate_area(constant.INV,1, 0, self.width_pmos_precharger, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT*1.3, self.tech)
        w_eq, h_eq, _ = logicGate.calculate_logicgate_area(constant.INV,1, 0, self.width_pmos_equalizer, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT*1.3, self.tech)

        w_unit = 4*w_pre + w_eq *2
        h_unit = max(h_pre, h_eq)

        if new_width and option == 'NONE':
            if new_width < w_unit:
                raise ValueError("Precharger width is larger than assigned width.")

            num_unit_per_row = int(new_width // w_unit)
            num_unit_per_row = min(num_unit_per_row, num_col)
            num_row_unit = math.ceil(num_col / num_unit_per_row)

            width = new_width
            height = num_row_unit * h_unit
        else:
            width = num_col * w_unit
            height = h_unit

        area = height * width
        _,cap_output_BL_precharger = logicGate.calculate_logicgate_cap(constant.INV,1, 0 , self.width_pmos_precharger, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        _,cap_output_BL_equalizer = logicGate.calculate_logicgate_cap(constant.INV,1, 0 , self.width_pmos_equalizer, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        self.cap_output_BL = cap_output_BL_precharger + cap_output_BL_equalizer
        return area, height, width, self.cap_output_BL

    

    def calculate_latency(self, cap_load, num_read, num_write):
        # compute pull-up resistance
        res_pull_up = logicGate.calculate_on_resistance(self.width_pmos_precharger, constant.PMOS, self.temp, self.tech)

        tau = res_pull_up * (cap_load + self.cap_output_BL) + self.res_load * cap_load / 2

        # conductance of PMOS
        gm = logicGate.calculate_transconductance(self.width_pmos_precharger, constant.PMOS, self.tech)

        # Horowitz model parameters beta
        beta = 1 / (res_pull_up * gm)

        # Horowitz calculation for delay
        base_latency,_ = logicGate.horowitz(tau, beta, ramp_input = 1e20)

        read_latency = base_latency * num_read
        write_latency = base_latency * num_write

        return read_latency, write_latency


    def calculate_power(self, cap_load, num_read, num_write):
        self.leakage = logicGate.calculate_logicgate_leakage(
            constant.INV, 1, 0, self.width_pmos_precharger, self.temp, self.tech) * self.vdd * self.num_col
        # min_read = min(self.num_read_cell_op, self.num_col)
        # min_write = min(self.num_write_cell_op, self.num_col * self.activity_col_write)
        min_read = min(int(self.num_read_cell_op), int(self.num_col))
        min_write = min(int(self.num_write_cell_op), int(self.num_col) * float(self.activity_col_write))

        self.read_energy = cap_load * self.vdd**2 * min_read * 2 * num_read
        self.write_energy = cap_load * self.vdd**2 * min_write * num_write
        return self.read_energy, self.write_energy, self.leakage



