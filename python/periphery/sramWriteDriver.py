import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology

class SRAMWriteDriver:
    """
    SRAM Write Driver module

    Parameters:
    - num_col: number of columns
    - res_load: resistance load of the bitline
    - tech: Technology object containing parameters
    - config: dictionary containing temperature, etc.
    - mapping: dictionary with 'activity_col_write', 'num_write_cell_op'

    """
    def __init__(self, num_col, res_load, tech, config, mapping):
        self.tech = tech
        self.config = config
        self.mapping = mapping

        self.num_col = num_col
        self.res_load = res_load
        self.activity_col_write = float(mapping['activity_col_write'])
        self.num_write_cell_op = int(mapping['num_write_cell_op'])

        self.feature_size = tech.get_param('featureSize')
        self.vdd = tech.get_param('vdd')
        self.temp = config['temperature']
        self.pn_ratio = tech.get_param('pnSizeRatio')

        # Define transistor sizes
        self.width_inv_n = constant.MIN_NMOS_SIZE * self.feature_size * 28
        self.width_inv_p = self.pn_ratio * self.width_inv_n
        self.height_transistor_region = constant.MAX_TRANSISTOR_HEIGHT * self.feature_size*1.9

        # Initialize capacitance placeholders
        self.cap_inv_input = 0
        self.cap_inv_output = 0
        self.cap_nmos_drain = 0
        self.initialized = True

    def calculate_area(self, new_height=0, new_width=0, option="NONE"):
        w_inv, h_inv, _ = logicGate.calculate_logicgate_area(gateType = constant.INV, num_Input = 1, width_NMOS = self.width_inv_n, width_PMOS = self.width_inv_p, height_Transistor_Region = self.height_transistor_region, tech = self.tech)
        w_nmos, h_nmos, _ = logicGate.calculate_logicgate_area(gateType = constant.INV, num_Input = 1, width_NMOS = self.width_inv_n, width_PMOS = 0, height_Transistor_Region = self.height_transistor_region, tech = self.tech)
        # w_inv = w_inv * 3
        h_unit = h_inv + h_nmos
        w_unit = max(w_inv, w_nmos) * 2
        if new_width and option == 'NONE':
            num_unit_per_row = int(new_width // w_unit)
            num_unit_per_row = min(num_unit_per_row, self.num_col)
            num_row_unit = math.ceil(self.num_col / num_unit_per_row)
            width = new_width
            height = num_row_unit * h_unit
        else:
            width = self.num_col * w_unit
            height = h_unit

        area = width * height
        # Capacitance computation
        self.cap_inv_input, self.cap_inv_output = logicGate.calculate_logicgate_cap(
            constant.INV, 1, self.width_inv_n, self.width_inv_p,
            h_inv, self.tech)

        self.cap_nmos_drain = logicGate.calculate_drain_cap(
            'nmos', self.width_inv_n, self.height_transistor_region, self.tech)
        return area, height, width

    def calculate_latency(self, cap_load, num_write, ramp_input=1e20):
        # First inverter (pull-up)
        res_pull_up = logicGate.calculate_on_resistance(self.width_inv_p, constant.PMOS, self.temp, self.tech)
        tr1 = res_pull_up * (self.cap_inv_output + self.cap_inv_input + self.cap_nmos_drain)
        gm1 = logicGate.calculate_transconductance(self.width_inv_p, constant.PMOS, self.tech)
        beta1 = 1 / (res_pull_up * gm1)
        delay1, ramp1 = logicGate.horowitz(tr1, beta1, ramp_input)

        # Second inverter (pull-down)
        res_pull_down = logicGate.calculate_on_resistance(self.width_inv_n, constant.NMOS, self.temp, self.tech)
        tr2 = res_pull_down * (self.cap_nmos_drain + self.cap_inv_output)
        gm2 = logicGate.calculate_transconductance(self.width_inv_n, constant.NMOS, self.tech)
        beta2 = 1 / (res_pull_down * gm2)
        delay2, ramp2 = logicGate.horowitz(tr2, beta2, ramp1)

        # Pass transistor stage
        tr3 = res_pull_down * (cap_load + self.cap_nmos_drain) + self.res_load * cap_load / 2
        gm3 = logicGate.calculate_transconductance(self.width_inv_n, constant.NMOS, self.tech)
        beta3 = 1 / (res_pull_down * gm3)
        delay3, _ = logicGate.horowitz(tr3, beta3, ramp2)

        write_latency = (delay1 + delay2 + delay3) * num_write

        return write_latency

    def calculate_power(self, num_write):
        gate_leak = logicGate.calculate_logicgate_leakage(
            constant.INV, 1, self.width_inv_n, self.width_inv_p, self.temp, self.tech)

        leakage = gate_leak * self.vdd * 2 * self.num_col

        # Active number of columns
        num_active = min(int(self.num_write_cell_op), int(self.num_col * self.activity_col_write))
        cap_total = self.cap_inv_input + self.cap_inv_output + self.cap_nmos_drain

        write_energy = cap_total * self.vdd**2 * num_active * num_write
        return write_energy, leakage

