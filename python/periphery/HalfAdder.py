import math
import sys
import yaml
sys.path.append('../../python/')
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology

class HalfAdder:
    """
    - num_col: number of columns
    - current_sense: True if current-mode sensing (not supported)
    - sense_voltage: voltage difference needed for sensing
    - pitch_sense_amp: layout pi
    """
    def __init__(self, clk_freq, tech, config, mapping):
        self.tech = tech
        self.config = config
        self.mapping = mapping
        self.clk_freq = clk_freq
        self.feature_size = self.tech.get_param('featureSize')
        self.vdd = self.tech.get_param('vdd')

        self.width_nand_n = 2 * constant.MIN_NMOS_SIZE * self.feature_size
        self.width_nand_p = self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.feature_size

        self.cap_nand_input = None
        self.cap_nand_output = None
        self.area = None
        self.height = None
        self.width = None
        self.read_latency = None
        self.read_energy = None
        self.leakage = None
        

    def calculate_area(self, new_height, new_width, option='NONE'):
        w_nand, h_nand, _ = logicGate.calculate_logicgate_area(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p,
            self.feature_size * constant.MAX_TRANSISTOR_HEIGHT, self.tech)

        w_adder = w_nand * 5
        h_adder = h_nand

        if new_height and option == 'NONE':
            if h_adder > new_height:
                raise ValueError("Adder height exceeds assigned height.")
            height = new_height
            width = w_adder * h_adder / new_height
        elif new_width and option == 'NONE':
            if w_adder > new_width:
                raise ValueError("Adder width exceeds assigned width.")
            width = new_width
            height = w_adder * h_adder / new_width
        else:
            width = w_adder
            height = h_adder

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

        # Delay stages - mimic C++ logic
        # NAND NMOS pull-down
        res_pd = logicGate.calculate_on_resistance(self.width_nand_n, constant.NMOS,
                                                   self.config['temperature'], self.tech) * 2
        gm_pd = logicGate.calculate_transconductance(self.width_nand_n, constant.NMOS, self.tech)
        tau_pd = res_pd * (cap_out + cap_in * 3)
        beta_pd = 1 / (res_pd * gm_pd)
        delay_pd, ramp_next = logicGate.horowitz(tau_pd, beta_pd, ramp_input)
        self.read_latency += delay_pd
        ramp_input = ramp_next
        # NAND PMOS pull-up
        res_pu = logicGate.calculate_on_resistance(self.width_nand_p, constant.PMOS,
                                                   self.config['temperature'], self.tech)
        gm_pu = logicGate.calculate_transconductance(self.width_nand_p, constant.PMOS, self.tech)
        tau_pu = res_pu * (cap_out + cap_in * 2)
        beta_pu = 1 / (res_pu * gm_pu)
        delay_pu, ramp_next = logicGate.horowitz(tau_pu, beta_pu, ramp_input)
        self.read_latency += delay_pu
        ramp_input = ramp_next

        # Final stage output drive
        res_pd = logicGate.calculate_on_resistance(self.width_nand_n, constant.NMOS,
                                                   self.config['temperature'], self.tech) * 2
        gm_pd = logicGate.calculate_transconductance(self.width_nand_n, constant.NMOS, self.tech)
        tau_pd = res_pd * (cap_out + cap_load)
        beta_pd = 1 / (res_pd * gm_pd)
        delay_out, _ = logicGate.horowitz(tau_pd, beta_pd, ramp_input)
        self.read_latency += delay_out

        self.read_latency *= num_read
        return self.read_latency

    def calculate_power(self, num_read, num_adder_per_op):
        vdd = self.vdd
        total_nand = 5 

        # Leakage
        self.leakage = logicGate.calculate_logicgate_leakage(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p,
            self.config['temperature'], self.tech) * vdd * total_nand

        # Dynamic Energy 
        # Calibration data pattern of critical path is A=1111111..., B=1000000... and Cin=1
		# Only count 0 to 1 transition for energy
        cap_in = self.cap_nand_input
        cap_out = self.cap_nand_output
        delta = self.config.get('delta', 0.15)

        energy_per_bit = (
            (cap_in * 4 + cap_out * 3) +       # First stage
            (cap_in * 2 )
        ) * vdd * vdd

        self.read_energy = energy_per_bit  * num_read * delta

        return self.read_energy, self.leakage


