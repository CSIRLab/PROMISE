import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class SenseAmp:
    """
    Sense Amplifier for SRAM/Neural In-Memory Readout
    - num_col: number of columns
    - current_sense: True if current-mode sensing (not supported)
    - sense_voltage: voltage difference needed for sensing
    - pitch_sense_amp: /* The maximum width allowed for one sense amplifier layout */       lengthRow/numCol
    - clk_freq: clock frequency
    - tech: Technology object
    - config: dict with environmental config (e.g., temperature)
    - mapping: dict with operational parameters (e.g., num_read_cell_op)
    """
    def __init__(self, num_col, current_sense, sense_voltage, pitch_sense_amp, clk_freq, tech, config, mapping):
        self.num_col = num_col
        self.current_sense = current_sense
        self.sense_voltage = sense_voltage
        self.pitch_sense_amp = pitch_sense_amp
        self.clk_freq = clk_freq
        self.tech = tech
        self.config = config
        self.mapping = mapping

        self.featureSize = self.tech.get_param('featureSize')
        self.vdd = self.tech.get_param('vdd')
        self.temp = self.config['temperature']
        self.num_read_cell_op = mapping['num_read_cell_op']

        # Width scaling below 14nm
        scale = 2 if self.featureSize <= 14e-9 else 1
        self.width_sense_p = scale * constant.W_SENSE_P * self.featureSize
        self.width_sense_n = scale * constant.W_SENSE_N * self.featureSize
        self.width_sense_iso = scale * constant.W_SENSE_ISO * self.featureSize
        self.width_sense_en = scale * constant.W_SENSE_EN * self.featureSize
        self.width_sense_mux = scale * constant.W_SENSE_MUX * self.featureSize

        self.cap_load = None
        self.area = 0
        self.height = 0
        self.width = 0
        

    def calculate_area(self, new_height, new_width, option='NONE'):
        if self.current_sense:
            raise NotImplementedError("Current sensing is not supported yet.")
        # Compute gate areas
        w_sense_p, h_sense_p, _ = logicGate.calculate_logicgate_area(constant.INV, 1, 0, self.width_sense_p, self.pitch_sense_amp, self.tech)   #pmos
        w_sense_n, h_sense_n, _ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_sense_n, 0, self.pitch_sense_amp, self.tech)     #nmos
        w_sense_iso, h_sense_iso, _ = logicGate.calculate_logicgate_area(constant.INV, 1, 0, self.width_sense_iso, self.pitch_sense_amp, self.tech)       #pmos
        w_sense_en, h_sense_en, _ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_sense_en, 0, self.pitch_sense_amp, self.tech)        #nmos

        area_per_col = (w_sense_p * h_sense_p) * 2 + (w_sense_n * h_sense_n) * 2 + (w_sense_iso * h_sense_iso) + (w_sense_en * h_sense_en)
        self.area = area_per_col * self.num_col

        if new_width and option == 'NONE':
            self.width = new_width
            self.height = self.area / new_width
        elif new_height and option == 'NONE':
            self.height = new_height
            self.width = self.area / new_height
        else:
            self.height = max(h_sense_p, h_sense_n, h_sense_iso, h_sense_en) * 2
            self.width = self.area / self.height
        # Capacitance calculation
        cap_gate_sense_p = logicGate.calculate_logicgate_cap(constant.INV, 1, 0, self.width_sense_p, self.pitch_sense_amp, self.tech)[0]
        cap_gate_sense_n = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_sense_n, 0, self.pitch_sense_amp, self.tech)[0]
        cap_drain_sense_p = logicGate.calculate_drain_cap(constant.PMOS,self.width_sense_p,  self.pitch_sense_amp, self.tech)
        cap_drain_sense_n = logicGate.calculate_drain_cap(constant.NMOS,self.width_sense_n,  self.pitch_sense_amp, self.tech)
        cap_drain_sense_iso = logicGate.calculate_drain_cap(constant.PMOS,self.width_sense_iso,  self.pitch_sense_amp, self.tech)
        cap_drain_sense_mux = logicGate.calculate_drain_cap(constant.NMOS,self.width_sense_mux,  self.pitch_sense_amp, self.tech)
        # Total load capacitance seen by the sense amplifier
        

        self.cap_load = cap_gate_sense_p + cap_gate_sense_n + cap_drain_sense_p + cap_drain_sense_n

        
        return self.area, self.height, self.width, self.cap_load

    

    def calculate_latency(self, num_read):
        # gm = gm_NMOS + gm_PMOS
        gm_n = logicGate.calculate_transconductance(self.width_sense_n, constant.NMOS, self.tech)
        gm_p = logicGate.calculate_transconductance(self.width_sense_p, constant.PMOS, self.tech)
        gm_total = gm_n + gm_p

        # print(f"SenseAmp: gm_n {gm_n}, gm_p {gm_p}, gm_total {gm_total}, cap_load {self.cap_load}")

        tau = self.cap_load / gm_total
        latency = tau * math.log(self.vdd / self.sense_voltage)
        read_latency = latency * num_read
        return read_latency

    def calculate_power(self, num_read):
        # Leakage
        gate_leak = logicGate.calculate_logicgate_leakage(
            constant.INV, 1, self.width_sense_en, 0, self.temp, self.tech)
        leakage = gate_leak * self.vdd * self.num_col

        # Dynamic
        min_read = min(int(self.num_read_cell_op), int(self.num_col))
        read_energy = self.cap_load * self.vdd ** 2 * min_read * num_read

        return read_energy, leakage


