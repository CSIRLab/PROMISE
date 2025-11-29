import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class LevelShifter:
    def __init__(self, tech, config,param, mapping,clk_freq,num_output):
        self.tech = tech
        self.param = param
        self.config = config
        self.mapping = mapping

        self.featureSize = tech.get_param('featureSize')
        self.vdd = tech.get_param('vdd')
        self.temp = config['temperature']

        self.width_nmos = constant.MIN_NMOS_SIZE * self.featureSize
        self.width_pmos = self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.featureSize
        self.writeVoltage = self.param['writeVoltage']

        self.cap_low_drain = 0
        self.cap_high_drain = 0
        self.cap_mid_gate_n = 0

        self.num_output = num_output
        self.activity_row_read = self.mapping['activity_row_read']
        self.clk_freq = clk_freq

    def calculate_area(self, new_height, new_width, option):
        MAGIC = 'MAGIC'
        OVERRIDE = 'OVERRIDE'
        NONE = 'NONE'

        # 3 types of inverter in level shifter
        #one high voltage inverter, one low voltage inverter, two mid latch
        wlow, hlow, _ = logicGate.calculate_logicgate_area(constant.INV, 1,
                                                           self.width_nmos * 15,
                                                           self.width_pmos * 20,
                                                           self.featureSize * constant.MAX_TRANSISTOR_HEIGHT*2,
                                                           self.tech)

        wlatch, hlatch, _ = logicGate.calculate_logicgate_area(constant.INV, 1,
                                                               self.width_nmos * 32,
                                                               self.width_pmos * 10,
                                                               self.featureSize * constant.MAX_TRANSISTOR_HEIGHT*2,
                                                               self.tech)

        whigh, hhigh, _ = logicGate.calculate_logicgate_area(constant.INV, 1,
                                                             self.width_nmos * 64,
                                                             self.width_pmos * 82,
                                                             self.featureSize * constant.MAX_TRANSISTOR_HEIGHT*2,
                                                             self.tech)

        print("wlow, hlow, wlatch, hlatch, whigh, hhigh:", wlow, hlow, wlatch, hlatch, whigh, hhigh)
        hLS = max(hlow, hlatch, hhigh)
        wLS = wlow + (2 * wlatch + whigh)*1.2

        width = new_width if new_width and option == NONE else wLS
        height = new_height if new_height and option == NONE else hLS

        area = height * width * self.num_output
        self.area = area
        self.height = height
        self.width = width

        # Capacitances
        self.cap_mid_gate_n = logicGate.calculate_mos_gate_cap(self.width_nmos * 32, self.tech)
        _, self.cap_low_drain = logicGate.calculate_logicgate_cap(constant.INV, 1,
                                                                   self.width_nmos * 15,
                                                                   self.width_pmos * 20,
                                                                   hlow,
                                                                   self.tech)
        _, self.cap_high_drain = logicGate.calculate_logicgate_cap(constant.INV, 1,
                                                                    self.width_nmos * 64,
                                                                    self.width_pmos * 82,
                                                                    hhigh,
                                                                    self.tech)
        return area, height, width

    def calculate_latency(self, cap_load, num_read, num_write):
        # First stage: low voltage pull up
        res_pull_up = logicGate.calculate_on_resistance(self.width_pmos * 20, constant.PMOS, self.temp, self.tech)
        tr1 = res_pull_up * (self.cap_low_drain + self.cap_mid_gate_n * 2)
        gm1 = logicGate.calculate_transconductance(self.width_pmos * 20, constant.PMOS, self.tech)
        beta1 = 1 / (res_pull_up * gm1)
        base_latency1, _ = logicGate.horowitz(tr1, beta1, ramp_input=1e20)

        # Second stage: high voltage pull up
        res_pull_up = logicGate.calculate_on_resistance(self.width_pmos * 82, constant.PMOS, self.temp, self.tech)
        tr2 = res_pull_up * (cap_load + self.cap_high_drain)
        gm2 = logicGate.calculate_transconductance(self.width_pmos * 82, constant.PMOS, self.tech)
        beta2 = 1 / (res_pull_up * gm2)
        base_latency2, _ = logicGate.horowitz(tr2, beta2, ramp_input=1e20)

        read_latency = (base_latency1 + base_latency2) * num_read
        write_latency = (base_latency1 + base_latency2) * num_write

        return read_latency, write_latency

    def calculate_power(self, cap_load, num_read, num_write):
        # Read dynamic energy
        read_energy = (self.cap_low_drain + self.cap_mid_gate_n * 2) * self.vdd**2 * self.num_output * self.activity_row_read
        read_energy *= num_read

        # Write dynamic energy
        write_energy = (self.cap_low_drain*4 + self.cap_mid_gate_n * 8) * self.vdd**2
        write_energy += (cap_load + self.cap_high_drain*4) * 1 * self.writeVoltage**2
        write_energy *= num_write
        write_energy *= 1.4  

        # Leakage (not modeled in C++ code, set to zero or extend later)
        leakage = 0

        return write_energy, write_energy, leakage

