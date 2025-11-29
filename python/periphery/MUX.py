import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class Mux:
    def __init__(self, tech, param, mapping,config, num_input, num_selection, res_tg,FPGA):
        self.tech = tech
        self.param = param
        self.mapping = mapping
        self.num_input = num_input
        self.config = config
        self.num_selection = num_selection  #number of mux selection lines
        self.FPGA = FPGA
        self.vdd = tech.get_param('vdd')
        self.feature_size = tech.get_param('featureSize')
        self.temp = config['temperature']
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.readVoltage = self.param['readVoltage']

        if FPGA:
            self.width_tg_n = constant.MIN_NMOS_SIZE * self.feature_size
            self.width_tg_p = self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.feature_size
            # Simplified without EnlargeSize
            self.res_tg = 1 / (
                1 / logicGate.calculate_on_resistance(self.width_tg_n, constant.NMOS, self.temp, self.tech) +
                1 / logicGate.calculate_on_resistance(self.width_tg_p, constant.PMOS, self.temp, self.tech)
            )
        else:
            self.width_tg_n = constant.MIN_NMOS_SIZE * self.feature_size
            self.width_tg_p = self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.feature_size
            self.res_tg = 1 / (
                1 / logicGate.calculate_on_resistance(self.width_tg_n, constant.NMOS, self.temp, self.tech) +
                1 / logicGate.calculate_on_resistance(self.width_tg_p, constant.PMOS, self.temp, self.tech)
            )
            # self.res_tg = res_tg
            # Width calculation for analog MUX
            scale = 2 if self.feature_size <= 14e-9 else 1
            res_width_n = logicGate.calculate_on_resistance(self.feature_size, constant.NMOS, self.temp, self.tech)
            res_width_p = logicGate.calculate_on_resistance(self.feature_size, constant.PMOS, self.temp, self.tech)
            self.width_tg_n = res_width_n * self.feature_size * scale / (self.res_tg * 2)
            self.width_tg_p = res_width_p * self.feature_size * scale / (self.res_tg * 2)
            self.res_tg = 1 / (
                1 / logicGate.calculate_on_resistance(self.width_tg_n, constant.NMOS, self.temp, self.tech) +
                1 / logicGate.calculate_on_resistance(self.width_tg_p, constant.PMOS, self.temp, self.tech)
            )

        self.initialized = True

    def calculate_area(self, new_height=None, new_width=None, option='NONE'):
        if not self.initialized:
            raise ValueError("MUX must be initialized before area calculation.")

        num_tg = self.num_input * self.num_selection

        if self.FPGA:   #digital MUX
            w_tg, h_tg, _ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_tg_n, self.width_tg_p,constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
            w_tg = w_tg * 1.5
            if new_width and option == 'NONE':
                num_tg_per_row = int(new_width // w_tg)
                num_tg_per_row = min(num_tg_per_row, num_tg)
                num_row_tg = math.ceil(num_tg / num_tg_per_row)
                self.width = new_width
                self.height = h_tg * num_row_tg
            else:
                self.width = w_tg * num_tg
                self.height = h_tg
        else:   #analog MUX
            #############################################
            # to be done: for the technology below 14nm, there will be some tuning for the cell width and height and islation region
            if self.feature_size == 14e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_14nm
                self.min_cell_width = (constant.POLY_WIDTH_FINFET + constant.MIN_GAP_BET_GATE_POLY_FINFET )*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_14nm
            elif self.feature_size == 10e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_10nm
                self.min_cell_width = constant.CPP_10nm*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_10nm
            elif self.feature_size == 7e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_7nm
                self.min_cell_width = constant.CPP_7nm*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_7nm
            elif self.feature_size == 5e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_5nm
                self.min_cell_width = constant.CPP_5nm*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_5nm
            elif self.feature_size == 3e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_3nm
                self.min_cell_width = constant.CPP_3nm*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_3nm
            elif self.feature_size == 2e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_2nm
                self.min_cell_width = constant.CPP_2nm*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_2nm
            elif self.feature_size == 1e-9:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT_1nm
                self.min_cell_width = constant.CPP_1nm*2
                self.siolation_region = constant.OUTER_HEIGHT_REGION_1nm
            else:
                self.min_cell_height = constant.MAX_TRANSISTOR_HEIGHT
                self.min_cell_width = constant.MIN_GAP_BET_GATE_POLY + constant.POLY_WIDTH *2
                self.siolation_region = constant.MIN_POLY_EXT_DIFF *2 + constant.MIN_GAP_BET_FIELD_POLY
            #############################################
            w_tg, h_tg, _ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_tg_n, self.width_tg_p,constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
            if new_width and option == 'NONE':
                num_tg_per_row = int(new_width // self.min_cell_width)
                num_tg_per_row = min(num_tg_per_row, num_tg)
                num_row_tg = math.ceil(num_tg / num_tg_per_row)
                tgWidth = new_width/num_tg_per_row
                numFold = math.ceil(tgWidth / (0.5*self.min_cell_width))-1

                self.width = new_width
                self.height = h_tg * num_row_tg
            else:
                self.width = w_tg * num_tg
                self.height = h_tg

        self.area = self.width * self.height

        self.capTgGateN = logicGate.calculate_mos_gate_cap(self.width_tg_n, self.tech)
        self.capTgGateP = logicGate.calculate_mos_gate_cap(self.width_tg_p, self.tech)
        _,self.capTgDrain = logicGate.calculate_logicgate_cap(gate_type = constant.NMOS, num_Input = 1, width_NMOS = self.width_tg_n, width_PMOS = self.width_tg_p, height_transistor_region = h_tg, tech = self.tech)

        return self.area, self.height, self.width
    
    def calculate_latency(self, cap_load, num_read):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
            return 0

        self.ramp_input = 1e20
        self.cap_load = cap_load

        tr = self.res_tg * (
            self.capTgDrain + 0.5 * self.capTgGateN + 0.5 * self.capTgGateP + self.cap_load
        )

        # assume 2.3*tau for 90% voltage swing
        read_latency = 2.3 * tr * num_read

        self.read_latency = read_latency
        return read_latency
    
    def calculate_power(self, num_read):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
            return 0, 0

        leakage = 0  

        energy_gate = self.capTgGateN * self.num_input * self.vdd ** 2

        energy_drain = 2 * self.capTgDrain * self.num_input * self.vdd ** 2

        read_dynamic_energy = (energy_gate + energy_drain) * num_read

        self.read_dynamic_energy = read_dynamic_energy
        self.leakage = leakage

        return read_dynamic_energy, leakage