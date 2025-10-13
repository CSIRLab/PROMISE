import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.DFF import DFF
from periphery.MUX import Mux
from periphery.Technology import Technology


class RNG_block:
    def __init__(self, tech, param, mapping, RNG, config, num_block):
        self.tech = tech
        self.param = param
        self.mapping = mapping
        self.RNG = RNG
        self.config = config
        self.num_block = num_block

        self.vdd = tech.get_param('vdd')
        self.feature_size = tech.get_param('featureSize')
        self.temp = config['temperature']
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.readVoltage = self.param['readVoltage']

        self.precision_sigma = self.config['precision_sigma']
        self.clk_freq = self.config['frequency']

        #RNG
        self.RNG_height = self.RNG['Hight']/65e-9*self.feature_size
        self.RNG_width = self.RNG['Width']/65e-9*self.feature_size
        self.RNG_energy_per_sample = self.RNG['EnergyPerSample']
        self.RNG_throughput = self.RNG['Throughtput']
        self.RNG_SamplingMode = self.RNG['SamplingMode']

        self.MUX = Mux(num_input=2,num_selection=1,param=self.param,mapping=self.mapping, config = self.config, tech=self.tech,res_tg=None,FPGA=False)
        self.dff = DFF(num_dff=self.precision_sigma+1,tech=self.tech,config=self.config,param=self.param,clk_freq=self.clk_freq)

        scale = 2 if self.feature_size <= 14e-9 else 1
        res_width_n = logicGate.calculate_on_resistance(self.feature_size, constant.NMOS, self.temp, self.tech)
        res_width_p = logicGate.calculate_on_resistance(self.feature_size, constant.PMOS, self.temp, self.tech)
        self.width_tg_n = constant.MIN_NMOS_SIZE * self.feature_size
        self.width_tg_p = self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.feature_size
        # Simplified without EnlargeSize
        self.res_tg = 1 / (
            1 / logicGate.calculate_on_resistance(self.width_tg_n, constant.NMOS, self.temp, self.tech) +
            1 / logicGate.calculate_on_resistance(self.width_tg_p, constant.PMOS, self.temp, self.tech)
        )

        self.initialized = True

    def calculate_area(self):
        if not self.initialized:
            raise ValueError("MUX must be initialized before area calculation.")
        
        w_tg, h_tg, _ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_tg_n, self.width_tg_p,constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
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
        ############################################

        MUX_area,MUX_height,MUX_width = self.MUX.calculate_area(new_height=None,new_width=None,option='NONE')
        dff_area,dff_height,dff_width = self.dff.calculate_area(new_height=None,new_width=None,option='NONE')

        self.width = ((w_tg + MUX_width)* self.precision_sigma + self.RNG_width)  * self.num_block
        self.width = max(self.width, dff_width)
        self.height = max(h_tg, MUX_height, self.RNG_height) + dff_height

        self.area = self.width * self.height

        self.capTgGateN = logicGate.calculate_mos_gate_cap(self.width_tg_n, self.tech)
        self.capTgGateP = logicGate.calculate_mos_gate_cap(self.width_tg_p, self.tech)
        _,self.capTgDrain = logicGate.calculate_logicgate_cap(gate_type = constant.NMOS, num_Input = 1, width_NMOS = self.width_tg_n, width_PMOS = self.width_tg_p, height_transistor_region = h_tg, tech = self.tech)

        return self.area, self.height, self.width
    
    def calculate_latency(self, num_read):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
            return 0

        self.ramp_input = 1e20

        tr = self.res_tg * (
            self.capTgDrain + 0.5 * self.capTgGateN + 0.5 * self.capTgGateP)

        # assume 2.3*tau for 90% voltage swing
        read_latency = 2.3 * tr * num_read

        dff_read_latency,dff_write_latency = self.dff.calculate_latency(num_read=1)

        self.read_latency = read_latency + 1/self.RNG_throughput + dff_read_latency
        return read_latency
    
    def calculate_power(self, num_read):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
            return 0, 0

        leakage = 0  

        energy_gate = self.capTgGateN * 2 * self.vdd ** 2

        energy_drain = 2 * self.capTgDrain * 2 * self.readVoltage ** 2
        dff_read_energy,dff_write_energy,dff_leakage = self.dff.calculate_power(num_read=1, num_dff_per_op=2, validated=False)

        read_dynamic_energy = (energy_gate + energy_drain + self.RNG_energy_per_sample + dff_read_energy) * num_read * self.num_block

        self.read_dynamic_energy = read_dynamic_energy
        self.leakage = leakage

        return read_dynamic_energy, leakage
