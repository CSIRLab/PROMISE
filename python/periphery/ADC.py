import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class SarADC:
    """
    Parameters:
    - num_col: number of ADCs (usually one per column)
    - level_output: ADC resolution (e.g., 8-bit -> 256 levels)
    - clk_freq: operating clock frequency
    - num_read_cell: number of active read cells per operation
    - tech: Technology object (provides feature size, vdd, etc.)
    - config: dict containing simulation environment parameters (e.g., temperature)
    - mapping: dict containing array mapping details (optional)
    - param: additional parameters 
    -column_resistance_list- from the mapping, list of column resistance for each column

    """
    def __init__(self, num_col, level_output, clk_freq, num_read_cell, tech, config,mapping,param):
        self.tech = tech
        self.config = config
        self.mapping = mapping
        self.param = param
        self.num_col = num_col
        self.level_output = level_output
        self.clk_freq = clk_freq
        self.num_read_cell = num_read_cell
        self.feature_size = self.tech.get_param('featureSize')
        self.vdd = self.tech.get_param('vdd')
        self.temp = self.config['temperature']
        self.width_nmos = self.feature_size * constant.MIN_NMOS_SIZE
        self.width_pmos = self.tech.get_param('pnSizeRatio') * self.width_nmos
        self.initialized = True

    def calculate_unit_area(self):
        height_tx = self.feature_size * constant.MAX_TRANSISTOR_HEIGHT
        level = math.log2(self.level_output)

        area_nmos = height_tx * self.width_nmos
        area_pmos = height_tx * self.width_pmos

        self.area_unit = (area_nmos * (269 + (level - 1) * 109)) + \
                         (area_pmos * (209 + (level - 1) * 73))

    def calculate_area(self, height_array, width_array, option='NONE'):
        total_area = self.area_unit * self.num_col

        if option == 'NONE':
            if width_array:
                self.width = width_array
                self.height = total_area / width_array
            elif height_array:
                self.height = height_array
                self.width = total_area / height_array
            else:
                self.width = math.sqrt(total_area)
                self.height = math.sqrt(total_area)
        else:
            # Extend here for MAGIC / OVERRIDE layout options
            self.width = math.sqrt(total_area)
            self.height = math.sqrt(total_area)

        self.area = self.width * self.height
        return self.area, self.height, self.width

    def calculate_latency(self, num_read):
        level = math.log2(self.level_output)
        cycles = level + 1
        self.read_latency = cycles * num_read / self.clk_freq
        return self.read_latency

    def calculate_power(self, column_resistance_list, num_read):
        self.read_dynamic_energy = 0
        for res in column_resistance_list:
            self.read_dynamic_energy += self.get_column_power(res)
        self.read_dynamic_energy *= num_read
        return self.read_dynamic_energy

    def get_column_power(self, column_res):
        # convert the column resistance to 0.5 read voltage
        read_voltage = self.param['readVoltage']
        roadmap = self.param['deviceroadmap']
        # roadmap = 'HP'
        col_res = column_res * 0.5 / read_voltage
        # tech_node = self.feature_size * 1e9  # in nm
        tech_node = self.tech.get_param('node_nm')  # in nm
        level = math.log2(self.level_output)

        #model is based on HP and LP roadmap
        # base = A + B * level
        # exp_term = C * exp(D * log10(col_res))
        #A, B, C, D are different for each technology node
        if roadmap == 'HP':  # HP
            if tech_node == 130:
                base = (6.4806 * level + 49.047) * 1e-6
                exp_term = 0.207452 * math.exp(-2.367 * math.log10(col_res))
            elif tech_node == 90:
                base = (4.3474 * level + 31.782) * 1e-6
                exp_term = 0.1649 * math.exp(-2.345 * math.log10(col_res))
            elif tech_node == 65:
                base = (2.9503 * level + 22.047) * 1e-6
                exp_term = 0.128483 * math.exp(-2.321 * math.log10(col_res))
            elif tech_node == 45:
                base = (2.1843 * level + 11.931) * 1e-6
                exp_term = 0.097754 * math.exp(-2.296 * math.log10(col_res))
            elif tech_node == 32:
                base = (1.0157 * level + 7.6286) * 1e-6
                exp_term = 0.083709 * math.exp(-2.313 * math.log10(col_res))
            elif tech_node == 22:
                base = (0.7213 * level + 3.3041) * 1e-6
                exp_term = 0.084273 * math.exp(-2.311 * math.log10(col_res))
            elif tech_node == 14:
                base = (0.4710 * level + 1.9529) * 1e-6
                exp_term = 0.060584 * math.exp(-2.311 * math.log10(col_res))
            elif tech_node == 10:
                base = (0.3076 * level + 1.1543) * 1e-6
                exp_term = 0.049418 * math.exp(-2.311 * math.log10(col_res))
            else:  # 7nm
                base = (0.2008 * level + 0.6823) * 1e-6
                exp_term = 0.040310 * math.exp(-2.311 * math.log10(col_res))
        else:  # LP
            if tech_node == 130:
                base = (8.4483 * level + 65.243) * 1e-6
                exp_term = 0.16938 * math.exp(-2.303 * math.log10(col_res))
            elif tech_node == 90:
                base = (5.9869 * level + 37.462) * 1e-6
                exp_term = 0.144323 * math.exp(-2.303 * math.log10(col_res))
            elif tech_node == 65:
                base = (3.7506 * level + 25.844) * 1e-6
                exp_term = 0.121272 * math.exp(-2.303 * math.log10(col_res))
            elif tech_node == 45:
                base = (2.1691 * level + 16.693) * 1e-6
                exp_term = 0.100225 * math.exp(-2.303 * math.log10(col_res))
            elif tech_node == 32:
                base = (1.1294 * level + 8.8998) * 1e-6
                exp_term = 0.079449 * math.exp(-2.297 * math.log10(col_res))
            elif tech_node == 22:
                base = (0.538 * level + 4.3753) * 1e-6
                exp_term = 0.072341 * math.exp(-2.303 * math.log10(col_res))
            elif tech_node == 14:
                base = (0.3132 * level + 2.5681) * 1e-6
                exp_term = 0.061085 * math.exp(-2.303 * math.log10(col_res))
            elif tech_node == 10:
                base = (0.1823 * level + 1.5073) * 1e-6
                exp_term = 0.05158 * math.exp(-2.303 * math.log10(col_res))
            else:  # 7nm
                base = (0.1061 * level + 0.8847) * 1e-6
                exp_term = 0.043555 * math.exp(-2.303 * math.log10(col_res))

        # consider temperature effect, power increase by 0.13% per degree Celsius
        column_power = (base + exp_term) * (1 + 1.3e-3 * (self.temp- 300))
        column_energy = column_power * (level + 1) * 1/ self.clk_freq  # Energy = Power * Time
        return column_energy

