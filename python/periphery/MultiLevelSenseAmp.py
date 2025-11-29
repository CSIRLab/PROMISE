import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology
from periphery.SenseAmp import SenseAmp

class MultilevelSenseAmp:
    """
    Parameters:
    - tech: Technology object (provides feature size, vdd, etc.)
    - param: dict of simulation parameters from param.yaml
    - config: dict of configuration values (temperature etc.)

    Initialize Parameters:
    - num_col: number of columns
    - level_output: resolution (e.g., 8-bit → 256 levels)
    - clk_freq: clock frequency
    - num_read_cell: number of cells read per operation
    - parallel: whether ADCs are operating in parallel
    - current_mode: use current sensing mode or not
    """

    def __init__(self, num_col, level_output, clk_freq, columncap,pitch_sense_amp, current_mode, tech, config,mapping,param):
        self.tech = tech
        self.param = param
        self.config = config
        self.mapping = mapping
        self.pitch_sense_amp = pitch_sense_amp
        self.Rref = []

        self.num_col = num_col
        self.columncap = columncap
        self.level_output = level_output
        self.clk_freq = clk_freq
        self.current_mode = current_mode

        self.feature_size = self.tech.get_param('featureSize')
        self.tech_node = self.tech.get_param('node_nm')
        self.roadmap = self.tech.get_param('roadmap')
        self.temp = self.config['temperature']
        self.vdd = self.tech.get_param('vdd')
        self.temp = self.config.get('temperature', 300)
        self.width_nmos = constant.MIN_NMOS_SIZE * self.feature_size
        self.width_pmos = self.tech.get_param('pnSizeRatio') * self.width_nmos
        self.width_NandN = constant.MIN_NMOS_SIZE * self.feature_size * 4
        self.width_NandP = self.tech.get_param('pnSizeRatio') * self.width_NandN * 2
        scale = 2 if self.feature_size <= 14e-9 else 1
        self.width_sense_p = scale * constant.W_SENSE_P * self.feature_size
        self.width_sense_n = scale * constant.W_SENSE_N * self.feature_size

        self.read_voltage = self.param['readVoltage']
        self.resistance_on = self.param['resistanceOn']
        self.resistance_off = self.param['resistanceOff']
        self.height_region = self.feature_size * constant.MAX_TRANSISTOR_HEIGHT

        self.level = math.log2(self.level_output)

        self.gatecap_senseamp_P,self.junctioncap_senseamp_P = logicGate.calculate_logicgate_cap(constant.INV, 1, 0, self.width_sense_p, self.height_region, self.tech)
        self.gatecap_senseamp_N,self.junctioncap_senseamp_N = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_sense_n, 0, self.height_region, self.tech)
        self.capNandInput,self.capNandOutput = logicGate.calculate_logicgate_cap(constant.NAND, 2, self.width_NandN, self.width_NandP, self.height_region, self.tech)
        #the pitch size need to be updated based on the actual design
        self.sense_amp = SenseAmp(num_col=1,current_sense=False,sense_voltage=self.param['minSenseVoltage'],clk_freq = None,pitch_sense_amp= self.pitch_sense_amp,tech=self.tech,config=self.config,mapping=self.mapping)

        # NOR
        self.width_nor_n = constant.MIN_NMOS_SIZE * self.feature_size
        self.num_input_nor = self.level_output / 2
        self.width_nor_p = self.num_input_nor * self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.feature_size
        self.num_nor = self.level

        for i in range(level_output - 1):
            R_start = self.resistance_on
            R_end = self.resistance_off
            G_start = 1 / R_start
            G_end = 1 / R_end
            G_index = G_end - G_start       # Conductance range
            G_offset = G_index / (2 * (level_output - 1))
            R_this = 1 / (G_start + G_offset + i * G_index / (level_output - 1))

            self.Rref.append(R_this)

        self.initialized = True


    def calculate_area(self, height_array=None, width_array=None, option='NONE'):
        if not self.initialized:
            raise Exception("MultilevelSenseAmp not initialized.")
        wNmos,hNmos,_ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_nmos, 0, self.height_region, self.tech)
        wPmos,hPmos,_ = logicGate.calculate_logicgate_area(constant.INV, 1, 0, self.width_pmos, self.height_region, self.tech)
        h_nor, w_nor, _ = logicGate.calculate_logicgate_area(constant.NOR,self.num_input_nor, self.width_nor_n , self.width_nor_p, self.feature_size * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        self.cap_nor_input, self.cap_nor_output = logicGate.calculate_logicgate_cap(constant.NOR, self.num_input_nor, self.width_nor_n, self.width_nor_p, h_nor, self.tech)
        sense_amp_area,sense_amp_height,sense_amp_width,_ = self.sense_amp.calculate_area(new_height=None,new_width=None,option='NONE')
        # total_area = (hNmos * wNmos) * 9 * (self.level_output - 1) * self.num_col
        # total_area += (hPmos * wPmos) * 7 * (self.level_output - 1) * self.num_col
        total_area = sense_amp_area * (self.level_output - 1) * self.num_col * 0.8

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
            self.width = math.sqrt(total_area)
            self.height = math.sqrt(total_area)

        self.area = self.width * self.height
        return self.area, self.height, self.width

    def calculate_latency(self, num_col_muxed, num_read,cap_load=47e-15):
        nor_read_latency = 0
        if len(self.Rref) < 2:
            raise ValueError("Rref must contain at least two reference levels.")

        # Latency for min and max reference resistance
        latency_minrefcon = self.column_latency_table(self.Rref[-1])
        latency_maxrefcon = self.column_latency_table(self.Rref[1])

        latency_col = max(latency_minrefcon, latency_maxrefcon)

        if self.num_nor:
            res_pull_up = logicGate.calculate_on_resistance(self.width_nor_p, constant.PMOS, self.config['temperature'], self.tech) * 2
            tr = res_pull_up * (self.cap_nor_output)
            gm = logicGate.calculate_transconductance(self.width_nor_p, constant.PMOS, self.tech)
            beta = 1 / (res_pull_up * gm)
            nor_read_latency += logicGate.horowitz(tr, beta, 1e20)[0]
        ####################
        #change 1ns to the latency of the voltage sense amp
        ####################
        # No current mode → fixed delay of 1ns per column MUX
        _,_,_,_ = self.sense_amp.calculate_area(new_height=None,new_width=None,option='NONE')
        sense_amp_read_latency = self.sense_amp.calculate_latency(num_read=1,cap_load=cap_load)

         # Total read latency
        read_latency = (sense_amp_read_latency + nor_read_latency) * num_col_muxed * num_read

        return read_latency

    def calculate_power(self, column_resistance_list, num_read):
        self.read_dynamic_energy = 0
        for res in column_resistance_list:
            self.P_Col = self.get_column_power(res)

            Column_SwitchingE = (self.columncap * 2) * (1 / (self.level_output - 1)) / 2 * (self.read_voltage ** 2)
            # print(f"Initial Column_SwitchingE: {Column_SwitchingE:.3e} J")
            # Column_SwitchingE += param.reference_energy_peri 

            Column_SwitchingE += (
                (self.gatecap_senseamp_N * 2 +
                 (self.junctioncap_senseamp_N) / (self.level_output - 1) +
                 (self.junctioncap_senseamp_N)) * (self.read_voltage ** 2)
            )

            Column_SwitchingE += (
                (self.gatecap_senseamp_P * 2 + self.gatecap_senseamp_N) * (self.vdd ** 2)
            )

            Column_SwitchingE += (
                (self.gatecap_senseamp_P * 3 + self.gatecap_senseamp_N * 3 +
                 (self.junctioncap_senseamp_P + self.junctioncap_senseamp_N) * 3 +
                 self.junctioncap_senseamp_P * 1) * (self.vdd ** 2)
            )

            Column_SwitchingE += (
                (self.gatecap_senseamp_P + self.gatecap_senseamp_N) * (self.vdd ** 2)
            )
            self.read_dynamic_energy += Column_SwitchingE * (self.level_output - 1)

            self.read_dynamic_energy += max(self.P_Col * 1e-9, 0)
             # NOR stage
            self.read_dynamic_energy += (self.cap_nor_output) * self.vdd ** 2 * self.num_nor    #one NOR output activated
        self.read_dynamic_energy *= num_read
        self.read_dynamic_energy *= self.num_col

        return self.read_dynamic_energy

    def get_column_power(self, column_res):
        
        col_res = column_res * 0.5 / self.read_voltage
        level = math.log2(self.level_output)

        #skip the current mode for now
        if 1/col_res == 0:
            column_power = 1e-6
        elif col_res == 0:
            column_power = 0
        else:
            #leakage power due to the current mirror part
            # negligible in the voltage mode case
            if self.roadmap == 'HP':
                column_power = 0
            else:
                column_power = 0
        column_power *= (1+1.3e-3*(self.temp-300))
        return column_power
    
    def column_latency_table(self, res):
        x = res
        latency = 0
        refcap = 0
        dC = 0              #fitting parameter
        R2 = 0              #fitting parameter
        resthreshold = 0    #shows cap dependency above a threshold

        if self.tech_node == 130:
            if x < 1832:
                latency = -1.6656E-10 * math.log(x) + 2.86608E-09
            else:
                latency = 3.36654E-14 * x + 2.33047E-09
            refcap = 26.62E-15
            dC = 2.12E-9
            R2 = 0.1E+6
            resthreshold = 1.95E+3

        elif self.tech_node == 90:
            if x < 1956:
                latency = -8.13238E-11 * math.log(x) + 1.9981E-09
            else:
                latency = 2.76205E-14 * x + 1.87593E-09
            refcap = 18.4E-15
            dC = 1.8E-9
            R2 = 0.1E+6
            resthreshold = 4.83E+3

        elif self.tech_node == 65:
            if x < 1210:
                latency = -1.15612E-10 * math.log(x) + 2.08218E-09
            elif x < 3161:
                latency = -1.00285E-11 * math.log(x) + 1.35472E-09
            else:
                latency = 2.03232E-14 * x + 1.61564E-09
            refcap = 13.3E-15
            dC = 1.3E-9
            R2 = 0.1E+6
            resthreshold = 4.83E+3

        elif self.tech_node == 45:
            if x < 695:
                latency = -2.66629E-10 * math.log(x) + 3.07867E-09
            elif x < 4832:
                latency = -6.27624E-11 * math.log(x) + 1.78932E-09
            else:
                latency = 1.43578E-14 * x + 1.46251E-09
            refcap = 9.21E-15
            dC = 0.9E-9
            R2 = 0.1E+6
            resthreshold = 4.83E+3

        elif self.tech_node == 32:
            if x < 4832:
                latency = -1.8746E-10 * math.log(x) + 2.44191E-09
            else:
                latency = 1.07272E-14 * x + 1.05695E-09
            refcap = 6.55E-15
            dC = 0.7E-9
            R2 = 0.1E+6
            resthreshold = 12700

        elif self.tech_node == 22:
            if x < 4832:
                latency = -2.01894E-10 * math.log(x) + 2.2559E-09
            else:
                latency = 7.45392E-15 * x + 6.72844E-10
            refcap = 4.5E-15
            dC = 0.47E-9
            R2 = 0.1E+6
            resthreshold = 12700

        elif self.tech_node == 14:
            if x < 695:
                latency = -1.17191E-10 * math.log(x) + 1.51724E-09
            elif x < 1832:
                latency = -3.56669E-11 * math.log(x) + 1.00049E-09
            else:
                latency = 5.484E-15 * x + 7.4812E-10
            refcap = 2.86E-15
            dC = 0.34E-9
            R2 = 0.1E+6
            resthreshold = 12700

        elif self.tech_node == 10:
            if x < 2976:
                latency = -5.88954E-11 * math.log(x) + 1.12213E-09
            elif x < 7847:
                latency = -1.03546E-11 * math.log(x) + 7.46313E-10
            else:
                latency = 4.02595E-15 * x + 6.64175E-10
            refcap = 2.04E-15
            dC = 0.24E-9
            R2 = 0.1E+6
            resthreshold = 12700

        elif self.tech_node == 7:
            if x < 1956:
                latency = -9.29209E-11 * math.log(x) + 1.36675E-09
            elif x < 8251:
                latency = -2.32202E-11 * math.log(x) + 8.55312E-10
            else:
                latency = 2.96941E-15 * x + 6.3388E-10
            refcap = 1.43E-15
            dC = 0.17E-9
            R2 = 0.1E+6
            resthreshold = 12700

        else:
            raise ValueError(f"Unsupported technology node")

        # Capacitance-dependent latency extension (only for res > threshold)
        if x > resthreshold:
            columncap = self.columncap + self.gatecap_senseamp_N * (len(self.Rref) - 2)
            extra = dC / (R2 - resthreshold) * (x - resthreshold) * ((columncap - refcap) / refcap)
            latency += extra

        return latency