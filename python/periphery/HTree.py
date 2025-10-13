import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology



class HTree:
    def __init__(self, tech, param, config, num_row, num_col, delay_tolerance, bus_width):
        self.tech = tech
        self.param = param
        self.config = config
        self.num_row = num_row
        self.num_col = num_col
        self.delay_tolerance = delay_tolerance
        self.bus_width = bus_width


        self.feature_size = self.tech.get_param('featureSize')
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.vdd = self.tech.get_param('vdd')
        self.height_Transistor_Region = constant.MAX_TRANSISTOR_HEIGHT * self.feature_size


        self.temp = self.config['temperature']
        self.clk_freq = self.config['frequency']
        self.latency_mode = config['latency_mode']

        self.unit_length_wire_resistance = self.param['unitLengthWireResistance']
        self.wireWidth = self.param['wireWidth']

        self.unit_length_wire_cap = 0.2e-15 / 1e-6
        self.num_stage = 2*math.ceil(math.log2(max(self.num_row, self.num_col))) + 1######################why time 2

        self.width_min_inv_n = constant.MIN_NMOS_SIZE * self.feature_size
        self.width_min_inv_p = self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.feature_size
        w_min_INV,w_min_INV,_ =  logicGate.calculate_logicgate_area(constant.INV, 1, self.width_min_inv_n, self.width_min_inv_p,self.height_Transistor_Region, self.tech)
        self.cap_min_inv_input,self.cap_min_inv_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_min_inv_n, self.width_min_inv_p, self.height_Transistor_Region, self.tech)
        res_on_rep = (logicGate.calculate_on_resistance(self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.width_min_inv_p, constant.PMOS, self.temp, self.tech))/2


        self.repeater_size = math.floor(math.sqrt(
            res_on_rep * self.unit_length_wire_cap /
            (self.cap_min_inv_input * self.unit_length_wire_resistance)
        ))

        self.min_dist = math.sqrt(
            2 * res_on_rep * (self.cap_min_inv_input + self.cap_min_inv_output) /
            (self.unit_length_wire_resistance * self.unit_length_wire_cap)
        )
        w_rep, h_rep, rep_area = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_min_inv_n * self.repeater_size, self.width_min_inv_p * self.repeater_size, self.height_Transistor_Region, self.tech)
        cap_rep_input, cap_rep_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.repeater_size * self.width_min_inv_n, self.repeater_size * self.width_min_inv_p, h_rep, self.tech)
        res_on_rep = (logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_p, constant.PMOS, self.temp, self.tech))/2
        min_unit_length_delay = 0.7 * (res_on_rep * (cap_rep_input + cap_rep_output + self.unit_length_wire_cap * self.min_dist)+ 
                                       0.54 * self.unit_length_wire_resistance * self.min_dist * self.unit_length_wire_cap* self.min_dist + 
                                       self.unit_length_wire_resistance * self.min_dist * cap_rep_input)/ self.min_dist
        max_unit_length_Energy = (cap_rep_input + cap_rep_output + self.unit_length_wire_cap * self.min_dist) * self.vdd * self.vdd / self.min_dist

        # // tradeoff: increase delay to decrease energy
        if self.delay_tolerance:
            delay = 0
            energy = 100
            while delay < min_unit_length_delay*(1+self.delay_tolerance) and self.repeater_size >= 1:  # Simplified condition
                self.repeater_size /= 12
                self.min_dist *= 0.9
                w_rep, h_rep, rep_area = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_min_inv_n * self.repeater_size, self.width_min_inv_p * self.repeater_size, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
                cap_rep_input, cap_rep_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.repeater_size * self.width_min_inv_n, self.repeater_size * self.width_min_inv_p, h_rep, self.tech)
                res_on_rep = (logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.repeater_size * self.width_min_inv_p, constant.PMOS, self.temp, self.tech))/2
                delay = 0.7 * (res_on_rep * (cap_rep_input + cap_rep_output + self.unit_length_wire_cap * self.min_dist)+ 
                                       0.54 * self.unit_length_wire_resistance * self.min_dist * self.unit_length_wire_cap* self.min_dist + 
                                       self.unit_length_wire_resistance * self.min_dist * cap_rep_input)/ self.min_dist
                energy = (cap_rep_input + cap_rep_output + self.unit_length_wire_cap * self.min_dist) * self.vdd * self.vdd / self.min_dist

        self.width_inv_n = max(1, self.repeater_size) * self.feature_size * constant.MIN_NMOS_SIZE
        self.width_inv_p = self.width_inv_n * self.pnSizeRatio

        num_row = pow(2, (self.num_stage-1)/2)
        num_col = pow(2, (self.num_stage-1)/2)

        #define center of the H-tree
        self.center_x = (num_row)/2
        self.center_y = (num_col)/2

        self.orc = 1    #over-routing constraint: (important for unbalanced tree) avoid routing outside chip boundray
        if num_col - self.center_x < self.orc:
            self.center_x -= self.orc 
        if num_row - self.center_y < self.orc:
            self.center_y -= self.orc

        self.find_stage = 0 #assume the top stage as self.find_stage = 0
	
        self.initialized = True

    def calculate_area(self, unit_height, unit_width, folded_ratio):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating area.")
        w_inv,h_inv,area_inv = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_inv_n, self.width_inv_p, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
        wire_length_vertical = unit_height * pow(2, (self.num_stage-1)/2)
        wire_length_horizontal = unit_width * pow(2, (self.num_stage-1)/2)
        wire_width_vertical = 0
        wire_width_horizontal = 0
        area = 0
        self.total_wire_length = 0
        for i in range(1, int((self.num_stage - 1) / 2)):
            #vertical stages
            wire_length_vertical /= 2
            wire_width, unit_length_res = self.get_unit_length_res(wire_length_vertical)
            num_repeater = math.ceil(wire_length_vertical / self.min_dist)
            if num_repeater > 0:
                wire_width_vertical += self.bus_width * w_inv/folded_ratio
            else:
                wire_width_vertical += self.bus_width * wire_width/folded_ratio
            area += wire_width_vertical * wire_length_vertical/2

            #horizontal stages
            wire_length_horizontal /= 2
            wire_width, unit_length_res = self.get_unit_length_res(wire_length_horizontal)
            num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
            if num_repeater > 0:
                wire_width_horizontal += self.bus_width * w_inv/folded_ratio
            else:
                wire_width_horizontal += self.bus_width * wire_width/folded_ratio
            area += wire_width_horizontal * wire_length_horizontal/2

            #count total wire length
            self.total_wire_length += wire_length_vertical + wire_length_horizontal
        self.total_wire_length += min(self.num_col-self.center_x, self.center_x) * unit_width
        area += (self.bus_width * h_inv/folded_ratio) * min(self.num_col-self.center_x, self.center_x) * unit_width
        self.cap_inv_input, self.cap_inv_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_inv_n, self.width_inv_p, h_inv, self.tech)
        return area

    def calculate_latency(self, x_init,y_init,x_end,y_end,unit_height,unit_width,num_read):
        if not self.initialized:
            raise RuntimeError("[HTree] Error: Require initialization first!")

        read_latency = 0
        wire_length_vertical = unit_height * 2 ** ((self.num_stage - 1) / 2)
        wire_length_horizontal = unit_width * 2 ** ((self.num_stage - 1) / 2)
        res_on_rep = (logicGate.calculate_on_resistance(self.width_min_inv_n, constant.NMOS, self.temp, self.tech) + logicGate.calculate_on_resistance(self.width_min_inv_p, constant.PMOS, self.temp, self.tech))/2

    # unit_latency_rep = 0.7 * (
    #     res_on_rep * (cap_inv_input + cap_inv_output + unit_length_wire_cap * min_dist)
    #     + 0.54 * unit_length_wire_resistance * min_dist * unit_length_wire_cap * min_dist
    #     + unit_length_wire_resistance * min_dist * cap_inv_input
    # ) / min_dist

    # unit_latency_wire = (
    #     0.7 * unit_length_wire_resistance * min_dist * unit_length_wire_cap * min_dist
    # ) / min_dist

        if (x_init == 0 and y_init == 0) or (x_end == 0 and y_end == 0):
            # root-leaf communication
            for _ in range(int((self.num_stage - 1) / 2)):
                # vertical stages
                wire_length_vertical /= 2
                wire_width, unit_length_res = self.get_unit_length_res(wire_length_vertical)
                unit_latency_rep = 0.7 * (res_on_rep * (self.cap_inv_input + self.cap_inv_output + self.unit_length_wire_cap * self.min_dist) + 
                                        0.54 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist +
                                        unit_length_res * self.min_dist * self.cap_inv_input) / self.min_dist
                unit_latency_wire = (0.7 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist) / self.min_dist
                num_repeater = math.ceil(wire_length_vertical / self.min_dist)
                if num_repeater > 0:
                    read_latency += wire_length_vertical * unit_latency_rep
                else:
                    read_latency += wire_length_vertical * unit_latency_wire

                # horizontal stages
                wire_length_horizontal /= 2
                wire_width, unit_length_res = self.get_unit_length_res(wire_length_horizontal)
                unit_latency_rep = 0.7 * (res_on_rep * (self.cap_inv_input + self.cap_inv_output + self.unit_length_wire_cap * self.min_dist) + 
                                        0.54 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist +
                                        unit_length_res * self.min_dist * self.cap_inv_input) / self.min_dist
                unit_latency_wire = (0.7 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist) / self.min_dist
                num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
                if num_repeater > 0:
                    read_latency += wire_length_horizontal * unit_latency_rep
                else:
                    read_latency += wire_length_horizontal * unit_latency_wire
            # main bus
            read_latency += min(self.num_col - self.center_x, self.center_x) * unit_width * unit_latency_rep
        else:       # leaf-leaf communication
            # find the common ancestor stage
            # /*** firstly need to find the zone of two units ***/
			# /*** in each level, the units are defined as 4 zones, which used to decide the travel distance
			#     ______________________
			# 	|          |          |
			# 	|          |          |
			# 	|    0     |     1    |
			# 	|          |          |
			# 	|__________|__________|
			# 	|          |          |
			# 	|          |          |
			# 	|    2     |     3    |      
			# 	|          |          |
			# 	|__________|__________|                       ***/
            self.find_stage = 0
            while not hit and self.find_stage < (self.num_stage - 1) / 2:
                max_coor_diff = 2 ** ((self.num_stage - 1) / 2 - self.find_stage - 1) - 1
                if abs(x_init - x_end) > max_coor_diff or abs(y_init - y_end) > max_coor_diff:
                    #hit means the belongs to different zone in this stage, stop searching
                    hit = True
                    if abs(x_init - x_end) < max_coor_diff and abs(y_init - y_end) > max_coor_diff: #// two zone belong to same row, do not pass the longest vertical bus at this stage
                        self.skip_ver = True
                else:   #// keep searching in next stage
                    self.find_stage += 1
            #count the top self.find_stage, whether pass the vertical bus or not)
            wire_length_vertical /= 2 ** self.find_stage
            wire_length_horizontal /= 2 ** self.find_stage
            unit_latency_rep = 0.7 * (res_on_rep * (self.cap_inv_input + self.cap_inv_output + self.unit_length_wire_cap * self.min_dist) + 
                                        0.54 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist +
                                        unit_length_res * self.min_dist * self.cap_inv_input) / self.min_dist
            unit_latency_wire = (0.7 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist) / self.min_dist

            #horizontal stage
            num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
            if num_repeater > 0:
                read_latency += wire_length_horizontal * unit_latency_rep
            else:
                read_latency += wire_length_horizontal * unit_latency_wire

            if not self.skip_ver:
                #vertical stage
                num_repeater = math.ceil(wire_length_vertical / self.min_dist)
                if num_repeater > 0:
                    read_latency += wire_length_vertical * unit_latency_rep
                else:
                    read_latency += wire_length_vertical * unit_latency_wire
            #count the following stages
            for _ in range(int(self.find_stage + 1), int((self.num_stage - 1) / 2)):
                # vertical stages
                wire_length_vertical /= 2
                wire_width, unit_length_res = self.get_unit_length_res(wire_length_vertical)
                unit_latency_rep = 0.7 * (res_on_rep * (self.cap_inv_input + self.cap_inv_output + self.unit_length_wire_cap * self.min_dist) +
                                        0.54 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist +
                                        unit_length_res * self.min_dist * self.cap_inv_input) / self.min_dist
                unit_latency_wire = (0.7 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist) / self.min_dist
                num_repeater = math.ceil(wire_length_vertical / self.min_dist)
                if num_repeater > 0:
                    read_latency += wire_length_vertical * unit_latency_rep
                else:
                    read_latency += wire_length_vertical * unit_latency_wire
                # horizontal stages
                wire_length_horizontal /= 2
                wire_width, unit_length_res = self.get_unit_length_res(wire_length_horizontal)
                unit_latency_rep = 0.7 * (res_on_rep * (self.cap_inv_input + self.cap_inv_output + self.unit_length_wire_cap * self.min_dist) +
                                        0.54 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist +
                                        unit_length_res * self.min_dist * self.cap_inv_input) / self.min_dist
                unit_latency_wire = (0.7 * unit_length_res * self.min_dist * self.unit_length_wire_cap * self.min_dist) / self.min_dist
                num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
                if num_repeater > 0:
                    read_latency += wire_length_horizontal * unit_latency_rep
                else:
                    read_latency += wire_length_horizontal * unit_latency_wire

            if self.latency_mode == 'synchronous':
                read_latency = math.ceil(read_latency * self.clk_freq)

        return read_latency * num_read
    
    def calculate_power(self, x_init,y_init,x_end,y_end,unit_height,unit_width, num_Bit_Access, num_read):
        if not self.initialized:
            raise RuntimeError("[HTree] Error: Require initialization first!")

        read_energy = 0
        unit_length_leakage = logicGate.calculate_logicgate_leakage(constant.INV, 1, self.width_inv_n, self.width_inv_p,self.temp, self.tech)
        leakage = unit_length_leakage * self.total_wire_length
        unit_leagth_energy_rep = (self.cap_inv_input + self.cap_inv_output + self.unit_length_wire_cap * self.min_dist) * self.vdd * self.vdd/ self.min_dist * 0.25
        unit_leagth_energy_wire = self.unit_length_wire_cap * self.min_dist * self.vdd * self.vdd/ self.min_dist * 0.25
        wire_length_vertical = unit_height * 2 ** ((self.num_stage - 1) / 2)
        wire_length_horizontal = unit_width * 2 ** ((self.num_stage - 1) / 2)

        if (x_init == 0 and y_init == 0) or (x_end == 0 and y_end == 0):
            #// root-leaf communicate (fixed addr)
            #ignore main bus here, but need to count until last stage (diff from area calculation)
            for _ in range(int((self.num_stage - 1) / 2)):
                # vertical stages
                wire_length_vertical /= 2
                wire_width, unit_length_res = self.get_unit_length_res(wire_length_vertical)
                num_repeater = math.ceil(wire_length_vertical / self.min_dist)
                if num_repeater > 0:
                    read_energy += wire_length_vertical * unit_leagth_energy_rep
                else:
                    read_energy += wire_length_vertical * unit_leagth_energy_wire

                # horizontal stages
                wire_length_horizontal /= 2
                wire_width, unit_length_res = self.get_unit_length_res(wire_length_horizontal)
                num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
                if num_repeater > 0:
                    read_energy += wire_length_horizontal * unit_leagth_energy_rep
                else:
                    read_energy += wire_length_horizontal * unit_leagth_energy_wire

            # main bus
            read_energy += min(self.num_col - self.center_x, self.center_x) * unit_width * unit_leagth_energy_rep
            read_energy *= num_Bit_Access
        else:       # leaf-leaf communication
            #/*** count the top self.find_stage, whether pass the vertical bus or not) ***/
            wire_length_vertical /= 2 ** self.find_stage
            wire_length_horizontal /= 2 ** self.find_stage

            # horizontal stage
            num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
            if num_repeater > 0:
                read_energy += wire_length_horizontal * unit_leagth_energy_rep
            else:
                read_energy += wire_length_horizontal * unit_leagth_energy_wire
            if not self.skip_ver:
                # vertical stage
                num_repeater = math.ceil(wire_length_vertical / self.min_dist)
                if num_repeater > 0:
                    read_energy += wire_length_vertical * unit_leagth_energy_rep
                else:
                    read_energy += wire_length_vertical * unit_leagth_energy_wire
            # count the following stages

            for _ in range(int(self.find_stage + 1), int((self.num_stage - 1) / 2)):
                # vertical stages
                wire_length_vertical /= 2
                num_repeater = math.ceil(wire_length_vertical / self.min_dist)
                if num_repeater > 0:
                    read_energy += wire_length_vertical * unit_leagth_energy_rep
                else:
                    read_energy += wire_length_vertical * unit_leagth_energy_wire

                # horizontal stages
                wire_length_horizontal /= 2
                num_repeater = math.ceil(wire_length_horizontal / self.min_dist)
                if num_repeater > 0:
                    read_energy += wire_length_horizontal * unit_leagth_energy_rep
                else:
                    read_energy += wire_length_horizontal * unit_leagth_energy_wire

            read_energy *= num_Bit_Access
        return read_energy * num_read, leakage
        
    def get_unit_length_res(self, wire_length):
        # Determine wire width based on wire length
        ratio = wire_length / self.feature_size
        if ratio >= 100000:
            wire_width = 4 * self.wireWidth
        elif 10000 <= ratio <= 100000:
            wire_width = 2 * self.wireWidth
        else:
            wire_width = 1 * self.wireWidth

        # Set AR, Rho, and barrierthickness based on wireWidth
        if wire_width >= 175:
            AR = 1.6
            Rho = 2.01e-8
            barrierthickness = 10.0e-9
        elif 110 <= wire_width < 175:
            AR = 1.6
            Rho = 2.20e-8
            barrierthickness = 10.0e-9
        elif 105 <= wire_width < 110:
            AR = 1.7
            Rho = 2.21e-8
            barrierthickness = 7.0e-9
        elif 80 <= wire_width < 105:
            AR = 1.7
            Rho = 2.37e-8
            barrierthickness = 5.0e-9
        elif 56 <= wire_width < 80:
            AR = 1.8
            Rho = 2.63e-8
            barrierthickness = 4.0e-9
        elif 40 <= wire_width < 56:
            AR = 1.9
            Rho = 2.97e-8
            barrierthickness = 3.0e-9
        elif 32 <= wire_width < 40:
            AR = 2.0
            Rho = 3.25e-8
            barrierthickness = 2.5e-9
        elif 22 <= wire_width < 32:
            AR = 2.0
            Rho = 3.95e-8
            barrierthickness = 2.5e-9
        elif 20 <= wire_width < 22:
            AR = 2.0
            Rho = 4.17e-8
            barrierthickness = 2.5e-9
        elif 15 <= wire_width < 20:
            AR = 2.0
            Rho = 4.98e-8
            barrierthickness = 2.0e-9
        elif 12 <= wire_width < 15:
            AR = 2.0
            Rho = 5.8e-8
            barrierthickness = 1.5e-9
        elif 10 <= wire_width < 12:
            AR = 3.0
            Rho = 6.65e-8
            barrierthickness = 0.5e-9
        elif 8 <= wire_width < 10:
            AR = 3.0
            Rho = 7.87e-8
            barrierthickness = 0.5e-9
        else:
            raise ValueError("Wire width out of range")

        # Adjust Rho based on barrier thickness
        Rho /= (1 - ((2 * AR * wire_width + wire_width) * barrierthickness / (AR * (wire_width ** 2))))

        # Adjust for temperature
        Rho *= (1 + 0.00451 * (self.temp - 300))

        # Calculate unit length wire resistance
        if wire_width == -1:
            unit_length_wire_resistance = 1.0
        else:
            wire_width_m = wire_width * 1e-9
            unit_length_wire_resistance = Rho / (wire_width_m * wire_width_m * AR)

        return wire_width, unit_length_wire_resistance



