import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology


class WLNewDecoderDriver:
    def __init__(self, tech, param, config, mapping,numWLRow):
        self.tech = tech
        self.param = param
        self.config = config
        self.mapping = mapping

        self.featureSize = self.tech.get_param('featureSize')
        self.vdd = self.tech.get_param('vdd')
        self.writeVoltage = self.param['writeVoltage']
        self.readVoltage = self.param['readVoltage']
        self.temp = self.config['temperature']

        self.num_WL_row = numWLRow

        # NAND2
        self.width_nand_n = 2 * constant.MIN_NMOS_SIZE * self.featureSize
        self.width_nand_p = self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.featureSize

        # INV
        self.width_inv_n = constant.MIN_NMOS_SIZE * self.featureSize
        self.width_inv_p = self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.featureSize

        # TG
        self.width_tg_n = constant.MIN_NMOS_SIZE * self.featureSize
        self.width_tg_p = self.tech.get_param('pnSizeRatio') * constant.MIN_NMOS_SIZE * self.featureSize

        self.cap_nand_input = 0
        self.cap_nand_output = 0
        self.cap_inv_input = 0
        self.cap_inv_output = 0
        self.cap_tg_gate_n = 0
        self.cap_tg_gate_p = 0
        self.cap_tg_drain = 0
        self.res_tg = 0

    def calculate_area(self, new_height, new_width, option):
        MAGIC = 'MAGIC'
        OVERRIDE = 'OVERRIDE'
        NONE = 'NONE'

        min_cell_height = constant.MAX_TRANSISTOR_HEIGHT * self.featureSize
        node_adjust = {
            14e-9: constant.MAX_TRANSISTOR_HEIGHT_14nm,
            10e-9: constant.MAX_TRANSISTOR_HEIGHT_10nm,
            7e-9: constant.MAX_TRANSISTOR_HEIGHT_7nm,
            5e-9: constant.MAX_TRANSISTOR_HEIGHT_5nm,
            3e-9: constant.MAX_TRANSISTOR_HEIGHT_3nm,
            1e-9: constant.MAX_TRANSISTOR_HEIGHT_1nm
        }

        if self.featureSize in node_adjust:
            min_cell_height *= node_adjust[self.featureSize] / constant.MAX_TRANSISTOR_HEIGHT

        # Gate areas
        h_nand, w_nand, _ = logicGate.calculate_logicgate_area(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p, min_cell_height, self.tech)
        h_inv, w_inv, _ = logicGate.calculate_logicgate_area(
            constant.INV, 1, self.width_inv_n, self.width_inv_p, min_cell_height, self.tech)
        h_tg, w_tg, _ = logicGate.calculate_logicgate_area(
            constant.INV, 1, self.width_tg_n, self.width_tg_p, min_cell_height, self.tech)

        if new_height and option == NONE:
            Tg_height = new_height / self.num_WL_row
            if Tg_height < min_cell_height:
                num_col_tg = math.ceil(min_cell_height / Tg_height)
                if num_col_tg > self.num_WL_row:
                    raise ValueError("Pass gate height exceeds array height.")
                Tg_height = new_height / math.ceil(self.num_WL_row / num_col_tg)

            h_tg, w_tg, _ = logicGate.calculate_logicgate_area(
                constant.INV, 1, self.width_tg_n, self.width_tg_p, Tg_height, self.tech)

            h_unit = max(h_nand, h_inv, h_tg)
            w_unit = 3 * w_nand + w_inv + 2 * w_tg
            num_unit_per_col = int(new_height // h_unit)
            num_col_unit = math.ceil(self.num_WL_row / num_unit_per_col)

            self.height = new_height
            self.width = w_unit * num_col_unit

        else:
            self.height = max(h_nand, h_inv, h_tg) * self.num_WL_row
            self.width = 3 * w_nand + w_inv + 2 * w_tg

        self.area = self.height * self.width

        # Resistance
        res_tg_n = logicGate.calculate_on_resistance(self.width_tg_n, constant.NMOS, self.temp, self.tech) * constant.LINEAR_REGION_RATIO
        res_tg_p = logicGate.calculate_on_resistance(self.width_tg_p, constant.PMOS, self.temp, self.tech) * constant.LINEAR_REGION_RATIO
        self.res_tg = 1 / (1 / res_tg_n + 1 / res_tg_p)

        # Capacitance
        self.cap_nand_input, self.cap_nand_output = logicGate.calculate_logicgate_cap(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p, h_nand, self.tech)
        
        self.cap_inv_input, self.cap_inv_output = logicGate.calculate_logicgate_cap(
            constant.INV, 1, self.width_inv_n, self.width_inv_p, h_inv, self.tech)
        self.cap_tg_gate_n = logicGate.calculate_mos_gate_cap(self.width_tg_n, self.tech)
        self.cap_tg_gate_p = logicGate.calculate_mos_gate_cap(self.width_tg_p, self.tech)
        _, self.cap_tg_drain = logicGate.calculate_logicgate_cap(
            constant.INV, 1, self.width_tg_n, self.width_tg_p, h_tg, self.tech)

        return self.area, self.height, self.width

    def calculate_latency(self, cap_load, res_load, num_read, num_write):
        ramp_input = 1e20
        read_latency = write_latency = 0

        # Stage 1: NAND
        res_pd = logicGate.calculate_on_resistance(self.width_nand_n, constant.NMOS, self.temp, self.tech) * 2  #// pulldown 2 NMOS in series
        tr_nand = res_pd * (self.cap_nand_output + self.cap_inv_input)          #// connect to INV
        gm_nand = logicGate.calculate_transconductance(self.width_nand_n, constant.NMOS, self.tech)
        beta_nand = 1 / (res_pd * gm_nand)
        read_latency += logicGate.horowitz(tr_nand, beta_nand, ramp_input)[0]
        write_latency += logicGate.horowitz(tr_nand, beta_nand, ramp_input)[0]

        # Stage 2: INV
        res_pu = logicGate.calculate_on_resistance(self.width_inv_p, constant.PMOS, self.temp, self.tech)
        tr_inv = res_pu * (self.cap_inv_output + 2 * self.cap_nand_input)
        gm_inv = logicGate.calculate_transconductance(self.width_inv_p, constant.PMOS, self.tech)
        beta_inv = 1 / (res_pu * gm_inv)
        read_latency += logicGate.horowitz(tr_inv, beta_inv, ramp_input)[0]
        write_latency += logicGate.horowitz(tr_inv, beta_inv, ramp_input)[0]

        # Stage 3: NAND
        res_pd = logicGate.calculate_on_resistance(self.width_nand_n, constant.NMOS, self.temp, self.tech) * 2
        tr_nand2 = res_pd * (self.cap_nand_output + self.cap_tg_gate_n + self.cap_tg_gate_p)            #// connect to 2 transmission gates
        gm_nand2 = logicGate.calculate_transconductance(self.width_nand_n, constant.NMOS, self.tech)
        beta_nand2 = 1 / (res_pd * gm_nand2)
        read_latency += logicGate.horowitz(tr_nand2, beta_nand2, ramp_input)[0]
        write_latency += logicGate.horowitz(tr_nand2, beta_nand2, ramp_input)[0]

        # Stage 4: TG
        cap_output = 2 * self.cap_tg_drain
        tr_tg = self.res_tg * (cap_output + cap_load) + res_load * cap_load / 2
        read_latency += logicGate.horowitz(tr_tg, 0, 1e20)[0]
        write_latency += logicGate.horowitz(tr_tg, 0, 1e20)[0]

        return read_latency * num_read, write_latency * num_write

    def calculate_power(self, num_read, num_write):
        # Leakage
        leakage = 0
        leakage += logicGate.calculate_logicgate_leakage(constant.NAND, 2, self.width_nand_n, self.width_nand_p,
                                                    self.temp, self.tech) * self.vdd * self.num_WL_row * 2
        leakage += logicGate.calculate_logicgate_leakage(constant.INV, 1, self.width_inv_n, self.width_inv_p,
                                                    self.temp, self.tech) * self.vdd * self.num_WL_row * 2

        # Read energy
        read_energy = 0
        read_energy += self.cap_nand_input * self.vdd**2                                                #// NAND2 input charging ( 0 to 1 )
        read_energy += (self.cap_inv_output + self.cap_tg_gate_n) * self.vdd**2                         #// INV output charging ( 0 to 1 )
        read_energy += (self.cap_nand_output + self.cap_tg_gate_n + self.cap_tg_gate_p) * self.vdd**2   #// NAND2 output charging ( 0 to 1 )
        read_energy += self.cap_tg_drain * self.readVoltage**2                        #// TG gate energy
        read_energy *= num_read

        # Write energy
        write_energy = 0
        write_energy += self.cap_nand_input * self.vdd**2
        write_energy += (self.cap_inv_output + self.cap_tg_gate_n) * self.vdd**2
        write_energy += (self.cap_nand_output + self.cap_tg_gate_n + self.cap_tg_gate_p) * self.vdd**2
        write_energy += self.cap_tg_drain * self.writeVoltage**2
        write_energy *= num_write

        return read_energy, write_energy, leakage


