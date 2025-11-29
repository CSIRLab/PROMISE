import numpy as np
import math
import sys
import yaml
sys.path.append('../../python/')  
from periphery import constant
from periphery import logicGate
from periphery.Technology import Technology

class RowDecoder:
    def __init__(self, mode, num_addr_row, mux, parallel, tech, config,mapping):
        self.tech = tech
        self.config = config
        self.mapping = mapping
        self.initialized = False

        self.mode = mode
        self.numAddrRow = num_addr_row
        self.MUX = mux
        self.parallel = parallel
        self.featureSize = self.tech.get_param('featureSize') 
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')

        scale = 8 if parallel else 1

        # use 2 bit predecoder for 2^n address lines
        # INV
        self.width_inv_n = scale * constant.MIN_NMOS_SIZE * self.featureSize
        self.width_inv_p = scale * self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.featureSize
        self.num_inv = num_addr_row # the INV at output driver stage does not count here

        # NAND2
        self.width_nand_n = scale * 2 * constant.MIN_NMOS_SIZE  * self.featureSize
        self.width_nand_p = scale * self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.featureSize
        self.num_nand = 4 * int(math.floor(num_addr_row / 2))

        # NOR2
        self.width_nor_n = scale * constant.MIN_NMOS_SIZE * self.featureSize
        self.num_input_nor = int(math.ceil(num_addr_row / 2))
        self.width_nor_p = scale * self.num_input_nor * self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.featureSize
        self.num_nor = int(2 ** num_addr_row) if num_addr_row > 2 else 0

        self.num_metal_connection = self.num_nand + (num_addr_row % 2) * 2 if num_addr_row > 2 else 0

        self.width_driver_inv_n = scale * 3 * constant.MIN_NMOS_SIZE * self.featureSize
        self.width_driver_inv_p = scale * 3 * self.pnSizeRatio * constant.MIN_NMOS_SIZE * self.featureSize
        self.initialized = True

    def calculate_area(self, new_height, new_width, option):
        if not self.initialized:
            raise RuntimeError("RowDecoder must be initialized before area calculation.")

        # Magic layout options
        MAGIC, OVERRIDE, NONE = 'MAGIC', 'OVERRIDE', 'NONE'
        Metal2_pitch = constant.M2_PITCH
        Metal3_pitch = constant.M3_PITCH
        if self.featureSize == 14e-9:
            Metal2_pitch = constant.M2_PITCH_14nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_14nm/Metal3_pitch
        elif self.featureSize == 10e-9:
            Metal2_pitch = constant.M2_PITCH_10nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_10nm/Metal3_pitch
        elif self.featureSize == 7e-9:
            Metal2_pitch = constant.M2_PITCH_7nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_7nm/Metal3_pitch
        elif self.featureSize == 5e-9:
            Metal2_pitch = constant.M2_PITCH_5nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_5nm/Metal3_pitch
        elif self.featureSize == 3e-9:
            Metal2_pitch = constant.M2_PITCH_3nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_3nm/Metal3_pitch
        elif self.featureSize == 2e-9:
            Metal2_pitch = constant.M2_PITCH_2nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_2nm/Metal3_pitch
        elif self.featureSize == 1e-9:
            Metal2_pitch = constant.M2_PITCH_1nm/Metal2_pitch
            Metal3_pitch = constant.M3_PITCH_1nm/Metal3_pitch

        # Gate areas
        h_inv, w_inv, _ = logicGate.calculate_logicgate_area(constant.INV,1, self.width_inv_n , self.width_inv_p, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        h_nand, w_nand, _ = logicGate.calculate_logicgate_area(constant.NAND,2, self.width_nand_n, self.width_nand_p, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        h_nor, w_nor, _ = logicGate.calculate_logicgate_area(constant.NOR,self.num_input_nor, self.width_nor_n , self.width_nor_p, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT, self.tech)
        h_driver_inv, w_driver_inv, _ = logicGate.calculate_logicgate_area(constant.INV, 1, self.width_driver_inv_n , self.width_driver_inv_p, self.featureSize * constant.MAX_TRANSISTOR_HEIGHT, self.tech)

        if self.mode == "REGULAR_ROW":
            if new_height and option == NONE:
                if max(h_inv, h_nand, h_nor) > new_height:
                    raise ValueError("RowDecoder cell height exceeds assigned layout height.")

                # NOR
                num_nor_per_col = int(new_height / h_nor)
                num_nor_per_col = min(num_nor_per_col, self.num_nor)
                num_col_nor = math.ceil(self.num_nor / num_nor_per_col) if num_nor_per_col > 0 else 0

                # NAND
                num_nand_per_col = int(new_height / h_nand)
                num_nand_per_col = min(num_nand_per_col, self.num_nand)
                num_col_nand = math.ceil(self.num_nand / num_nand_per_col) if num_nand_per_col > 0 else 0

                # INV
                num_inv_per_col = int(new_height / h_inv)
                num_inv_per_col = min(num_inv_per_col, self.num_inv)
                num_col_inv = math.ceil(self.num_inv / num_inv_per_col)

                height = new_height
                width = (
                    w_inv * num_col_inv +
                    w_nand * num_col_nand +
                    Metal3_pitch * self.num_metal_connection * self.featureSize +
                    w_nor * num_col_nor
                )
                if self.MUX:
                    width += (w_nand + w_inv * 2) * num_col_nor
                else:
                    width += w_driver_inv * 2 * num_col_nor
            else:
                height = max(h_nor * self.num_nor, h_nand * self.num_nand)
                width = (
                    w_inv + w_nand +
                    Metal3_pitch * self.num_metal_connection * self.featureSize +
                    w_nor
                )
                if self.MUX:
                    width += w_nand + w_inv * 2
                else:
                    width += w_driver_inv * 2

        elif self.mode == "REGULAR_COL":
            if new_width and option == NONE:
                if max(w_inv, w_nand, w_nor) > new_width:
                    raise ValueError("RowDecoder cell width exceeds assigned layout width.")

                # NOR
                num_nor_per_row = int(new_width / w_nor)
                num_nor_per_row = min(num_nor_per_row, self.num_nor)
                num_row_nor = math.ceil(self.num_nor / num_nor_per_row) if num_nor_per_row > 0 else 0

                # NAND
                num_nand_per_row = int(new_width / w_nand)
                num_nand_per_row = min(num_nand_per_row, self.num_nand)
                num_row_nand = math.ceil(self.num_nand / num_nand_per_row) if num_nand_per_row > 0 else 0

                # INV
                num_inv_per_row = int(new_width / w_inv)
                num_inv_per_row = min(num_inv_per_row, self.num_inv)
                num_row_inv = math.ceil(self.num_inv / num_inv_per_row)

                width = new_width
                height = (
                    h_inv * num_row_inv +
                    h_nand * num_row_nand +
                    Metal2_pitch * self.num_metal_connection * self.featureSize +
                    h_nor * num_row_nor
                )
                if self.MUX:
                    height += (h_nand + h_inv * 2) * num_row_nor
                else:
                    height += h_driver_inv * 2 * num_row_nor
            else:
                height = (
                    h_inv + h_nand +
                    Metal2_pitch * self.num_metal_connection * self.featureSize +
                    h_nor
                )
                width = max(w_nor * self.num_nor, w_nand * self.num_nand)
                if self.MUX:
                    height += h_nand + h_inv * 2
                else:
                    height += h_driver_inv * 2

        else:
            raise ValueError("Unsupported mode for RowDecoder.")

        area = height * width

        # capacitance
        # inv
        self.cap_inv_input, self.cap_inv_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_inv_n, self.width_inv_p, h_inv, self.tech)
        # nand
        if self.num_nand > 0:
            self.cap_nand_input, self.cap_nand_output = logicGate.calculate_logicgate_cap(constant.NAND, 2, self.width_nand_n, self.width_nand_p, h_nand, self.tech)
        else:
            self.cap_nand_input, self.cap_nand_output = 0, 0
        # nor
        if self.num_nor > 0:
            self.cap_nor_input, self.cap_nor_output = logicGate.calculate_logicgate_cap(constant.NOR, self.num_input_nor, self.width_nor_n, self.width_nor_p, h_nor, self.tech)
        else:
            self.cap_nor_input, self.cap_nor_output = 0, 0

        # driver inv
        self.cap_driver_inv_input, self.cap_driver_inv_output = logicGate.calculate_logicgate_cap(constant.INV, 1, self.width_driver_inv_n, self.width_driver_inv_p, h_driver_inv, self.tech)
        return area, height, width

    def calculate_latency(self, cap_load1, cap_load2, num_read, num_write):
        if not self.initialized:
            raise RuntimeError("RowDecoder must be initialized before latency calculation.")

        read_latency = 0
        write_latency = 0

        ramp_inv_output = 1e20
        ramp_nand_output = 1e20
        ramp_nor_output = 1e20
        ramp_output = 1e20

        # --- INV stage ---
        res_pull_down = logicGate.calculate_on_resistance(self.width_inv_n, constant.NMOS, self.config['temperature'], self.tech)
        if self.num_nand:
            tr = res_pull_down * (self.cap_inv_output + self.cap_nand_input * 2) # one address line connects to 2 NAND inputs
        else:
            tr = res_pull_down * (self.cap_inv_output + cap_load1)
        gm = logicGate.calculate_transconductance(self.width_inv_n, constant.NMOS, self.tech)
        beta = 1 / (res_pull_down * gm)
        read_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]
        write_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]
        if not self.num_nand:
            ramp_output = ramp_inv_output

        # --- NAND stage ---
        if self.num_nand:
            res_pull_down = logicGate.calculate_on_resistance(self.width_nand_n, constant.NMOS, self.config['temperature'], self.tech) * 2
            if self.num_nor:
                tr = res_pull_down * (self.cap_nand_output + self.cap_nor_input * self.num_nor / 4)
            else:
                tr = res_pull_down * (self.cap_nand_output + cap_load1)
            gm = logicGate.calculate_transconductance(self.width_nand_n, constant.NMOS, self.tech)
            beta = 1 / (res_pull_down * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]
            if not self.num_nor:
                ramp_output = ramp_nand_output

        # --- NOR stage ---
        if self.num_nor:
            res_pull_up = logicGate.calculate_on_resistance(self.width_nor_p, constant.PMOS, self.config['temperature'], self.tech) * 2
            if self.MUX:
                tr = res_pull_up * (self.cap_nor_output + self.cap_nand_input)
            else:
                tr = res_pull_up * (self.cap_nor_output + self.cap_inv_input)
            gm = logicGate.calculate_transconductance(self.width_nor_p, constant.PMOS, self.tech)
            beta = 1 / (res_pull_up * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_nand_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_nand_output)[0]
            ramp_output = ramp_nor_output

        # --- Output driver or MUX ---
        if self.MUX:
            # NAND
            res_pull_down = logicGate.calculate_on_resistance(self.width_nand_n, constant.NMOS, self.config['temperature'], self.tech)
            tr = res_pull_down * (self.cap_nand_output + self.cap_inv_input)
            gm = logicGate.calculate_transconductance(self.width_nand_n, constant.NMOS, self.tech)
            beta = 1 / (res_pull_down * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_output)[0]

            # 1st INV
            res_pull_up = logicGate.calculate_on_resistance(self.width_inv_p, constant.PMOS, self.config['temperature'], self.tech)
            tr = res_pull_up * (self.cap_inv_output + self.cap_inv_input + cap_load1)
            gm = logicGate.calculate_transconductance(self.width_inv_p, constant.PMOS, self.tech)
            beta = 1 / (res_pull_up * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_nand_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_nand_output)[0]

            # 2nd INV
            res_pull_down = logicGate.calculate_on_resistance(self.width_inv_n, constant.NMOS, self.config['temperature'], self.tech)
            tr = res_pull_down * (self.cap_inv_output + cap_load2)
            gm = logicGate.calculate_transconductance(self.width_inv_n, constant.NMOS, self.tech)
            beta = 1 / (res_pull_down * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]

        else:
            # REGULAR: 2 INV as output driver
            res_pull_down = logicGate.calculate_on_resistance(self.width_driver_inv_n, constant.NMOS, self.config['temperature'], self.tech)
            tr = res_pull_down * (self.cap_driver_inv_output + self.cap_driver_inv_input)
            gm = logicGate.calculate_transconductance(self.width_driver_inv_n, constant.NMOS, self.tech)
            beta = 1 / (res_pull_down * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_output)[0]

            res_pull_up = logicGate.calculate_on_resistance(self.width_driver_inv_p, constant.PMOS, self.config['temperature'], self.tech)
            tr = res_pull_up * (self.cap_driver_inv_output + cap_load1)
            gm = logicGate.calculate_transconductance(self.width_driver_inv_p, constant.PMOS, self.tech)
            beta = 1 / (res_pull_up * gm)
            read_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]
            write_latency += logicGate.horowitz(tr, beta, ramp_inv_output)[0]

        read_latency *= num_read
        write_latency *= num_write

        return read_latency, write_latency
    

    def calculate_power(self, num_read, num_write):
        if not self.initialized:
            raise RuntimeError("RowDecoder must be initialized before power calculation.")

        self.leakage = 0
        self.read_dynamic_energy = 0
        self.write_dynamic_energy = 0
        vdd = self.tech.get_param('vdd')

        # Leakage
        # INV
        self.leakage += logicGate.calculate_logicgate_leakage(
            constant.INV, 1, self.width_inv_n, self.width_inv_p, self.config['temperature'], self.tech) * vdd * self.num_inv

        self.leakage += logicGate.calculate_logicgate_leakage(
            constant.NAND, 2, self.width_nand_n, self.width_nand_p, self.config['temperature'], self.tech) * vdd * self.num_nand

        self.leakage += logicGate.calculate_logicgate_leakage(
            constant.NOR, self.num_input_nor, self.width_nor_n, self.width_nor_p, self.config['temperature'], self.tech) * vdd * self.num_nor

        # Output driver or MUX selector
        if self.MUX:
            self.leakage += logicGate.calculate_logicgate_leakage(
                constant.NAND, 2, self.width_nand_n, self.width_nand_p, self.config['temperature'], self.tech) * vdd * self.num_nor
            self.leakage += logicGate.calculate_logicgate_leakage(
                constant.INV, 1, self.width_inv_n, self.width_inv_p, self.config['temperature'], self.tech) * vdd * 2 * self.num_nor
        else:
            self.leakage += logicGate.calculate_logicgate_leakage(
                constant.INV, 1, self.width_driver_inv_n, self.width_driver_inv_p, self.config['temperature'], self.tech) * vdd * 2 * self.num_nor

        # Dynamic energy calculation
        floor_half = int(self.numAddrRow // 2)
        read_factor = vdd ** 2

        # --- Read energy ---
        # INV stage
        self.read_dynamic_energy += (
            (self.cap_inv_input + self.cap_nand_input * 2) * read_factor * floor_half * 2
        )
        self.read_dynamic_energy += (
            (self.cap_inv_input + self.cap_nor_input * (self.num_nor / 2)) * read_factor * (self.numAddrRow - floor_half * 2)
        )
        # NAND stage
        self.read_dynamic_energy += (
            (self.cap_nand_output + self.cap_nor_input * self.num_nor / 4) * read_factor * self.num_nand / 4
        )

        # NOR stage
        if self.MUX:
            self.read_dynamic_energy += (self.cap_nor_output + self.cap_nand_input) * read_factor   #one NOR output activated
        else:
            self.read_dynamic_energy += (self.cap_nor_output + self.cap_inv_input) * read_factor    #one NOR output activated
        # Output driver or MUX selector
        if self.MUX:
            self.read_dynamic_energy += (self.cap_nand_output + self.cap_inv_input) * read_factor
            self.read_dynamic_energy += (self.cap_inv_output + self.cap_inv_input) * read_factor
            self.read_dynamic_energy += self.cap_inv_output * read_factor
        else:
            self.read_dynamic_energy += (self.cap_driver_inv_input + self.cap_driver_inv_output) * read_factor * 2

        # --- Write energy (same pattern) ---
        # INV stage
        self.write_dynamic_energy += (
            (self.cap_inv_input + self.cap_nand_input * 2) * read_factor * floor_half * 2
        )
        self.write_dynamic_energy += (
            (self.cap_inv_input + self.cap_nor_input * (self.num_nor / 2)) * read_factor * (self.numAddrRow - floor_half * 2)
        )
        # NAND stage
        self.write_dynamic_energy += (
            (self.cap_nand_output + self.cap_nor_input * self.num_nor / 4) * read_factor * self.num_nand / 4
        )
        # NOR stage
        if self.MUX:
            self.write_dynamic_energy += (self.cap_nor_output + self.cap_nand_input) * read_factor
            # Output driver or MUX selector
            self.write_dynamic_energy += (self.cap_nand_output + self.cap_inv_input) * read_factor
            self.write_dynamic_energy += (self.cap_inv_output + self.cap_inv_input) * read_factor
            self.write_dynamic_energy += self.cap_inv_output * read_factor
        else:
            self.write_dynamic_energy += (self.cap_nor_output + self.cap_inv_input) * read_factor
            self.write_dynamic_energy += (self.cap_driver_inv_input + self.cap_driver_inv_output) * read_factor * 2

        # Scale by access counts
        self.read_dynamic_energy *= num_read
        self.write_dynamic_energy *= num_write
        return self.read_dynamic_energy, self.write_dynamic_energy, self.leakage
