import math
import sys
import yaml
import numpy as np
sys.path.append('../../python/')  
from periphery import logicGate
from periphery import constant
from periphery.Technology import Technology
from periphery.sramWriteDriver import SRAMWriteDriver
from periphery.precharger import Precharger
from periphery.WLdecoder import RowDecoder
from periphery.SenseAmp import SenseAmp
from periphery.DFF import DFF
from periphery.MUX import Mux
from periphery.levelShifter import LevelShifter
from periphery.WLDecoderDriver import WLNewDecoderDriver
from periphery.adder import Adder
from periphery.ADC import SarADC
from periphery.RNGBlock import RNG_block
from periphery.MultiLevelSenseAmp import MultilevelSenseAmp

class SubArray:
    def __init__(self, tech, param, config, mapping, RNG, numRow, numCol,array_x_overlap, array_y_overlap, num_mu, num_sigma, relaxArrayCellWidth = False,relaxArrayCellHeight = False):
        self.tech = tech
        self.param = param  
        self.config = config 
        self.mapping = mapping
        self.RNG = RNG
        self.relaxArrayCellWidth = relaxArrayCellWidth
        self.relaxArrayCellHeight = relaxArrayCellHeight

        self.numRow = numRow
        self.numCol = numCol
        self.num_mu = num_mu
        self.num_sigma = num_sigma
        self.array_x_overlap = array_x_overlap
        self.array_y_overlap = array_y_overlap

        self.feature_size = self.tech.get_param('featureSize')
        self.PitchFin = self.tech.get_param('PitchFin')
        self.widthFin = self.tech.get_param('widthFin')
        self.heightFin = self.tech.get_param('heightFin')
        self.vdd = self.tech.get_param('vdd')
        self.temp = self.config['temperature']
        self.cell_type = self.config['device_type']
        self.precision_sigma = self.config['precision_sigma']
        self.precision_ADC = self.config['precision_ADC']
        self.clk_freq = self.config['frequency']

        min_cell_height = constant.MAX_TRANSISTOR_HEIGHT
        min_cell_width = constant.MIN_GAP_BET_GATE_POLY + constant.POLY_WIDTH *2
        siolation_region = constant.MIN_POLY_EXT_DIFF *2 + constant.MIN_GAP_BET_FIELD_POLY

        self.sram_mode = self.param['operationmode']
        self.unitWireRes = self.param['unitLengthWireResistance']
        self.widthInFeatureSize = self.param['widthInFeatureSize']
        self.heightInFeatureSize = self.param['heightInFeatureSize']
        self.widthAccessCMOS = self.param['widthAccessCMOS']
        self.widthSRAMCellPMOS = self.param['widthSRAMCellPMOS']
        self.widthSRAMCellNMOS = self.param['widthSRAMCellNMOS']
        self.minSenseVoltage = self.param['minSenseVoltage']
        self.wireResistanceRowPerCell = self.param['wireResistanceRowPerCell']
        self.wireResistanceColPerCell = self.param['wireResistanceColPerCell']
        self.resistanceOn = self.param['resistanceOn']
        self.resistanceOff = self.param['resistanceOff']

        self.writeVoltage = self.param['writeVoltage']
        self.readVoltage = self.param['readVoltage']
        self.accessVoltage = self.param['accessVoltage']
        self.avgWeightBit = self.param['cellBit']
        self.accesstype = self.param['accesstype']
        self.mem_mode = self.param['operationmode']
        self.cell_bit = self.param['cellBit']
        
        self.activity_col_write = self.mapping['activity_col_write']
        self.activity_row_write = self.mapping['activity_row_write']
        self.activity_col_read = self.mapping['activity_col_read']
        self.activity_row_read = self.mapping['activity_row_read']
        self.num_read_cell_op =  self.mapping['num_read_cell_op']
        self.num_write_cell_op =  self.mapping['num_write_cell_op']
        self.numRowParallel = self.mapping['numRowParallel']
        self.numColMuxed = self.mapping['numColMuxed']

        #RNG
        self.RNG_height = self.RNG['Hight']
        self.RNG_width = self.RNG['Width']
        self.RNG_energy_per_sample = self.RNG['EnergyPerSample']
        self.RNG_throughput = self.RNG['Throughtput']
        self.RNG_SamplingMode = self.RNG['SamplingMode']

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

        if self.cell_type == 'SRAM':
            if self.relaxArrayCellWidth:    #if want to relax the cell width
                width = max(self.widthInFeatureSize, min_cell_width)
                height = max(self.heightInFeatureSize, min_cell_height)
            else:
                width = self.widthInFeatureSize
            if self.relaxArrayCellHeight:
                height = max(self.heightInFeatureSize, min_cell_height)
            else:
                height = self.heightInFeatureSize

            self.lengthRow = numCol * width * self.feature_size 
            self.lengthRow = self.lengthRow - width * (numCol - 1)* self.array_x_overlap * self.feature_size
            self.lengthCol = numRow * height * self.feature_size
            self.lengthCol = self.lengthCol - height * (numRow - 1)* self.array_y_overlap * self.feature_size
            # #consider RNG
            # if self.RNG_SamplingMode == 'parallel':
            #     num_RNG_per_row = math.floor(self.numCol/self.precision_sigma)  #17/4 = 4
            #     word_width = self.precision_sigma * self.lengthRow/numCol
            #     word_height = self.lengthCol/numRow
            #     _, (sigm_epsilon_w, sigm_epsilon_h, desc) = self.min_combined_area(word_width, word_height, self.RNG_width, self.RNG_height)
                
            #     self.lengthCol = sigm_epsilon_w * num_RNG_per_row
            #     self.lengthRow = sigm_epsilon_h * numRow

            self.arraywidthunit = self.widthInFeatureSize * self.feature_size
            self.arrayheightunit = numRow * self.heightInFeatureSize * self.feature_size
        elif self.cell_type in ['RRAM', 'FeFET']:
            cellWidth = self.widthInFeatureSize
            cellHeight = self.heightInFeatureSize
            width = max(cellWidth, self.min_cell_width * 2)      # Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
            height = max(cellHeight, self.min_cell_height)
            self.lengthRow = numCol * cellWidth * self.feature_size
            self.lengthCol = numRow * cellHeight * self.feature_size
            if self.accesstype == 'CMOS_access':
                if self.relaxArrayCellWidth:
                    self.lengthRow = numCol * width * self.feature_size
                else:
                    self.lengthRow = numCol * cellWidth * self.feature_size
                if self.relaxArrayCellHeight:
                    self.lengthCol = numRow * height * self.feature_size
                else:
                    self.lengthCol = numRow * cellHeight * self.feature_size
            else:   
                if self.relaxArrayCellWidth:
                    self.lengthRow = numCol * width * self.feature_size
                else:
                    self.lengthRow = numCol * cellWidth * self.feature_size
                if self.relaxArrayCellHeight:
                    self.lengthCol = numRow * height * self.feature_size
                else:
                    self.lengthCol = numRow * cellHeight * self.feature_size

            # #consider RNG
            # if self.RNG_SamplingMode == 'parallel':
            #     num_RNG_per_row = math.floor(self.numCol/self.precision_sigma)  #17/4 = 4
            #     word_width = self.precision_sigma * self.lengthRow/numCol
            #     word_height = self.lengthCol/numRow
            #     _, (sigm_epsilon_w, sigm_epsilon_h, desc) = self.min_combined_area(word_width, word_height, self.RNG_width, self.RNG_height)
                
            #     self.lengthCol = sigm_epsilon_w * num_RNG_per_row
            #     self.lengthRow = sigm_epsilon_h * numRow

            # self.arraywidthunit = cellWidth * self.feature_size
            # self.arrayheightunit = numRow * cellHeight * self.feature_size

        print("lengthcol:", self.lengthCol)
        # cap and resistance calculation
        self.capRow1 = self.lengthRow * 0.2e-15 / 1e-6      # BL for 1T1R, WL for Cross-point and SRAM
        self.capRow2 = self.capRow1                         # WL for 1T1R
        self.capCol = self.lengthCol * 0.2e-15 / 1e-6
        self.resRow = self.lengthRow * self.param['Metal1_unitwireresis']
        self.resCol = self.lengthCol * self.param['Metal0_unitwireresis']
        self.output_levle = (1 << self.precision_ADC) - 1
        self.output_levle_mu_rram = (1 << self.cell_bit) - 1
        #start to initialize the subarray modules
        if self.cell_type == 'SRAM':
            #firstly calculate the CMOS resistance and capacitance
            scale_factor = 2 if self.feature_size <= 14e-9 else 1
            effective_width = self.widthAccessCMOS * scale_factor * self.feature_size

            effective_width_nmos = self.widthSRAMCellNMOS * scale_factor * self.feature_size
            effective_width_pmos = self.widthSRAMCellPMOS * scale_factor * self.feature_size

            # resCellAccess = logicGate.calculate_on_resistance(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
            self.resCellAccess = logicGate.calculate_on_resistance(effective_width,constant.NMOS,self.temp, self.tech)
            self.capCellAccess = logicGate.calculate_drain_cap(constant.NMOS, effective_width, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
             
            #cap of the Q and Qb node
            self.capSRAMCell = (self.capCellAccess + logicGate.calculate_drain_cap(constant.NMOS, effective_width_nmos, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech) 
                        + logicGate.calculate_drain_cap(constant.PMOS, effective_width_pmos, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech) 
                        + logicGate.calculate_mos_gate_cap(effective_width_nmos, self.tech) + logicGate.calculate_mos_gate_cap(effective_width_pmos, self.tech))
            
            unit_cap = self.capRow1/self.param['numColSubArray']
            unit_res = self.resRow/self.param['numColSubArray']
            if self.feature_size <= 14e-9:
                # self.capCol += self.tech['cap_draintotal'] * self.cell.widthAccessCMOS * self.tech['effective_width'] * self.numRow
                print("to be done")
            else:
                drain_cap = logicGate.calculate_drain_cap(constant.NMOS, effective_width, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech)
                self.capCol += drain_cap * self.numRow

            ##########################################################################
            #initialize the periphery
            ##########################################################################
            self.sram_write_driver = SRAMWriteDriver(num_col=self.numCol,res_load=self.resCol,tech=self.tech,config=self.config,mapping=self.mapping)
            self.precharger = Precharger(num_col=self.numCol,res_load=self.resCol,tech=self.tech,config=self.config,mapping=self.mapping)
            self.row_decoder = RowDecoder(mode="REGULAR_ROW",num_addr_row=int(math.ceil(math.log2(self.numRow))),mux=False,parallel=False,tech=self.tech,config=self.config,mapping=self.mapping)
            self.RNG_bloclk = RNG_block(num_block = self.numCol/self.precision_sigma/self.numColMuxed, param=self.param, tech=self.tech, config=self.config, mapping=self.mapping, RNG=self.RNG)
            

            if self.sram_mode == "conventionalSequential":
                #sum up all the gate cap of access CMOS, as the row cap
                gate_cap = logicGate.calculate_mos_gate_cap(effective_width, self.tech)
                #to be determined why the factor is 2
                ##################################################################################
                self.capRow1 += 2 * gate_cap * numCol  # 2 access transistors per cell ß
                #######initialize the ADC########################
                self.sense_amp_mu = SenseAmp(num_col=self.num_mu/self.numColMuxed,current_sense=False,sense_voltage=self.param['minSenseVoltage'],clk_freq = None,pitch_sense_amp= self.lengthRow/self.numCol,tech=self.tech,config=self.config,mapping=self.mapping)
                self.sense_amp_sigma = SenseAmp(num_col=self.num_sigma/self.numColMuxed,current_sense=False,sense_voltage=self.param['minSenseVoltage'],clk_freq = None,pitch_sense_amp= self.lengthRow/self.numCol,tech=self.tech,config=self.config,mapping=self.mapping)
                
                # self.SarADC_mu = SarADC(num_col=self.num_mu/self.precision_sigma/self.numColMuxed,level_output = self.output_levle,clk_freq=self.clk_freq,num_read_cell = self.activity_row_read,tech=self.tech,config=self.config,mapping=self.mapping,param=self.param)
                # self.SarADC_sigma = SarADC(num_col=self.num_sigma/self.precision_sigma/self.numColMuxed,level_output = self.output_levle,clk_freq=self.clk_freq,num_read_cell = self.activity_row_read,tech=self.tech,config=self.config,mapping=self.mapping,param=self.param)
                # self.dff_mu = DFF(num_dff=self.num_mu,tech=self.tech,config=self.config,param=self.param,clk_freq=self.clk_freq)
                # self.dff_sigma = DFF(num_dff=self.num_sigma * self.precision_ADC,tech=self.tech,config=self.config,param=self.param,clk_freq=self.clk_freq)
                self.mux_bus = Mux(num_input=self.numCol*self.output_levle/self.numColMuxed/self.precision_sigma,num_selection=1,param=self.param,mapping=self.mapping,tech=self.tech, config = self.config, res_tg=None,FPGA=True)
                
                if self.numColMuxed > 1:
                    self.MUX = Mux(num_input=self.numCol/self.numColMuxed,num_selection=self.numColMuxed,param=self.param,mapping=self.mapping,tech=self.tech, config = self.config, res_tg=None,FPGA=True)
                    self.muxDecoder = RowDecoder(mode="REGULAR_ROW",num_addr_row=int(math.ceil(math.log2(self.numColMuxed))),mux=True,parallel=False,tech=self.tech,config=self.config,mapping=self.mapping)
                self.MultilevelSenseAmp_mu = MultilevelSenseAmp(num_col=self.num_mu/self.numColMuxed/self.precision_sigma,level_output=self.output_levle,clk_freq = self.clk_freq,current_mode=False,columncap=self.capCol, pitch_sense_amp = self.lengthRow/self.numCol,param= self.param,tech=self.tech,config=self.config,mapping=self.mapping)
                self.MultilevelSenseAmp_sigma = MultilevelSenseAmp(num_col=self.num_sigma/self.numColMuxed/self.precision_sigma,level_output=self.output_levle,clk_freq = self.clk_freq,current_mode=False,columncap=self.capCol, pitch_sense_amp = self.lengthRow/self.numCol,param= self.param,tech=self.tech,config=self.config,mapping=self.mapping)
                # print("num_col of adc", self.num_mu/self.numColMuxed/self.precision_sigma)
                # print("level of adc", self.output_levle)
        elif self.cell_type in ['RRAM', 'FeFET']:
            if self.accesstype == 'CMOS_access':
                self.resCellAccess = self.resistanceOn * constant.IR_DROP_TOLERANCE
                #firstly calculate the CMOS resistance and capacitance
                scale_factor = 2 if self.feature_size <= 14e-9 else 1
                self.widthAccessCMOS = logicGate.calculate_on_resistance(scale_factor*self.feature_size, constant.NMOS, self.temp, self.tech) * constant.LINEAR_REGION_RATIO / self.resCellAccess
                widthAccessInFeatureSize = self.widthAccessCMOS
                if self.feature_size <= 14e-9:
                    widthAccessInFeatureSize = ((self.widthAccessCMOS-1) * self.PitchFin + self.widthFin) / self.feature_size;  #convert #fin to F
                if (widthAccessInFeatureSize > self.widthInFeatureSize):
                    print("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", self.widthAccessCMOS, self.widthInFeatureSize)
                    exit(1)
                self.resMemCellOn = self.resistanceOn + self.resCellAccess      #calculate single memory cell resistance_ON
                self.resMemCellOff = self.resistanceOff + self.resCellAccess    #calculate single memory cell resistance_OFF
                #assume half of the weights are ON and half are OFF
                self.resMemCellAvg = 1/(1/(self.resistanceOn + self.resCellAccess ) * self.numRowParallel/2.0 + 1/(self.resistanceOff + self.resCellAccess )* self.numRowParallel/2.0) * self.numRowParallel       #calculate single memory cell resistance_AVG
                self.capRow2 += logicGate.calculate_mos_gate_cap(self.widthAccessCMOS * scale_factor * self.feature_size, self.tech) * self.numCol;          #sum up all the gate cap of access CMOS, as the row cap
                unitcap= self.capRow2/self.param['numColSubArray']
                unitres= self.resRow/self.param['numColSubArray']
                self.capCol += logicGate.calculate_drain_cap(constant.NMOS, self.widthAccessCMOS * scale_factor * self.feature_size, constant.MAX_TRANSISTOR_HEIGHT * self.feature_size, self.tech) * numRow;	# If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit
                columncap = self.capCol
            else:
                self.resMemCellOn = self.resistanceOn
                self.resMemCellOff = self.resistanceOff
                self.resMemCellOnAtHalfVw = self.resistanceOn
                self.resMemCellOffAtHalfVw = self.resistanceOff
                self.resMemCellOnAtVw = self.resistanceOn
                self.resMemCellOffAtVw = self.resistanceOff

                self.resMemCellAvg = 1/(1/(self.resistanceOn + self.resCellAccess ) * self.numRowParallel/2.0 + 1/(self.resistanceOff + self.resCellAccess )* self.numRowParallel/2.0) * self.numRowParallel
                self.resistanceAvg = (self.resistanceOn + self.resistanceOff)/2;            #// Average resistance (for energy estimation)
                self.resMemCellAvgAtHalfVw = self.resistanceAvg
                self.resMemCellAvgAtVw = self.resistanceAvg

            if (self.writeVoltage > 1.5):
                self.wllevelshifter = LevelShifter(num_output=numRow,param=self.param,clk_freq=None,tech=self.tech,config=self.config,mapping=self.mapping)
                self.bllevelshifter = LevelShifter(num_output=numRow,param=self.param,clk_freq=None,tech=self.tech,config=self.config,mapping=self.mapping)
                self.sllevelshifter = LevelShifter(num_output=numCol,param=self.param,clk_freq=None,tech=self.tech,config=self.config,mapping=self.mapping)
            if self.mem_mode == "conventionalSequential":
                self.capBL = self.lengthCol * 0.2e-15/1e-6
                numAdder = math.ceil(self.numCol/self.numColMuxed)   # numCol is divisible by numCellPerSynapse
                numInput = numAdder       #XXX input number of MUX,
                resTg = self.resMemCellOn     #transmission gate resistance
                adderBit = math.ceil(math.log2(self.numRow)) + self.avgWeightBit
                self.row_decoder = RowDecoder(mode="REGULAR_ROW",num_addr_row=int(math.ceil(math.log2(self.numRow))),mux=False,parallel=False,tech=self.tech,config=self.config,mapping=self.mapping)
                self.wlDecoderDriver = WLNewDecoderDriver(numWLRow=self.numRow,param=self.param,tech=self.tech,config=self.config,mapping=self.mapping)
                self.RNG_bloclk = RNG_block(num_block = self.numCol/self.precision_sigma/self.numColMuxed, param=self.param, tech=self.tech, config=self.config, mapping=self.mapping, RNG=self.RNG)
                if self.numColMuxed > 1:
                    self.MUX = Mux(num_input=numInput,num_selection=self.numColMuxed,param=self.param,mapping=self.mapping,tech=self.tech, config = self.config, res_tg=None,FPGA=True)
                    self.muxDecoder = RowDecoder(mode="REGULAR_ROW",num_addr_row=int(math.ceil(math.log2(self.numColMuxed))),mux=True,parallel=False,tech=self.tech,config=self.config,mapping=self.mapping)
                    
                # self.SarADC_mu = SarADC(num_col=self.num_mu/self.numColMuxed,level_output = self.output_levle,clk_freq=self.clk_freq,num_read_cell = self.activity_row_read,tech=self.tech,config=self.config,mapping=self.mapping,param=self.param)
                # self.SarADC_sigma = SarADC(num_col=self.num_sigma/self.numColMuxed,level_output = self.output_levle,clk_freq=self.clk_freq,num_read_cell = self.activity_row_read,tech=self.tech,config=self.config,mapping=self.mapping,param=self.param)
                # self.dff_mu = DFF(num_dff=self.num_mu * self.output_levle_mu_rram/self.numColMuxed,tech=self.tech,config=self.config,param=self.param,clk_freq=self.clk_freq)
                # self.dff_sigma = DFF(num_dff=self.num_sigma * self.output_levle/self.numColMuxed,tech=self.tech,config=self.config,param=self.param,clk_freq=self.clk_freq)
                self.mux_bus = Mux(num_input=self.numCol*self.output_levle/self.numColMuxed,num_selection=2,param=self.param,mapping=self.mapping,tech=self.tech, config = self.config, res_tg=None,FPGA=True)
                # self.MultilevelSenseAmp = MultilevelSenseAmp(num_col=self.numCol/self.numColMuxed,level_output=self.output_levle,clk_freq = self.clk_freq,current_mode=False,columncap=self.capCol,param= self.param,tech=self.tech,config=self.config,mapping=self.mapping)
                self.MultilevelSenseAmp_mu = MultilevelSenseAmp(num_col=self.num_mu/self.numColMuxed/self.precision_sigma,level_output=self.output_levle,clk_freq = self.clk_freq,current_mode=False,columncap=self.capCol, pitch_sense_amp = self.lengthRow/self.numCol,param= self.param,tech=self.tech,config=self.config,mapping=self.mapping)
                self.MultilevelSenseAmp_sigma = MultilevelSenseAmp(num_col=self.num_sigma/self.numColMuxed/self.precision_sigma,level_output=self.output_levle,clk_freq = self.clk_freq,current_mode=False,columncap=self.capCol, pitch_sense_amp = self.lengthRow/self.numCol,param= self.param,tech=self.tech,config=self.config,mapping=self.mapping)
        self.initialized = True
    def get_column_resistance(self, input_vector, weight_matrix, parallel_read, res_cell_access):
        """
        Compute column resistances for a given input and weight matrix.
        Only SRAM logic is implemented. RRAM and FeFET blocks are preserved as placeholders.

        Args:
            input_vector (List[float]): input activation vector
            weight_matrix (List[List[float]]): conductance matrix [rows][cols]
            cell (MemCell): a memory cell object with memCellType
            parallel_read (bool): whether the read is parallel
            res_cell_access (float): access transistor resistance for SRAM

        Returns:
            List[float]: per-column equivalent resistance
        """

        resistance = []
        conductance = []

        num_rows = len(weight_matrix)
        num_cols = len(weight_matrix[0])

        for j in range(num_cols):
            column_g = 0.0
            activated_row = 0

            for i in range(num_rows):
                if self.cell_type == "RRAM":
                    if self.accesstype == "CMOS_access":
                        total_wire_res = (1.0 / weight_matrix[i][j]) \
                                     + (j + 1) * self.wireResistanceRowPerCell \
                                     + (num_rows - i) * self.wireResistanceColPerCell \
                                     + res_cell_access
                    else:  # Cross-point
                        total_wire_res = (1.0 / weight_matrix[i][j]) \
                                     + (j + 1) * self.wireResistanceRowPerCell \
                                     + (num_rows - i) * self.wireResistanceColPerCell
                    if int(input_vector[i]) == 1:
                        column_g += 1.0 / total_wire_res
                        activated_row += 1
                

                elif self.cell_type == "FeFET":
                    total_wire_res = (1.0 / weight_matrix[i][j]) \
                                 + (j + 1) * self.wireResistanceRowPerCell \
                                 + (num_rows - i) * self.wireResistanceColPerCell

                    if int(input_vector[i]) == 1:
                        column_g += 1.0 / total_wire_res
                        activated_row += 1

                elif self.cell_type == "SRAM":
                    # In SRAM, weight[i][j] has no impact on energy
                    total_wire_res = res_cell_access + self.wireResistanceColPerCell
                    if int(input_vector[i]) == 1:
                        column_g += 1.0 / total_wire_res
                        activated_row += 1
                    else:
                        column_g += 0

            if self.cell_type in ["RRAM", "FeFET"]:
                if not parallel_read:
                    if activated_row > 0:
                        conductance.append(column_g / activated_row)
                    else:
                        conductance.append(0)
                else:
                    num_add = math.ceil(self.param.numRowSubArray / self.param.numRowParallel)
                    conductance.append(column_g / num_add)
            else:  # SRAM
                conductance.append(column_g)

        # Convert conductance to resistance
        for g in conductance:
            if g > 0:
                resistance.append(1.0 / g)
            else:
                resistance.append(float('inf'))  # open circuit

        return resistance
    def min_combined_area(self, w1, h1, w2, h2):
        # Define all 8 configurations of placement (including rotations)
        configs = [
            # 1. R1 and R2 side by side (no rotation)
            (w1 + w2, max(h1, h2), "R1 and R2 side by side (no rotation)"),

            # 2. R1 on top of R2 (no rotation)
            (max(w1, w2), h1 + h2, "R1 on top of R2 (no rotation)"),

            # 3. R1 normal, R2 rotated 90 degrees (side by side)
            (w1 + h2, max(h1, w2), "R1 normal, R2 rotated 90° (side by side)"),

            # 4. R1 normal, R2 rotated 90 degrees (stacked vertically)
            (max(w1, h2), h1 + w2, "R1 normal, R2 rotated 90° (stacked)"),

            # 5. R1 rotated 90 degrees, R2 normal (side by side)
            (h1 + w2, max(w1, h2), "R1 rotated 90°, R2 normal (side by side)"),

            # 6. R1 rotated 90 degrees, R2 normal (stacked vertically)
            (max(h1, w2), w1 + h2, "R1 rotated 90°, R2 normal (stacked)"),

            # 7. Both rotated 90 degrees (stacked vertically)
            (h1 + h2, max(w1, w2), "Both R1 and R2 rotated 90° (stacked)"),

            # 8. Both rotated 90 degrees (side by side)
            (max(h1, h2), w1 + w2, "Both R1 and R2 rotated 90° (side by side)")
        ]
        # Find the configuration with minimum area
        min_area = float('inf')
        best_config = None
        for w, h, description in configs:
            area = w * h
            if area < min_area:
                min_area = area
                best_config = (w, h, description)

        return min_area, best_config
    def calculate_area(self):
        if not self.initialized:
            raise Exception("SubArray not initialized. Please call initialize() method first.")
        else:
            area = 0
            used_area = 0
            if self.cell_type  == 'SRAM':
                #array only
                height_array = self.lengthCol
                width_array = self.lengthRow
                area_array = height_array * width_array

                #precharger and writeDriver are always needed for all different designs
                pre_charge_area,pre_charge_height,pre_charge_width,pre_charge_cap_output_BL = self.precharger.calculate_area(num_col=self.numCol, new_height=None, new_width=None, option='NONE')
                sram_write_driver_area,sram_write_driver_height,sram_write_driver_width = self.sram_write_driver.calculate_area(new_height=None,new_width=None,option='NONE')
                WL_decoder_area,WL_decoder_height,WL_decoder_width = self.row_decoder.calculate_area(new_height=None,new_width=None,option='NONE')
                # self.SarADC_mu.calculate_unit_area()
                # self.SarADC_sigma.calculate_unit_area()
                sense_amp_mu_area,sense_amp_mu_height,sense_amp_mu_width,_ = self.sense_amp_mu.calculate_area(new_height=None,new_width=None,option='NONE')
                sense_amp_sigma_area,sense_amp_sigma_height,sense_amp_sigma_width,_ = self.sense_amp_sigma.calculate_area(new_height=None,new_width=None,option='NONE')
                # SarADC_mu_area,SarADC_mu_height,SarADC_mu_width = self.SarADC_mu.calculate_area(height_array=None,width_array=None,option='NONE')
                # SarADC_sigma_area,SarADC_sigma_height,SarADC_sigma_width = self.SarADC_sigma.calculate_area(height_array=None,width_array=None,option='NONE')
                # dff_mu_area,dff_mu_height,dff_mu_width = self.dff_mu.calculate_area(new_height=None,new_width=None,option='NONE')
                # dff_sigma_area,dff_sigma_height,dff_sigma_width = self.dff_sigma.calculate_area(new_height=None,new_width=None,option='NONE')
                RNG_bloclk_area,RNG_bloclk_height,RNG_bloclk_width = self.RNG_bloclk.calculate_area()
                MUX_bus_area,MUX_bus_height,MUX_bus_width = self.mux_bus.calculate_area(new_height=None,new_width=None,option='NONE')
                if self.numColMuxed > 1:
                    MUX_area,MUX_height,MUX_width = self.MUX.calculate_area(new_height=None,new_width=None,option='NONE')
                    muxDecoder_area,muxDecoder_height,muxDecoder_width = self.muxDecoder.calculate_area(new_height=None,new_width=None,option='NONE')
                else:
                    MUX_area,MUX_height,MUX_width = 0,0,0
                    muxDecoder_area,muxDecoder_height,muxDecoder_width = 0,0,0
                MultilevelSenseAmp_mu_area,MultilevelSenseAmp_mu_height,MultilevelSenseAmp_mu_width = self.MultilevelSenseAmp_mu.calculate_area(height_array=None,width_array=None,option='NONE')
                MultilevelSenseAmp_sigma_area,MultilevelSenseAmp_sigma_height,MultilevelSenseAmp_sigma_width = self.MultilevelSenseAmp_sigma.calculate_area(height_array=None,width_array=None,option='NONE')
                if self.mem_mode == "conventionalSequential":
                    height = pre_charge_height + sram_write_driver_height + height_array + MultilevelSenseAmp_mu_height + RNG_bloclk_height + MUX_bus_height + MUX_height + sense_amp_mu_height
                    width = WL_decoder_width + width_array
                    width = max(width, RNG_bloclk_width, MultilevelSenseAmp_mu_width + MultilevelSenseAmp_sigma_width)
                    area = height * width
                    # MultilevelSenseAmp_sigma_area = 0
                    # MultilevelSenseAmp_mu_area = 0
                    # RNG_bloclk_area = 0
                    used_area = (area_array + pre_charge_area + sram_write_driver_area +
                               WL_decoder_area + MultilevelSenseAmp_mu_area + MultilevelSenseAmp_sigma_area + RNG_bloclk_area + sense_amp_mu_area + sense_amp_sigma_area)
                    used_area += MUX_bus_area
                    used_area += MUX_area + muxDecoder_area
                    memory_density = self.numRow * self.numCol/area * 1e-6  # in bits/mm2
                    used_memory_density = self.numRow * self.numCol/used_area * 1e-6  # in bits/mm2
                    # area_data = {
                    #     "num_rows": self.numRow,
                    #     "num_cols": self.numCol,
                    #     "num_muxed_cols": self.numColMuxed,
                    #     "adc_precision": self.precision_ADC,
                    #     "Array": area_array,
                    #     "WL_Decoder": WL_decoder_area,
                    #     "Pre_Charge": pre_charge_area,
                    #     "SRAM_Write_Driver": sram_write_driver_area,
                    #     "Sense_Amp": sense_amp_mu_area + sense_amp_sigma_area,
                    #     # "SAR ADC": SarADC_mu_area + SarADC_sigma_area,
                    #     "ADC": MultilevelSenseAmp_mu_area + MultilevelSenseAmp_sigma_area,
                    #     "RNG_Block": RNG_bloclk_area,
                    #     "MUX_Bus": MUX_bus_area,
                    #     "MUX": MUX_area,
                    #     "MUX_Decoder": muxDecoder_area,
                    #     "MUX_Block": MUX_area + muxDecoder_area + MUX_bus_area,
                    #     "periphery": (pre_charge_area + sram_write_driver_area +
                    #                         WL_decoder_area + MUX_bus_area + MUX_area + muxDecoder_area),
                    #     "Total Area": area,
                    #     "Used Area": used_area,
                    #     "Memory_Density (bits/um^2)": memory_density,
                    #     "Used_memory_Density (bits/um^2)": used_memory_density
                    # }
                    # # filename = f"../../Data/simulation/ps_sram/3bitADC/ARNG1/mux16/area/ps_sram_area_data_{self.numRow}x{self.numCol}.json"
                    # filename = (
                    #     f"../../Data/simulation/ps_sram/{self.precision_ADC}bitADC/"
                    #     f"ARNG1/mux{self.numColMuxed}/area/"
                    #     f"ps_sram_area_data_{self.numRow}x{self.numCol}.json"
                    # )

                    # with open(filename, "w") as f:
                    #     json.dump(area_data, f, indent=4)


            elif self.cell_type in ['RRAM', 'FeFET']:
                height_array = self.lengthCol
                width_array = self.lengthRow
                area_array = height_array * width_array
                if (self.writeVoltage > 1.5):
                    wllevelshifter_area,wllevelshifter_height,wllevelshifter_width = self.wllevelshifter.calculate_area(new_height=None,new_width=None,option='NONE')
                    bllevelshifter_area,bllevelshifter_height,bllevelshifter_width = self.bllevelshifter.calculate_area(new_height=None,new_width=None,option='NONE')
                    sllevelshifter_area,sllevelshifter_height,sllevelshifter_width = self.sllevelshifter.calculate_area(new_height=None,new_width=None,option='NONE')	
                else:
                    wllevelshifter_area,wllevelshifter_height,wllevelshifter_width = 0,0,0
                    bllevelshifter_area,bllevelshifter_height,bllevelshifter_width = 0,0,0
                    sllevelshifter_area,sllevelshifter_height,sllevelshifter_width = 0,0,0	
			
                WL_decoder_area,WL_decoder_height,WL_decoder_width = self.row_decoder.calculate_area(new_height=None,new_width=None,option='NONE')
                wlDecoderDriver_area,wlDecoderDriver_height,wlDecoderDriver_width = self.wlDecoderDriver.calculate_area(new_height=None,new_width=None,option='NONE')
                if self.numColMuxed > 1:
                    MUX_area,MUX_height,MUX_width = self.MUX.calculate_area(new_height=None,new_width=None,option='NONE')
                    muxDecoder_area,muxDecoder_height,muxDecoder_width = self.muxDecoder.calculate_area(new_height=None,new_width=None,option='NONE')
                else:
                    MUX_area,MUX_height,MUX_width = 0,0,0
                    muxDecoder_area,muxDecoder_height,muxDecoder_width = 0,0,0
                # self.SarADC_mu.calculate_unit_area()
                # self.SarADC_sigma.calculate_unit_area()
                # SarADC_mu_area,SarADC_mu_height,SarADC_mu_width = self.SarADC_mu.calculate_area(height_array=None,width_array=None,option='NONE')
                # SarADC_sigma_area,SarADC_sigma_height,SarADC_sigma_width = self.SarADC_sigma.calculate_area(height_array=None,width_array=None,option='NONE')
                # dff_mu_area,dff_mu_height,dff_mu_width = self.dff_mu.calculate_area(new_height=None,new_width=None,option='NONE')
                # dff_sigma_area,dff_sigma_height,dff_sigma_width = self.dff_sigma.calculate_area(new_height=None,new_width=None,option='NONE')
                RNG_bloclk_area,RNG_bloclk_height,RNG_bloclk_width = self.RNG_bloclk.calculate_area()
                # SarADC_area = SarADC_mu_area + SarADC_sigma_area
                MUX_bus_area,MUX_bus_height,MUX_bus_width = self.mux_bus.calculate_area(new_height=None,new_width=None,option='NONE')
                # MultilevelSenseAmp_area,MultilevelSenseAmp_height,MultilevelSenseAmp_width = self.MultilevelSenseAmp.calculate_area(height_array=None,width_array=None,option='NONE')
                MultilevelSenseAmp_mu_area,MultilevelSenseAmp_mu_height,MultilevelSenseAmp_mu_width = self.MultilevelSenseAmp_mu.calculate_area(height_array=None,width_array=None,option='NONE')
                MultilevelSenseAmp_sigma_area,MultilevelSenseAmp_sigma_height,MultilevelSenseAmp_sigma_width = self.MultilevelSenseAmp_sigma.calculate_area(height_array=None,width_array=None,option='NONE')
                # dff_area = dff_mu_area + dff_sigma_area

                if self.mem_mode == "conventionalSequential":
                    height = sllevelshifter_height + height_array + MUX_height + MultilevelSenseAmp_mu_height + RNG_bloclk_height + MUX_bus_height
                    width = wllevelshifter_width + bllevelshifter_width +  WL_decoder_width + width_array + wlDecoderDriver_width
                    width = max(width, RNG_bloclk_width)
                    area = height * width
                    MultilevelSenseAmp_area = MultilevelSenseAmp_mu_area*4 + MultilevelSenseAmp_sigma_area*4
                    used_area = (area_array + WL_decoder_area + wlDecoderDriver_area + wllevelshifter_area + bllevelshifter_area + sllevelshifter_area + MUX_area + muxDecoder_area + MultilevelSenseAmp_area + RNG_bloclk_area)
                    used_area += MUX_bus_area
                    empty_area = area - used_area

                    memory_density = self.numRow * self.numCol/area * 1e-6  # in bits/um2
                    used_memory_density = self.numRow * self.numCol/used_area * 1e-6
                    # area_data = {
                    #     "num_rows": self.numRow,
                    #     "num_cols": self.numCol,
                    #     "num_muxed_cols": self.numColMuxed,
                    #     "adc_precision": self.precision_ADC,
                    #     "Array": area_array,
                    #     "WL_Decoder": WL_decoder_area,
                    #     "wlDecoderDriver": wlDecoderDriver_area,
                    #     "levelshifter": wllevelshifter_area+bllevelshifter_area+sllevelshifter_area,
                    #     "ADC": MultilevelSenseAmp_area,
                    #     "RNG_Block": RNG_bloclk_area,
                    #     "MUX_Bus": MUX_bus_area,
                    #     "MUX": MUX_area,
                    #     "MUX_Decoder": muxDecoder_area,
                    #     "MUX_Block": MUX_area + muxDecoder_area + MUX_bus_area,
                    #     "periphery": (WL_decoder_area + wlDecoderDriver_area + MUX_bus_area + MUX_area + muxDecoder_area),
                    #     "Used Area": used_area,
                    #     "Memory_Density (bits/um^2)": memory_density,
                    #     "Used_memory_Density (bits/um^2)": used_memory_density,
                    #     "Total Area": area
                    # }
                    # filename = (
                    #     f"../../Data/simulation/ps_{self.cell_type}/{self.precision_ADC}bitADC/"
                    #     f"ARNG1/mux{self.numColMuxed}/area/"
                    #     f"ps_{self.cell_type}_area_data_{self.numRow}x{self.numCol}.json"
                    # )

                    # with open(filename, "w") as f:
                    #     json.dump(area_data, f, indent=4)

                else:
                    #if pure RRAM/FeFET array, only array area is considered
                    height = height_array
                    width = width_array
                    area = height * width
                    used_area = area_array
                    empty_area = area - used_area
        return area, height, width, used_area
    
    def calculate_latency(self, calculate_clk_freq, validated=False):
        if not self.initialized:
            raise Exception("SubArray not initialized. Please call initialize() method first.")
        else:
            read_latency = 0
            read_latency_adc = 0
            read_latency_other = 0
            write_latency = 0
            # print("cell_type", self.cell_type)
            # print("mem_mode", self.mem_mode)
            if self.cell_type == 'SRAM':
                if self.mem_mode == "conventionalSequential":
                    #calculate the read latency
                    # num_read_op_per_row = self.numCol/self.num_read_cell_op  #in case not all the columns are read out
                    # num_write_op_per_row = self.numCol * self.activity_col_write / self.num_write_cell_op #in case not all the columns are written
                    if (calculate_clk_freq):

                        wl_decoder_read_latency,wl_decoder_wrtie_latency = self.row_decoder.calculate_latency(cap_load1=self.capRow1,cap_load2=self.capRow1,num_read=1,num_write=1)
                        precharger_read_latency,precharger_write_latency = self.precharger.calculate_latency(cap_load=self.capCol,num_read=1,num_write=1)
                        sense_amp_mu_read_latency = self.sense_amp_mu.calculate_latency(num_read=1,cap_load=self.capCol)
                        sense_amp_sigma_read_latency = self.sense_amp_sigma.calculate_latency(num_read=1,cap_load=self.capCol)
                        # SarADC_mu_read_latency = self.SarADC_mu.calculate_latency(num_read=1)
                        # SarADC_sigma_read_latency = self.SarADC_sigma.calculate_latency(num_read=1)
                        MultilevelSenseAmp_read_latency = self.MultilevelSenseAmp_mu.calculate_latency(num_col_muxed=1,num_read=1,cap_load=self.capCol)
                        # dff_mu_read_latency,dff_mu_write_latency = self.dff_mu.calculate_latency(num_read=1)
                        # dff_sigma_read_latency,dff_sigma_write_latency = self.dff_sigma.calculate_latency(num_read=1)
                        if self.numColMuxed > 1:
                            MUX_read_latency = self.MUX.calculate_latency(cap_load=self.capCol,num_read=1)
                            MUX_decoder_read_latency,MUX_decoder_wrtie_latency = self.muxDecoder.calculate_latency(cap_load1=self.MUX.capTgGateN*math.ceil(self.numCol/self.numColMuxed),cap_load2=self.MUX.capTgGateP*math.ceil(self.numCol/self.numColMuxed),num_read=1,num_write=0)
                        else:
                            MUX_read_latency = 0
                            MUX_decoder_read_latency = 0
                        MUX_decoder_wrtie_latency = 0
                        MUX_bus_read_latency = self.mux_bus.calculate_latency(cap_load=self.capCol,num_read=1)
                        # print("capCol", self.capCol)
                        if (self.RNG_SamplingMode == 'parallel'):
                            rng_latency = self.RNG_bloclk.calculate_latency(num_read=1)


                        #read
                        scale_factor = 2 if self.feature_size <= 14e-9 else 1
                        res_pull_down = logicGate.calculate_on_resistance(
                            self.widthSRAMCellNMOS * self.feature_size * scale_factor, constant.NMOS, self.temp, self.tech)
                        BL_cap_per_cell = self.capCol / self.numRow + self.capCellAccess
                        BL_res_per_cell = self.resCol / self.numRow
                        Elmore_BL = (self.resCellAccess + res_pull_down) * BL_cap_per_cell * self.numRow + (BL_cap_per_cell * self.numRow * (self.numRow + 1)/2)
                        col_delay = Elmore_BL * math.log(self.vdd / (self.vdd - self.minSenseVoltage/2))
                        
                        read_latency = wl_decoder_read_latency + precharger_read_latency + col_delay + MultilevelSenseAmp_read_latency + rng_latency + MUX_bus_read_latency + MUX_read_latency + MUX_decoder_read_latency
                        if validated:
                            readLatency *= param.beta
                        read_latency_cycles = read_latency * self.clk_freq
                        throughtput = self.numCol/self.numColMuxed / read_latency
                    #     latency_data = {
                    #     "num_rows": self.numRow,
                    #     "num_cols": self.numCol,
                    #     "num_muxed_cols": self.numColMuxed,
                    #     "adc_precision": self.precision_ADC,
                    #     "WL_Decoder": wl_decoder_read_latency,
                    #     "Precharger": precharger_read_latency,
                    #     "Col_Delay": col_delay,
                    #     "ADC": MultilevelSenseAmp_read_latency,
                    #     "RNG_Block": rng_latency,
                    #     "MUX_Bus": MUX_bus_read_latency,
                    #     "MUX": MUX_read_latency,
                    #     "MUX_Decoder": MUX_decoder_read_latency,
                    #     "MUX_Block": MUX_read_latency + MUX_decoder_read_latency + MUX_bus_read_latency,
                    #     "periphery": (wl_decoder_read_latency + precharger_read_latency +
                    #                         MUX_bus_read_latency + MUX_read_latency + MUX_decoder_read_latency),
                    #     "Total Read Latency (s)": read_latency,
                    #     "Throughput (bits/s)": throughtput,
                    #     "Total Read Latency (cycles)": read_latency_cycles
                    # }
                    # # filename = f"../../Data/simulation/ps_sram/3bitADC/ARNG1/mux16/latency/ps_sram_latency_data_{self.numRow}x{self.numCol}.json"
                    # filename = (
                    #     f"../../Data/simulation/ps_sram/{self.precision_ADC}bitADC/"
                    #     f"ARNG1/mux{self.numColMuxed}/latency/"
                    #     f"ps_sram_latency_data_{self.numRow}x{self.numCol}.json"
                    # )

                    # with open(filename, "w") as f:
                    #     json.dump(latency_data, f, indent=4)
            elif self.cell_type in ['RRAM', 'FeFET']:
                print("read_latency", read_latency)
                if self.mem_mode == "conventionalSequential":
                    capBL = self.lengthCol * 0.2e-15/1e-6
                    colRamp = 0
                    tau = (self.capCol)*(self.resMemCellAvg)
                    colDelay,colRamp = logicGate.horowitz(tau, 0, 1e20)	# Just to generate colRamp
                    colDelay = tau * 0.2;  # assume the 15~20% voltage drop is enough for sensing
                    num_write_op_per_row = self.numCol * self.activity_col_write / self.num_write_cell_op
                    wl_decoder_read_latency,wl_decoder_wrtie_latency = self.row_decoder.calculate_latency(cap_load1=self.capRow2,cap_load2=0,num_read=1,num_write=1)
                    wlDecoderDriver_read_latency,wlDecoderDriver_write_latency = self.wlDecoderDriver.calculate_latency(cap_load=self.capRow2,res_load=self.resRow,num_read=1,num_write=1)
                    if self.numColMuxed > 1:
                        MUX_read_latency = self.MUX.calculate_latency(cap_load=self.capCol,num_read=1)
                        MUX_decoder_read_latency,MUX_decoder_wrtie_latency = self.muxDecoder.calculate_latency(cap_load1=self.MUX.capTgGateN*math.ceil(self.numCol/self.numColMuxed),cap_load2=self.MUX.capTgGateP*math.ceil(self.numCol/self.numColMuxed),num_read=1,num_write=0)
                    else:
                        MUX_read_latency = 0
                        MUX_decoder_read_latency = 0
                        MUX_decoder_wrtie_latency = 0
                    # SarADC_mu_read_latency = self.SarADC_mu.calculate_latency(num_read=1)
                    # SarADC_sigma_read_latency = self.SarADC_sigma.calculate_latency(num_read=1)
                    # MultilevelSenseAmp_read_latency = self.MultilevelSenseAmp.calculate_latency(num_col_muxed=1,num_read=1)
                    MultilevelSenseAmp_read_latency = self.MultilevelSenseAmp_mu.calculate_latency(num_col_muxed=1,num_read=1,cap_load=self.capCol)
                    MUX_bus_read_latency = self.mux_bus.calculate_latency(cap_load=self.capCol,num_read=1)
                    # print("capCol", self.capCol)
                    # dff_mu_read_latency,dff_mu_write_latency = self.dff_mu.calculate_latency(num_read=1)
                    # dff_sigma_read_latency,dff_sigma_write_latency = self.dff_sigma.calculate_latency(num_read=1)
                    if (self.RNG_SamplingMode == 'parallel'):
                        rng_latency = self.RNG_bloclk.calculate_latency(num_read=1)
                    read_latency = wl_decoder_read_latency + wlDecoderDriver_read_latency + colDelay + MUX_decoder_read_latency + MUX_read_latency + MultilevelSenseAmp_read_latency + rng_latency + MUX_bus_read_latency
                    if validated:
                        read_latency *= self.param['beta']
                    read_latency_cycles = read_latency * self.clk_freq
                    throughtput = self.numCol/self.numColMuxed / read_latency
                    # latency_data = {
                    #     "num_rows": self.numRow,
                    #     "num_cols": self.numCol,
                    #     "num_muxed_cols": self.numColMuxed,
                    #     "adc_precision": self.precision_ADC,
                    #     "WL_Decoder": wl_decoder_read_latency,
                    #     "wlDecoderDriver":wlDecoderDriver_read_latency,
                    #     "Col_Delay": colDelay,
                    #     "ADC": MultilevelSenseAmp_read_latency,
                    #     "RNG_Block": rng_latency,
                    #     "MUX_Bus": MUX_bus_read_latency,
                    #     "MUX": MUX_read_latency,
                    #     "MUX_Decoder": MUX_decoder_read_latency,
                    #     "MUX_Block": MUX_read_latency + MUX_decoder_read_latency + MUX_bus_read_latency,
                    #     "periphery": (wl_decoder_read_latency + wlDecoderDriver_read_latency +
                    #                         MUX_bus_read_latency + MUX_read_latency + MUX_decoder_read_latency),
                    #     "Total Read Latency (s)": read_latency,
                    #     "Throughput (bits/s)": throughtput,
                    #     "Total Read Latency (cycles)": read_latency_cycles
                    # }
                    # # filename = f"../../Data/simulation/ps_sram/3bitADC/ARNG1/mux16/latency/ps_sram_latency_data_{self.numRow}x{self.numCol}.json"
                    # filename = (
                    #     f"../../Data/simulation/ps_{self.cell_type}/{self.precision_ADC}bitADC/"
                    #     f"ARNG1/mux{self.numColMuxed}/latency/"
                    #     f"ps_{self.cell_type}_latency_data_{self.numRow}x{self.numCol}.json"
                    # )

                    # with open(filename, "w") as f:
                    #     json.dump(latency_data, f, indent=4)
        return read_latency * self.numColMuxed, read_latency_cycles

    def calculate_power(self, input_vector, weight_matrix):
        if not self.initialized:
            raise Exception("SubArray not initialized. Please call initialize() method first.")
        else:
            readDynamicEnergy = 0
            write_dynamic_energy = 0
            read_dynamic_energy_array = 0
            if (self.numCol > self.num_read_cell_op):
                num_read_op_per_row = self.numCol / self.num_read_cell_op
            else:
                num_read_op_per_row = 1
            if (self.numCol * self.activity_col_write > self.num_write_cell_op):
                num_write_op_per_row = self.numCol * self.activity_col_write / self.num_write_cell_op
            else:
                num_write_op_per_row = 1


            if self.cell_type == 'SRAM':
                #array leakage (assume 2 INV)
                leakage = 0
                scale_factor = 2 if self.feature_size <= 14e-9 else 1
                leakage += logicGate.calculate_logicgate_leakage(gate_type = constant.INV, num_input = 1, width_nmos = self.widthSRAMCellNMOS * self.feature_size * scale_factor, 
                                                                width_pmos = self.widthSRAMCellPMOS * self.feature_size * scale_factor,temperature = self.temp, tech = self.tech)
                leakage_sram_in_use = leakage
                leakage *= self.numRow * self.numCol

                if self.sram_mode == "conventionalSequential":
                    wl_decoder_read_energy,wl_decoder_write_energy,wl_decoder_leakage = self.row_decoder.calculate_power(num_read=1, num_write=self.numRow*self.activity_row_write)
                    precharger_read_energy,precharger_write_energy,precharger_leakage = self.precharger.calculate_power(cap_load=self.capCol,num_read=1, 
                                                                                                                        num_write=num_write_op_per_row * self.numRow*self.activity_row_write)
                    sram_write_driver_write_energy,sram_write_driver_leakage = self.sram_write_driver.calculate_power(num_write=num_write_op_per_row * self.numRow*self.activity_row_write)
                    columnResistance = self.get_column_resistance(input_vector, weight_matrix, parallel_read = None, res_cell_access = self.resCellAccess)
                    sense_amp_mu_read_energy,sense_amp_mu_leakage = self.sense_amp_mu.calculate_power(num_read=1)
                    sense_amp_sigma_read_energy,sense_amp_sigma_leakage = self.sense_amp_sigma.calculate_power(num_read=1)
                    # SarADC_mu_read_energy = self.SarADC_mu.calculate_power(column_resistance_list=columnResistance,num_read=1)
                    # SarADC_sigma_read_energy = self.SarADC_sigma.calculate_power(column_resistance_list=columnResistance,num_read=1)
                    column_resistance_list = [columnResistance[1]]
                    MultilevelSenseAmp_mu_read_energy = self.MultilevelSenseAmp_mu.calculate_power(column_resistance_list=column_resistance_list, num_read=1)
                    MultilevelSenseAmp_sigma_read_energy = self.MultilevelSenseAmp_sigma.calculate_power(column_resistance_list=column_resistance_list, num_read=1)
                    # print("MultilevelSenseAmp_sigma_read_energy",MultilevelSenseAmp_sigma_read_energy)
                    # dff_mu_read_energy,dff_mu_write_energy,dff_mu_leakage = self.dff_mu.calculate_power(num_read=1, num_dff_per_op=self.numCol, validated=False)
                    # dff_sigma_read_energy,dff_sigma_write_energy,dff_sigma_leakage = self.dff_sigma.calculate_power(num_read=1, num_dff_per_op=self.numCol/self.num_read_cell_op * self.output_levle, validated=False)
                    RNG_energy,RNG_leakage = self.RNG_bloclk.calculate_power(num_read=1)
                    MUX_bus_read_energy,MUX_bus_leakage = self.mux_bus.calculate_power(num_read=1)
                    if self.numColMuxed > 1:
                        MUX_read_energy,MUX_leakage = self.MUX.calculate_power(num_read=self.numColMuxed)
                        muxDecoder_read_energy,muxDecoder_write_energy,muxDecoder_leakage = self.muxDecoder.calculate_power(num_read=1, num_write=1)
                    else:
                        MUX_read_energy,MUX_leakage = 0,0
                        muxDecoder_read_energy,muxDecoder_write_energy,muxDecoder_leakage = 0,0,0

                    readDynamicEnergyArray = self.capRow1 * self.vdd * self.vdd * 1;  #// Just BL discharging // -added, wordline charging
                    readDynamicEnergyArray += self.capCol * self.vdd * self.vdd * (self.numCol/self.numColMuxed)  #// Just BL discharging
                    # Read
                    readDynamicEnergy += wl_decoder_read_energy
                    # readDynamicEnergy += precharger_read_energy
                    readDynamicEnergy += MultilevelSenseAmp_sigma_read_energy + sense_amp_mu_read_energy + sense_amp_sigma_read_energy
                    readDynamicEnergy += readDynamicEnergyArray
                    # readDynamicEnergy += dff_mu_read_energy + dff_sigma_read_energy
                    readDynamicEnergy += RNG_energy
                    readDynamicEnergy += MUX_bus_read_energy
                    readDynamicEnergy += MUX_read_energy
                    readDynamicEnergy += muxDecoder_read_energy
                    
                    # Write
				    # writeDynamicEnergy += wlDecoder.writeDynamicEnergy
				    # writeDynamicEnergy += precharger.writeDynamicEnergy
				    # writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy
				    # writeDynamicEnergy += writeDynamicEnergyArray
				
				    # Leakage
                    leakage += wl_decoder_leakage
                    leakage += precharger_leakage
                    leakage += sram_write_driver_leakage
                    leakage += sense_amp_mu_leakage + sense_amp_sigma_leakage

                    # leakage += dff_mu_leakage + dff_sigma_leakage
                    leakageSRAMInUse = RNG_leakage
                    # leakageSRAMInUse *= (numRow-1) * numCol
                    # energy_data = {
                    #     "num_rows": self.numRow,
                    #     "num_cols": self.numCol,
                    #     "num_muxed_cols": self.numColMuxed,
                    #     "adc_precision": self.precision_ADC,
                    #     "Array": readDynamicEnergyArray,
                    #     "WL_Decoder": wl_decoder_read_energy,
                    #     "Sense_Amp": sense_amp_mu_read_energy + sense_amp_sigma_read_energy,
                    #     "ADC": MultilevelSenseAmp_sigma_read_energy,
                    #     "RNG_Block": RNG_energy,
                    #     "MUX_Bus": MUX_bus_read_energy,
                    #     "MUX": MUX_read_energy,
                    #     "MUX_Decoder": muxDecoder_read_energy,
                    #     "MUX_Block": MUX_read_energy + muxDecoder_read_energy + MUX_bus_read_energy,
                    #     "periphery": (wl_decoder_read_energy + + MUX_bus_read_energy + MUX_read_energy + muxDecoder_read_energy),

                    #     "Total Read Dynamic Energy (J)": readDynamicEnergy
                    # }
                    # # filename = f"../../Data/simulation/ps_sram/3bitADC/ARNG1/mux16/energy/ps_sram_energy_data_{self.numRow}x{self.numCol}.json"
                    # filename = (
                    #     f"../../Data/simulation/ps_sram/{self.precision_ADC}bitADC/"
                    #     f"ARNG1/mux{self.numColMuxed}/energy/"
                    #     f"ps_sram_energy_data_{self.numRow}x{self.numCol}.json"
                    # )

                    # with open(filename, "w") as f:
                    #     json.dump(energy_data, f, indent=4)
                    # print("dff_leakage", dff_mu_leakage + dff
            elif self.cell_type in ['RRAM', 'FeFET']:
                print("checking cell_type", self.cell_type)
                leakageSRAMInUse = 0
                if self.mem_mode == "conventionalSequential":
                    numReadCells = math.ceil(self.numCol/self.numColMuxed)    # similar parameter as numReadCellPerOperationNeuro, which is for SRAM
                    numWriteCells = numReadCells 
                    num_write_op_per_row = math.ceil(self.numCol * self.activity_col_write / self.num_write_cell_op)
                    capBL = self.lengthCol * 0.2e-15/1e-6
                    wl_decoder_read_energy,wl_decoder_write_energy,wl_decoder_leakage = self.row_decoder.calculate_power(num_read=1, num_write=2 * num_write_op_per_row *self.numRow*self.activity_row_write)
                    wlDecoderDriver_read_energy,wlDecoderDriver_write_energy,wlDecoderDriver_leakage = self.wlDecoderDriver.calculate_power(num_read=1, num_write=2 * num_write_op_per_row *self.numRow*self.activity_row_write)
                    if self.numColMuxed > 1:
                        MUX_read_energy,MUX_leakage = self.MUX.calculate_power(num_read=self.numColMuxed)
                        muxDecoder_read_energy,muxDecoder_write_energy,muxDecoder_leakage = self.muxDecoder.calculate_power(num_read=1, num_write=1)
                    else:
                        MUX_read_energy,MUX_leakage = 0,0
                        muxDecoder_read_energy,muxDecoder_write_energy,muxDecoder_leakage = 0,0,0
                    columnResistance = self.get_column_resistance(input_vector, weight_matrix, parallel_read = None, res_cell_access = self.resCellAccess)
                    # SarADC_mu_read_energy = self.SarADC_mu.calculate_power(column_resistance_list=columnResistance,num_read=1)
                    # SarADC_sigma_read_energy = self.SarADC_sigma.calculate_power(column_resistance_list=columnResistance,num_read=1)
                    # MultilevelSenseAmp_read_energy = self.MultilevelSenseAmp.calculate_power(column_resistance_list=columnResistance, num_read=1)
                    column_resistance_list = [columnResistance[1]]
                    MultilevelSenseAmp_mu_read_energy = self.MultilevelSenseAmp_mu.calculate_power(column_resistance_list=column_resistance_list, num_read=1)
                    MultilevelSenseAmp_sigma_read_energy = self.MultilevelSenseAmp_sigma.calculate_power(column_resistance_list=column_resistance_list, num_read=1)
                    # dff_mu_read_energy,dff_mu_write_energy,dff_mu_leakage = self.dff_mu.calculate_power(num_read=1, num_dff_per_op=self.numCol, validated=False)
                    # dff_sigma_read_energy,dff_sigma_write_energy,dff_sigma_leakage = self.dff_sigma.calculate_power(num_read=1, num_dff_per_op=self.numCol/self.num_read_cell_op * self.output_levle, validated=False)
                    RNG_energy,RNG_leakage = self.RNG_bloclk.calculate_power(num_read=1)
                    MUX_bus_read_energy,MUX_bus_leakage = self.mux_bus.calculate_power(num_read=1)
                    MultilevelSenseAmp_energy  = MultilevelSenseAmp_sigma_read_energy + MultilevelSenseAmp_mu_read_energy
                    # dff_read_energy = dff_mu_read_energy + dff_sigma_read_energy
                    # dff_leakage = dff_mu_leakage + dff_sigma_leakage


                    selected_bl_energy = self.capBL * self.readVoltage * self.readVoltage * numReadCells
                    selected_wl_energy = self.capRow2 * self.accessVoltage * self.accessVoltage 
                    selected_row_energy = (selected_bl_energy + selected_wl_energy) 

                    readDynamicEnergyArray = selected_row_energy + MultilevelSenseAmp_energy + MUX_read_energy + muxDecoder_read_energy + wl_decoder_read_energy + wlDecoderDriver_read_energy + RNG_energy + MUX_bus_read_energy
            #         leakage = wl_decoder_leakage + wlDecoderDriver_leakage + MUX_leakage + muxDecoder_leakage
            #         energy_data = {
            #             "num_rows": self.numRow,
            #             "num_cols": self.numCol,
            #             "num_muxed_cols": self.numColMuxed,
            #             "adc_precision": self.precision_ADC,
            #             "Array": selected_row_energy,
            #             "WL_Decoder": wl_decoder_read_energy,
            #             "wlDecoderDriver": wlDecoderDriver_read_energy,
            #             "ADC": MultilevelSenseAmp_energy,
            #             "RNG_Block": RNG_energy,
            #             "MUX_Bus": MUX_bus_read_energy,
            #             "MUX": MUX_read_energy,
            #             "MUX_Decoder": muxDecoder_read_energy,
            #             "MUX_Block": MUX_read_energy + muxDecoder_read_energy + MUX_bus_read_energy,
            #             "periphery": (wl_decoder_read_energy + wlDecoderDriver_read_energy + MUX_bus_read_energy + MUX_read_energy + muxDecoder_read_energy),

            #             "Total Read Dynamic Energy (J)": readDynamicEnergyArray
            #         }
            #         # filename = f"../../Data/simulation/ps_sram/3bitADC/ARNG1/mux16/energy/ps_sram_energy_data_{self.numRow}x{self.numCol}.json"
            #         filename = (
            #             f"../../Data/simulation/ps_{self.cell_type}/{self.precision_ADC}bitADC/"
            #             f"ARNG1/mux{self.numColMuxed}/energy/"
            #             f"ps_{self.cell_type}_energy_data_{self.numRow}x{self.numCol}.json"
            #         )

            #         with open(filename, "w") as f:
            #             json.dump(energy_data, f, indent=4)
            # print("shape of columnResistance", len(columnResistance))
        return readDynamicEnergyArray, write_dynamic_energy, leakage
            