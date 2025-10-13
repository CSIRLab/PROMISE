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
from periphery.Bus import Bus
from simulator.subarray import SubArray



class MAT:
    def __init__(self, tech,config, param, mapping, RNG, numSubArrayRow, numSubArrayCol, numCol, numRow, num_mu, num_sigma):

        self.tech = tech
        self.param = param
        self.config = config
        self.mapping = mapping
        self.RNG = RNG
        self.initialized = False
        self.num_col = numCol
        self.num_row = numRow
        self.numSubArrayRow = numSubArrayRow
        self.numSubArrayCol = numSubArrayCol
        self.num_mu = num_mu
        self.num_sigma = num_sigma

        self.feature_size = self.tech.get_param('featureSize')
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.vdd = self.tech.get_param('vdd')
        self.cell_type = self.config['device_type']
        self.clk_freq = self.config['frequency']
        self.temp = self.config['temperature']

        self.unitWireRes = self.param['unitLengthWireResistance']
        self.wireWidth = self.param['wireWidth']
        self.unitWireCap = 0.2e-15 / 1e-6  # 0.2 fF/um = 0.2e-15 F/micron

        self.resistanceOn = self.param['resistanceOn']
        self.resistanceOff = self.param['resistanceOff']
        self.resistanceAvg = (self.resistanceOn + self.resistanceOff) / 2
        self.writeVoltage = self.param['writeVoltage']
        self.readVoltage = self.param['readVoltage']
        self.accessVoltage = self.param['accessVoltage']
        self.avgWeightBit = self.param['cellBit']
        self.accesstype = self.param['accesstype']
        self.mem_mode = self.param['operationmode']
        self.readPulseWidth = self.param['readPulseWidth']
        self.heightInFeatureSizeSRAM = self.param['heightInFeatureSizeSRAM']
        self.widthInFeatureSizeSRAM = self.param['widthInFeatureSizeSRAM']
        self.widthSRAMCellNMOS = self.param['widthSRAMCellNMOS']
        self.widthSRAMCellPMOS = self.param['widthSRAMCellPMOS']
        self.widthAccessCMOS = self.param['widthAccessCMOS']
        self.minSenseVoltage = self.param['minSenseVoltage']
        self.heightInFeatureSize1T1R = self.param['heightInFeatureSize1T1R']
        self.heightInFeatureSizeCrossbar = self.param['heightInFeatureSizeCrossbar']
        self.widthInFeatureSize1T1R = self.param['widthInFeatureSize1T1R']
        self.widthInFeatureSizeCrossbar = self.param['widthInFeatureSizeCrossbar']

        if self.cell_type == 'SRAM':
            self.heightInFeatureSize = self.heightInFeatureSizeSRAM
            self.widthInFeatureSize = self.widthInFeatureSizeSRAM
        else:
            self.heightInFeatureSize = self.heightInFeatureSize1T1R if self.accessType == 'CMOS_access' else self.heightInFeatureSizeCrossbar

        self.SubArray = SubArray(numCol=self.num_col,numRow = self.num_row,num_mu= self.num_mu,num_sigma=self.num_sigma,relaxArrayCellWidth = False,relaxArrayCellHeight = False,tech=self.tech,config=config,mapping=self.mapping,param=param,RNG=self.RNG)
        self.SubArray_area, self.SubArray_height, self.SubArray_width,_ = self.SubArray.calculate_area()
        self.BusOutput = Bus(mode='VERTICAL',num_row=self.numSubArrayRow,num_col=self.numSubArrayCol,delay_tolerance=0,bus_width=64,unit_height=self.SubArray_height,unit_width=self.SubArray_width,clk_freq=self.clk_freq,param=param,config=config,tech=self.tech)

        self.initialized = True

    def calculate_area(self, overlap=False):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating area.")

        BusOutput_area = self.BusOutput.calculate_area(folded_ratio=1.0, overlap=False)
        area = self.numSubArrayRow * self.numSubArrayCol * self.SubArray_area + BusOutput_area

        height = math.sqrt(area)
        width = area / height

        return area, height, width

    def calculate_latency(self, num_read):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating latency.")
        SubArray_read_latency = self.SubArray.calculate_latency(calculate_clk_freq = self.clk_freq,validated=False)
        BusOutput_read_latency = self.BusOutput.calculate_latency(num_read = 1)
        
        read_latency = SubArray_read_latency + BusOutput_read_latency
        return read_latency * num_read

    def calculate_power(self,input_vector,weight_matrix, num_bit_access, num_read):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating power.")
        SubArray_read_energy,SubArray_write_energy,SubArray_leakage = self.SubArray.calculate_power(input_vector, weight_matrix)
        BusOutput_read_energy,BusOutput_leakage = self.BusOutput.calculate_power(num_bit_access = 64, num_read = 1)
        read_dynamic_energy = SubArray_read_energy + BusOutput_read_energy
        leakage = SubArray_leakage + BusOutput_leakage
        

        return read_dynamic_energy * num_bit_access * num_read, leakage

