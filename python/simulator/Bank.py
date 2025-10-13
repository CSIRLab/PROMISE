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
from periphery.HTree import HTree
from simulator.subarray import SubArray
from simulator.MAT import MAT

class Bank:
    def __init__(self, tech,config, param, mapping, RNG, numMATRow, numMATCol, numSubArrayRow, numSubArrayCol, numCol, numRow, num_mu, num_sigma):

        self.tech = tech
        self.param = param
        self.config = config
        self.mapping = mapping
        self.RNG = RNG
        self.initialized = False
        self.numMATRow = numMATRow
        self.numMATCol = numMATCol
        self.num_col = numCol
        self.num_row = numRow
        self.numSubArrayRow = numSubArrayRow
        self.numSubArrayCol = numSubArrayCol
        self.num_mu = num_mu
        self.num_sigma = num_sigma

        self.feature_size = self.tech.get_param('featureSize')
        self.pnSizeRatio = self.tech.get_param('pnSizeRatio')
        self.temp = self.config['temperature']
        self.vdd = self.tech.get_param('vdd')
        self.cell_type = self.config['device_type']
        self.clk_freq = self.config['frequency']

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

        self.MAT = MAT(numSubArrayRow=self.numSubArrayRow,numSubArrayCol=self.numSubArrayCol,numCol=self.num_col,numRow=self.num_row,num_mu=self.num_mu,num_sigma=self.num_sigma,param=self.param, mapping = self.mapping, config=self.config, RNG = self.RNG, tech=self.tech)
        self.HTree = HTree(num_row=self.numMATRow,num_col=self.numMATCol,delay_tolerance=0,bus_width=self.num_col,param=self.param,config=self.config,tech=self.tech)
        

        self.initialized = True

    def calculate_area(self, overlap=False):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating area.")
        self.MAT_area, self.MAT_height, self.MAT_width = self.MAT.calculate_area(overlap=False)
        self.HTree_area = self.HTree.calculate_area(unit_height=self.MAT_height , unit_width = self.MAT_width, folded_ratio=1)

        area = self.numMATRow * self.numMATCol * self.MAT_area + self.HTree_area

        height = math.sqrt(area)
        width = area / height

        return area, height, width

    def calculate_latency(self, num_read):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating latency.")
        MAT_read_latency = self.MAT.calculate_latency(num_read=1)
        HTree_read_latency = self.HTree.calculate_latency(x_init=0,y_init=0,x_end=0,y_end=0,unit_height=self.MAT_height,unit_width=self.MAT_width,num_read=1)
        
        read_latency = MAT_read_latency + HTree_read_latency
        return read_latency * num_read

    def calculate_power(self,input_vector,weight_matrix, num_bit_access, num_read):
        if not self.initialized:
            raise ValueError("Bus must be initialized before calculating power.")
        MAT_read_energy,MAT_leakage = self.MAT.calculate_power(input_vector,weight_matrix, num_bit_access=num_bit_access, num_read=num_read)
        HTree_read_energy,HTree_leakage = self.HTree.calculate_power(x_init=0,y_init=0,x_end=0,y_end=0,unit_height=self.MAT_height,unit_width=self.MAT_width, num_Bit_Access=1, num_read=1)
        
        read_dynamic_energy = MAT_read_energy + HTree_read_energy
        leakage = MAT_leakage + HTree_leakage
        

        return read_dynamic_energy * num_bit_access * num_read, leakage

