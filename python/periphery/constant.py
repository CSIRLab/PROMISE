# constants.py

# Gate Types
INV = 0
NOR = 1
NAND = 2

# Transistor Types
NMOS = 0
PMOS = 1

# Transistor Size Constraints
MAX_NMOS_SIZE = 100
MIN_NMOS_SIZE = 2

# Transistor Region Heights
MAX_TRANSISTOR_HEIGHT = 28
MAX_TRANSISTOR_HEIGHT_FINFET = 34


MAX_TRANSISTOR_HEIGHT_14nm = 41.142 # Samsung
MAX_TRANSISTOR_HEIGHT_10nm = 33.0 # TSMC
MAX_TRANSISTOR_HEIGHT_7nm = 34.285 # TSMC
MAX_TRANSISTOR_HEIGHT_5nm = 36.0 # IRDS 2021
MAX_TRANSISTOR_HEIGHT_3nm =  48.0 # IRDS 2022
MAX_TRANSISTOR_HEIGHT_2nm =  57.0 # IRDS 2022
MAX_TRANSISTOR_HEIGHT_1nm =  80.0 # IRDS 2022

# // 1.4 update : PN separation, region outside fin (half of the inter-spacing between outermost fins in adjacent standard cells) - updated
# // OUTER_HEIGHT_REGION = Cell height - MIN_GAP_BET_P_AND_N_DIFFS (PN gap) - Fin region
MIN_GAP_BET_P_AND_N_DIFFS_14nm  = 9.71 # Fin pitch*3 - Fin width
OUTER_HEIGHT_REGION_14nm =  10.285
MIN_GAP_BET_P_AND_N_DIFFS_10nm = 10.0 # Fin pitch*3 - Fin width
OUTER_HEIGHT_REGION_10nm = 8.2
MIN_GAP_BET_P_AND_N_DIFFS_7nm = 11.85 # Fin pitch*3 - Fin width
OUTER_HEIGHT_REGION_7nm = 12.28
MIN_GAP_BET_P_AND_N_DIFFS_5nm = 12.8 # IRDS 2021
OUTER_HEIGHT_REGION_5nm = 19.4
MIN_GAP_BET_P_AND_N_DIFFS_3nm = 15.0 # IRDS 2022
OUTER_HEIGHT_REGION_3nm = 12.666
MIN_GAP_BET_P_AND_N_DIFFS_2nm = 20.0 # IRDS 2022
OUTER_HEIGHT_REGION_2nm = 22.0 
MIN_GAP_BET_P_AND_N_DIFFS_1nm = 15.0  # IRDS 2022
OUTER_HEIGHT_REGION_1nm = 40.0 
# Contacted Poly Pitch and Width Trends
CPP_14nm = 5.571            # Samsung
POLY_WIDTH_14nm =  1.857    # IRDS 2016
CPP_10nm =  6.4             # TSMC
POLY_WIDTH_10nm =  2.2      # IRDS 2017
CPP_7nm = 8.1428            # TSMC
POLY_WIDTH_7nm = 3.14       # IRDS 2018
CPP_5nm = 10.2              # IRDS 2021
POLY_WIDTH_5nm = 4.0        # IRDS 2021
CPP_3nm = 16.0              # IRDS 2022
POLY_WIDTH_3nm = 6.0        # IRDS 2022
CPP_2nm = 22.5              # IRDS 2022
POLY_WIDTH_2nm = 7.0        # IRDS 2022
CPP_1nm = 40.0              # IRDS 2022
POLY_WIDTH_1nm = 12.0       # IRDS 2022

# Layout Design Rules
MIN_GAP_BET_P_AND_N_DIFFS = 3.5
MIN_GAP_BET_SAME_TYPE_DIFFS = 1.6
MIN_GAP_BET_GATE_POLY = 2.8
MIN_GAP_BET_GATE_POLY_FINFET = 3.9
MIN_GAP_BET_CONTACT_POLY = 0.7
CONTACT_SIZE = 1.3
MIN_WIDTH_POWER_RAIL = 3.4
MIN_POLY_EXT_DIFF = 1.0
MIN_GAP_BET_FIELD_POLY = 1.6
POLY_WIDTH = 1.0
POLY_WIDTH_FINFET = 1.4

# Routing Pitch
M2_PITCH = 3.2
M3_PITCH = 2.8

# Leakage Ratios
AVG_RATIO_LEAK_2INPUT_NAND = 0.48
AVG_RATIO_LEAK_3INPUT_NAND = 0.31
AVG_RATIO_LEAK_2INPUT_NOR = 0.95
AVG_RATIO_LEAK_3INPUT_NOR = 0.62

# Sense Amplifier Transistor Widths
W_SENSE_P = 7.5
W_SENSE_N = 3.75
W_SENSE_ISO = 12.5
W_SENSE_EN = 5.0
W_SENSE_MUX = 9.0

# IR Drop and Region Ratios
IR_DROP_TOLERANCE = 0.25
LINEAR_REGION_RATIO = 0.20
HEIGHT_WIDTH_RATIO_LIMIT = 5
RATIO_READ_THRESHOLD_VS_VOLTAGE = 0.2

# Routing Modes
ROW_MODE = 0
COL_MODE = 1

#Metal Pitch trend -> needed for row-decoder layout
#Assume single patterning for 14 nm and beyond and assume Minimum Metal Pitch=M1 pitch=M2 pitch=M3 pitch

M2_PITCH      = 3.2  # along width
M2_PITCH_14nm = 4.57
M2_PITCH_10nm = 4.4
M2_PITCH_7nm  = 5.71
M2_PITCH_5nm  = 6.0
M2_PITCH_3nm  = 8.0
M2_PITCH_2nm  = 10.0
M2_PITCH_1nm  = 16.0

M3_PITCH      = 2.8  # along height
M3_PITCH_14nm = 4.57
M3_PITCH_10nm = 4.4
M3_PITCH_7nm  = 5.71
M3_PITCH_5nm  = 6.0
M3_PITCH_3nm  = 8.0
M3_PITCH_2nm  = 10.0
M3_PITCH_1nm  = 16.0
