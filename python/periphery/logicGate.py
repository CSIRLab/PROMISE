import math
import sys
sys.path.append('../../python/')  
from periphery import constant
print(constant.INV)

def calculate_mos_gate_cap(width, tech):
    """
    calculate the gate capacitance of a transistor (including ideal gate capacitance, overlap, fringe, and polywire capacitance)

    """
    # use the effective width for FinFET or bulk CMOS
    if tech.get_param('featureSize') >= 22e-9 or tech.get_param('transistorType') != 'conventional':
        # Bulk CMOS
        width_eff = width
    else:
        # FinFET: convert width to number of fins
        width_scaled = width * (tech.get_param('PitchFin') / (2 * tech.get_param('featureSize')))
        num_fins = math.ceil(width_scaled / tech.get_param('PitchFin'))
        fin_surface_area = 2 * tech.get_param('heightFin') + tech.get_param('widthFin')
        width_eff = num_fins * fin_surface_area

    gate_cap = (tech.get_param('capIdealGate') + tech.get_param('capOverlap') + tech.get_param('capFringe')) * width_eff
    gate_cap += tech.get_param('phyGateLength') * tech.get_param('capPolywire')

    return gate_cap

def calculate_logicgate_area(gateType, num_Input, width_NMOS, width_PMOS,height_Transistor_Region, tech):
    # if tech.get_param('featureSize') <= 14e-9:  # FinFET
    #     width_NMOS *= tech('PitchFin') / (2 * tech.get_param('featureSize'))
    #     width_PMOS *= tech('PitchFin') / (2 * tech.get_param('featureSize'))
    #     height_Transistor_Region *= (MAX_TRANSISTOR_HEIGHT_FINFET / MAX_TRANSISTOR_HEIGHT)

    ratio = width_PMOS / (width_PMOS + width_NMOS) if (width_PMOS + width_NMOS) != 0 else 0
    num_Folded_PMOS = num_Folded_NMOS = 1

    if tech.get_param('featureSize') >= 22e-9 or tech.get_param('transistorType') != 'conventional':
        #consider electrical width of PMOS and NMOS which is y direction
        # Bulk
        if ratio == 0:  # no PMOS
            maxwidth_PMOS = 0
            maxwidth_NMOS = height_Transistor_Region - (constant.MIN_POLY_EXT_DIFF + constant.MIN_GAP_BET_FIELD_POLY / 2) * 2 * tech.get_param('featureSize')
        elif ratio == 1:    # no NMOS
            maxwidth_PMOS = height_Transistor_Region - (constant.MIN_POLY_EXT_DIFF + constant.MIN_GAP_BET_FIELD_POLY / 2) * 2 * tech.get_param('featureSize')
            maxwidth_NMOS = 0
        else:
            temp = height_Transistor_Region - constant.MIN_GAP_BET_P_AND_N_DIFFS * tech.get_param('featureSize') - (constant.MIN_POLY_EXT_DIFF + constant.MIN_GAP_BET_FIELD_POLY / 2) * 2 * tech.get_param('featureSize')
            maxwidth_PMOS = ratio * temp
            maxwidth_NMOS = (1 - ratio) * temp 
        #consider physical width of PMOS and NMOS which is x direction
        unit_Width_x_P, heightRegionP = 0, 0
        if width_PMOS > 0:
            if width_PMOS <= maxwidth_PMOS:
                unit_Width_x_P = 2 * (constant.POLY_WIDTH + constant.MIN_GAP_BET_GATE_POLY) * tech.get_param('featureSize') #########################################I don't know why this is multiplied by 2
                heightRegionP = width_PMOS
            else:
                num_Folded_PMOS = math.ceil(width_PMOS / maxwidth_PMOS)
                unit_Width_x_P = (num_Folded_PMOS + 1) * (constant.POLY_WIDTH + constant.MIN_GAP_BET_GATE_POLY) * tech.get_param('featureSize') #########################################sane problem, it add one more unit width
                heightRegionP = maxwidth_PMOS

        unit_Width_x_N, heightRegionN = 0, 0
        if width_NMOS > 0:
            if width_NMOS <= maxwidth_NMOS:
                unit_Width_x_N = 2 * (constant.POLY_WIDTH + constant.MIN_GAP_BET_GATE_POLY) * tech.get_param('featureSize')
                heightRegionN = width_NMOS
            else:
                num_Folded_NMOS = math.ceil(width_NMOS / maxwidth_NMOS)
                unit_Width_x_N = (num_Folded_NMOS + 1) * (constant.POLY_WIDTH + constant.MIN_GAP_BET_GATE_POLY) * tech.get_param('featureSize')
                heightRegionN = maxwidth_NMOS

    else:
        # # FinFET
        # def calc_max_fin(ratio_part):
        #     return math.floor(
        #         ratio_part * (height_Transistor_Region - MIN_GAP_BET_P_AND_N_DIFFS * tech.get_param('featureSize') -
        #                       (MIN_POLY_EXT_DIFF + MIN_GAP_BET_FIELD_POLY / 2) * 2 * tech.get_param('featureSize')) / tech('PitchFin')
        #     ) + 1

        # maxNumPFin, maxNumNFin = 0, 0
        # if ratio == 0:
        #     maxNumNFin = calc_max_fin(1)
        # elif ratio == 1:
        #     maxNumPFin = calc_max_fin(1)
        # else:
        #     maxNumPFin = calc_max_fin(ratio)
        #     maxNumNFin = calc_max_fin(1 - ratio)

        # unit_Width_x_P, heightRegionP = 0, 0
        # NumPFin = math.ceil(width_PMOS / tech('PitchFin'))
        # if NumPFin > 0:
        #     if NumPFin <= maxNumPFin:
        #         unit_Width_x_P = 2 * (POLY_WIDTH_FINFET + MIN_GAP_BET_GATE_POLY_FINFET) * tech.get_param('featureSize')
        #         heightRegionP = (NumPFin - 1) * tech('PitchFin') + tech('widthFin')
        #     else:
        #         num_Folded_PMOS = math.ceil(NumPFin / maxNumPFin)
        #         unit_Width_x_P = (num_Folded_PMOS + 1) * (POLY_WIDTH_FINFET + MIN_GAP_BET_GATE_POLY_FINFET) * tech.get_param('featureSize')
        #         heightRegionP = (maxNumPFin - 1) * tech('PitchFin') + tech('widthFin')

        # unit_Width_x_N, heightRegionN = 0, 0
        # NumNFin = math.ceil(width_NMOS / tech('PitchFin'))
        # if NumNFin > 0:
        #     if NumNFin <= maxNumNFin:
        #         unit_Width_x_N = 2 * (POLY_WIDTH_FINFET + MIN_GAP_BET_GATE_POLY_FINFET) * tech.get_param('featureSize')
        #         heightRegionN = (NumNFin - 1) * tech('PitchFin') + tech('widthFin')
        #     else:
        #         num_Folded_NMOS = math.ceil(NumNFin / maxNumNFin)
        #         unit_Width_x_N = (num_Folded_NMOS + 1) * (POLY_WIDTH_FINFET + MIN_GAP_BET_GATE_POLY_FINFET) * tech.get_param('featureSize')
        #         heightRegionN = (maxNumNFin - 1) * tech('PitchFin') + tech('widthFin')
        print("FinFET support is not implemented in this version.")

    # # gate type width computation
    # def shared_region_correction(unit_width, isFinFET=False):
    #     if isFinFET:
    #         poly_width = POLY_WIDTH_FINFET + MIN_GAP_BET_GATE_POLY_FINFET
    #     else:
    #         poly_width = POLY_WIDTH + MIN_GAP_BET_GATE_POLY
    #     return unit_width * num_Input - (num_Input - 1) * tech.get_param('featureSize') * poly_width

    # # gate type width computation
    if gateType == constant.INV:
        width_x_P = unit_Width_x_P
        width_x_N = unit_Width_x_N
    elif gateType in (constant.NOR, constant.NAND):
        isFinFET = tech.get_param('featureSize') < 22e-9 and tech.get_param('transistorType') == 'conventional'
        if num_Folded_PMOS == 1 and num_Folded_NMOS == 1:   # no folding
            width_x_P = unit_Width_x_P/2 * (num_Input+1)
            width_x_N = unit_Width_x_N/2 * (num_Input+1)
        else:
            width_x_P = unit_Width_x_P/2 * (num_Input * num_Folded_PMOS+1)
            width_x_N = unit_Width_x_N/2 * (num_Input * num_Folded_NMOS+1)
    else:
        width_x_P = width_x_N = 0

    print("maxtransistor height:", height_Transistor_Region)
    print("maxwidth_PMOS:", maxwidth_PMOS)
    print("maxwidth_NMOS:", maxwidth_NMOS)
    # print("unit_Width_x_P:", unit_Width_x_P)
    # print("unit_Width_x_N:", unit_Width_x_N)
    # print("num_Input:", num_Input)
    print("num_Folded_PMOS:", num_Folded_PMOS)
    # print(unit_Width_x_P/2)
    # print((num_Input+1))
    # print(unit_Width_x_N/2 * (num_Input+1))
    # print("width_x_P:", width_x_P)
    # print("width_x_N:", width_x_N)
    width_x = max(width_x_P, width_x_N)
    height_y = height_Transistor_Region
    return width_x, height_y, width_x * height_y


def calculate_logicgate_cap(gate_type, num_Input, width_NMOS, width_PMOS, height_transistor_region, tech):
    
    ratio = width_PMOS / (width_PMOS + width_NMOS) if (width_PMOS + width_NMOS) > 0 else 0
    num_folded_pmos = num_folded_nmos = 1
    # FinFET adjustment
    if tech.get_param('featureSize') <= 14e-9:
        width_NMOS *= tech('PitchFin') / (2 * tech.get_param('featureSize'))
        width_PMOS *= tech('PitchFin') / (2 * tech.get_param('featureSize'))
        height_transistor_region *= (34 / 28)

    #consider electrical width of PMOS and NMOS which is y direction
    if tech.get_param('featureSize') >= 22e-9 or tech.get_param('transistorType') != 'conventional':  
        # Bulk
        if ratio == 0:  # no PMOS
            max_width_pmos = 0
            max_width_nmos = height_transistor_region - (constant.MIN_POLY_EXT_DIFF + constant.MIN_GAP_BET_FIELD_POLY / 2)* 2 * tech.get_param('featureSize')
        elif ratio == 1:    # no NMOS
            max_width_pmos = height_transistor_region - (constant.MIN_POLY_EXT_DIFF + constant.MIN_GAP_BET_FIELD_POLY / 2) * 2 * tech.get_param('featureSize')
            max_width_nmos = 0
        else:
            temp = height_transistor_region - constant.MIN_GAP_BET_P_AND_N_DIFFS * tech.get_param('featureSize') - (constant.MIN_POLY_EXT_DIFF + constant.MIN_GAP_BET_FIELD_POLY / 2) * 2 * tech.get_param('featureSize')
            maxwidth_PMOS = ratio * temp
            maxwidth_NMOS = (1 - ratio) * temp 
            
        #consider physical width of PMOS and NMOS which is x direction\
        if width_PMOS > 0:
            if width_PMOS <= maxwidth_PMOS:
                unit_width_Drain_P = constant.MIN_GAP_BET_GATE_POLY * tech.get_param('featureSize') 
                unit_Width_Source_P = unit_width_Drain_P;
                height_Drain_P = width_PMOS
            else:
                num_Folded_PMOS = math.ceil(width_PMOS / maxwidth_PMOS)
                unit_width_Drain_P = math.ceil((num_Folded_PMOS+1)/2) * constant.MIN_GAP_BET_GATE_POLY * tech.get_param('featureSize') 
                unit_width_Source_P = math.floor((num_Folded_PMOS+1)/2) * constant.MIN_GAP_BET_GATE_POLY * tech.get_param('featureSize') 
                height_Drain_P = maxwidth_PMOS
        else:
            unit_width_Drain_P = unit_Width_Source_P = height_Drain_P = 0

        if width_NMOS > 0:
            if width_NMOS <= maxwidth_NMOS:
                unit_width_Drain_N = constant.MIN_GAP_BET_GATE_POLY * tech.get_param('featureSize')
                unit_Width_Source_N = unit_width_Drain_N
                height_Drain_N = width_NMOS
            else:
                num_Folded_NMOS = math.ceil(width_NMOS / maxwidth_NMOS)
                unit_width_Drain_N = math.ceil((num_Folded_NMOS + 1) / 2) * tech.get_param('featureSize') * constant.MIN_GAP_BET_GATE_POLY
                unit_Width_Source_N = math.floor((num_Folded_NMOS + 1) / 2) * tech.get_param('featureSize') * constant.MIN_GAP_BET_GATE_POLY
                height_Drain_N = maxwidth_NMOS
        else:
            unit_width_Drain_N = unit_Width_Source_N = height_Drain_N = 0
    
    else:  # FinFET
        print("FinFET support is not implemented in this version.")
        # pitch = tech('PitchFin')
        # tech.get_param('featureSize') = tech.get_param('featureSize')
        # height_effective = height_transistor_region - (1.0 + 1.6 / 2) * 2 * tech.get_param('featureSize')
        # max_num_pfin = int((ratio * (height_effective - 3.5 * tech.get_param('featureSize'))) / pitch) + 1
        # max_num_nfin = int(((1 - ratio) * (height_effective - 3.5 * tech.get_param('featureSize'))) / pitch) + 1

        # num_pfin = math.ceil(width_pmos / pitch)
        # num_nfin = math.ceil(width_nmos / pitch)

        # def calc_drain_finfet(num_fin, max_fin, tech_val):
        #     if num_fin <= max_fin:
        #         unit_drain = tech_val
        #         unit_source = tech_val
        #         height_drain = (num_fin - 1) * pitch + tech('widthFin')
        #     else:
        #         folds = math.ceil(num_fin / max_fin)
        #         unit_drain = math.ceil((folds + 1) / 2) * tech_val
        #         unit_source = math.floor((folds + 1) / 2) * tech_val
        #         height_drain = (max_fin - 1) * pitch + tech('widthFin')
        #     return unit_drain, unit_source, height_drain, folds if num_fin > max_fin else 1

        # unit_width_drain_p, unit_width_source_p, height_drain_p, num_folded_pmos = calc_drain_finfet(
        #     num_pfin, max_num_pfin, tech.get_param('featureSize') * 3.9)
        # unit_width_drain_n, unit_width_source_n, height_drain_n, num_folded_nmos = calc_drain_finfet(
        #     num_nfin, max_num_nfin, tech.get_param('featureSize') * 3.9)

    # Gate-specific drain capacitance model (INV, NOR, NAND)
    if gate_type == constant.INV:
        if width_PMOS > 0:
            width_drain_p = unit_width_Drain_P * ((num_folded_pmos + 1) // 2)
            width_drain_sidewall_p = (unit_width_Drain_P + height_Drain_P) * 2 * ((num_folded_pmos+1) // 2)
        if width_NMOS > 0:
            width_drain_n = unit_width_Drain_N * ((num_folded_pmos + 1) // 2)
            width_drain_sidewall_n = (unit_width_Drain_N + height_Drain_N) * 2 * ((num_folded_nmos+1) // 2)
    elif gate_type == constant.NOR:
        if width_PMOS > 0:              #pmos only has one drain as output
            width_drain_p = unit_width_Drain_P * ((num_folded_pmos + 1) // 2)
            width_drain_sidewall_p = (unit_width_Drain_P + height_Drain_P) * 2 * ((num_folded_pmos+1) // 2)
        if width_NMOS > 0:              #nmos has all the drains as output
            width_drain_n = unit_width_Drain_N * ((num_folded_nmos * num_Input + 1) // 2)
            width_drain_sidewall_n = (unit_width_Drain_N + height_Drain_N) * 2 * ((num_folded_nmos * num_Input+1) // 2)
    elif gate_type == constant.NAND:
        if width_PMOS > 0:              #pmos has all the drains as output
            width_drain_p = unit_width_Drain_P * ((num_folded_pmos * num_Input + 1) // 2)
            width_drain_sidewall_p = (unit_width_Drain_P + height_Drain_P) * 2 * ((num_folded_pmos * num_Input+1) // 2)
        if width_NMOS > 0:              #nmos only has one drain as output
            width_drain_n = unit_width_Drain_N * ((num_folded_pmos + 1) // 2)
            width_drain_sidewall_n = (unit_width_Drain_N + height_Drain_N) * 2 * ((num_folded_nmos+1) // 2)
            


    cap_drain_bottom_p = width_drain_p * height_Drain_P * tech.get_param('capJunction')
    cap_drain_bottom_n = width_drain_n * height_Drain_N * tech.get_param('capJunction')
    cap_drain_side_p = width_drain_sidewall_p * tech.get_param('capSidewall')
    cap_drain_side_n = width_drain_sidewall_n * tech.get_param('capSidewall')
    cap_drain_channel_p = num_folded_pmos * height_Drain_P * tech.get_param('capDrainToChannel')
    cap_drain_channel_n = num_folded_nmos * height_Drain_N * tech.get_param('capDrainToChannel')

    cap_output = cap_drain_bottom_p + cap_drain_bottom_n + cap_drain_side_p + cap_drain_side_n + \
                 cap_drain_channel_p + cap_drain_channel_n
    cap_input = calculate_mos_gate_cap(width_NMOS, tech) + calculate_mos_gate_cap(width_PMOS, tech)

    return cap_input, cap_output


def calculate_drain_cap(mos_type, width,height_transistor_region,tech ):
    if mos_type == 'nmos':
        drain_cap = calculate_logicgate_cap(constant.INV,1, width, 0, height_transistor_region, tech)[1]
    else:
        drain_cap = calculate_logicgate_cap(constant.INV,1, 0, width, height_transistor_region, tech)[1]

    return drain_cap

def calculate_logicgate_leakage(gate_type, num_input, width_nmos, width_pmos, temperature, tech):
    temp_index = int(temperature) - 300
    if temp_index > 100 or temp_index < 0:
        raise ValueError("Error: Temperature is out of range (300K to 400K)")

    leak_n = tech.get_param('currentOffNmos')
    leak_p = tech.get_param('currentOffPmos')

    if tech.get_param('featureSize') >= 22e-9 or tech.get_param('transistorType') != 'conventional':
        # Bulk CMOS
        width_nmos_eff = width_nmos
        width_pmos_eff = width_pmos
    else:  # FinFET
        width_nmos *= tech.get_param('PitchFin') / (2 * tech.get_param('featureSize'))
        width_pmos *= tech.get_param('PitchFin') / (2 * tech.get_param('featureSize'))
        width_nmos_eff = math.ceil(width_nmos / tech.get_param('PitchFin')) * (2 * tech.get_param('heightFin') + tech.get_param('widthFin'))
        width_pmos_eff = math.ceil(width_pmos / tech.get_param('PitchFin')) * (2 * tech.get_param('heightFin') + tech.get_param('widthFin'))

    if gate_type == constant.INV:  # INV
        leakage_n = width_nmos_eff * leak_n[temp_index]
        leakage_p = width_pmos_eff * leak_p[temp_index]
        return (leakage_n + leakage_p) / 2
    elif gate_type == constant.NOR:  # NOR
        leakage_n = width_nmos_eff * leak_n[temp_index] * num_input
        if num_input == 2:
            return constant.AVG_RATIO_LEAK_2INPUT_NOR * leakage_n
        else:
            return constant.AVG_RATIO_LEAK_3INPUT_NOR * leakage_n
    elif gate_type == constant.NAND: # NAND
        leakage_p = width_pmos_eff * leak_p[temp_index] * num_input
        if num_input == 2:
            return constant.AVG_RATIO_LEAK_2INPUT_NAND * leakage_p
        else:
            return constant.AVG_RATIO_LEAK_3INPUT_NAND * leakage_p
    else:
        return 0.0


def calculate_on_resistance(width, mos_type, temperature, tech):
    """
    calculate the on-resistance (R_on) of a MOS transistor given its width, type, and temperature
    returns the on-resistance in ohms (Ω)
    """
    temp_index = int(temperature) - 300
    if temp_index < 0 or temp_index > 100:
        raise ValueError("Temperature is out of range [300K, 400K]")

    # calculate effective width considering FinFET or bulk CMOS
    if tech.get_param('featureSize') >= 22e-9 or tech.get_param('transistorType') != 'conventional':
        # Bulk CMOS
        width_eff = width
    else:
        # FinFET: convert width to number of fins
        width *= tech('PitchFin') / (2 * tech.get_param('featureSize'))
        width_eff = math.ceil(width / tech('PitchFin')) * (2 * tech('heightFin') + tech('widthFin'))

    # based on lookup table for current on
    if mos_type == 0:  # NMOS
        I_on = tech.get_param('currentOnNmos')[temp_index]
    else:              # PMOS
        I_on = tech.get_param('currentOnPmos')[temp_index]

    # calculate the on-resistance
    resistance = tech.get_param('effectiveResistanceMultiplier') * tech.get_param('vdd') / (I_on * width_eff)
    return resistance


def calculate_transconductance(width, mos_type, tech):
    """
    calculate the transconductance (g_m) of a MOS transistor given its width and type
    returns the transconductance in siemens (S)
    """
    # assume Vgs is at 70% of Vdd for overdrive voltage calculation
    v_ov = 0.7 * tech.get_param('vdd') - tech.get_param('vth')  # effective overdrive voltage

    # gm = (2 * I_on) / V_ov
    if mos_type == 0:  # NMOS
        gm = (2 * tech.get_param('current_gmNmos')) * width / v_ov
    else:  # PMOS
        gm = (2 * tech.get_param('current_gmPmos')) * width / v_ov

    return gm


def horowitz(tr, beta, ramp_input):
    """
    Horowitz delay model estimation.
    
    Parameters:
    tr (float): intrinsic delay of the gate (time constant, tau)
    beta (float): output response factor (usually set to 0.5 in simplified form)
    ramp_input (float): input signal ramp time (transition time)
    
    Returns:
    result (float): propagation delay
    ramp_output (float): estimated output ramp rate (1/result)

    t_delay = tr * sqrt(ln(vs)^2 + 2 * alpha * beta * (1 - vs))
    alpha = 1 / (ramp_input * tr) is the ramp factor
    beta is gate drive factor like gm/C
    VS is switching voltage, typically normalized to 0.5 in many models.
    """
    vs = 0.5  # Normalized switching voltage
    alpha = 1 / (ramp_input * tr)
    beta = 0.5  # Simplified model (used in CACTI and similar tools)

    result = tr * math.sqrt(math.log(vs)**2 + 2 * alpha * beta * (1 - vs))
    ramp_output = (1 - vs) / result if result != 0 else 0

    return result, ramp_output

def nonlinear_resistance(R, NL, Vw, Vr, V):
    """
    calculate the nonlinear resistance based on the given parameters.
    parameters:
    R   : resistance at reference voltage Vr
    NL  : nonlinear coefficient (I(Vw/2)/I(V))
    Vw  : write voltage window (e.g., ±1V)
    Vr  : reference voltage (e.g., 0.1V)
    V   : actual applied voltage

    returns:
    R_NL : nonlinear resistance at voltage V
    """
    if V == 0:
        return float('inf')  # when V is zero, resistance is infinite
    exponent = (Vr - V) / (Vw / 2)
    R_NL = R * V / Vr * (NL ** exponent)
    return R_NL
