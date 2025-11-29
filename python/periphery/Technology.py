class Technology:
    def __init__(self, node_nm=45, roadmap='HP', transistor_type='conventional'):
        self.node_nm = node_nm
        self.roadmap = roadmap
        self.transistor_type = transistor_type
        self.initialized = False
        self.params = {}
        self._select_tech_params()

    def _select_tech_params(self):
        if self.initialized:
            print("Warning: Already initialized!")
            return
        if self.transistor_type == 'conventional':
            PitchFin= 0
            heightFin = 0
            widthFin = 0
            PitchFin = 0
            if self.node_nm == 130:
                if self.roadmap == 'HP':
                    # /* PTM model: 130nm_HP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.3
                    vth = 128.4855e-3
                    phyGateLength = 7.5e-8
                    capIdealGate = 6.058401e-10
                    capFringe = 6.119807e-10
                    effectiveResistanceMultiplier = 1.54;	#/* from CACTI */
                    current_gmNmos=3.94E+02
                    current_gmPmos=2.61E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        0.93e3, 0.91e3, 0.89e3, 0.87e3, 0.85e3, 
                        0.83e3, 0.81e3, 0.79e3, 0.77e3, 0.75e3, 0.74e3
                    ]
                    currentOnPmos = [
                        0.43e3, 0.41e3, 0.38e3, 0.36e3, 0.34e3, 
                        0.32e3, 0.30e3, 0.28e3, 0.26e3, 0.25e3, 0.24e3
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.00e-3, 119.90e-3, 142.20e-3, 167.00e-3, 194.30e-3, 
                        224.30e-3, 256.80e-3, 292.00e-3, 329.90e-3, 370.50e-3, 413.80e-3
                    ]

                    currentOffPmos = [
                        100.20e-3, 113.60e-3, 127.90e-3, 143.10e-3, 159.10e-3, 
                        175.80e-3, 193.40e-3, 211.70e-3, 230.80e-3, 250.70e-3, 271.20e-3
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
                else:
                #     //(deviceRoadmap == LSTP)
				# /* PTM model: 130nm_LP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.3
                    vth = 466.0949e-3
                    phyGateLength = 7.5e-8
                    capIdealGate = 1.8574e-9
                    capFringe = 9.530642e-10
                    cap_draintotal = capFringe/2
                    effectiveResistanceMultiplier = 1.54;	#/* from CACTI */
                    current_gmNmos=3.87E+01
                    current_gmPmos=5.67E+01
                    # On-current (A/m)
                    currentOnNmos = [
                        300.70, 273.40, 249.40, 228.40, 209.90, 
                        193.50, 179.00, 166.00, 154.40, 144.00, 134.60
                    ]
                    currentOnPmos = [
                        150.70, 136.20, 123.60, 112.70, 103.20, 
                        94.88, 87.54, 81.04, 75.25, 70.08, 65.44
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.20e-6, 135.90e-6, 181.20e-6, 237.80e-6, 307.30e-6, 
                        391.90e-6, 493.30e-6, 613.70e-6, 755.30e-6, 920.20e-6, 1111.0e-6
                    ]

                    currentOffPmos = [
                        100.20e-6, 132.80e-6, 173.00e-6, 221.90e-6, 280.70e-6, 
                        350.40e-6, 432.20e-6, 527.20e-6, 636.80e-6, 761.90e-6, 903.80e-6
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 90:
                if self.roadmap == 'HP':
                    # /* PTM model: 90nm_HP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.2
                    vth = 146.0217e-3
                    phyGateLength = 5.5e-8
                    capIdealGate = 5.694423e-10
                    capFringe = 5.652302e-10
                    effectiveResistanceMultiplier = 1.54
                    current_gmNmos=4.95E+02
                    current_gmPmos=3.16E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        1.07e3, 1.05e3, 1.03e3, 1.01e3, 0.99e3,
                        0.97e3, 0.95e3, 0.93e3, 0.90e3, 0.88e3, 0.86e3
                    ]
                    currentOnPmos = [
                        0.54e3, 0.50e3, 0.47e3, 0.44e3, 0.41e3,
                        0.39e3, 0.37e3, 0.34e3, 0.32e3, 0.31e3, 0.29e3
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.8e-3, 120.8e-3, 143.4e-3, 168.6e-3, 196.6e-3,
                        227.4e-3, 261.1e-3, 297.7e-3, 337.3e-3, 379.8e-3, 425.4e-3
                    ]
                    currentOffPmos = [
                        100.00e-3, 114.00e-3, 128.90e-3, 144.80e-3, 161.60e-3,
                        179.30e-3, 197.90e-3, 217.40e-3, 237.90e-3, 259.10e-3, 281.30e-3
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
                else:
                    # /* PTM model: 90nm_LP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.2
                    vth = 501.3229e-3
                    phyGateLength = 5.5e-8
                    capIdealGate = 1.5413e-9
                    capFringe = 9.601334e-10

                    effectiveResistanceMultiplier = 1.77	#/* from CACTI */
                    current_gmNmos=4.38E+01
                    current_gmPmos=5.99E+01
                    # On-current (A/m)
                    currentOnNmos = [
                        346.30, 314.50, 286.80, 262.50, 241.20,
                        222.30, 205.60, 190.80, 177.50, 165.60, 155.00
                    ]
                    currentOnPmos = [
                        200.30, 179.50, 161.90, 146.90, 133.90,
                        122.60, 112.80, 104.10, 96.47, 89.68, 83.62
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.00e-6, 135.70e-6, 181.10e-6, 238.00e-6, 308.50e-6,
                        394.60e-6, 498.50e-6, 622.60e-6, 769.30e-6, 941.20e-6, 1141.0e-6
                    ]
                    currentOffPmos = [
                        100.30e-6, 133.20e-6, 174.20e-6, 224.40e-6, 285.10e-6,
                        357.60e-6, 443.40e-6, 543.70e-6, 660.00e-6, 793.80e-6, 946.40e-6
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 65:
                if self.roadmap == 'HP':
                    # /* PTM model: 65nm_HP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.1
                    vth = 166.3941e-3
                    phyGateLength = 3.5e-8
                    capIdealGate = 4.868295e-10
                    capFringe = 5.270361e-10
                    cap_draintotal = capFringe/2
                    effectiveResistanceMultiplier = 1.54    #/* from CACTI */
                    current_gmNmos=5.72E+02
                    current_gmPmos=3.99E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        1.12e3, 1.10e3, 1.08e3, 1.06e3, 1.04e3,
                        1.02e3, 1.00e3, 0.98e3, 0.95e3, 0.93e3, 0.91e3
                    ]
                    currentOnPmos = [
                        0.70e3, 0.66e3, 0.62e3, 0.58e3, 0.55e3,
                        0.52e3, 0.49e3, 0.46e3, 0.44e3, 0.41e3, 0.39e3
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.00e-3, 119.70e-3, 141.90e-3, 166.80e-3, 194.40e-3,
                        224.80e-3, 258.10e-3, 294.40e-3, 333.60e-3, 375.90e-3, 421.20e-3
                    ]
                    currentOffPmos = [
                        100.10e-3, 115.20e-3, 131.50e-3, 149.00e-3, 167.60e-3,
                        187.40e-3, 208.40e-3, 230.50e-3, 253.70e-3, 278.10e-3, 303.60e-3
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
                else:
                    # /* PTM model: 65nm_LP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.1
                    vth = 501.6636e-3
                    phyGateLength = 3.5e-8
                    capIdealGate = 1.1926e-9
                    capFringe = 9.62148e-10
                    effectiveResistanceMultiplier = 1.77    #/* from CACTI */
                    current_gmNmos=5.90E+01
                    current_gmPmos=6.75E+01
                    # On-current (A/m)
                    currentOnNmos = [
                        400.00, 363.90, 332.30, 304.70, 280.40,
                        258.90, 239.90, 223.00, 207.90, 194.30, 182.10
                    ]
                    currentOnPmos = [
                        238.70, 216.10, 196.60, 179.70, 164.90,
                        152.00, 140.50, 130.40, 121.40, 113.30, 106.10
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.20e-6, 137.50e-6, 185.80e-6, 247.20e-6, 324.20e-6,
                        419.30e-6, 535.40e-6, 675.70e-6, 843.10e-6, 1041.0e-6, 1273.0e-6
                    ]
                    currentOffPmos = [
                        100.20e-6, 135.40e-6, 179.70e-6, 234.90e-6, 302.50e-6,
                        384.30e-6, 482.20e-6, 598.00e-6, 733.90e-6, 891.60e-6, 1073.0e-6
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 45:
                if self.roadmap == 'HP':
                    # /* PTM model: 45nm_HP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.0
                    vth = 171.0969e-3
                    phyGateLength = 3.0e-8
                    capIdealGate = 4.091305e-10
                    capFringe = 4.957928e-10
                    effectiveResistanceMultiplier = 1.54;	#/* from CACTI */
                    current_gmNmos=7.37E+02
                    current_gmPmos=6.30E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        1.27e3, 1.24e3, 1.22e3, 1.19e3, 1.16e3,
                        1.13e3, 1.11e3, 1.08e3, 1.05e3, 1.02e3, 1.00e3
                    ]
                    currentOnPmos = [
                        1.08e3, 1.04e3, 1.00e3, 0.96e3, 0.92e3,
                        0.88e3, 0.85e3, 0.81e3, 0.78e3, 0.75e3, 0.72e3
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.00e-3, 120.70e-3, 144.10e-3, 170.50e-3, 199.80e-3,
                        232.30e-3, 268.00e-3, 307.10e-3, 349.50e-3, 395.40e-3, 444.80e-3
                    ]
                    currentOffPmos = [
                        100.20e-3, 118.70e-3, 139.30e-3, 162.00e-3, 186.80e-3,
                        213.90e-3, 243.30e-3, 274.90e-3, 308.90e-3, 345.20e-3, 383.80e-3
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
                else:
                    # /* PTM model: 45nm_LP.pm, from http://ptm.asu.edu/ */
                    vdd = 1.0
                    vth = 464.3718e-3
                    phyGateLength = 3.0e-8
                    capIdealGate = 8.930709e-10
                    capFringe = 8.849901e-10
                    effectiveResistanceMultiplier = 1.77    #/* from CACTI */
                    current_gmNmos=1.32E+02
                    current_gmPmos=8.65E+01
                    # On-current (A/m)
                    currentOnNmos = [
                        500.20, 462.00, 427.80, 397.10, 369.40,
                        344.50, 322.10, 301.80, 283.40, 266.70, 251.50
                    ]
                    currentOnPmos = [
                        300.00, 275.70, 254.20, 235.10, 218.10,
                        202.80, 189.20, 176.90, 165.80, 155.80, 146.70
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.00e-6, 140.50e-6, 193.90e-6, 263.10e-6, 351.40e-6,
                        462.50e-6, 600.30e-6, 769.20e-6, 973.90e-6, 1219.0e-6, 1511.0e-6
                    ]
                    currentOffPmos = [
                        100.20e-6, 138.40e-6, 187.60e-6, 250.10e-6, 328.10e-6,
                        424.10e-6, 540.90e-6, 681.30e-6, 848.30e-6, 1045.0e-6, 1275.0e-6
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 32:
                if self.roadmap == 'HP':
                    # /* PTM model: 32nm_HP.pm, from http://ptm.asu.edu/ */
                    vdd = 0.9
                    vth = 194.4951e-3
                    phyGateLength = 2.8e-8
                    capIdealGate = 3.767721e-10
                    capFringe = 4.713762e-10
                    effectiveResistanceMultiplier = 1.54   #/* from CACTI */
                    current_gmNmos=9.29E+02
                    current_gmPmos=6.73E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        1.41e3, 1.38e3, 1.35e3, 1.31e3, 1.28e3,
                        1.25e3, 1.21e3, 1.18e3, 1.15e3, 1.12e3, 1.08e3
                    ]
                    currentOnPmos = [
                        1.22e3, 1.17e3, 1.12e3, 1.07e3, 1.02e3,
                        0.98e3, 0.94e3, 0.89e3, 0.86e3, 0.82e3, 0.78e3
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.30e-3, 120.40e-3, 143.10e-3, 168.60e-3, 197.00e-3,
                        228.40e-3, 262.90e-3, 300.60e-3, 341.70e-3, 386.10e-3, 433.90e-3
                    ]
                    currentOffPmos = [
                        100.10e-3, 119.00e-3, 140.00e-3, 163.30e-3, 188.80e-3,
                        216.70e-3, 247.00e-3, 279.70e-3, 314.90e-3, 352.60e-3, 392.80e-3
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
                else:
                    # /* PTM model: 32nm_LP.pm, from http://ptm.asu.edu/ */
                    vdd = 0.9
                    vth = 442.034e-3
                    phyGateLength = 2.8e-8
                    capIdealGate = 8.375279e-10
                    capFringe = 6.856677e-10
                    effectiveResistanceMultiplier = 1.77    #/* from CACTI */
                    current_gmNmos=2.56E+02
                    current_gmPmos=1.19E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        600.20, 562.80, 528.20, 496.20, 466.80,
                        439.70, 414.80, 391.90, 370.70, 351.30, 333.30
                    ]
                    currentOnPmos = [
                        400.00, 368.40, 340.30, 315.30, 292.90,
                        272.80, 254.80, 238.50, 223.80, 210.50, 198.40
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.10e-6, 143.60e-6, 202.10e-6, 279.30e-6, 379.50e-6,
                        507.50e-6, 668.80e-6, 869.20e-6, 1115.0e-6, 1415.0e-6, 1774.0e-6
                    ]
                    currentOffPmos = [
                        100.10e-6, 140.70e-6, 194.00e-6, 262.50e-6, 349.30e-6,
                        457.70e-6, 591.20e-6, 753.70e-6, 949.30e-6, 1182.0e-6, 1457.0e-6
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 22:
                if self.roadmap == 'HP':
                    # /* PTM model: 22nm_HP.pm, from http://ptm.asu.edu/ */
                    vdd = 0.85
                    vth = 208.9006e-3
                    phyGateLength = 2.6e-8
                    capIdealGate = 3.287e-10
                    capFringe = 4.532e-10
                    effectiveResistanceMultiplier = 1.54    #/* from CACTI */
                    current_gmNmos=1.08E+03
                    current_gmPmos=6.98E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        1.50e3, 1.47e3, 1.43e3, 1.39e3, 1.35e3,
                        1.31e3, 1.28e3, 1.24e3, 1.20e3, 1.17e3, 1.13e3
                    ]
                    currentOnPmos = [
                        1.32e3, 1.25e3, 1.19e3, 1.13e3, 1.07e3,
                        1.02e3, 0.97e3, 0.92e3, 0.88e3, 0.84e3, 0.80e3
                    ]
                    # Off-current (A/m)
                    currentOffNmos = [
                        100.20e-3, 120.30e-3, 143.50e-3, 169.50e-3, 198.70e-3,
                        231.20e-3, 267.00e-3, 306.30e-3, 349.30e-3, 396.00e-3, 446.60e-3
                    ]
                    currentOffPmos = [
                        100.2e-3, 119.4e-3, 140.8e-3, 164.6e-3, 190.9e-3,
                        219.5e-3, 250.7e-3, 284.5e-3, 320.9e-3, 359.8e-3, 401.5e-3
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
                else:
                    # /* PTM model: 22nm_LP.pm, from http://ptm.asu.edu/ */
                    vdd = 0.85
                    vth = 419.915e-3
                    phyGateLength = 2.6e-8
                    capIdealGate = 5.245e-10
                    capFringe = 8.004e-10
                    effectiveResistanceMultiplier = 1.77    #/* from CACTI */
                    current_gmNmos=4.56E+02
                    current_gmPmos=1.85E+02
                    # On-current (A/m)
                    currentOnNmos = [
                        791.90, 756.40, 722.20, 689.40, 658.10,
                        628.30, 600.00, 573.30, 548.00, 524.20, 501.70
                    ]

                    currentOnPmos = [
                        600.20, 561.30, 525.50, 492.50, 462.20,
                        434.30, 408.70, 385.10, 363.40, 343.30, 324.80
                    ]

                    currentOffNmos = [
                        100.00e-6, 147.30e-6, 212.10e-6, 299.60e-6, 415.30e-6,
                        565.80e-6, 758.90e-6, 1003.00e-6, 1307.00e-6, 1682.00e-6, 2139.00e-6
                    ]

                    currentOffPmos = [
                        100.00e-6, 147.30e-6, 212.10e-6, 299.60e-6, 415.30e-6,
                        565.80e-6, 758.90e-6, 1003.00e-6, 1307.00e-6, 1682.00e-6, 2139.00e-6
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 14:
                if self.roadmap == 'HP':
                    raise RuntimeError("14nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2016
                    vdd = 0.8
                    vth = 0.1
                    heightFin = 4.2e-8
                    widthFin = 8.0e-9
                    PitchFin = 4.8e-8

                    max_fin_num =4
                    effective_width=widthFin+heightFin*2

                    phyGateLength = 2.6e-8
                    capIdealGate = 103.816  * 1E-18 / (effective_width)
                    cap_draintotal = 2.499e-17 / (effective_width)
                    capFringe = 0
                    effectiveResistanceMultiplier = 2.09	#/* from CACTI */
                    current_gmNmos=1415.34
                    current_gmPmos=1415.34
                    gm_oncurrent = 1415.34  #// gm at on current

                    # On-current (A/m)
                    currentOnNmos = [
                        595.045, 853, 814, 777, 742,
                        708, 677, 646, 618, 591, 565
                    ]

                    currentOnPmos = [
                        595.045, 767, 718, 672, 631,
                        593, 558, 526, 496, 469, 443
                    ]

                    currentOffNmos = [
                        0.0001, 184.4553e-6, 328.7707e-6, 566.8658e-6, 948.1816e-6,
                        1.5425e-3, 2.4460e-3, 3.7885e-3, 5.7416e-3, 8.5281e-3, 1.24327e-2
                    ]

                    currentOffPmos = [
                        102.3333e-6, 203.4774e-6, 389.0187e-6, 717.5912e-6, 1.2810e-3,
                        2.2192e-3, 3.7395e-3, 6.1428e-3, 9.8554e-3, 1.54702e-2, 2.37959e-2
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 10:
                if self.roadmap == 'HP':
                    raise RuntimeError("10nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2016
                    vdd = 0.7
                    vth = 0.1
                    heightFin = 4.5e-8
                    widthFin = 8.0e-9
                    PitchFin = 3.6e-8

                    max_fin_num =3
                    effective_width=widthFin+heightFin*2

                    phyGateLength = 2.2e-8
                    capIdealGate = 97.549  * 1E-18 / (effective_width)
                    cap_draintotal = 2.668e-17 / (effective_width)
                    capFringe = 0
                    effectiveResistanceMultiplier = 2.09
                    current_gmNmos=1803.50
                    current_gmPmos=1803.50
                    gm_oncurrent = 1803.50

                    # On-current (A/m)
                    currentOnNmos = [
                        599.237, 824, 787, 751, 717,
                        684, 654, 624, 597, 571, 546
                    ]

                    currentOnPmos = [
                        599.237, 725, 678, 636, 597,
                        561, 527, 497, 469, 443, 419
                    ]

                    currentOffNmos = [
                        0.000127, 184.4892e-6, 329.1615e-6, 568.0731e-6, 951.0401e-6,
                        1.5484e-3, 2.4574e-3, 3.8090e-3, 5.7767e-3, 8.5862e-3, 1.2525e-2
                    ]

                    currentOffPmos = [
                        100.5839e-6, 200.2609e-6, 383.3239e-6, 707.8499e-6, 1.2649e-3,
                        2.1932e-3, 3.6987e-3, 6.0804e-3, 9.7622e-3, 1.53340e-2, 2.36007e-2
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 7:
                if self.roadmap == 'HP':
                    raise RuntimeError("7nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2017
                    vdd = 0.7
                    vth = 0.1
                    heightFin = 5.0e-8
                    widthFin = 7.0e-9
                    PitchFin = 3.0e-8

                    max_fin_num =2
                    effective_width=107e-9

                    phyGateLength = 2.2e-8
                    capIdealGate = 100.497  * 1E-18 / (effective_width)
                    cap_draintotal = 2.224e-17 / (effective_width)
                    capFringe = 0
                    effectiveResistanceMultiplier = 2.05
                    current_gmNmos=1785.37
                    current_gmPmos=1785.37
                    gm_oncurrent = 1785.37

                    # On-current (A/m)
                    currentOnNmos = [
                        562.048, 786, 750, 716, 684,
                        653, 624, 595, 569, 545, 521
                    ]

                    currentOnPmos = [
                        562.048, 689, 645, 605, 567,
                        533, 501, 473, 446, 421, 398
                    ]

                    currentOffNmos = [
                        0.000147, 1.85e-4, 3.32e-4, 5.74e-4, 9.62e-4,
                        1.5695e-3, 2.4953e-3, 3.8744e-3, 5.8858e-3, 8.7624e-3, 1.28025e-2
                    ]

                    currentOffPmos = [
                        100.9536e-6, 201.3937e-6, 386.2086e-6, 714.4288e-6, 1.2788e-3,
                        2.2207e-3, 3.7509e-3, 6.1750e-3, 9.9278e-3, 1.56146e-2, 2.40633e-2
                    ]
                    # Interpolation to full 0-100
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 5:
                if self.roadmap == 'HP':
                    raise RuntimeError("5nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2021
                    vdd = 0.7
                    vth = 0.1
                    widthFin=6.0e-9
                    PitchFin=28.0e-9
                    phyGateLength = 2.0e-8                  
                    effective_width = 106.0*1e-9
                    max_fin_num =2                  
                    capIdealGate = 81.859 * 1E-18 / (effective_width )
                    cap_draintotal = 2.076e-17/ (effective_width)
                    capFringe =0
                    effectiveResistanceMultiplier = 2.10	#/* from CACTI */
                    current_gmNmos= 1820.90
                    current_gmPmos= 1820.90
                    gm_oncurrent = 1820.90  #// gm at on current

                    # On-current (A/m)
                    currentOnNmos = [
                        578.494, 786, 750, 716, 684,
                        653, 624, 595, 569, 545, 521
                    ]

                    currentOnPmos = [
                        578.494, 689, 645, 605, 567,
                        533, 501, 473, 446, 421, 398
                    ]

                    currentOffNmos = [
                        0.000138, 1.85e-4, 3.32e-4, 5.74e-4, 9.62e-4,
                        1.5695e-3, 2.4953e-3, 3.8744e-3, 5.8858e-3, 8.7624e-3, 1.28025e-2
                    ]

                    currentOffPmos = [
                        100.9536e-6, 201.3937e-6, 386.2086e-6, 714.4288e-6, 1.2788e-3,
                        2.2207e-3, 3.7509e-3, 6.1750e-3, 9.9278e-3, 1.56146e-2, 2.40633e-2
                    ]
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 3:
                if self.roadmap == 'HP':
                    raise RuntimeError("5nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2022
                    vdd = 0.7
                    vth = 0.1
                    widthFin=5.0e-9 	
                    PitchFin=24.0e-9
                    phyGateLength = 1.8e-8                  
                    effective_width = 101.0*1e-9
                    max_fin_num =2

                    capIdealGate = 72.572 * 1E-18 / (effective_width);   #//6.44E-10; //8.91E-10;
                    cap_draintotal = 1.791e-17/ (effective_width);
                    capFringe = 0
                    effectiveResistanceMultiplier = 2.14 	#/* from CACTI */
                    current_gmNmos= 2018.04
                    current_gmPmos= 2018.04
                    gm_oncurrent = 2018.04  #// gm at on current

                    # On-current (A/m)
                    currentOnNmos = [
                        641.463, 786, 750, 716, 684,
                        653, 624, 595, 569, 545, 521
                    ]

                    currentOnPmos = [
                        641.463, 689, 645, 605, 567,
                        533, 501, 473, 446, 421, 398
                    ]

                    currentOffNmos = [
                        0.000158, 1.85e-4, 3.32e-4, 5.74e-4, 9.62e-4,
                        1.5695e-3, 2.4953e-3, 3.8744e-3, 5.8858e-3, 8.7624e-3, 1.28025e-2
                    ]

                    currentOffPmos = [
                        100.9536e-6, 201.3937e-6, 386.2086e-6, 714.4288e-6, 1.2788e-3,
                        2.2207e-3, 3.7509e-3, 6.1750e-3, 9.9278e-3, 1.56146e-2, 2.40633e-2
                    ]
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 2:
                if self.roadmap == 'HP':
                    raise RuntimeError("3nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2022
                    vdd = 0.65
                    vth = 0.1
                    PitchFin= 26e-9
                    phyGateLength = 1.4e-8                 
                    #// 1.4 update: GAA-specific parameters
                    max_fin_per_GAA=1
                    max_sheet_num=3
                    thickness_sheet=6*1e-9
                    width_sheet=15*1e-9              
                    widthFin=width_sheet; #// for drain height calculation 	
                    effective_width=(thickness_sheet+width_sheet)                   
                    capIdealGate = 79.74 * 1E-18 /  (effective_width*max_sheet_num) 
                    cap_draintotal = 1.543e-17/ (effective_width)
                    capFringe = 0
                    effectiveResistanceMultiplier = 1.98 	#/* from CACTI */
                    current_gmNmos= 1968.85
                    current_gmPmos= 1968.85
                    gm_oncurrent = 1968.85

                    # On-current (A/m)
                    currentOnNmos = [
                        526.868, 786, 750, 716, 684,
                        653, 624, 595, 569, 545, 521
                    ]

                    currentOnPmos = [
                        526.868, 689, 645, 605, 567,
                        533, 501, 473, 446, 421, 398
                    ]

                    currentOffNmos = [
                        0.0000733, 1.85e-4, 3.32e-4, 5.74e-4, 9.62e-4,
                        1.5695e-3, 2.4953e-3, 3.8744e-3, 5.8858e-3, 8.7624e-3, 1.28025e-2
                    ]

                    currentOffPmos = [
                        100.9536e-6, 201.3937e-6, 386.2086e-6, 714.4288e-6, 1.2788e-3,
                        2.2207e-3, 3.7509e-3, 6.1750e-3, 9.9278e-3, 1.56146e-2, 2.40633e-2
                    ]
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]
            elif self.node_nm == 1:
                if self.roadmap == 'HP':
                    raise RuntimeError("2nm HP not supported")
                elif self.roadmap == 'LP':
                    #device specifications follow IRDS 2022
                    vdd = 0.6
                    vth = 0.1
                    PitchFin= 24e-9
                    phyGateLength = 1.2e-8           
                    #// IRDS 2022 - GAA specfic parameters
                    max_fin_per_GAA=1
                    max_sheet_num=4
                    thickness_sheet=6*1e-9
                    width_sheet=10*1e-9

                    widthFin= width_sheet #// for drain height calculation 
                    effective_width=(thickness_sheet+width_sheet)*2

                    capIdealGate = 66.94 * 1E-18 /  (effective_width*max_sheet_num) 
                    cap_draintotal = 1.409e-17/ (effective_width)
                    capFringe = 0
                    effectiveResistanceMultiplier = 2.05 	#/* from CACTI */
                    current_gmNmos= 2401.75
                    current_gmPmos= 2401.75
                    gm_oncurrent = 2401.75
                    # On-current (A/m)
                    currentOnNmos = [
                        460.979, 786, 750, 716, 684,
                        653, 624, 595, 569, 545, 521
                    ]

                    currentOnPmos = [
                        460.979, 689, 645, 605, 567,
                        533, 501, 473, 446, 421, 398
                    ]

                    currentOffNmos = [
                        0.000169, 1.85e-4, 3.32e-4, 5.74e-4, 9.62e-4,
                        1.5695e-3, 2.4953e-3, 3.8744e-3, 5.8858e-3, 8.7624e-3, 1.28025e-2
                    ]

                    currentOffPmos = [
                        100.9536e-6, 201.3937e-6, 386.2086e-6, 714.4288e-6, 1.2788e-3,
                        2.2207e-3, 3.7509e-3, 6.1750e-3, 9.9278e-3, 1.56146e-2, 2.40633e-2
                    ]
                    currentOnNmos = self._interpolate_full(currentOnNmos)
                    currentOnPmos = self._interpolate_full(currentOnPmos)
                    currentOffNmos = self._interpolate_full(currentOffNmos)
                    currentOffPmos = self._interpolate_full(currentOffPmos)
                    pnSizeRatio = currentOnNmos[0]/currentOnPmos[0]


        capOverlap = capIdealGate * 0.2 if self.node_nm >= 22 else 0.0
        # Junction cap model (from BSIM4)
        buildInPotential = 0.9
        cjd = 1e-3
        cjswd = 2.5e-10
        cjswgd = 0.5e-10
        mjd = 0.5
        mjswd = 0.33
        mjswgd = 0.33
        #/* Properties not used so far */
        capPolywire = 0.0

        capJunction = cjd / pow(1 + vdd / buildInPotential, mjd)
        capSidewall = cjswd / pow(1 + vdd / buildInPotential, mjswd)
        capDrainToChannel = cjswgd / pow(1 + vdd / buildInPotential, mjswgd)

        #junction capacitance for 14 nm and beyond; 
        if self.node_nm == 14:  capJunction= 0.0120
        elif self.node_nm == 10: capJunction= 0.0134
        elif self.node_nm == 7:  capJunction= 0.0137
        elif self.node_nm == 5:  capJunction= 0.0119
        elif self.node_nm == 3:  capJunction= 0.0128
        elif self.node_nm == 2:  capJunction= 0.0091
        elif self.node_nm == 1:  capJunction= 0.0102
        else: capJunction = cjd / pow(1 + vdd / buildInPotential, mjd)

        self.params = {
                    'vdd': vdd,
                    'vth': vth,
                    'roadmap': self.roadmap,
                    'phyGateLength': phyGateLength,
                    'PitchFin': PitchFin,
                    'widthFin': widthFin,
                    'heightFin': heightFin,
                    'capIdealGate': capIdealGate,
                    'capFringe': capFringe,
                    'capOverlap': capOverlap,
                    'capJunction': capJunction,
                    'capSidewall': capSidewall,
                    'capDrainToChannel': capDrainToChannel,
                    'effectiveResistanceMultiplier': effectiveResistanceMultiplier,  # could be adjusted based on design
                    'current_gmNmos': current_gmNmos,   # conductance for gm unit is μS/μm
                    'current_gmPmos': current_gmPmos,
                    'currentOnNmos': currentOnNmos,
                    'currentOnPmos': currentOnPmos,
                    'currentOffNmos': currentOffNmos,
                    'currentOffPmos': currentOffPmos,
                    'pnSizeRatio': pnSizeRatio,
                    'transistorType': self.transistor_type,   
                    'capPolywire': capPolywire,  
                    'node_nm': self.node_nm,
                    'featureSize': self.node_nm * 1e-9       
                }

    def _interpolate_full(self, base):
        'gemerate a full 0-100 array from a base array with 10 steps'
        full = [0.0] * 101
        for i in range(0, 101, 10):
            full[i] = base[i // 10]
        for i in range(1, 100):
            if i % 10 != 0:
                low = (i // 10) * 10
                high = low + 10
                alpha = (i - low) / 10
                full[i] = full[low] * (1 - alpha) + full[high] * alpha
        return full

    def get_param(self, name):
        return self.params.get(name, None)

    def print_summary(self):
        print(f"Tech Node: {self.node_nm}nm, Roadmap: {self.roadmap}")
        for k, v in self.params.items():
            if isinstance(v, list):
                print(f"{k}: {v[:3]} ... {v[-3:]}")
            else:
                print(f"{k}: {v:.3e}" if isinstance(v, float) else f"{k}: {v}")
