class Technology:
    def __init__(self, node_nm=45, roadmap='HP', transistor_type='conventional'):
        self.node_nm = node_nm
        self.roadmap = roadmap
        self.transistor_type = transistor_type
        self.initialized = False
        self.params = {}
        self._initialize()

    def _initialize(self):
        if self.initialized:
            print("Warning: Already initialized!")
            return
        if self.transistor_type == 'conventional':
            if self.node_nm == 45 and self.roadmap == 'HP':
                vdd = 1.0
                vth = 0.18
                phyGateLength = 45e-9
                capIdealGate = 4e-10
                capFringe = 5e-10
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
                capPolywire = 0.0;	#/* TO-DO: we need to find the values */

                capJunction = cjd / pow(1 + vdd / buildInPotential, mjd)
                capSidewall = cjswd / pow(1 + vdd / buildInPotential, mjswd)
                capDrainToChannel = cjswgd / pow(1 + vdd / buildInPotential, mjswgd)

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

                self.params = {
                    'vdd': vdd,
                    'vth': vth,
                    'phyGateLength': phyGateLength,
                    'capIdealGate': capIdealGate,
                    'capFringe': capFringe,
                    'capOverlap': capOverlap,
                    'capJunction': capJunction,
                    'capSidewall': capSidewall,
                    'capDrainToChannel': capDrainToChannel,
                    'effectiveResistanceMultiplier': 1.0,  # could be adjusted based on design
                    'current_gmNmos': 1e-3,  # conductance for gm unit is μS/μm
                    'current_gmPmos': 0.8e-3,
                    'currentOnNmos': currentOnNmos,
                    'currentOnPmos': currentOnPmos,
                    'currentOffNmos': currentOffNmos,
                    'currentOffPmos': currentOffPmos,
                    'pnSizeRatio': 2.0,
                    'transistorType': self.transistor_type,   
                    'capPolywire': capPolywire,  
                    'featureSize': self.node_nm * 1e-9       
                }

            elif self.node_nm == 14 and self.roadmap == 'LSTP':
                vdd = 0.8
                vth = 0.45
                phyGateLength = 14e-9
                capIdealGate = 2.5e-10
                capFringe = 3e-10
                capOverlap = 0.0  # don't use overlap cap in finFET nodes

                buildInPotential = 0.9
                cjd = 1e-3
                cjswd = 2.5e-10
                cjswgd = 0.5e-10
                mjd = 0.5
                mjswd = 0.33
                mjswgd = 0.33
                #/* Properties not used so far */
                capPolywire = 0.0;	#/* TO-DO: we need to find the values */

                capJunction = cjd / pow(1 + vdd / buildInPotential, mjd)
                capSidewall = cjswd / pow(1 + vdd / buildInPotential, mjswd)
                capDrainToChannel = cjswgd / pow(1 + vdd / buildInPotential, mjswgd)

                currentOnNmos = [600e-6 - i * 2e-6 for i in range(0, 101, 10)]
                currentOnPmos = [500e-6 - i * 1.5e-6 for i in range(0, 101, 10)]
                currentOffNmos = [10e-9 + i * 0.1e-9 for i in range(0, 101, 10)]
                currentOffPmos = [8e-9 + i * 0.1e-9 for i in range(0, 101, 10)]

                currentOnNmos = self._interpolate_full(currentOnNmos)
                currentOnPmos = self._interpolate_full(currentOnPmos)
                currentOffNmos = self._interpolate_full(currentOffNmos)
                currentOffPmos = self._interpolate_full(currentOffPmos)

                self.params = {
                    'vdd': vdd,
                    'vth': vth,
                    'phyGateLength': phyGateLength,
                    'capIdealGate': capIdealGate,
                    'capFringe': capFringe,
                    'capOverlap': capOverlap,
                    'capJunction': capJunction,
                    'capSidewall': capSidewall,
                    'capDrainToChannel': capDrainToChannel,
                    'effectiveResistanceMultiplier': 1.2,
                    'current_gmNmos': 2.0e-3,
                    'current_gmPmos': 1.6e-3,
                    'currentOnNmos': currentOnNmos,
                    'currentOnPmos': currentOnPmos,
                    'currentOffNmos': currentOffNmos,
                    'currentOffPmos': currentOffPmos,
                    'capPolywire': capPolywire,  
                    'pnSizeRatio': 2.0
                }
            else:
                raise ValueError(f"Unsupported node {self.node_nm}nm or roadmap {self.roadmap}")

            self.initialized = True

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

    def interpolate_current(self, name, bias_index):
        'Interpolate current values based on bias index (0-100)'
        if name not in self.params:
            return None
        arr = self.params[name]
        i_low = int(bias_index)
        i_high = min(100, i_low + 1)
        alpha = bias_index - i_low
        return arr[i_low] * (1 - alpha) + arr[i_high] * alpha

    def print_summary(self):
        print(f"Tech Node: {self.node_nm}nm, Roadmap: {self.roadmap}")
        for k, v in self.params.items():
            if isinstance(v, list):
                print(f"{k}: {v[:3]} ... {v[-3:]}")
            else:
                print(f"{k}: {v:.3e}" if isinstance(v, float) else f"{k}: {v}")
