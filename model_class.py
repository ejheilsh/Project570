from dataclasses import dataclass
from characteristics import *


@dataclass
class ship_model:
    E_D: float              # endurance (nm)
    W_C: float              # cargo deadweight (tons)
    V: float                # speed (kt)
    tank_type: str          # tank type/location ("on-deck" or "in-hold")

    def __post_init__(self):
        # check that ship speed is above 10 kts
        if self.V < 10:
            raise ValueError(f"ship speed is too low... V = {self.V} < 10 kts")

        # check that tank type is acceptable
        if self.tank_type not in {"on-deck", "in-hold"}:
            raise ValueError(f"tank_type is not \"on-deck\" or \"in-hold\"")

        # determine fuel oil consumption based on speed (tons FO / day)
        SFC_FO = 0.3206 * self.V**2 - 1.2558 * self.V  

        # --- find ammonia volume required ---
        E_T = self.E_D / (24 * self.V)        # days
        W_FO = SFC_FO * E_T         # tons
        vol_FO = W_FO / rho_FO      # m^3
        E_R = W_FO * LHV_FO         # GJ
        W_NH3 = E_R / LHV_NH3       # tons
        self.vol_NH3 = W_NH3 / rho_NH3   # m^3

        # --- resolve impact on londitudinal strength ---
        W_FO_red = -W_FO / 2
        M_FO = -W_FO_red * d_FO / 2

        if self.tank_type == "in-hold":
            W_tank_hold = 0                         # tons
            H_hold, D_hold = 22, 38                     # m
            L_hold = self.vol_NH3 / (H_hold * D_hold)   # m
            d_NH3_hold = 28 - L_hold / 2                     # m

            M_C_hold = -self.vol_NH3 * rho_cargo * d_NH3_hold / 2    # ton-m
            M_NH3_hold = (W_NH3 + W_tank_hold) * d_NH3_hold / 2     # ton-m
            M_add = M_C_hold + M_NH3_hold + M_FO # ton-m

        elif self.tank_type == "on-deck":
            W_tank_deck = 0.000008 * self.vol_NH3**2 + 0.0198 * self.vol_NH3 + 83.631   # tons
            # d_NH3_deck = 130 - 49 - d_FO    # m

            M_C_deck = (self.W_C - 175000) * d_c / 2         # ton-m
            M_NH3_deck = (W_NH3 + W_tank_deck) * d_NH3_deck / 2     # ton-m
            M_add = M_C_deck + M_NH3_deck + M_FO # ton-m

        else:
            raise ValueError("bad tank_type definition...")

        self.M_add_kNm = M_add * 9.81   # kN-m
        # W_steel = max(0, 0.036 * self.M_add_kNm / M_SW_H * disp) # tons, don't want to remove steel
        W_steel = 0.036 * self.M_add_kNm / M_SW_H * disp # tons, don't want to remove steel

        # --- weight balance ---
        if self.tank_type == "in-hold":
            self.W_add = -self.vol_NH3 * rho_cargo + W_NH3 + W_tank_hold + W_FO_red + W_steel   # tons
        elif self.tank_type == "on-deck":
            self.W_add = (self.W_C - 175000) + W_NH3 + W_tank_deck + W_FO_red + W_steel   # tons
        else:
            raise ValueError("bad tank_type definition...")

        # --- transverse stability ---
        if self.tank_type == "in-hold":
            disp_hold = W_L + (175000 - self.vol_NH3 * rho_cargo) + W_NH3 + W_tank_hold + W_steel - W_FO_red   # tons
            M_vert_hold = KG_L * W_L + KG_C * (175000 - self.vol_NH3 * rho_cargo) + KG_NH3_hold * W_NH3 + KG_NH3_hold * W_tank_hold - KG_FO * W_FO_red + KG_L * W_steel #ton-m
            vcg_hold = M_vert_hold / disp_hold      # m
            self.GM = KM - vcg_hold     # m
        elif self.tank_type == "on-deck":
            disp_deck = W_L + self.W_C + W_NH3 + W_tank_deck - W_FO_red + W_steel   # tons
            M_vert_deck = KG_L * W_L + KG_C * self.W_C + KG_NH3_deck * W_NH3 + KG_NH3_deck * W_tank_deck - KG_FO * W_FO_red + KG_L * W_steel    # ton-m
            vcg_deck = M_vert_deck / disp_deck      # m
            self.GM = KM - vcg_deck     # m
        else:
            raise ValueError("bad tank_type definition...")

        # --- cost of ownership ---
        E_ratio = 24000 / self.E_D
        B_num = np.ceil(E_ratio)
        C_fuel_tot = E_R * P_NH3 * E_ratio
        C_fuel_pen = 0.18 * B_num**2 - 0.25 * B_num + 1.06
        C_fuel = C_fuel_tot * C_fuel_pen

        C_capex_base = 0.15 * P_new * 1e6
        C_capex_tank = P_tank_deck * self.vol_NH3 if self.tank_type == "on-deck" else P_tank_hold * self.vol_NH3
        C_capex = (C_capex_base + C_capex_tank) / ((3650 / E_T) * E_ratio)

        C_LR = 24 * (175000 - self.W_C) * self.V * R_TW if self.tank_type == "on-deck" else 24 * self.vol_NH3 * rho_cargo * self.V * R_TW     # $ / day
        C_rev = E_T * E_ratio * C_LR
        self.C_TCO = C_fuel + C_capex + C_rev

        # print()



    def print_outputs(self):
        print(f"E_D = {self.E_D:.2f} nm")
        print(f"W_C = {self.W_C:.2f} tons")
        print(f"V = {self.V:.2f} kts")
        print(f"--- {self.tank_type} ---")
        print(f"vol_NH3 = {self.vol_NH3:.2f} m^3")
        print(f"M_add_kNm = {self.M_add_kNm:.2f} kN-m")
        print(f"W_add = {self.W_add:.2f} tons")
        print(f"GM = {self.GM:.2f} m")
        print(f"C_TCO = ${self.C_TCO:.2f}")


if __name__ == "__main__":
    print()
    inputs = {
        "E_D": 9e3,
        "W_C": 151e3,
        "V": 10,
        "tank_type": "on-deck",
    }
    # s1 = ship_model(**inputs)
    # s1.print_outputs()


    print()
    inputs["tank_type"] = "in-hold"
    s2 = ship_model(**inputs)
    s2.print_outputs()


    # print()
    # lower = 9e3
    # upper = 24e3
    # print(f"{(upper - lower)/3 + lower}")

    # lower = 151e3
    # upper = 175e3
    # print(f"{(upper - lower)/3 + lower}")

    # lower = 10
    # upper = 16
    # print(f"{(upper - lower)/3 + lower}")
    

