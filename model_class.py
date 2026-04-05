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
        self.SFC_FO = 0.3206 * self.V**2 - 1.2558 * self.V

        # --- find ammonia volume required ---
        self.E_T = self.E_D / (24 * self.V)        # days
        self.W_FO = self.SFC_FO * self.E_T         # tons
        self.vol_FO = self.W_FO / rho_FO      # m^3
        self.E_R = self.W_FO * LHV_FO         # GJ
        self.W_NH3 = self.E_R / LHV_NH3       # tons
        self.vol_NH3 = self.W_NH3 / rho_NH3   # m^3

        # --- resolve impact on londitudinal strength ---
        self.W_FO_red = -self.W_FO / 2
        self.M_FO = -self.W_FO_red * d_FO / 2

        if self.tank_type == "in-hold":
            self.W_tank = 0                         # tons
            self.H_hold, self.D_hold = 22, 38                     # m
            self.L_hold = self.vol_NH3 / (self.H_hold * self.D_hold)   # m
            self.d_NH3 = 28 - self.L_hold / 2                     # m

            self.M_C = -self.vol_NH3 * rho_cargo * self.d_NH3 / 2    # ton-m
            self.M_NH3 = (self.W_NH3 + self.W_tank) * self.d_NH3 / 2     # ton-m
            M_add = self.M_C + self.M_NH3 + self.M_FO # ton-m

        elif self.tank_type == "on-deck":
            self.W_tank = 0.000008 * self.vol_NH3**2 + 0.0198 * self.vol_NH3 + 83.631   # tons
            self.d_NH3 = d_NH3_deck

            self.M_C = (self.W_C - 175000) * d_c / 2         # ton-m
            self.M_NH3 = (self.W_NH3 + self.W_tank) * self.d_NH3 / 2     # ton-m
            M_add = self.M_C + self.M_NH3 + self.M_FO # ton-m

        else:
            raise ValueError("bad tank_type definition...")

        self.M_add_kNm = M_add * 9.81   # kN-m
        # W_steel = max(0, 0.036 * self.M_add_kNm / M_SW_H * disp) # tons, don't want to remove steel
        self.W_steel = 0.036 * self.M_add_kNm / M_SW_H * disp # tons, don't want to remove steel

        # --- weight balance ---
        if self.tank_type == "in-hold":
            self.W_add = -self.vol_NH3 * rho_cargo + self.W_NH3 + self.W_tank + self.W_FO_red + self.W_steel   # tons
        elif self.tank_type == "on-deck":
            self.W_add = (self.W_C - 175000) + self.W_NH3 + self.W_tank + self.W_FO_red + self.W_steel   # tons
        else:
            raise ValueError("bad tank_type definition...")

        # --- transverse stability ---
        if self.tank_type == "in-hold":
            self.disp_total = W_L + (175000 - self.vol_NH3 * rho_cargo) + self.W_NH3 + self.W_tank + self.W_steel - self.W_FO_red   # tons
            self.M_vert = KG_L * W_L + KG_C * (175000 - self.vol_NH3 * rho_cargo) + KG_NH3_hold * self.W_NH3 + KG_NH3_hold * self.W_tank - KG_FO * self.W_FO_red + KG_L * self.W_steel #ton-m
            self.vcg = self.M_vert / self.disp_total      # m
            self.GM = KM - self.vcg     # m
        elif self.tank_type == "on-deck":
            self.disp_total = W_L + self.W_C + self.W_NH3 + self.W_tank - self.W_FO_red + self.W_steel   # tons
            self.M_vert = KG_L * W_L + KG_C * self.W_C + KG_NH3_deck * self.W_NH3 + KG_NH3_deck * self.W_tank - KG_FO * self.W_FO_red + KG_L * self.W_steel    # ton-m
            self.vcg = self.M_vert / self.disp_total      # m
            self.GM = KM - self.vcg     # m
        else:
            raise ValueError("bad tank_type definition...")

        # --- cost of ownership ---
        self.E_ratio = 24000 / self.E_D
        self.B_num = np.ceil(self.E_ratio)
        self.C_fuel_tot = self.E_R * P_NH3 * self.E_ratio
        self.C_fuel_pen = 0.18 * self.B_num**2 - 0.25 * self.B_num + 1.06
        self.C_fuel = self.C_fuel_tot * self.C_fuel_pen

        self.C_capex_base = 0.15 * P_new * 1e6
        self.C_capex_tank = P_tank_deck * self.vol_NH3 if self.tank_type == "on-deck" else P_tank_hold * self.vol_NH3
        self.C_capex = (self.C_capex_base + self.C_capex_tank) / ((3650 / self.E_T) * self.E_ratio)

        self.C_LR = 24 * (175000 - self.W_C) * self.V * R_TW if self.tank_type == "on-deck" else 24 * self.vol_NH3 * rho_cargo * self.V * R_TW     # $ / day
        self.C_rev = self.E_T * self.E_ratio * self.C_LR
        self.C_TCO = self.C_fuel + self.C_capex + self.C_rev

        self.constraint_metrics = {
            "volume_m3": self.vol_NH3,
            "bending_moment_kNm": abs(self.M_add_kNm),
            "weight_change_tons": abs(self.W_add),
            "GM_m": self.GM,
        }

    def constraint_violations(
        self,
        volume_limit_m3: float,
        bending_moment_limit_knm: float,
        weight_limit_tons: float,
        gm_min_m: float,
        gm_max_m: float,
    ) -> dict[str, float]:
        return {
            "volume": max(0.0, (self.constraint_metrics["volume_m3"] - volume_limit_m3) / volume_limit_m3),
            "bending_moment": max(
                0.0,
                (self.constraint_metrics["bending_moment_kNm"] - bending_moment_limit_knm) / bending_moment_limit_knm,
            ),
            "weight": max(
                0.0,
                (self.constraint_metrics["weight_change_tons"] - weight_limit_tons) / weight_limit_tons,
            ),
            "gm_min": max(0.0, (gm_min_m - self.constraint_metrics["GM_m"]) / gm_min_m),
            "gm_max": max(0.0, (self.constraint_metrics["GM_m"] - gm_max_m) / gm_max_m),
        }

    def is_feasible(
        self,
        volume_limit_m3: float,
        bending_moment_limit_knm: float,
        weight_limit_tons: float,
        gm_min_m: float,
        gm_max_m: float,
        tolerance: float = 1.0e-12,
    ) -> bool:
        return all(
            violation <= tolerance
            for violation in self.constraint_violations(
                volume_limit_m3=volume_limit_m3,
                bending_moment_limit_knm=bending_moment_limit_knm,
                weight_limit_tons=weight_limit_tons,
                gm_min_m=gm_min_m,
                gm_max_m=gm_max_m,
            ).values()
        )

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
        print(f"W_NH3 = {self.W_NH3:.2f} tons")
        print(f"W_tank = {self.W_tank:.2f} tons")
        print(f"W_steel = {self.W_steel:.2f} tons")


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
    
