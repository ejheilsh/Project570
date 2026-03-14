import numpy as np

# principal characteristics of tanker given in project description
L_pp = 278          # m
B = 45              # m
D = 25.4            # m
T_D = 17.6          # m
C_B = 0.85          
disp = 200000       # tons
DWT = 175000        # tons
V_D = 14.5          # kt

# fuel constants
LHV_FO = 40.4       # GJ/t
LHV_NH3 = 18.8      # GJ/t
rho_FO = 0.93       # t/m^3
rho_NH3 = 0.7       # t/m^3

# structural assumptions
M_SW_H = 4e6        # kN-m
d_c = 56.5          # m
d_NH3_deck = 60.5   # m
d_FO = 20.5         # m
rho_cargo = 0.8     # tons / m^3

# stability assumptions
W_L = 23645         # tons
KG_L = 12           # m
KG_C = 13           # m
KG_FO = 21.9        # m
KG_NH3_deck = 28.4  # m
KG_NH3_hold = 13    # m
KM = 16             # m

# TCO assumptions
R_TW = 0.000328     # $ / ton-nm
P_NH3 = 25          # $ / GJ
P_new = 65          # $M
P_tank_hold = 1500  # $ / m^3
P_tank_deck = 1600  # $ / m^3
