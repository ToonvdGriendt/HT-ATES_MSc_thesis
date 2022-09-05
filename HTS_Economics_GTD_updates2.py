import numpy as np
# import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
sns.set()

#%% INPUT SECTION

Base = [7.4] # Geothermal base load [MW]
df_eco = pd.DataFrame(Base, columns = ['Base'])    # Create empty DataFrame for economic results
 
c_w        = 4182     # Fluid specific heat [J/kg*K]
rho_w      = 1000     # Water mass density [kg/m³]

T_cutoff                 = 50
df_eco['T_cutoff [C]']   = T_cutoff       # HT-ATES cut-off temperature [C]
df_eco['T_network [C]']  = 80             # Heat network temperature [C], needed for heatpump

df_eco['p_ref [kPa]']          = 1550.0   # Reference pressure [kPa]
df_eco['mu_pump [-]']          = 0.5      # Pump efficiency [-]
df_eco['Price_elec [EUR/MWh]'] = 145      # Electricity price [EUR/MWh]

Multiply   = 2        # Fluid rate multiplication factor --> 2 when doublet (mirrored) and 4 when quarter-5-spot pattern

df_eco['T [C]']          = 80       # Initial storage temperature [C]
df_eco['T_warm [C]']     = T_cutoff       # Re-injection temperature [C]
df_eco['Q [m3/day]']     = 4800     # Flow rate [m3/day] -> FULL FLOW RATES
df_eco['V [m3]']         = 1000000  # Total stored volume [m3]

df_eco['Lifetime [yrs]'] = 30       # Project lifetime [years] (Delft HT-ATES report)
Year                     = 4        # The year to be used for HTS calculations

df_eco['r_annual [-]']   = 0.24#0.055    # Annual discount rate (from Delft HT-ATES report)
# r_daily    = (1+r_annual)**(1/365)-1 # Daily compounded discount rate

#%% IMPORT DEMAND DATA

Demand_curve_hrs = np.genfromtxt('Demand_curve_hourly_GTD.txt')        # Hourly heat demand data GTD

# Get daily demand data from hourly heat demand
Demand_curve_GTD = [] # Create empty list for daily demand data

for i in range(len(Demand_curve_hrs[:,0])):
    if Demand_curve_hrs[i,0] == 24:                                    # Find all days in data
        Demand_curve_GTD.append(np.mean(Demand_curve_hrs[i-23:i+1,1])) # Append the mean values to the list

df_demand = pd.DataFrame(Demand_curve_GTD, columns = ['Demand curve [MW]'])

#%% DIFFERENT ENERGY SOURCES

#1 Geothermal

# Add 'Geothermal Base load [MW]' to the DataFrame
df_demand['Geothermal Base load [MW]'] = df_eco.loc[0,'Base']

# Calculate the amount of directly used geothermal energy ('Geothermal used [MW]') based on demand curve
df_demand.loc[df_demand['Geothermal Base load [MW]'] >  df_demand['Demand curve [MW]'], 'Geothermal used [MW]'] = df_demand['Demand curve [MW]']
df_demand.loc[df_demand['Geothermal Base load [MW]'] <= df_demand['Demand curve [MW]'], 'Geothermal used [MW]'] = df_demand['Geothermal Base load [MW]']

# Store the Excess geothermal heat ('Geothermal Excess [MW]') in a separate column in the DataFrame
df_demand.loc[df_demand['Geothermal Base load [MW]'] >  df_demand['Demand curve [MW]'], 'Geothermal Excess [MW]'] = df_demand['Geothermal Base load [MW]'] - df_demand['Demand curve [MW]']


#2 HT-ATES
# Read data retrieved from GEM simulations

#%% 2020 VERSION
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\Faster Simulations\Base case\Base_case.xlsx', skiprows=2) # Base Case
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\Scenario_1\Scenario_1.xlsx', skiprows=2)                  # Scenario 1

#%% 2022 VERSION

# BASE CASE
df_ates = pd.read_excel(r'Excel_input_files\Base_case.xlsx', skiprows=2)

# S1 SMALL STORAGE VOLUME
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S1_small_storage_V\S1_small_storage_V.xlsx', skiprows=2)

# S2 SMALL WELL SPACING
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S2_small_well_spacing\S2_small_well_spacing.xlsx', skiprows=2)

# S3 LARGE WELL SPACING
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S3_large_well_spacing\S3_large_well_spacing.xlsx', skiprows=2)

# S4 SMALL SCREEN LENGTH
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S4_screen_length\S4_screen_length.xlsx', skiprows=2)

# S5 T_INJ = 90 AND T_RET = 60
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S5_Tinj90_Tret60\S5_Tinj90_Tret60.xlsx', skiprows=2)

# S6 T_INJ = 90 AND T_RET = 30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S6_Tinj90_Tret30\S6_Tinj90_Tret30.xlsx', skiprows=2)

# S7 T_INJ = 70 AND T_RET = 40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S7_Tinj70_Tret40\S7_Tinj70_Tret40.xlsx', skiprows=2)

# S8 HIGH PERMEABILITY
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S8_high_perm\S8_high_perm.xlsx', skiprows=2)

# S9 LOW PERMEABILITY
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S9_low_perm\S9_low_perm.xlsx', skiprows=2)

# S10 LOW POROSITY
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S10_low_poro\S10_low_poro.xlsx', skiprows=2)

# S11 LOW ROCK COMPRESSIBILITY (CPOR)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S11_low_CPOR\S11_low_CPOR.xlsx', skiprows=2)

# S12 LOW ROCK SPECIFIC HEAT (CP-ROCK)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S12_low_CP-ROCK\S12_low_CP-ROCK.xlsx', skiprows=2)

# S13 HIGH THERMAL CONDUCTIVITY (THCONR0)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S13_high_THCONR0\S13_high_THCONR0.xlsx', skiprows=2)

# S14 LOW AQUIFER THICKNESS (20M)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S14_low_aquifer_thickness\S14_low_aquifer_thickness.xlsx', skiprows=2)

# S15 HIGH AQUIFER THICKNESS (100M)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S15_high_aquifer_thickness\S15_high_aquifer_thickness.xlsx', skiprows=2)

# S16 HIGH AMBIENT T (40C)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S16_high_ambient_T\S16_high_ambient_T.xlsx', skiprows=2)

# S17 GROUNDWATER FLOW
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S17_groundwater_flow\S17_groundwater_flow.xlsx', skiprows=2)

# S18 ONE CLAY LAYER IN AQUIFER
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S18_one_clay_layer\S18_one_clay_layer.xlsx', skiprows=2)

# S19 TWO CLAY LAYERS IN AQUIFER
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S19_two_clay_layers\S19_two_clay_layers.xlsx', skiprows=2)

# S20 QUARTER-5-SPOT PATTERN
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S20_5_spot_pattern\S20_5_spot_pattern.xlsx', skiprows=2)
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S20_5_spot_pattern\S20_5_spot_pattern_smallerwellspacing.xlsx', skiprows=2)   # Smaller well spacing

# S21 FOLLOW DEMAND CURVE
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S21_follow_demand_curve\S21_follow_demand_curve.xlsx', skiprows=2)

# S22 5 MONTHS LOADING 7 MONTHS PRODUCING
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\S22_5_months_loading\S22_5_months_loading.xlsx', skiprows=2)

#%% FOCUSSING ON DIFFERENT STORAGE VOLUMES

#1 200k See small storage V above (S1)

#2 400k
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Storage_V\Storage_400k.xlsx', skiprows=2)

#3 600k
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Storage_V\Storage_600k.xlsx', skiprows=2)

#4 800k
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Storage_V\Storage_800k.xlsx', skiprows=2)

#5 1000k
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Storage_V\Storage_1000k.xlsx', skiprows=2)

#%% FOCUSSING ON DIFFERENT AQUIFER PERMEABILITIES

#1 0.1 Darcy See low perm (S9)

#2 1   Darcy
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Permeability\Perm1Darcy.xlsx', skiprows=2)

#3 10  Darcy See base case

#4 25  Darcy
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Permeability\Perm25Darcy.xlsx', skiprows=2)

#5 50  Darcy See high perm (S8)

#6 100 Darcy
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Permeability\Perm100Darcy.xlsx', skiprows=2)

#7 200 Darcy
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Permeability\Perm200Darcy.xlsx', skiprows=2)

#%% PARAMETERSTUDY PERMEABILITY VOLUME AND DELTA T

#1 P10 V200 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V200_dt20.xlsx', skiprows=2)

#2 P10 V200 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V200_dt30.xlsx', skiprows=2)

#3 P10 V200 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V200_dt40.xlsx', skiprows=2)

#4 P10 V600 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V600_dt20.xlsx', skiprows=2)

#5 P10 V600 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V600_dt30.xlsx', skiprows=2)

#6 P10 V600 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V600_dt40.xlsx', skiprows=2)

#7 P10 V1000 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V1000_dt20.xlsx', skiprows=2)

#8 P10 V1000 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V1000_dt30.xlsx', skiprows=2)

#9 P10 V1000 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm10_V1000_dt40.xlsx', skiprows=2)

#10 P50 V200 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V200_dt20.xlsx', skiprows=2)

#11 P50 V200 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V200_dt30.xlsx', skiprows=2)

#12 P50 V200 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V200_dt40.xlsx', skiprows=2)

#13 P50 V600 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V600_dt20.xlsx', skiprows=2)

#14 P50 V600 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V600_dt30.xlsx', skiprows=2)

#15 P50 V600 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V600_dt40.xlsx', skiprows=2)

#16 P50 V1000 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V1000_dt20.xlsx', skiprows=2)

#17 P50 V1000 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V1000_dt30.xlsx', skiprows=2)

#18 P50 V1000 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm50_V1000_dt40.xlsx', skiprows=2)

#19 P100 V200 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V200_dt20.xlsx', skiprows=2)

#20 P100 V200 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V200_dt30.xlsx', skiprows=2)

#21 P100 V200 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V200_dt40.xlsx', skiprows=2)

#22 P100 V600 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V600_dt20.xlsx', skiprows=2)

#23 P100 V600 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V600_dt30.xlsx', skiprows=2)

#24 P100 V600 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V600_dt40.xlsx', skiprows=2)

#25 P100 V1000 T20
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V1000_dt20.xlsx', skiprows=2)

#26 P100 V1000 T30
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V1000_dt30.xlsx', skiprows=2)

#27 P100 V1000 T40
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_perm_storV_dT\perm100_V1000_dt40.xlsx', skiprows=2)

#%% PARAMETERSTUDY HORIZONTAL VERTICAL PERMEABILITY

#1 kh1 kv1
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh1_kv1.xlsx', skiprows=2)

#2 kh1 kv5
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh1_kv5.xlsx', skiprows=2)

#3 kh1 kv10
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh1_kv10.xlsx', skiprows=2)

#4 kh10 kv1
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh10_kv1.xlsx', skiprows=2)

#5 kh10 kv5
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh10_kv5.xlsx', skiprows=2)

#6 kh10 kv10
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh10_kv10.xlsx', skiprows=2)

#7 kh50 kv1
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh50_kv1.xlsx', skiprows=2)

#8 kh50 kv5
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh50_kv5.xlsx', skiprows=2)

#9 kh50 kv10
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh50_kv10.xlsx', skiprows=2)

#10 kh100 kv1
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh100_kv1.xlsx', skiprows=2)

#11 kh100 kv5
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh100_kv5.xlsx', skiprows=2)

#12 kh100 kv10
# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\2022\Parameterstudy_kh_kv\kh100_kv10.xlsx', skiprows=2)

#%% INJECTIE FULL PRODUCTIE TOP HALF

# df_ates = pd.read_excel(r'C:\Users\924224\OneDrive - EBN BV\Documents\CMG MODEL\SCENARIOS\INJfull_PRODtop_TNO.xlsx', skiprows=2)

#%% CREATE DATAFRAME WITH ALL RELEVANT DATA

# # Way to delete first two rows in Excel file
# df_ates = df_ates.iloc[1:,:]
# headers = df_ates.iloc[0]
# df_ates = pd.DataFrame(df_ates.values[1:], columns=headers)

# Drop duplicate values for column 'Date'
df_ates = df_ates.drop_duplicates(subset='Date')

# Assign datetime as index
df_ates['Datetime'] = pd.to_datetime('5/1/2010') + pd.to_timedelta(
    df_ates['Time (day)'] * pd.Timedelta(('1 days')), unit='s')
df_ates.set_index('Datetime', inplace=True)

# Set the new time interval
set_interval = '1D'

# Generate the new index on which the data should correspond based on original index range
resample_index = pd.date_range(df_ates.index[0] - pd.Timedelta(set_interval),
                                df_ates.index[-1] + pd.Timedelta(set_interval), freq=set_interval, normalize=True,
                                closed='right')

# Create a new index as the union of the old and new index and interpolate on the combined index
# Then reindex the dataframe to the new index
tdata = df_ates.reindex(df_ates.index.union(resample_index)).interpolate('index').reindex(resample_index)

# Maintain initial values of original dataset in the newly indexed one at the edges
tdata.iloc[0] = df_ates.iloc[0]
tdata.iloc[-1] = df_ates.iloc[-1]

# Add the number of periods for further economic assessment
tdata['econ_periods'] = np.arange(len(tdata))

# compute the deltahours for economic and energy assessments
tdata['Deltahours'] = tdata.index.to_series().diff(1) / pd.Timedelta('1 hour')

# # verify that we only have one delta
# tdata['Deltahours'].unique()

# Add the Fluid specific heat capacity and Water mass density to DataFrame
tdata['Fluid specific heat [J/kg*K]'] = c_w
tdata['Water mass density [kg/m³]']   = rho_w

# Multiply Fluid rate data times 2 -> simulations were mirrored
tdata['Hot well INJ-Fluid Rate SC (m³/day)'  ] = tdata['Hot well INJ-Fluid Rate SC (m³/day)'  ] * Multiply
tdata['Hot well PROD-Fluid Rate SC (m³/day)' ] = tdata['Hot well PROD-Fluid Rate SC (m³/day)' ] * Multiply
tdata['Warm well INJ-Fluid Rate SC (m³/day)' ] = tdata['Warm well INJ-Fluid Rate SC (m³/day)' ] * Multiply
tdata['Warm well PROD-Fluid Rate SC (m³/day)'] = tdata['Warm well PROD-Fluid Rate SC (m³/day)'] * Multiply

# Allign the two DataFrames -> starting on same date and with same length
Start_date  = tdata.iloc[0]['Date']                # Starting date of GEM simulation
Start_demand = date(2010,1,1)                      # Start date of data from demand curve
Num_days = (Start_date.date() - Start_demand)      # Retrieve total number of days
Num_days = int(Num_days.total_seconds()/(3600*24)) # Convert from datetime to float

#df_demand['Demand curve alligned [MW]', 'Geothermal used alligned [MW]'] = df_demand[df_demand.loc[:, 'Demand curve [MW]']+Num_days, df_demand.loc[:, 'Geothermal used [MW]']+Num_days]

df_demand_merge = pd.concat([df_demand, df_demand], ignore_index=True)      # Create DataFrame with two full years of df_demand data
df_demand_merge = df_demand_merge.drop(df_demand_merge.index[:Num_days])    # Drop the values before the first day of GEM simulations
df_demand_merge = df_demand_merge.drop(df_demand_merge.index[365:])         # Drop the values after one year
df_demand_merge.reset_index(drop=True)                                      # Set index from 0
df_demand_merge = pd.concat([df_demand_merge]*int(len(tdata.index)/365), ignore_index=True)    # Set length of demand DataFrame to length of tdata
tdata = tdata.drop(tdata.index[int(len(df_demand_merge.index)):])           # Set length of tdata to length of demand DataFrame

# Merge the two DataFrames into one:
df_demand_merge['Data'] = tdata.index               # Add timedata as column to df_demand_merge
df_demand_merge = df_demand_merge.set_index('Data') # Set timedata as index of df_demand_merge
df = pd.concat([df_demand_merge, tdata], axis=1)    # Concatenate the two DataFrames

# Test to see effect of double demand
# df['Demand curve [MW]'] = 2*df['Demand curve [MW]']

# Calculate Heat output from HTS [MW]
# P = (Q*rho*c*dT)/(1e6*24*3600) [MW]
df['HTS power [MW]'] = df['Hot well PROD-Fluid Rate SC (m³/day)']*df['Water mass density [kg/m³]']*df['Fluid specific heat [J/kg*K]']*(df['Hot well PROD-Well bottom hole temperature (C)']-df['Warm well INJ-Well bottom hole temperature (C)'])/(1e6*24*3600)

# Define HTS power after cut-off
df.loc[df['Hot well PROD-Well bottom hole temperature (C)'] > T_cutoff,  'HTS power cutoff [MW]'] = df['HTS power [MW]']
df.loc[df['Hot well PROD-Well bottom hole temperature (C)'] <= T_cutoff, 'HTS power cutoff [MW]'] = 0

HTS_power = df['HTS power cutoff [MW]'].to_numpy() # Used for plotting the power
Dates = df.index.to_numpy()                        # Used for plotting the power

# Add 'Geothermal used [MW]' and 'HTS power [MW]' in order to plot
df['Geo + HTS [MW]'] = df['Geothermal used [MW]'] + df['HTS power cutoff [MW]']

# Check whether all values are below actual demand curve
df.loc[df['Geo + HTS [MW]'] > df['Demand curve [MW]'], 'Geo + HTS [MW]'] = df['Demand curve [MW]']

# Create column for 'HTS used [MW]' which is difference between HTS and demand curve
df['HTS used [MW]'] = df['Geo + HTS [MW]'] - df['Geothermal used [MW]']

# Calculate Heat supplied by a gas boiler
df.loc[df['Demand curve [MW]'] > df['Geothermal used [MW]'] + df['HTS power cutoff [MW]'], 'Gas boiler [MW]' ] = df['Demand curve [MW]'] - df['Geothermal used [MW]'] - df['HTS power cutoff [MW]']

## PUMPING POWER CALCULATIONS

# Obtain flow rate for hot well INJ+PROD
df['Hot well volumetric flow rate SC total [m3/day]'] = df['Hot well INJ-Fluid Rate SC (m³/day)'] + df['Hot well PROD-Fluid Rate SC (m³/day)']

# Obtain a column with correct BHP's for INJ/PROD periods
df.loc[df['Hot well INJ-Well Bottom-hole Pressure (kPa)']  >  df['Hot well PROD-Well Bottom-hole Pressure (kPa)'], 'Hot Well BHP (kPa)']   = df['Hot well INJ-Well Bottom-hole Pressure (kPa)']
df.loc[df['Hot well INJ-Well Bottom-hole Pressure (kPa)']  <= df['Hot well PROD-Well Bottom-hole Pressure (kPa)'], 'Hot Well BHP (kPa)']   = df['Hot well PROD-Well Bottom-hole Pressure (kPa)']
df.loc[df['Warm well INJ-Well Bottom-hole Pressure (kPa)'] >  df['Warm well PROD-Well Bottom-hole Pressure (kPa)'], 'Warm Well BHP (kPa)'] = df['Warm well INJ-Well Bottom-hole Pressure (kPa)']
df.loc[df['Warm well INJ-Well Bottom-hole Pressure (kPa)'] <= df['Warm well PROD-Well Bottom-hole Pressure (kPa)'], 'Warm Well BHP (kPa)'] = df['Warm well PROD-Well Bottom-hole Pressure (kPa)']

# Calculate pumping power
df['Pumping power [MW]'] = ((df['Hot well volumetric flow rate SC total [m3/day]']/24/3600) * (abs(df['Warm Well BHP (kPa)'] - df['Hot Well BHP (kPa)'])/1e3)) / df_eco.loc[0,'mu_pump [-]']

# Calculate the pumping costs
df['Pumping costs [EUR]'] = df['Pumping power [MW]'] * df['Deltahours'] * df_eco.loc[0,'Price_elec [EUR/MWh]']

# Heatpump calculations
df.loc[df['Hot well PROD-Fluid Rate SC (m³/day)'] > 0, 'Delta T (C)'] = df_eco.loc[0,'T_network [C]'] - df['Hot well PROD-Well bottom hole temperature (C)'] # Obtain delta T only for production periods

df['Heatpump power [MW]'] = df['Hot well PROD-Fluid Rate SC (m³/day)']*df['Water mass density [kg/m³]']*df['Fluid specific heat [J/kg*K]']*df['Delta T (C)']/(1e6*24*3600)

#%% RECOVERY EFFICIENCY CALCULATIONS
# T = df.at(0, 'Hot well INJ-Well bottom hole temperature (C)')

# Take the sum of columns of specific intervals for every year
Total_T_yr1 = df.iloc[    :364 ].sum()
Total_T_yr2 = df.iloc[365 :729 ].sum()
Total_T_yr3 = df.iloc[730 :1094].sum()
Total_T_yr4 = df.iloc[1095:1459].sum()

# Calculate the recovery efficiency for every year based on power in/output
df_eco['RE_yr1 [-]'] = Total_T_yr1['HTS power cutoff [MW]']/((df_eco['Q [m3/day]']*rho_w*c_w*(df_eco['T [C]']-df_eco['T_warm [C]']))/(24*3600*df_eco['V [m3]'])*(365/2))
df_eco['RE_yr2 [-]'] = Total_T_yr2['HTS power cutoff [MW]']/((df_eco['Q [m3/day]']*rho_w*c_w*(df_eco['T [C]']-df_eco['T_warm [C]']))/(24*3600*df_eco['V [m3]'])*(365/2))
df_eco['RE_yr3 [-]'] = Total_T_yr3['HTS power cutoff [MW]']/((df_eco['Q [m3/day]']*rho_w*c_w*(df_eco['T [C]']-df_eco['T_warm [C]']))/(24*3600*df_eco['V [m3]'])*(365/2))
df_eco['RE_yr4 [-]'] = Total_T_yr4['HTS power cutoff [MW]']/((df_eco['Q [m3/day]']*rho_w*c_w*(df_eco['T [C]']-df_eco['T_warm [C]']))/(24*3600*df_eco['V [m3]'])*(365/2))

# format_RE_yr4 = "{:.2f}".format(df_eco['RE_yr4 [-]'].to_numpy)
print('Recovery efficiency in year 4:',df_eco.loc[0,'RE_yr4 [-]'])


#%% ECONOMICS
# Source: WarmingUp PPT 'Inpassing van HTO in warmtenetten' Slide 13

# CO2_tax               = 41.21          # EUR/ton CO2 (https://www.change.inc/finance/zeven-vragen-over-de-nederlandse-co2-heffing-36999)
CO2_tax               = 125            # EUR/ton CO2 (https://www.pwc.nl/en/topics/sustainability/green-deal-monitor/green-deal-monitor-4.html#:~:text=To%20ensure%20achieving%20this%20goal,Economic%20Affairs%20and%20Climate%20Policy.)


# Parameters to define range to be used from the DataFrame
Year_start = (Year-1)*365
Year_end   = Year*365

# Geothermal costs PBL 2022
# df_eco.loc[df_eco['Base'] < 12, 'Geothermal_instal [EUR/MW]'] = 2.333e6        # PBL 2022
# df_eco.loc[12 < df_eco['Base'], 'Geothermal_instal [EUR/MW]'] = 1.395e6        # PBL 2022
# df_eco.loc[df_eco['Base'] > 20, 'Geothermal_instal [EUR/MW]'] = 1.014e6        # PBL 2022

df_eco['Geothermal_instal [EUR/MW]']        = 1.646e6        # EUR/MW (1.36e6 = TNO, 1.646e6 = PBL)
df_eco['Geothermal_fixed_opex [EUR/MW/yr]'] = 91e3           # EUR/MW/yr
df_eco['Geothermal_varia_opex [EUR/MWh]']   = 2              # EUR/MWh
df_eco['Geothermal_SDE [EUR/MWh]']          = 51.8           # EUR/MWh (51.8 = Fase 4 bedrag uit SDE++ 2021, 42 = bedrag uit GTD rappport)
df_eco['Geothermal_correction [EUR/MWh]']   = 20.1           # EUR/MWh (correctiebedrag uit SDE++ 2021)
df_eco['Geothermal_CO2 [kg/MWh]']           = 23             # kg/MWh

df2 = df.set_index(['econ_periods'], inplace=False)                     # Create second df to extract data from -> no datetime index
Base_MWh = np.sum(df2.loc[Year_start:Year_end, 'Geothermal used [MW]']) # [MWh]
df_eco['Geothermal_CO2_emis [ton/yr]']   = df_eco['Geothermal_CO2 [kg/MWh]']*Base_MWh/1e3                     # ton per year

df_eco['Geothermal_instal_cost [EUR]']     = df_eco['Geothermal_instal [EUR/MW]']*np.max(df['Geothermal Base load [MW]'])     # At t=0
df_eco['Geothermal_fixed_opex_cost [EUR]'] = df_eco['Geothermal_fixed_opex [EUR/MW/yr]']*np.max(df['Geothermal Base load [MW]']) # Every year
df_eco['Geothermal_varia_opex_cost [EUR]'] = df_eco['Geothermal_varia_opex [EUR/MWh]']*Base_MWh                                # Every year
df_eco['Geothermal_SDE_rev [EUR]']         = df_eco['Geothermal_SDE [EUR/MWh]']*Base_MWh                                       # Every year
df_eco['Geothermal_CO2_cost [EUR]']        = df_eco['Geothermal_CO2_emis [ton/yr]']*CO2_tax                                    # Every year

# HTS costs
df_eco['ATES_instal']           = 0.9e6                               # EUR/MW
df_eco['ATES_fixed_opex']       = 33e3#180e3                          # EUR/yr
df_eco['ATES_varia_opex']       = 12e3                                # EUR/MW/yr
df_eco['ATES_varia_opex_GT']    = df_eco['Geothermal_varia_opex [EUR/MWh]']     # EUR/MWh (costs to pump the excess geothermal energy)
df_eco['ATES_CO2']              = 7.675*3.6                           # kg/MWh  (source: vraag Raymond)
df_eco['ATES_CO2_GT']           = df_eco['Geothermal_CO2 [kg/MWh]']            # kg/MWh
df_eco['ATES_price']            = df_eco['Geothermal_SDE [EUR/MWh]']            # EUR/MWh (assumption: price equal to geothermal)

df_eco['HTS_total_MW'] = np.max(df2.loc[Year_start:Year_end, 'HTS power cutoff [MW]'])                                                                       # [MW] in one year
df_eco['ATES_CO2_emis'] = (np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]'])/1e3 + np.sum(df2.loc[Year_start:Year_end, 'Geothermal Excess [MW]'])/1e3)*df_eco['ATES_CO2'] # ton per year

# Costs from Delft HT-ATES report
# ATES_instal_cost           = 3048000 #ATES_instal*HTS_total_MW                         # At t=0
# ATES_fixed_opex_cost       = 0 #ATES_fixed_opex                                  # Every year
# ATES_varia_opex_cost       = 0 #ATES_varia_opex*HTS_total_MW                     # Every year
# ATES_varia_opex_GT_cost    = 0 #ATES_varia_opex_GT*1e6*np.sum(HTS_used)          # Every year
# ATES_CO2_cost              = 376000 #ATES_CO2_emis*CO2_tax                            # Every year
# ATES_rev                   = ATES_price*np.sum(HTS_used)*1e6                  # Every year

df_eco['ATES_instal_cost [EUR]']           = df_eco['ATES_instal']*df_eco['HTS_total_MW']                                                     # At t=0
df_eco['ATES_fixed_opex_cost [EUR]']       = df_eco['ATES_fixed_opex']                                                              # Every year
# ATES_varia_opex_cost       = ATES_varia_opex*HTS_total_MW                                                 # Every year
df_eco['ATES_varia_opex_cost [EUR]']       = df2.loc[Year_start:Year_end, 'Pumping costs [EUR]'].sum()
df_eco['ATES_varia_opex_GT_cost [EUR]']    = df_eco['ATES_varia_opex_GT']*np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]'])*24  # Every year
df_eco['ATES_CO2_cost [EUR]']              = df_eco['ATES_CO2_emis']*CO2_tax                                                        # Every year
df_eco['ATES_rev [EUR]']                   = df_eco['ATES_price']*np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]'])*24          # Every year

# Boiler costs
df_eco['Boiler_instal']         = 0.1e6          # EUR/MW
df_eco['Boiler_fixed_opex']     = 0.02           # factor of CAPEX per year 
df_eco['Boiler_varia_opex']     = 100#17             # EUR/MWh
df_eco['Boiler_CO2']            = 200            # kg/MWh (https://www.duec.nl/warmtepompen-besparen-veel-co2-uitstoot-ten-opzichte-van-gasverwarming/#:~:text=Bij%20verbranding%20van%20aardgas%20komt,uitgestoten%20(1800%2F9).)
df_eco['Boiler_price']          = 35             # EUR/MWh (assumption: price equal to geothermal)

# Boiler HTS included
df_eco['Boiler_total_MW_HTS']   = np.max(df2.loc[Year_start:Year_end, 'Gas boiler [MW]'])                # Boiler power [MW]
df_eco['Boiler_CO2_emis_HTS']   = df_eco['Boiler_CO2']*np.sum(df2.loc[Year_start:Year_end, 'Gas boiler [MW]'])/1e3 # ton per year

df_eco['Boiler_instal_cost_HTS [EUR]']         = df_eco['Boiler_instal']*df_eco['Boiler_total_MW_HTS']                                            # At t=0
df_eco['Boiler_fixed_opex_cost_HTS [EUR]']     = df_eco['Boiler_fixed_opex']*df_eco['Boiler_instal_cost_HTS [EUR]']                                     # Every year
df_eco['Boiler_varia_opex_cost_HTS [EUR]']     = df_eco['Boiler_varia_opex']*np.sum(df2.loc[Year_start:Year_end, 'Gas boiler [MW]'])*24 # Every year
df_eco['Boiler_CO2_cost_HTS [EUR]']            = df_eco['Boiler_CO2_emis_HTS']*CO2_tax                                                  # Every year
df_eco['Boiler_rev_HTS [EUR]']                 = df_eco['Boiler_price']*np.sum(df2.loc[Year_start:Year_end, 'Gas boiler [MW]'])         # Every year

# Boiler HTS excluded
df_eco['Boiler_total_MW']       = np.max(df2.loc[Year_start:Year_end, 'Gas boiler [MW]'] + df2.loc[Year_start:Year_end, 'HTS used [MW]'])                          # Boiler energy [MW]
df_eco['Boiler_CO2_emis']       = df_eco['Boiler_CO2']*(np.sum(df2.loc[Year_start:Year_end, 'Gas boiler [MW]']) + np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]']))/1e3 # ton per year

df_eco['Boiler_instal_cost [EUR]']         = df_eco['Boiler_instal']*df_eco['Boiler_total_MW']                                                                                                          # At t=0
df_eco['Boiler_fixed_opex_cost [EUR]']     = df_eco['Boiler_fixed_opex']*df_eco['Boiler_instal_cost [EUR]']                                                                                                   # Every year
df_eco['Boiler_varia_opex_cost [EUR]']     = df_eco['Boiler_varia_opex']*(np.sum(df2.loc[Year_start:Year_end, 'Gas boiler [MW]']) + np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]']))*24 # Every year
df_eco['Boiler_CO2_cost [EUR]']            = df_eco['Boiler_CO2_emis']*CO2_tax                                                                                                                # Every year
df_eco['Boiler_rev [EUR]']                 = df_eco['Boiler_price']*(np.sum(df2.loc[Year_start:Year_end, 'Gas boiler [MW]']) + np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]']))         # Every year

# Heatpump costs
df_eco['Heatpump_instal [EUR/MW]']      = 0.5e6    # EUR/MW
df_eco['Heatpump_fixed_opex [factor]']  = 0.02     # factor of CAPEX per year
df_eco['Heatpump_varia_opex [EUR/MWh]']  = 80       # EUR/MWh

df_eco['Heatpump_instal_cost [EUR]']     = df_eco['Heatpump_instal [EUR/MW]']*np.max(df['Heatpump power [MW]'])
df_eco['Heatpump_fixed_opex_cost [EUR]'] = df_eco['Heatpump_fixed_opex [factor]']*df_eco['Heatpump_instal_cost [EUR]']
df_eco['Heatpump_varia_opex_cost [EUR]'] = df_eco['Heatpump_varia_opex [EUR/MWh]']*np.sum(df2.loc[Year_start:Year_end, 'Heatpump power [MW]'])*24*3600

# Calculate total CO2 emissions for both scenarios
df_eco['Total CO2 emissions HTS'] = df_eco['Geothermal_CO2_emis [ton/yr]'] + df_eco['ATES_CO2_emis'] + df_eco['Boiler_CO2_emis_HTS']
df_eco['Total CO2 emissions']     = df_eco['Geothermal_CO2_emis [ton/yr]'] + df_eco['Boiler_CO2_emis']

#%% LCOH CALCULATIONS

def Calculate_LCOH(CAPEX, Lifetime, OPEX):
    time_yrs = np.linspace(1,Lifetime,Lifetime) # Array including all the years in Lifetime
    LCOH_top = np.zeros(Lifetime)               # Create empty array for LCOH
    LCOH_bot = np.zeros(Lifetime)
    LCOH     = np.zeros(Lifetime)
    for i in range(Lifetime):
        LCOH_top[i] = (CAPEX + np.sum(OPEX[:i])) / (1+df_eco['r_annual [-]'])**time_yrs[i]
        LCOH_bot[i] = (i+1)*df2.loc[Year_start:Year_end, 'Demand curve [MW]'].sum()*24 / (1+df_eco['r_annual [-]'])**time_yrs[i]
        LCOH[i]     = np.sum(LCOH_top[:i]) / np.sum(LCOH_bot[:i])
    # LCOH = LCOH_top / LCOH_bot
    return LCOH, time_yrs

# LCOH Calculations HTS included
df_eco['Instal_total_HTS'] = df_eco['Geothermal_instal_cost [EUR]'] + df_eco['ATES_instal_cost [EUR]'] + df_eco['Boiler_instal_cost_HTS [EUR]']  # Total installation costs
OPEX_HTS = np.zeros(df_eco['Lifetime [yrs]']) # All annual cashflows, excluding the investment costs at t=0

for i in range(df_eco.loc[0,'Lifetime [yrs]']):
    OPEX_HTS[i] = df_eco['Geothermal_fixed_opex_cost [EUR]'] + df_eco['Geothermal_varia_opex_cost [EUR]'] + df_eco['Geothermal_CO2_cost [EUR]'] \
                + df_eco['ATES_fixed_opex_cost [EUR]'] + df_eco['ATES_varia_opex_cost [EUR]'] + df_eco['ATES_varia_opex_GT_cost [EUR]'] + df_eco['ATES_CO2_cost [EUR]'] \
                + df_eco['Boiler_fixed_opex_cost_HTS [EUR]'] + df_eco['Boiler_varia_opex_cost_HTS [EUR]'] + df_eco['Boiler_CO2_cost_HTS [EUR]']

LCOH_HTS, time_yrs = Calculate_LCOH(df_eco['Instal_total_HTS'], df_eco.loc[0,'Lifetime [yrs]'], OPEX_HTS)

# Print the LCOH at the end of the lifetime
# format_LCOH_HTS = "{:.2f}".format(LCOH_HTS[df_eco['Lifetime [yrs]']-1])
print('LCOH_HTS at end of lifetime:', LCOH_HTS[df_eco['Lifetime [yrs]']-1])

# LCOH Calculations HTS excluded
df_eco['Instal_total'] = df_eco['Geothermal_instal_cost [EUR]'] + df_eco['Boiler_instal_cost [EUR]']  # Total installation costs
OPEX = np.zeros(df_eco['Lifetime [yrs]']) # All annual cashflows, excluding the investment costs at t=0

for i in range(df_eco.loc[0,'Lifetime [yrs]']):
    OPEX[i] = df_eco['Geothermal_fixed_opex_cost [EUR]'] + df_eco['Geothermal_varia_opex_cost [EUR]'] + df_eco['Geothermal_CO2_cost [EUR]'] \
            + df_eco['Boiler_fixed_opex_cost [EUR]'] + df_eco['Boiler_varia_opex_cost [EUR]'] + df_eco['Boiler_CO2_cost [EUR]']

LCOH, _            = Calculate_LCOH(df_eco['Instal_total'],     df_eco.loc[0,'Lifetime [yrs]'], OPEX    )

# Print the LCOH at the end of the lifetime
print('LCOH at end of lifetime:', LCOH[df_eco['Lifetime [yrs]']-1]) 

# Print the total energy output from HTS in the stated year
format_HTS_energy = "{:.2f}".format(np.sum(df2.loc[Year_start:Year_end, 'HTS power [MW]'])*24/1e3)
print('HTS energy output [GWh]', format_HTS_energy)

#%% LOAD FACTOR CALCULATIONS

df_eco['LF_geothermal']     = np.sum(df2.loc[0:365,'Geothermal used [MW]']) / (365*df_eco['Base'])
df_eco['LF_geothermal_HTS'] = np.sum(df2.loc[0:365,'Geo + HTS [MW]']) / (365*df_eco['Base'])
df_eco['LF_HT-ATES']        = np.sum(df2.loc[0:365,'HTS used [MW]']) / np.sum(df2.loc[0:365,'HTS power [MW]'])

#%% DATAFRAME EXTRACTION FOR DATA ANALYSIS
# To be used for data analysis in spotfire along with GEM simulation result data

df_eco['LCOH_HTS']                       = LCOH_HTS[df_eco['Lifetime [yrs]']-1]
df_eco['LCOH']                           = LCOH[df_eco['Lifetime [yrs]']-1]
df_eco['HTS energy output [GWh]']        = np.sum(df2.loc[Year_start:Year_end, 'HTS power [MW]'])*24/1e3
df_eco['HTS energy used [GWh]']          = np.sum(df2.loc[Year_start:Year_end, 'HTS used [MW]'])*24/1e3
df_eco['Geothermal energy output [GWh]'] = np.sum(df2.loc[Year_start:Year_end,'Geothermal used [MW]'])*24/1e3
df_eco['Boiler energy output [GWh]']     = np.sum(df2.loc[Year_start:Year_end,'Gas boiler [MW]'])*24/1e3
df_eco.to_excel('Base_case_economics.xlsx', sheet_name='sheet1', index=False)

print(df_eco['HTS energy used [GWh]'])

#%% PLOTTING

# Plot HTS power over time
plt.figure(1)
plt.plot(Dates, HTS_power)
plt.title('HTS power over time')
plt.xlabel('Date')
plt.ylabel('Power [MW]')

# Plot heat demand/supply curves
plt.figure(2)
plt.plot(df['econ_periods'],         df['Demand curve [MW]']   , color='r', label='Gas boiler'  )
plt.plot(df['econ_periods'],         df['Geo + HTS [MW]']      , color='g', label='HTS'         )
plt.plot(df['econ_periods'],         df['Geothermal used [MW]'], color='b', label='Geothermal'  )
plt.fill_between(df['econ_periods'], df['Demand curve [MW]']   , color='r'                      )
plt.fill_between(df['econ_periods'], df['Geothermal used [MW]'], df['Geo + HTS [MW]'], color='g')
plt.fill_between(df['econ_periods'], df['Geothermal used [MW]'], color='b'                      )
plt.title('Heat supply and demand')
plt.xlabel('Time [Days]')
plt.ylabel('Average daily powrer [MW]')
plt.xlim(976, 1340)
plt.legend()

# Plot recovery efficiency over time
plt.figure(3)
plt.plot([1,2,3,4], [df_eco['RE_yr1 [-]'], df_eco['RE_yr2 [-]'], df_eco['RE_yr3 [-]'], df_eco['RE_yr4 [-]']])
plt.title('HT-ATES recovery efficiency over time')
plt.xlabel('Time [years]')
plt.ylabel('Recovery efficiency [-]')
plt.ylim(0,1)

# Plot LCOH over time
plt.figure(4)
plt.plot(time_yrs, LCOH,     label='LCOH HTS excluded')
plt.plot(time_yrs, LCOH_HTS, label='LCOH HTS included')
plt.title('LCOH over time')
plt.xlabel('Time [years]')
plt.ylabel('LCOH [EUR/MWh]')
plt.legend()

plt.figure(6)
plt.plot(df['econ_periods'], df['Pumping power [MW]'])
plt.title('Pumping power over time')
plt.xlabel('Time [days]')
plt.ylabel('Pumping power [MW]')

plt.figure(7)
plt.plot(df['econ_periods'], df2['Hot well INJ-Well bottom hole temperature (C)'], label='Hot well INJ')
plt.plot(df['econ_periods'], df2['Hot well PROD-Well bottom hole temperature (C)'], label='Hot well PROD')
plt.plot(df['econ_periods'], df2['Warm well INJ-Well bottom hole temperature (C)'], label='Warm well INJ')
plt.plot(df['econ_periods'], df2['Warm well PROD-Well bottom hole temperature (C)'], label='Warm well PROD')
plt.title('Bottom hole temperature [C] over time')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [C]')
plt.legend()

plt.figure(8)
plt.plot(df['econ_periods'], df['Pumping costs [EUR]'])
plt.title('Pumping costs over time')
plt.xlabel('Time [days]')
plt.ylabel('Pumping costs [EUR]')
# plt.legend()

# plt.figure(9)
# plt.plot(df['econ_periods'], df['Hot well INJ-Well Bottom-hole Pressure (kPa)'], label='INJ')
# plt.plot(df['econ_periods'], df['Hot well PROD-Well Bottom-hole Pressure (kPa)'], label='PROD')
# plt.plot(df['econ_periods'], df['Hot Well BHP (kPa)'], label='SUM')

# fig, axs = plt.subplots(2, 3)
# axs[0, 0].plot(df['econ_periods'], df2['Hot well INJ-Well bottom hole temperature (C)'], label='Hot well INJ')
# axs[0, 0].plot(df['econ_periods'], df2['Hot well PROD-Well bottom hole temperature (C)'], label='Hot well PROD')
# axs[0, 0].plot(df['econ_periods'], df2['Warm well INJ-Well bottom hole temperature (C)'], label='Warm well INJ')
# axs[0, 0].plot(df['econ_periods'], df2['Warm well PROD-Well bottom hole temperature (C)'], label='Warm well PROD')
# axs[0, 0].legend()
# axs[0, 0].set_title('Bottom-hole Temperature [C]')
# axs[0, 1].plot(df['econ_periods'], 'tab:orange')
# axs[0, 1].set_title('Axis [0,1]')
# axs[1, 0].plot(df['econ_periods'], 'tab:green')
# axs[1, 0].set_title('Axis [1,0]')
# axs[1, 1].plot(df['econ_periods'], 'tab:red')
# axs[1, 1].set_title('Axis [1,1]')
# axs[0, 2].plot(df['econ_periods'], 'tab:green')
# axs[0, 2].set_title('Axis [1,0]')
# axs[1, 2].plot(time_yrs, LCOH,     label='LCOH HTS excluded')
# axs[1, 2].plot(time_yrs, LCOH_HTS, label='LCOH HTS included')
# axs[1, 2].set_title('Axis [1,1]')

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# Pie chart containing total energy supply per source
plt.figure(9)
labels = 'Geothermal', 'HTS', 'Boiler'
sizes = [np.sum(df2.loc[Year_start:Year_end,'Geothermal used [MW]']), np.sum(df2.loc[Year_start:Year_end,'HTS used [MW]']), np.sum(df2.loc[Year_start:Year_end,'Gas boiler [MW]'])]
explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors = ['b','g','r']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors, shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Annual heat supply')
plt.show()
