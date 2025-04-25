import pandas as pd

# File path for input and output
input_file = 'data_pro/GHCNh_USW00023183_2023.csv'
output_file = 'data_filtered/GHCNh_USW00023183_filtered_2023.csv'

# Read the CSV file, setting low_memory=False to avoid DtypeWarning
df = pd.read_csv(input_file, dtype={'DATE': str}, low_memory=False)

# Select the desired columns
selected_columns = ['DATE', 'temperature', 'dew_point_temperature', 'station_level_pressure',
                    'sea_level_pressure', 'wind_direction', 'wind_speed', 'relative_humidity',
                    'wet_bulb_temperature', 'visibility', 'altimeter',
                    'sky_cover_1', 'sky_cover_baseht_1']

df_selected = df[selected_columns]

# Filter rows where the DATE column ends in '51:00'
df_filtered = df_selected[df_selected['DATE'].str.endswith('51:00')]

# Save the filtered data to a new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
