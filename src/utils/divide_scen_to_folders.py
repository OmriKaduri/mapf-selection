import glob
import os
import math

divide_to = 12
scenario_path = '../../data/from-vpn/scen/custom'
n_scenrios = len(glob.glob(scenario_path + '/*.scen'))
folder_prefix = '/scen'
scenarios_in_folder = math.ceil(n_scenrios / divide_to)
# for i in range(1, divide_to + 1):
#     os.mkdir(scenario_path + folder_prefix + str(i))

current_scenario_folder = 1
problem_types = ['cross-sides', 'swap-sides', 'inside-out', 'outside-in', 'tight-to-tight', 'tight-to-wide']
all_scens = [f for f in glob.glob(scenario_path + '/*.scen')]
for i in range(divide_to):
    curr_type = problem_types[i % 6]
    curr_scenario = (i // 6) + 1
    current_scenario_folder = i + 1
    curr_files = [f for f in all_scens if (curr_type in f) and int(f.split('-')[-1].split('.scen')[0]) == curr_scenario]
    print(len(curr_files))
    for file in curr_files:
        scenario_file_name = file.split('\\')[-1]
        print(scenario_file_name)
        os.rename(file,
                  scenario_path + folder_prefix + str(current_scenario_folder) + '/' + scenario_file_name)
