from utility import Utility
from stream import Stream
from synheat import SynheatModel

streams = {
    # Stream(name, fcp, temperature_in, temperature_out, stream_film_coefficient, stream_type)
    "H1" : Stream("H1", 4, 550, 400, 1, "hot"),
    "H2" : Stream("H2", 5, 400, 300, 1, "hot"),
    "H3" : Stream("H3", 3, 450, 350, 1, "hot"),
    "C1" : Stream("C1", 6, 460, 570, 1, "cold"),
    "C2" : Stream("C2", 5, 370, 460, 1, "cold"),
    "C3" : Stream("C3", 5, 300, 470, 1, "cold"),
}

utilities = {
    "Heating": Utility("Heating", 600, 590, 5, 0.01),
    "Cooling": Utility("Cooling", 290, 305, 1, 0.0025),
}

# Check on number of streams
Stream.number_of_streams()
# Check on number of utilities
Utility.number_of_utilities()

hot_streams = ["H1", "H2", "H3"]
cold_streams = ["C1", "C2", "C3"]

# Task A

sum_heat_duty = 0
for h_str in hot_streams:
    T_in_str = streams[h_str].t_in
    T_out_str = streams[h_str].t_out
    fcp = streams[h_str].fcp
    duty_str = (T_in_str - T_out_str) * fcp
    sum_heat_duty += duty_str
print(f'Task A: Heating: {sum_heat_duty} MW.')

sum_cooling_duty = 0
for c_str in cold_streams:
    T_in_str = streams[c_str].t_in
    T_out_str = streams[c_str].t_out
    fcp = streams[c_str].fcp
    duty_str = (T_in_str - T_out_str) * fcp
    sum_cooling_duty += duty_str
print(f'Task A: Cooling: {sum_cooling_duty} MW.')

# Task E
synheat_1 = SynheatModel()
synheat_1.solve_model()
synheat_1.print_data()

# # Task G
# synheat_2 = SynheatModel()
# # Load previous results if they exist
# results_file = '/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/JSON/Synheat_last_run.json'
# synheat_2.load_results_from_json(results_file)

# # Run the model with integer cuts if results are not already loaded
# if not synheat_2.results:
#     synheat_2.run_with_integer_cuts(max_cuts=50)
#     synheat_2.save_results_to_json(results_file)

# # Plot the results
# synheat_2.plot_results('/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/Figures/synheatPlot')