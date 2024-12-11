from utility import Utility
from stream import Stream
from synheat import SynheatModel
import matplotlib.pyplot as plt

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
print("==========Task A===============")

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

synheat_1 = SynheatModel(minimum_heat_flux=False, number_of_stages=2)
synheat_1.run_with_integer_cuts(0)
print("==========Task E===============")
synheat_1.print_data(toLatex=False)

# Task G

"""The below code runs the results for one integer cut only using
a specified amount of stages."""

print("----------One Int Cut Solution ---------------")
synheat_3 = SynheatModel(minimum_heat_flux=True, number_of_stages=2)
synheat_3.run_with_integer_cuts(1)
synheat_3.return_best_result(print_output=True)
synheat_3.return_worst_result(print_output=True)
synheat_3.print_data(toLatex=False)

"""Best solution with two stage model."""
print("----------10 Int Cut Solution ---------------")
synheat_4 = SynheatModel(minimum_heat_flux=True, number_of_stages=2)
synheat_4.run_with_integer_cuts(100)
synheat_4.return_best_result(print_output=True)
synheat_4.print_data(toLatex=False)

# report end
"""
Finding the best solution possible:
The below code is meant to run with multiple integer cuts and vary the number of stages used 
to model the HEN, 
it seeks to find the best/worst solution within a specified range of integer cuts (e.g. <100)
Not part of the Tasks: (see Appendix for results)"""

# Lists to store best and worst objective values for each stage
best_objective_values = []
worst_objective_values = []

# Stages from 1 to 6
for stages in range(1, 7):
    print(f"Running model for {stages} stages")
    
    # Initialize model
    synheat_2 = SynheatModel(minimum_heat_flux=True, number_of_stages=stages)
    
    # File paths for results and figures
    results_file = f'/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/JSON/Synheat_last_run_{stages}_stages.json'
    plot_path = f'/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/Figures/synheatPlot_stages_{stages}'
    
    # Load previous results if they exist
    synheat_2.load_results_from_json(results_file)
    
    # Run the model with integer cuts if results are not already loaded
    if not synheat_2.results:
        synheat_2.run_with_integer_cuts(max_cuts=100)
        synheat_2.save_results_to_json(results_file)
    
    # Plot the results for the current stage
    synheat_2.plot_results(save_path=plot_path) 
    
    # Retrieve the best and worst results
    best_result = synheat_2.return_best_result()
    worst_result = synheat_2.return_worst_result()
    
    # Extract objective values and store them
    best_objective_values.append(best_result[1])  
    worst_objective_values.append(worst_result[1])  
    
    # Print the best and worst results for each stage
    print(f"Best result for {stages} stages:")
    for item in best_result:
        print(item)
    
    print(f"Worst result for {stages} stages:")
    for item in worst_result:
        print(item)

# Plotting the best and worst objective values across all stages
plt.figure(figsize=(10, 6))
stage_range = list(range(1, 7))
plt.plot(stage_range, best_objective_values, label='Best Objective Value', marker='o')
plt.plot(stage_range, worst_objective_values, label='Worst Objective Value', marker='o')
plt.xlabel('Number of Stages')
plt.ylabel('Objective Value / $ \\$ \\, yr{-1}$')
plt.title('Best and Worst Objective Values for Different Stages')
plt.legend()
plt.grid(True)
plt.savefig('/Users/kairuth/Desktop/MasterStudium/PDD/Marked_2/Figures/Best_Worst_Objective_Values.png')
plt.show() 