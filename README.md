Heat Integration Model from the PDD class at ETHZ.
The model this is based upon was originally published here:

Yee, T F, and Grossmann, I E, Simultaneous Optimization of Models for
Heat Integration - Heat Exchanger Network Synthesis. Computers and
Chemical Engineering 14, 10 (1990), 1151-1184.

The code has been adapted based on the example code (synheat) available here:
https://www.gams.com/latest/gamslib_ml/libhtml/

However that code was written in the GAMS language for mathematical 
programming and the code presented in this repository makes 
use of the python package GAMSPy which interacts with the GAMS 
framework using a container based structure.
The advantage of this approach is that it is easier to 
modify the problems mathematical structure and rerun several
modified problems in loops and then store and manipulate or
plot the resulting data. 

To the user: 
Please modify the settings in the settings.py file and instantiate 
an instance of the class as shown below:

your_model_name = SynheatModel(minimum_heat_flux=False, number_of_stages=4)

This would lead to a model that doesn't impose any minimum heat flux upon
the internal heat exchangers in the heat exchanger network (HEN) and attempts
to model the HEN using a number of four intermediate stages.

To print your resulting data you can use the available methods of the SynheatModel 
class:

your_model_name.print_data(toLatex = False)

(This will print all the relevant data from the model to the system output.)

To run the model using a given number of integer cuts (e.g. 5 integer cuts)
and subsequently print the results of the last obtained integer cut
configuration constrained solution:

your_model_name.run_with_integer_cuts(5)
your_model_name.print_data()

By design the instance of the class will save the best (lowest objective cost) and 
worst (highest objective cost) results that were created using every single integer
cut. These results can be accessed as shown below:

index_of_best_value, best_result_objective_value, best_z_tuple_table = your_model_name.return_best_result(print_output=True)

where
-index_of_best_value is the index of the integer cut (e.g. 19) if the resulting objective cost was lowest after 
implementation of 19 integer cuts.
-best_result_objective_value is the objective value (in USD per year) at that solution with e.g. 19 integer cuts
-best_z_tuple_table is a table of the configuration of the heat exchangers inside of the HEN (neglecting the heat
exchangers to exchange heat with the utilities).

To access the worst attained value instead use:
index_of_worst_value, worst_result_objective_value, worst_z_tuple_table = your_model_name.return_worst_result(print_output=True)

If calculations take exeeding amounts of time you may want to save already obtained results into a JSON file.
Luckily can be done very easily as shown below (please create a folder named JSON somewhere to save the files there):

your_model_name.save_results_to_json("ENTER YOUR FILEPATH")

To later read the saved data from the JSON files use:

# Load previous results if they exist
your_model_name.load_results_from_json("ENTER YOUR FILEPATH")

# Run the model with 100 integer cuts if results are not already loaded
if not your_model_name.results:
    your_model_name.run_with_integer_cuts(max_cuts=100)
    your_model_name.save_results_to_json("ENTER YOUR FILEPATH")

After running your model with various integer cuts you may want to plot the obtained results, 
this can be done as shown below:

your_model_name.plot_results(save_path="ENTER YOUR SAVE PATH")

(If you don't want to store the plot you don't have to specify a save_path)
