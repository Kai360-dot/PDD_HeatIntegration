class Settings:
    """
    Class to store all the settings of the synheat model.
    Please modify the below data to vary the number of streams, 
    and their thermodynamic properties and requirements. 
    """

    def __init__(self, arg1=None, arg2=None):
        self.num_hot_streams = 3
        self.num_cold_streams = 3
        self.num_stages = 2
        self.annual_capital_charge_ratio = 0.205
        self.annual_operation_hours = 8000

        # hot streams
        self.fcp_hot = [['1', 4], ['2', 5], ['3', 3]] # stream id and FCp (flow heat capacity) [MW / K] 
        self.supply_temperature_hot_streams = [['1', 550], ['2', 400], ['3', 450]] # inlet temperatures [K]
        self.target_temperature_hot_streams = [['1', 400], ['2', 300], ['3', 350]] # outlet temperatures [K]

        # cold streams
        self.fcp_cold = [['1', 6], ['2', 5], ['3', 5]] # stream id and FCp (flow heat capacity) [MW / K] 
        self.supply_temperature_cold_streams = [['1', 460], ['2', 370], ['3', 300]] # inlet temperatures [K]
        self.target_temperature_cold_streams = [['1', 570], ['2', 460], ['3', 470]] # outlet temperatures [K]
        self.stream_film_coefficients_hot = [['1', 1], ['2', 1], ['3', 1]] # [MW/m2K]
        self.stream_film_coefficients_cold = [['1', 1], ['2', 1], ['3', 1]] # [MW/m2K]
        
        # general parameters
        self.heat_utility_cost = 0.01 # [$ / MWh]
        self.cooling_utility_cost = 0.0025 # [$ / MWh]
        self.fixed_cost_heat_exchangers = 10_000 # [$]
        self.area_cost_coefficient_exchangers = 800 # [$/m2]
        self.area_cost_coefficient_heaters = 800 # [$/m2]
        self.area_cost_coefficient_coolers = 800 # [$/m2]
        self.cost_exponent_exchangers = 0.8 
        self.film_coefficient_heating_utility = 5 # [MW/m2K]
        self.film_coefficient_cooling_utility = 1 # [MW/m2K]
        self.inlet_temperature_hot_utility = 600 # [K]
        self.outlet_temperature_hot_utility = 590 # [K]
        self.inlet_temperature_cold_utility = 290 # [K]
        self.outlet_temperature_cold_utility = 305 # [K]
        self.minimum_exchange_temperature = 10 # [K]
        self.big_m_parameter = 1e12 # very large number
        self.minimum_heat_exchanged_in_heat_exchanger = 5 # MW


        
