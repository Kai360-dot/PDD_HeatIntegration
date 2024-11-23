class Utility:
    utility_count = 0 # counts number of utilities
    def __init__(self, name, temperature_available, temperature_out, film_coefficient, cost):
        """
        Create instance of a utility
        [T] = K
        [h] = MW/ (m^2K)
        [cost] = $/MWh
        """
        self.name = name
        self.t_ava = temperature_available
        self.t_out = temperature_out
        self.h = film_coefficient
        self.cost = cost
        Utility.utility_count += 1 

    def print_properties(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")
    
    @classmethod
    def number_of_utilities(cls):
        print(f"{cls.utility_count} utilities were created.")
        return cls.utility_count

