class Stream:
    stream_count = 0 # counts the number of created streams
    def __init__(self, name, fcp, temperature_in, temperature_out, stream_film_coefficient, stream_type) -> None:
        """Initializes stream instance;
        [fcp] = MW/K
        [T] = K
        [h] = MW/ (m^2K)"""
        self.name = name
        self.fcp = fcp
        self.t_in = temperature_in
        self.t_out = temperature_out
        self.h = stream_film_coefficient
        self.type = stream_type # hot or cold
        Stream.stream_count += 1 # One more stream initialized
    
    def print_properties(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")
        
    @classmethod
    def number_of_streams(cls):
        print(f"{cls.stream_count} Streams were created.")
        return cls.stream_count