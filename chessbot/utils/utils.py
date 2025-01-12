class RunningAverage:
    """ Utility class to keep track of training metrics """
    
    def __init__(self):
        self.counts = {}
        self.averages = {}

    def add(self, var_names):
        var_names = [var_names] if isinstance(var_names, str) else var_names
        for var_name in var_names:
            if var_name not in self.averages:
                self.averages[var_name] = 0.0
                self.counts[var_name] = 0

    def update(self, var_name, value=None):
        if isinstance(var_name, dict):
            for k, v in var_name.items():
                if k not in self.averages:
                    print(f"Variable {k} is not being tracked. Use add method to track.")
                    continue
                self.update(k, v)
        else:
            if var_name not in self.averages:
                print(f"Variable {var_name} is not being tracked. Use add method to track.")
                return
            self.counts[var_name] += 1
            self.averages[var_name] += (value - self.averages[var_name]) / self.counts[var_name]

    def get_average(self, var_names):
        if isinstance(var_names, str):
            return self.averages.get(var_names, None)

        return {var_name: self.averages.get(var_name, None) for var_name in var_names}

    def reset(self, var_names=None):
        if var_names is None:
            self.counts = {}
            self.averages = {}
        else:
            var_names = [var_names] if isinstance(var_names, str) else var_names
            for var_name in var_names:
                if var_name in self.averages:
                    self.counts[var_name] = 0
                    self.averages[var_name] = 0.0
                else:
                    print(f"Variable {var_name} is not being tracked.")
