class Sequence:
    # def calculate_kspace(self, trajectory_delay, gradient_offset):

    def calculate_kspacePP(self, trajectory_delay, gradient_offset):
        self.calculate_kspace(trajectory_delay, gradient_offset)

    def calculate_pns(self, hardware, time_range, do_plots):
        raise NotImplementedError("Not callable in mr0 mode")

    def check_timing(self):
        return (True, [])

    def evaluate_labels(self, init, evolution):
        raise NotImplementedError("Not callable in mr0 mode")

    def plot(self, label, show_blocks, save, time_range, time_disp, grad_disp, plot_now):
        pass

    def write(self, name, create_signature, remove_duplicates):
        if create_signature:
            return ""
        else:
            return None
