from collections import OrderedDict
# waiting until we get the instructions of the tasks

class SomeEnv:
    def __init__(self):
        self.consecutive_grasp = 0

    def _start_episodeo(self):
        self.episode_starts = OrderedDict(
            
        )