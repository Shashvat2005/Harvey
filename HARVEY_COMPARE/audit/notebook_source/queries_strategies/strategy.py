
class Strategy:
    def __init__(self, model, data_unlabeled, NO_CLASSES, cfgs, device):
        self.model = model
        self.data_unlabeled = data_unlabeled
        self.subset = []
        self.labeled_set = []
        self.cycle = 0
        self.NO_CLASSES = NO_CLASSES
        self.cfgs = cfgs
        self.device = device
        self.init_set_seed = 0

        self.BATCH = self.cfgs.BATCH
        self.BUDGET = self.cfgs.BUDGET
        self.INIT_BUDGET = self.cfgs.INIT_BUDGET

    def query(self):
        pass

    def seed_random_init_set(self, init_set_seed):
        self.init_set_seed = init_set_seed
    
    def feed_current_state(self, cycle, subset, labeled_set):
        self.subset = subset
        self.labeled_set = labeled_set
        self.cycle = cycle

