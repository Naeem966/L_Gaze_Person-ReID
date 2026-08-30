import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
import copy

class RandomIdentitySampler(Sampler):
    
    def __init__(self, data_source, num_instances=4, batch_size=64):
        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            # item can be tuple (img_path, pid, camid, viewid)
            pid = item[1]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of samples
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

import numpy as np
