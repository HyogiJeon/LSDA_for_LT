import numpy as np
import torch
import torch.nn as nn

class feature_augmenation(nn.Module):
    def __init__(self, args, class_num, last_dim, class_num_list=None):
        super(feature_augmenation, self).__init__()
        self.args = args

        self.last_dim = last_dim
        self.class_num = class_num

        self.feature_mean = nn.Parameter(torch.zeros(self.class_num, last_dim), requires_grad=False)
        self.feature_var = nn.Parameter(torch.zeros(self.class_num, last_dim), requires_grad=False)

        if class_num_list != None:
            self.class_num_list = torch.tensor(class_num_list, dtype=torch.float32).cuda()
            self.prob_list = (1 - (torch.log10(self.class_num_list) / torch.log10(self.class_num_list).max())).cuda()

        self.feature_used = nn.Parameter(torch.zeros(self.class_num), requires_grad=False).cuda()

        self.ratio = 0.1
        self.epoch = 0


    def get_feature_mean_std(self, feature, labels):
        for label in torch.unique(labels):
            feat = feature[label == labels]
            feat_mean = feat.mean(dim=0)

            if self.feature_used[label] != 0:
                with torch.no_grad():
                    self.feature_mean[label] = (self.ratio * feat_mean) + ((1 - self.ratio) * self.feature_mean[label])
            else:
                self.feature_mean[label] = feat_mean

            feat_var = torch.sum((feat - self.feature_mean[label])**2, dim=0)

            n = feat.numel() / feat.size(1)
            if n > 1:
                feat_var = feat_var / (n - 1)
            else:
                feat_var = feat_var

            if self.feature_used[label] != 0:
                with torch.no_grad():
                    self.feature_var[label] = (self.ratio * feat_var) + ((1 - self.ratio) * self.feature_var[label])
            else:
                self.feature_var[label] = feat_var
                self.feature_used[label] += 1


    def generate_feature(self):
        embedding_list, label_list = [], []

        adaptive_std = 1 + (self.prob_list * self.args.k)
        epoch_ratio = (self.epoch - self.args.aug_start_epoch) / (self.args.max_epoch - self.args.aug_start_epoch)

        for i in range(self.args.n):
            rand = torch.rand(self.class_num).cuda()
            uniq_c = torch.where(rand < self.prob_list)[0]
            
            if len(uniq_c) > 0:
                for c in uniq_c:
                    c = int(c)
                    if self.feature_used[c] == 0:
                        continue
                    std = torch.sqrt(self.feature_var[c]) * (adaptive_std[c] * epoch_ratio).clamp(min=1)
                    new_sample = self.feature_mean[c] + std * torch.normal(0, 1, size=std.shape).cuda()

                    embedding_list.append(new_sample.unsqueeze(0))
                    label_list.append(c)

        if len(embedding_list) != 0:
            embedding_list = torch.cat(embedding_list, 0)
            label_list = torch.tensor(label_list, device=embedding_list.device)
            return embedding_list, label_list
        else:
            return [], []