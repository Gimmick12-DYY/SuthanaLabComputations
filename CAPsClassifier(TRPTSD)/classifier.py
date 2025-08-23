"""
Classifier Model for CAPs Result and Test Metrics

This module implements a classifier model using a stacking approach with XGBoost and RandomForest.


"""

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import random
import matplotlib

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This the Decision Tree Model used in the stacking ensemble
class SoftDecisionTree(nn.Module):
    def __init__(self, depth=3):
        super(SoftDecisionTree, self).__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth                  # Number of leaf nodes  
        self.n_internal_nodes = 2 ** depth - 1      # Number of internal decision nodes


        # Decision nodes parameters
        self.decision_nides = nn.ModuleList([nn.Linear(1, 1) for _ in range(self.n_internal_nodes)])

        # Leaf nodes parameters
        self.leaf_values = nn.Parameter(torch.randn(self.n_leaves, 1))

    def forward(self, x):
        batch_size = x.size(0)
        decision_probs = []

        for node in self.decision_nides:
            p = torch.sigmoid(node(x))
            decision_probs.append(p)
        
        decision_probs = torch.stack(decision_probs, dim=1)  # Shape: (batch_size, n_internal_nodes, 1)

        leaf_probs = []

        for leaf in range(self.n_leaves):
            path = []
            index = leaf

            for _ in range(self.depth):
                path.append(index % 2)
                index //= 2

            path = path[::-1]  # Reverse to get the path from root to leaf
            # Calculate the probability of reaching this leaf
            prob = torch.ones(batch_size, 1).to(device)
            node_index = 0
            for decision in path:
                p = decision_probs[:, node_index].unsqueeze(1)
                if decision == 0:
                    prob = prob * (1 - p)
                else:
                    prob = prob * p
                node_index += 1
            leaf_probs.append(prob)
        
        leaf_probs = torch.cat(leaf_probs, dim=1)  # Shape: (batch_size, n_leaves)
        output = torch.matmul(leaf_probs, self.leaf_values)  # Shape: (batch_size, 1)
        return output   