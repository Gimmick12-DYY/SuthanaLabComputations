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
    
# This is the Random Forest Model used in the stacking ensemble
class SoftRandomForest(nn.Module):
    def __init__(self, n_trees=10, tree_depth=3):
        super(SoftRandomForest, self).__init__()
        self.n_trees = n_trees
        self.trees = nn.ModuleList([SoftDecisionTree(depth=tree_depth) for _ in range(n_trees)])
    
    def forward(self, x):
        preds = [tree(x) for tree in self.trees]
        preds = torch.stack(preds, dim=0)  # Shape: (n_trees, batch_size, 1)
        return torch.mean(preds, dim=0)  # Shape: (batch_size, 1)

# This is the XGBoost Model used in the stacking ensemble
class SoftXGBoost(nn.Module):
    def __init__(self, n_estimators=10, tree_depth=3, learning_rate=0.1):
        super(SoftXGBoost, self).__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.trees = nn.ModuleList([SoftDecisionTree(depth=tree_depth) for _ in range(n_estimators)])
    
    def forward(self, x):
        preds = torch.zeros(x.size(0), 1).to(device)
        for tree in self.trees:
            preds += self.learning_rate * tree(x)
        return preds

# This is the Stacking Fusion Model that combines Random Forest and XGBoost predictions
# The fusion model is a simple feedforward neural network

class StackingFusionModel(nn.Module):
    def __init__(self, rf_model, xgb_model):
        super(StackingFusionModel, self).__init__()
        self.rf_model = rf_model
        self.xgb_model = xgb_model

        # Meta model
        self.meta_model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        rf_preds = self.rf_model(x)
        xgboost_preds = self.xgb_model(x)
        
        fusion_input = torch.cat((rf_preds, xgboost_preds), dim=1)
        fusion_output = self.meta_model(fusion_input)
        return fusion_output
    

# Training and Evaluation Functions

def train_model(model, optimizer, criterion, x, y, num_epochs=300):
    model.train()
    loss_list = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return loss_list

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x)
        loss = nn.MSELoss()(preds, y)
    
    return preds.cpu().numpy(), loss.item()

## Example Usage, this is where the data is loaded and the models are trained and evaluated