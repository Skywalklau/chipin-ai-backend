from flask import Flask, request, Blueprint, jsonify
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# pd.set_option("display.precision", 1)

recc = Blueprint('recc', __name__)

# class UserNN(nn.Module):
#     def __init__(self, num_user_features, num_outputs):
#         super(UserNN, self).__init__()
#         self.fc1 = nn.Linear(num_user_features, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_outputs)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# class ItemNN(nn.Module):
#     def __init__(self, num_item_features, num_outputs):
#         super(ItemNN, self).__init__()
#         self.fc1 = nn.Linear(num_item_features, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_outputs)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)
    
# class RecommendationModel(nn.Module):
#     def __init__(self, user_nn, item_nn):
#         super(RecommendationModel, self).__init__()
#         self.user_nn = user_nn
#         self.item_nn = item_nn

#     def forward(self, user_input, item_input):
#         vu = self.user_nn(user_input)
#         vu = nn.functional.normalize(vu, dim=1)
#         vm = self.item_nn(item_input)
#         vm = nn.functional.normalize(vm, dim=1)
#         output = torch.bmm(vu.unsqueeze(1), vm.unsqueeze(2)).squeeze()
#         return output
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_user_features = 31
# num_item_features = 35
# num_outputs = 64

# user_nn = UserNN(num_user_features, num_outputs).to(device)
# item_nn = ItemNN(num_item_features, num_outputs).to(device)
# model = RecommendationModel(user_nn, item_nn).to(device)

# model_path = "/server/website/model_weights.pth"
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()


@recc.route('/get-recc', methods=['GET'])
def get_recc():
    user_id = request.args.get('userId')
    # Logic for recommendation based on user history
    # Placeholder response for now
    return jsonify({'recommendations': 'Personalized restaurant recommendations based on history'}), 200