import torch
import torch.nn as nn


# Aggregation of data in the hidden layer
class AggregateLayer(nn.Module):
    def __init__(self, input_dimension):
        super(AggregateLayer, self).__init__()
        self.ForgetGateLayer = nn.Sequential(
            nn.Linear(input_dimension, input_dimension), nn.Sigmoid()
        )

        self.InputGateLayer = nn.Sequential(
            nn.Linear(input_dimension, input_dimension), nn.Sigmoid()
        )

        self.UpdateGateLayer = nn.Sequential(
            nn.Linear(input_dimension * 2, input_dimension), nn.Sigmoid()
        )

    def forward(self, current_data, previous_data):
        forget_gate = self.ForgetGateLayer(previous_data)
        forget_data = previous_data * forget_gate

        input_gate = self.InputGateLayer(current_data)
        input_data = current_data * input_gate

        integer_data = torch.cat([input_data, forget_data], dim=-1)
        update_gate = self.UpdateGateLayer(integer_data)
        update_data = update_gate * input_data + (1 - update_gate) * forget_data
        return update_data


class HiddenLayer(nn.ModuleList):
    def __init__(self, input_dimension, drop_rate):
        super(HiddenLayer, self).__init__()

        self.LinearLayer = nn.Sequential(
            nn.Linear(input_dimension, input_dimension // 2),
            nn.ReLU(),
            nn.Linear(input_dimension // 2, input_dimension),
            nn.Dropout(drop_rate),
        )

    def forward(self, data):
        data = self.LinearLayer(data)
        return data


class LDAGM(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        feature_num,
        hiddenLayer_num,
        drop_rate=0.05,
        use_aggregate=True,
    ):
        super(LDAGM, self).__init__()
        self.use_aggregate = use_aggregate
        self.hiddenLayer_num = hiddenLayer_num

        self.EmbeddingLayer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension), nn.Dropout(drop_rate)
        )

        self.HiddenLayers = nn.ModuleList(
            [HiddenLayer(hidden_dimension, drop_rate) for _ in range(hiddenLayer_num)]
        )
        self.AggregateLayers = nn.ModuleList(
            [AggregateLayer(hidden_dimension) for _ in range(hiddenLayer_num)]
        )

        # predictive layer
        self.Predict = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dimension * feature_num, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        data = self.EmbeddingLayer(data)
        if self.use_aggregate:
            agg_data = data
            for i in range(self.hiddenLayer_num):
                data = self.HiddenLayers[i](data)
                agg_data = self.AggregateLayers[i](data, agg_data)
            predict = self.Predict(agg_data).squeeze(-1)
        else:
            for i in range(self.hiddenLayer_num):
                data = self.HiddenLayers[i](data)
            predict = self.Predict(data).squeeze(-1)
        return predict
