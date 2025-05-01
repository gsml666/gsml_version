#!/mnt/binf/eric/anaconda3/envs/Py38/bin/python
import torch
import torch.nn as nn
import inspect

class DANN_1D(nn.Module):
    def __init__(
        self,
        input_size,
        out1,
        out2,
        conv1,
        pool1,
        drop1,
        conv2,
        pool2,
        drop2,
        fc1,
        fc2,
        drop3,
    ):
        super(DANN_1D, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=None
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out1),
            nn.Dropout(drop1),
            nn.MaxPool1d(kernel_size=pool1, stride=2),
            nn.Conv1d(
                in_channels=out1,
                out_channels=out2,
                kernel_size=conv2,
                stride=2,
                bias=None,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out2),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2),
        )

        self.fc_input_size = self._get_fc_input_size(input_size)

        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid(),
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid(),
        )

        # Domain classifier
        self.r01b_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid(),
        )

    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.feature_extractor(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size

    @staticmethod
    def initialize_lin(layer, bias=0):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    def forward(self, x, y, alpha):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # task classifier output
        task_output = self.task_classifier(features)

        # domain classifier output
        # reverse_features = ReverseLayerF.apply(features, alpha)
        # domain_output = self.domain_classifier(reverse_features)

        if y is not None:
            feature_r01b = self.feature_extractor(y)
            feature_r01b = feature_r01b.view(feature_r01b.size(0), -1)
            r01b_output = self.r01b_classifier(feature_r01b)

            return (
                task_output.squeeze(1),
                None,
                r01b_output.squeeze(1),
            )
        else:
            return task_output.squeeze(1), None, None

class DANNwithTrainingTuning_1D(DANN_1D):
    def __init__(self, config, input_size, gamma_r01b):
        model_config, _ = self._match_params(
            config
        )  # find the parameters for the original DANN class
        super(DANNwithTrainingTuning_1D, self).__init__(
            input_size, **model_config
        )  # pass the parameters into the original DANN class
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.loss_lambda = config["lambda"]
        self.gamma_r01b = gamma_r01b

        self.criterion_task = nn.BCELoss()
        self.criterion_domain = nn.BCELoss()
        self.criterion_r01b = nn.L1Loss()
        self.criterion_r01b_ranking = nn.MarginRankingLoss()

        self.optimizer_extractor = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.optimizer_task = torch.optim.Adam(
            self.task_classifier.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.optimizer_domain = torch.optim.Adam(
            self.domain_classifier.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.optimizer_r01b = torch.optim.Adam(
            self.r01b_classifier.parameters(), lr=1e-4, weight_decay=1e-4
        )

    def _match_params(self, config):
        model_config = {}
        args = inspect.signature(DANN_1D.__init__).parameters
        model_keys = [name for name in args if name != "self"]
        # model_keys = list(self.__init__.__code__.co_varnames)[1:]

        for key, value in config.items():
            if key in model_keys:
                model_config[key] = value
        return model_config, model_keys

    