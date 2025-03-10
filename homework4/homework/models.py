from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlannerLoss(nn.Module):
    def forward(self, 
        preds: torch.Tensor,
        labels: torch.Tensor,
        labels_mask: torch.Tensor
    ) -> dict[str, torch.tensor]:
        """
        Custom loss function for MLP Planner
        # copied from metrics.py
        Args:
            preds (torch.Tensor): (b, n, 2) float tensor with predicted waypoints
            labels (torch.Tensor): (b, n, 2) ground truth waypoints
            labels_mask (torch.Tensor): (b, n) bool mask for valid waypoints

        Returns:
            tensor, scalar loss
        """
        # # error = (preds - labels).abs()
        # # error_masked = error * labels_mask[..., None]
        # # error_sum = error_masked.sum(dim=(0, 1)).cpu().numpy()
        # # total = labels_mask.sum().item()

        # # return torch.nn.MSELoss(preds, labels)
        # loss = torch.sum((preds - labels) ** 2, dim=-1)  # Shape: (b, n)

        # # Take the mean over the batch and the sequence length (n)
        # return torch.mean(loss)
    
        # it is exact copy of metrics.PlannerMetric except its not in numpy
        error = (preds - labels).abs()
        total = labels_mask.sum().item()

        error_masked = error * labels_mask[..., None]
        error_sum = error_masked.sum(dim=(0, 1))
        longitudinal_error = error_sum[0] / total
        lateral_error = error_sum[1] / total
        l1_error = 2*longitudinal_error + lateral_error
        return {
            "l1_error": l1_error,
            "longitudinal_error": longitudinal_error,
            "lateral_error": lateral_error,
        }


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.input_size = self.n_track * 2 * 2 # 2 left and right, 2 for x,y coordinates
        self.output_size = self.n_waypoints * 2 # 2 for x,y coordinates
        layers = []
        layers.append(torch.nn.Flatten())
        # hidden features
        hidden_feature = 64
        layers.append(torch.nn.Linear(self.input_size, hidden_feature))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_feature, self.output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #mod_input = torch.cat((track_left, track_right), dim=-1)
        mod_input = torch.cat((track_left, track_right), dim=1)
        """
        # ignoring the batch size b
        track_left = [
            [1x, 1y],
            [3x, 3y],
            ...
        ]
        track_right = [
            [2x, 2y],
            [4x, 4y],
            ...
        ]

        mod_input = [
            [1x, 1y, 2x, 2y],
            [3x, 3y, 4x, 4y],
            ...
        ]
        """
        pred = self.model(mod_input)
        """
        pred = [ax, ay, bx, by, cx, cy]

        The view() function below changes it to
        [
            [ax, ay],
            [bx, by],
            [cx, cy]
        ]
        """
        return pred.view(-1, self.n_waypoints, 2)


# 1 self implemented Transformer Layer
class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.self_att = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim), torch.nn.ReLU(), torch.nn.Linear(4 * embed_dim, embed_dim)
        )
        self.in_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = x + self.self_att(x, x, x)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x
        
class TransformerPlanner(nn.Module):

    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        self.num_heads = 8
        self.num_layers = 6
        layers = []
        # convert input via Linear to embedding dimension
        layers.append(torch.nn.Linear(2, d_model))
        for _ in range(self.num_layers):
            layers.append(TransformerLayer(d_model, self.num_heads)) 

        # convert input via Linear to 2 dimension
        # at the end of bottom linear layer output will be [b X 20 X 2]
        layers.append(torch.nn.Linear(d_model,2))

        self.input_size = self.n_track * 2 * 2 # 2 left and right, 2 for x,y coordinates
        self.output_size = self.n_waypoints * 2 # 2 for x,y coordinates
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(self.input_size, self.output_size))
        
        self.network = torch.nn.Sequential(*layers)
        

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # (b, 2*n, 2)
        mod_input = torch.cat((track_left, track_right), dim=1)

        pred = self.network(mod_input)
        return pred.view(-1, self.n_waypoints, 2)


class CNNPlanner(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            stride=2

            layers = []
            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1,padding))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1,padding))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())
            self.model = torch.nn.Sequential(*layers)

            self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding)
            
        def forward(self, x):
            return self.skip(x) + self.model(x)
        
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # creata a wide first layer
        layers = []
        layers.append(torch.nn.Conv2d(3, 64, 11, 2, 5))
        layers.append(torch.nn.ReLU())
        
        channel_size = 64
        # down convolution that shrinks image but expands on channel
        for _ in range (2):
            new_channel_size = channel_size*2
            layers.append(self.Block(channel_size, new_channel_size))
            channel_size = new_channel_size

        # final layer to convert to desired number of classification classes against each pixel
        layers.append(torch.nn.Conv2d(channel_size, 6, 3, 1, 1))
        # layer that takes an average across all pixels and yields one final value against each classification class
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        logits = self.model(x)

        return logits.view(-1,self.n_waypoints, 2 )


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path

def save_model_epoch(model: torch.nn.Module, epoch:str) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}_{epoch}.th"
    torch.save(model.state_dict(), output_path)

    return output_path

def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
