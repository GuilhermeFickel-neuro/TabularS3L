import torch
from torch import nn
from typing import List

# --- Network Components ---

class Encoder(nn.Module):
    """Encoder network with a three-layer transformer."""
    def __init__(self, feature_size, num_heads=2):
        super(Encoder, self).__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_length, feature_size)
        # TransformerEncoderLayer with batch_first=True handles this directly
        return self.transformer_layers(x)

class Projector(nn.Module):
    """Projector network."""
    def __init__(self, feature_size):
        super(Projector, self).__init__()
        self.linear = nn.Linear(feature_size, feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class Decoder(nn.Module):
    """Decoder network."""
    def __init__(self, input_feature_size, output_feature_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_feature_size, output_feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# --- Matryoshka Components ---

class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: List[float] = None, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        # relative importance shape: [G]
        self.relative_importance = relative_importance

    def forward(self, output, target):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]

        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])
        
        # Set relative_importance to 1 if not specified
        if self.relative_importance is None:
            rel_importance = torch.ones_like(losses)
        else:
            rel_importance = torch.tensor(self.relative_importance, device=losses.device)

        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()


class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes # Number of classes for classification
        self.efficient = efficient
        if self.efficient:
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x):
        nesting_logits = []
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits.append(torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()))
                else:
                    nesting_logits.append(torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias)
            else:
                nesting_logits.append(getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]))

        return nesting_logits

# --- Main SwitchTab-MRL Model ---

class SwitchTabMRLModel(nn.Module):
    """
    A simplified implementation of the SwitchTab model with Matryoshka Representation Learning.
    """
    def __init__(self, nesting_list: List[int], num_classes: int, num_heads: int = 2, efficient: bool = False):
        super(SwitchTabMRLModel, self).__init__()
        feature_size = nesting_list[-1]
        
        self.encoder = Encoder(feature_size, num_heads)
        self.projector_s = Projector(feature_size)
        self.projector_m = Projector(feature_size)
        self.decoder = Decoder(2 * feature_size, feature_size)
        self.predictor = MRL_Linear_Layer(nesting_list, num_classes, efficient)

    def forward(self, x1, x2):
        # Encoder
        z1_encoded = self.encoder(x1)
        z2_encoded = self.encoder(x2)

        # Projectors
        s1_salient = self.projector_s(z1_encoded)
        m1_mutual = self.projector_m(z1_encoded)
        s2_salient = self.projector_s(z2_encoded)
        m2_mutual = self.projector_m(z2_encoded)

        # Decoder
        x1_reconstructed = self.decoder(torch.cat((m1_mutual, s1_salient), dim=2))
        x2_reconstructed = self.decoder(torch.cat((m2_mutual, s2_salient), dim=2))
        x1_switched = self.decoder(torch.cat((m2_mutual, s1_salient), dim=2))
        x2_switched = self.decoder(torch.cat((m1_mutual, s2_salient), dim=2))

        # MRL Predictor
        # .squeeze(1) assumes sequence length of 1 for tabular data
        z1_logits = self.predictor(z1_encoded.squeeze(1))
        z2_logits = self.predictor(z2_encoded.squeeze(1))
            
        return x1_reconstructed, x2_reconstructed, x1_switched, x2_switched, z1_logits, z2_logits

    def get_salient_embeddings(self, x):
        """Extracts salient embeddings for a given input."""
        z_encoded = self.encoder(x)
        s_salient = self.projector_s(z_encoded)
        return s_salient

    def get_embeddings(self, x):
        """Extracts embeddings for a given input."""
        return self.encoder(x).squeeze(1)
