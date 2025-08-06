import torch
from torch import nn

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

class Predictor(nn.Module):
    """Prediction network for fine-tuning."""
    def __init__(self, feature_size, num_classes):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- Main SwitchTab Model ---

class SwitchTabModel(nn.Module):
    """
    A simplified implementation of the SwitchTab model.
    """
    def __init__(self, feature_size, num_classes, num_heads=2):
        super(SwitchTabModel, self).__init__()
        self.encoder = Encoder(feature_size, num_heads)
        self.projector_s = Projector(feature_size)
        self.projector_m = Projector(feature_size)
        self.decoder = Decoder(2 * feature_size, feature_size)
        self.predictor = Predictor(feature_size, num_classes)

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

        # Predictor for pre-training (simulated semi-supervised)
        z1_logits = self.predictor(z1_encoded.squeeze(1)) # Remove seq_dim for predictor
        z2_logits = self.predictor(z2_encoded.squeeze(1))
            
        return x1_reconstructed, x2_reconstructed, x1_switched, x2_switched, z1_logits, z2_logits

    def get_salient_embeddings(self, x):
        """Extracts salient embeddings for a given input."""
        z_encoded = self.encoder(x)
        s_salient = self.projector_s(z_encoded)
        return s_salient
