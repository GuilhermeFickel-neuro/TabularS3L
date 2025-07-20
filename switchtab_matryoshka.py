"""
SwitchTab with Matryoshka Representation Learning

This module provides:
1. PaperExactSwitchTab: SwitchTab exactly as described in the paper (3-layer transformer, 2 heads, sigmoid activations)
2. PaperExactSwitchTabMatryoshka: Paper-exact SwitchTab + Matryoshka learning
3. SwitchTabMatryoshka: Existing codebase SwitchTab + Matryoshka (for compatibility)
4. Matryoshka_CE_Loss: Multi-granularity loss function
5. Helper functions for easy configuration
"""

import torch
from torch import nn
from typing import List, Tuple, Union

from ts3l.models.common import TS3LModule
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig
from ts3l.utils.backbone_utils import TransformerBackboneConfig


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
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
        
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
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
                else:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
            else:
                nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

        return nesting_logits


class PaperExactSwitchTab(TS3LModule):
    """SwitchTab implementation exactly as described in the paper:
    - 3-layer transformer with 2 heads
    - Sigmoid activations in projectors and decoder
    """
    def __init__(self,
                 embedding_config: BaseEmbeddingConfig,
                 backbone_config: BaseBackboneConfig,
                 output_dim: int,
                 temperature: float = -1.0,
                 **kwargs) -> None:
        # Force paper-exact transformer configuration
        if backbone_config.name == "transformer":
            backbone_config.encoder_depth = 3
            backbone_config.n_head = 2
        
        super(PaperExactSwitchTab, self).__init__(embedding_config, backbone_config)
        self.output_dim = output_dim
        self.t = temperature
        self.__return_salient_feature = False
        
        # Paper-exact projectors with sigmoid activation
        self.projector_m = self._PaperProjector(self.backbone_module.output_dim)
        self.projector_s = self._PaperProjector(self.backbone_module.output_dim)
        
        # Paper-exact decoder with sigmoid activation
        self.decoder = self._PaperDecoder(self.backbone_module.output_dim, self.embedding_module.input_dim)
        self.head = nn.Linear(self.backbone_module.output_dim, output_dim)
        self.activation = nn.SiLU()

    def _apply_logit_normalization(self, x):
        """Apply logit normalization if temperature > 0"""
        if self.t > 0:
            norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
            return torch.div(x, norms) / self.t
        return x

    class _PaperProjector(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.Sigmoid()  # Paper specifies sigmoid
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation(self.linear(x))  # Apply sigmoid after linear

    class _PaperDecoder(nn.Module):
        def __init__(self, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(hidden_dim * 2, output_dim)
            self.activation = nn.Sigmoid()  # Paper specifies sigmoid
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation(self.linear(x))  # Apply sigmoid after linear

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module
        
    @property
    def return_salient_feature(self) -> bool:
        return self.__return_salient_feature
    
    @return_salient_feature.setter
    def return_salient_feature(self, flag: bool) -> None:
        self.__return_salient_feature = flag

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        size = len(x) // 2
        x = self.embedding_module(x)
        zs = self.encoder(x)
        
        ms = self.projector_m(zs)
        ss = self.projector_s(zs)
        
        m1, s1 = ms[:size], ss[:size]
        m2, s2 = ms[size:], ss[size:]
        
        x1_tilde_hat = torch.concat([torch.concat([m1, s1], dim=1), torch.concat([m2, s1], dim=1)])
        x2_hat_tilde = torch.concat([torch.concat([m1, s2], dim=1), torch.concat([m2, s2], dim=1)])

        x1_recover_switch = self.decoder(x1_tilde_hat)
        x2_switch_recover = self.decoder(x2_hat_tilde)
        
        x_hat = torch.concat([x1_recover_switch, x2_switch_recover])
        y_hat = self.head(self.activation(zs))
        y_hat = self._apply_logit_normalization(y_hat)

        return x_hat, y_hat

    def _second_phase_step(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding_module(x)
        emb = self.encoder(x)
        y_hat = self.head(self.activation(emb))
        y_hat = self._apply_logit_normalization(y_hat)
        if not self.return_salient_feature:
            return y_hat
        else:
            salient_features = self.projector_s(emb)
            return y_hat, salient_features


class PaperExactSwitchTabMatryoshka(PaperExactSwitchTab):
    """Paper-exact SwitchTab with Matryoshka Representation Learning"""
    def __init__(self,
                 embedding_config: BaseEmbeddingConfig,
                 backbone_config: BaseBackboneConfig,
                 output_dim: int,
                 nesting_list: List[int],
                 efficient: bool = True,
                 temperature: float = -1.0,
                 **kwargs) -> None:
        super(PaperExactSwitchTabMatryoshka, self).__init__(embedding_config, backbone_config, output_dim, temperature=temperature, **kwargs)
        
        # Set up nesting dimensions
        self.nesting_list = nesting_list
        
        # Replace head with Matryoshka head
        self.head = MRL_Linear_Layer(nesting_list=self.nesting_list, num_classes=output_dim, efficient=efficient)
        
        if self.nesting_list[-1] != self.backbone_module.output_dim:
            print('ERROR: Nesting list dimension does not match backbone output dimension')
            exit(1)

    def _apply_logit_normalization_nested(self, nested_logits):
        """Apply logit normalization to nested outputs if temperature > 0"""
        if self.t > 0:
            return tuple(self._apply_logit_normalization(logits) for logits in nested_logits)
        return nested_logits

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        size = len(x) // 2
        x = self.embedding_module(x)
        zs = self.encoder(x)
        
        # Project if needed
        if hasattr(self, '_projection_layer'):
            zs = self._projection_layer(zs)
        
        ms = self.projector_m(zs)
        ss = self.projector_s(zs)
        
        m1, s1 = ms[:size], ss[:size]
        m2, s2 = ms[size:], ss[size:]
        
        x1_tilde_hat = torch.concat([torch.concat([m1, s1], dim=1), torch.concat([m2, s1], dim=1)])
        x2_hat_tilde = torch.concat([torch.concat([m1, s2], dim=1), torch.concat([m2, s2], dim=1)])

        x1_recover_switch = self.decoder(x1_tilde_hat)
        x2_switch_recover = self.decoder(x2_hat_tilde)
        
        x_hat = torch.concat([x1_recover_switch, x2_switch_recover])
        y_hat_nested = self.head(self.activation(zs))
        y_hat_nested = self._apply_logit_normalization_nested(y_hat_nested)

        return x_hat, y_hat_nested

    def _second_phase_step(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        x = self.embedding_module(x)
        emb = self.encoder(x)
        
        # Project if needed
        if hasattr(self, '_projection_layer'):
            emb = self._projection_layer(emb)
        
        y_hat_nested = self.head(self.activation(emb))
        y_hat_nested = self._apply_logit_normalization_nested(y_hat_nested)
        
        if not self.return_salient_feature:
            return y_hat_nested
        else:
            salient_features = self.projector_s(emb)
            return y_hat_nested, salient_features



def create_matryoshka_loss(relative_importance: List[float] = None, **kwargs):
    """Factory function to create Matryoshka CE Loss"""
    return Matryoshka_CE_Loss(relative_importance=relative_importance, **kwargs)


def create_paper_exact_transformer_config(d_model: int, **kwargs):
    """Factory function to create paper-exact transformer config (3 layers, 2 heads)
    
    Follows FT-Transformer defaults except for explicit SwitchTab modifications:
    - n_head: 2 (SwitchTab override, vs FT-Transformer's 8)
    - encoder_depth: 3 (same as FT-Transformer)
    - All other params follow FT-Transformer defaults
    """
    return TransformerBackboneConfig(
        d_model=d_model,
        encoder_depth=3,                    # Explicit in SwitchTab (matches FT-Transformer)
        n_head=2,                           # Explicit in SwitchTab (vs FT-Transformer's 8)
        ffn_factor=1.333333333333333,       # FT-Transformer default (not 2.0!)
        hidden_dim=192,                     # FT-Transformer d_token default (not 256!)
        dropout_encoder=0.2,                # FT-Transformer attention_dropout default
        **kwargs
    )
