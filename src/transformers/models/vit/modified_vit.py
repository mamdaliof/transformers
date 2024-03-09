# %%
from torchvision.models import resnet50, ResNet50_Weights
import math
import torch
import torch.nn as nn
from transformers import ViTConfig
from collections import OrderedDict 
from typing import Optional, Tuple, Union
from src.transformers.activations import ACT2FN
from src.transformers.modeling_outputs import BaseModelOutput
from src.transformers.models.vit.modeling_vit import ViTEmbeddings 

# %% [markdown]
# # Model

# %% [markdown]
# ## CNN BackBones

# %% [markdown]
# ### Forward Hook

# %%
class ModelHook(nn.Module):
    """
    A PyTorch module to retrieve the output of specified layers in a model using forward hooks.

    Args:
        model (nn.Module): The model from which the output is to be retrieved.
        output_layers (list): A list of layer names for which the output needs to be captured.

    Attributes:
        output_layers (list): A list of layer names for which the output needs to be captured.
        selected_out (OrderedDict): A dictionary to store the output of selected layers.
        model (nn.Module): The model from which the output is retrieved.
        fhooks (list): A list to hold the forward hooks registered for selected layers.

    Methods:
        forward_hook(layer_name): Method to create a forward hook for a specific layer.
        forward(x): Forward method of the module.

    Returns:
        out (torch.Tensor): The output tensor from the model's forward pass.
        selected_out (OrderedDict): A dictionary containing the output tensors of selected layers.

    Example:
        # Instantiate a ResNet model
        resnet_model = torchvision.models.resnet18(pretrained=True)

        # Define layers for which output needs to be captured
        output_layers = ['conv1', 'layer1', 'layer2']

        # Instantiate ModelHook module
        model_hook = ModelHook(resnet_model, output_layers)

        # Forward pass
        inputs = torch.randn(1, 3, 224, 224)
        out, selected_out = model_hook(inputs)

        # Output of selected layers can be accessed from 'selected_out' dictionary
        print(selected_out)
    """
    def __init__(self,model, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        # print(self.output_layers)
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.model = model
        self.fhooks = []

        for l in list(self.model._modules.keys()):
            if l in self.output_layers:
                self.fhooks.append(getattr(self.model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out

# %% [markdown]
# ### Modify CNN Model

# %%
class CNNBackBone(nn.Module):
    """
    A PyTorch module implementing a CNN backbone for feature extraction.

    Args:
        hidden_size (int): The length of embedded vector that will be mounted in ViT.
        hidden_dropout_prob (float): The dropout probability for the hidden layer.
        attention_probs_dropout_prob (float): The dropout probability for attention weights.

    Attributes:
        resnet50_module (nn.Module): ResNet50 module with pretrained weights as the CNN backbone.
        avg_pool (nn.Module): Average pooling layer.
        middle_linear (nn.Module): Linear layer to transform middle layer's output.
        end_linear (nn.Module): Linear layer to transform end layer's output.
        CNN_block (ModelHook): ModelHook instance to extract outputs from specific layers.

    Methods:
        forward(x): Forward method of the module.

    Returns:
        CNN_end_layer_out (torch.Tensor): The output tensor from the end layer of the CNN backbone.
        CNN_middle_layer_out (torch.Tensor): The output tensor from the middle layer of the CNN backbone.
    """
    def __init__(self, *args, hidden_size=768, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0):
        super().__init__(*args)
        # Load and implement models
        self.resnet50_module = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.middle_linear = nn.Linear(512, hidden_size)
        self.end_linear = nn.Linear(2048, hidden_size)
        
        # Remove extra layers from CNN block (ResNetx) and add hook to it
        layers_dict = {name: module for name,
        module in zip(list(self.resnet50_module._modules.keys()),
                             list(self.resnet50_module.children())[:-2])} #all layers except last two
        self.resnet50_module = torch.nn.Sequential(OrderedDict(layers_dict))
        self.CNN_block = ModelHook(self.resnet50_module, ["layer2", "layer4"])
        
    def forward(self, x):
        """
        Forward pass of the CNN backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            CNN_end_layer_out (torch.Tensor): Output tensor from the end layer of the CNN backbone.
            CNN_middle_layer_out (torch.Tensor): Output tensor from the middle layer of the CNN backbone.
        """
        # Send original input through the ResNetx model to extract middle and end layer output
        _, CNN_outputs = self.CNN_block(x)
        
        # Generate matrices of size (-1, 512, 16, 16) out of layer2 of ResNetx
        CNN_middle_layer_out = self.avg_pool(CNN_outputs["layer2"])
        # Generate matrices of size (-1, 2048, 16, 16) out of layer4 of ResNetx
        CNN_end_layer_out = CNN_outputs["layer4"]
        
        # Merge dimensions of height and width of matrices and swap dimensions to generate 256 vectors with lengths 512 and 2048
        CNN_middle_layer_out = CNN_middle_layer_out.permute(0, 2, 3, 1).contiguous().view(CNN_middle_layer_out.shape[0], -1, 512)
        CNN_end_layer_out = CNN_end_layer_out.permute(0, 2, 3, 1).contiguous().view(CNN_middle_layer_out.shape[0], -1, 2048)
        
        # Send vectors through an MLP layer to generate vectors with length of 768
        CNN_middle_layer_out = self.middle_linear(CNN_middle_layer_out)
        CNN_end_layer_out = self.end_linear(CNN_end_layer_out)   

        return CNN_end_layer_out, CNN_middle_layer_out


# %% [markdown]
# ## Transformer

# %% [markdown]
# ### ViTEmbedding

# %%
class ViTEmbeddings_modified(nn.Module):
    """
    A modified version of ViT embedding layer for the CNN backbone. it's add CLS token and positional encoding to the input vector.

    Args:
        config (ViTConfig): Configuration settings for the ViT model.
        num_patches (int): Number of patches in the input image.

    Attributes:
        cls_token (nn.Parameter): Learnable parameter representing the class token.
        position_embeddings (nn.Parameter): Learnable parameter representing position embeddings.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(pixel_values): Forward method of the module.

    Returns:
        embeddings (torch.Tensor): Output embeddings after processing the input pixel values.
    """
    def __init__(self, config, num_patches=16*16):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    
    
    def forward(self, pixel_values):
        """
        Forward pass of the ViT embedding layer.

        Args:
            pixel_values (torch.Tensor): Input pixel values of the image.

        Returns:
            embeddings (torch.Tensor): Output embeddings after processing the input pixel values.
        """
        batch_size = pixel_values.shape[0]
        
        embeddings = pixel_values
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# %% [markdown]
# ### ViTSelfAttention

# %%
""" this class does not modified"""
class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# %%
class ViTSelfAttention_modified(nn.Module):
    """
    Modified self-attention mechanism for the ViT model.

    Args:
        config (ViTConfig): Configuration settings for the ViT model.

    Attributes:
        num_attention_heads (int): Number of attention heads.
        attention_head_size (int): Size of each attention head.
        all_head_size (int): Total size of all attention heads.
        query (nn.Linear): Linear layer for query projection.
        key (nn.Linear): Linear layer for key projection.
        value (nn.Linear): Linear layer for value projection.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        transpose_for_scores(x): Reshape input tensor for attention scores calculation.
        forward(K, Q, V, head_mask=None, output_attentions=False): Forward method of the module.

    Returns:
        Tuple[torch.Tensor]: Tuple containing the context layer and attention probabilities.
    """
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        
        # NOTE: i think we should modify the following if
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input tensor for attention scores calculation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, K, Q, V, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass of the ViT self-attention mechanism.

        Args:
            K (torch.Tensor): Key tensor.
            Q (torch.Tensor): Query tensor.
            V (torch.Tensor): Value tensor.
            head_mask (Optional[torch.Tensor]): Optional tensor for masking heads.
            output_attentions (bool): Whether to output attention probabilities.

        Returns:
            Tuple[torch.Tensor]: Tuple containing the context layer and attention probabilities.
        """
        mixed_query_layer = self.query(Q)

        key_layer = self.transpose_for_scores(self.key(K))
        value_layer = self.transpose_for_scores(self.value(V))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply softmax to obtain attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Calculate context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# %%
class ViTSelfOutput(nn.Module):
    '''this class does not get modified'''
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# %% [markdown]
# ### ViTAttention

# %%
class ViTAttention_modified(nn.Module):
    """
    Modified attention layer for the ViT model.

    Args:
        config (ViTConfig): Configuration settings for the ViT model.

    Attributes:
        attention (ViTSelfAttention_modified): Self-attention mechanism.
        output (ViTSelfOutput): Output layer.

    Methods:
        forward(K, Q, V, head_mask=None, output_attentions=False): Forward method of the module.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]: Tuple containing output tensor and optionally attention probabilities.
    """
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention_modified(config)
        self.output = ViTSelfOutput(config)

    def forward(
        self,
        K: torch.Tensor,
        Q: torch.Tensor,
        V: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Forward pass of the ViT attention mechanism.

        Args:
            K (torch.Tensor): Key tensor.
            Q (torch.Tensor): Query tensor.
            V (torch.Tensor): Value tensor.
            head_mask (Optional[torch.Tensor]): Optional tensor for masking heads.
            output_attentions (bool): Whether to output attention probabilities.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]: Tuple containing output tensor and optionally attention probabilities.
        """
        self_outputs = self.attention(K, Q, V, head_mask, output_attentions)
        
        # Apply output layer
        attention_output = self.output(self_outputs[0], None)

        # Concatenate attention probabilities if needed
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        
        return outputs


# %% [markdown]
# ### ViTLayer

# %%
'this class does not get modified'
class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

# %%
'this class does not get modified'
class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states #+ input_tensor

        return hidden_states

# %%
class ViTLayer_modified(nn.Module):
    """
    Modified layer of the Vision Transformer model, corresponding to the Block class in the timm implementation.

    Args:
        config (ViTConfig): Configuration settings for the ViT model.

    Attributes:
        chunk_size_feed_forward (int): Chunk size for feed-forward operations.
        seq_len_dim (int): Sequence length dimension.
        attention (ViTAttention_modified): Modified self-attention mechanism.
        intermediate_K (ViTIntermediate): Intermediate layer for keys.
        intermediate_Q (ViTIntermediate): Intermediate layer for queries.
        intermediate_V (ViTIntermediate): Intermediate layer for values.
        output_K (ViTOutput): Output layer for keys.
        output_Q (ViTOutput): Output layer for queries.
        output_V (ViTOutput): Output layer for values.
        layernorm_before_k (nn.LayerNorm): Layer normalization before keys.
        layernorm_before_Q (nn.LayerNorm): Layer normalization before queries.
        layernorm_before_V (nn.LayerNorm): Layer normalization before values.
        layernorm_after_K (nn.LayerNorm): Layer normalization after keys.
        layernorm_after_Q (nn.LayerNorm): Layer normalization after queries.
        layernorm_after_V (nn.LayerNorm): Layer normalization after values.

    Methods:
        forward(K, Q, V, head_mask=None, output_attentions=False): Forward method of the module.

    Returns:
        Tuple[torch.Tensor]: Tuple containing modified output tensors for keys, queries, and values.
    """
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention_modified(config)
        
        self.intermediate_K = ViTIntermediate(config)
        self.intermediate_Q = ViTIntermediate(config)
        self.intermediate_V = ViTIntermediate(config)
        
        self.output_K = ViTOutput(config)
        self.output_Q = ViTOutput(config)
        self.output_V = ViTOutput(config)
        
        self.layernorm_before_k = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_before_Q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_before_V = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.layernorm_after_K = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after_Q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after_V = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        K: torch.Tensor,
        Q: torch.Tensor,
        V: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass of the ViT layer.

        Args:
            K (torch.Tensor): Key tensor.
            Q (torch.Tensor): Query tensor.
            V (torch.Tensor): Value tensor.
            head_mask (Optional[torch.Tensor]): Optional tensor for masking heads.
            output_attentions (bool): Whether to output attention probabilities.

        Returns:
            Tuple[torch.Tensor]: Tuple containing modified output tensors for keys, queries, and values.
        """
        # layer norm
        tk = self.layernorm_before_k(K)
        tq = self.layernorm_before_Q(Q)
        tv = self.layernorm_before_V(V)
        
        self_attention_outputs = self.attention(tk, tq, tv, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # residual connection
        K_prime = attention_output + K
        Q_prime = attention_output + Q
        V_prime = attention_output + V
        # layer norm
        layer_output_K = self.layernorm_after_K(K_prime)
        layer_output_Q = self.layernorm_after_Q(Q_prime)
        layer_output_V = self.layernorm_after_V(V_prime)
        # Feed Forward layer
        layer_output_K = self.intermediate_K(layer_output_K)
        layer_output_Q = self.intermediate_Q(layer_output_Q)
        layer_output_V = self.intermediate_V(layer_output_V)
        # second residual connection is done here
        layer_output_K = self.output_K(layer_output_K, None)
        layer_output_Q = self.output_Q(layer_output_Q, None)
        layer_output_V = self.output_V(layer_output_V, None)
        
        return (layer_output_K, layer_output_Q, layer_output_V)


# %%
#NOTE in this model we dont use this block modify docstring for future use.

# class ViTLayer_modified_last_layer(nn.Module):
#     """
#     Modified last layer of the Vision Transformer model, corresponding to the Block class in the timm implementation.

#     Args:
#         config (ViTConfig): Configuration settings for the ViT model.

#     Attributes:
#         chunk_size_feed_forward (int): Chunk size for feed-forward operations.
#         seq_len_dim (int): Sequence length dimension.
#         attention (ViTAttention_modified): Modified self-attention mechanism.
#         intermediate (ViTIntermediate): Intermediate layer.
#         output (ViTOutput): Output layer.
#         layernorm_before_k (nn.LayerNorm): Layer normalization before keys.
#         layernorm_before_Q (nn.LayerNorm): Layer normalization before queries.
#         layernorm_before_V (nn.LayerNorm): Layer normalization before values.
#         layernorm_after (nn.LayerNorm): Layer normalization after self-attention.

#     Methods:
#         forward(K, Q, V, head_mask=None, output_attentions=False): Forward method of the module.

#     Returns:
#         torch.Tensor: Output tensor after processing through the layer.
#     """
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.attention = ViTAttention_modified(config)
#         self.intermediate = ViTIntermediate(config)
#         self.output = ViTOutput(config)
#         self.layernorm_before_k = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.layernorm_before_Q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.layernorm_before_V = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


#     def forward(
#         self,
#         K: torch.Tensor,
#         Q: torch.Tensor,
#         V: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> torch.Tensor:
#         """
#         Forward pass of the ViT last layer.

#         Args:
#             K (torch.Tensor): Key tensor.
#             Q (torch.Tensor): Query tensor.
#             V (torch.Tensor): Value tensor.
#             head_mask (Optional[torch.Tensor]): Optional tensor for masking heads.
#             output_attentions (bool): Whether to output attention probabilities.

#         Returns:
#             torch.Tensor: Output tensor after processing through the layer.
#         """
#         tk = self.layernorm_before_k(K)
#         tq = self.layernorm_before_Q(Q)
#         tv = self.layernorm_before_V(V)
#         self_attention_outputs = self.attention(tk, tq, tv, head_mask, output_attentions=output_attentions)
#         attention_output = self_attention_outputs[0]
#         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

#         # first residual connection
#         hidden_states = attention_output + K + Q + V
       
#         # in ViT, layernorm is also applied after self-attention
#         layer_output = self.layernorm_after(hidden_states)
#         layer_output = self.intermediate(layer_output)

#         # second residual connection is done here
#         layer_output = self.output(layer_output, hidden_states)

#         return layer_output
        


# %% [markdown]
# ### ViTEncoder  

# %%
class ViTEncoder_modified(nn.Module):
    """
    Modified encoder layer of the Vision Transformer model.

    Args:
        config (ViTConfig): Configuration settings for the ViT model.

    Attributes:
        config (ViTConfig): Configuration settings for the ViT model.
        layer (nn.ModuleList): List of modified ViT layers.
        gradient_checkpointing (bool): Whether to use gradient checkpointing during training.

    Methods:
        forward(K, Q, V, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
            Forward method of the module.

    Returns:
        Union[tuple, BaseModelOutput]: Tuple containing output tensors if return_dict=True, otherwise a tuple
        of tensors.
    """
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

        
        self.layer = nn.ModuleList([ViTLayer_modified(config) for _ in range(config.num_hidden_layers)])
        # self.last_layer = ViTLayer_modified_last_layer(config)
        self.gradient_checkpointing = False

    def forward(
        self,
        K: torch.Tensor,
        Q: torch.Tensor,
        V: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        """
        Forward pass of the ViT encoder.

        Args:
            K (torch.Tensor): Key tensor.
            Q (torch.Tensor): Query tensor.
            V (torch.Tensor): Value tensor.
            head_mask (Optional[torch.Tensor]): Optional tensor for masking heads.
            output_attentions (bool): Whether to output attention probabilities.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return output as a dictionary.

        Returns:
            Union[tuple, BaseModelOutput]: Tuple containing output tensors if return_dict=True, otherwise a tuple
            of tensors.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        # original_K = deepcopy(K)
        # original_Q = deepcopy(Q)
        # original_V = deepcopy(V)
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    K,
                    Q,
                    V,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(K, Q, V, layer_head_mask, output_attentions)

            K = layer_outputs[0]
            Q = layer_outputs[1]
            V = layer_outputs[2]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # output = self.last_layer(K, Q, V, layer_head_mask, output_attentions)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return (K,Q,V)


# %% [markdown]
# ## Integration

# %%
class ViTIntegrated(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.CNN_backbone = CNNBackBone(hidden_size = config.hidden_size)
        self.ViT = ViTEncoder_modified(config)
        # NOTE: the following code is designed for image size of 512*512
        self.ViT_embedder_K = ViTEmbeddings_modified(config)
        self.ViT_embedder_Q = ViTEmbeddings_modified(config)
        self.ViT_embedder_V = ViTEmbeddings(config)
        
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        
    def forward(self, image):
        if image.shape[-3:] != torch.Size([3,512,512]):
            raise ValueError(f"Input image dimension is not (3,512,512).")
        else:
            K, Q = self.CNN_backbone(image)
            K = self.ViT_embedder_K(K)
            Q = self.ViT_embedder_Q(Q)
            V = self.ViT_embedder_V(image)
            if (K.shape != V.shape) or (K.shape != Q.shape):
                raise ValueError(f"Key, Quary, or Value dimension is not the same")

            ViT_outputs = self.ViT(
                K,
                Q,
                V,
            )

            sequence_output = ViT_outputs[0] + ViT_outputs[1] + ViT_outputs[2]

            logits = self.classifier(sequence_output[:, 0, :])
            return sequence_output, logits