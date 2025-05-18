import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt

def patchify(images: torch.Tensor, n_patches: int) -> torch.Tensor:
    """Create equally sized non-overlapping patches of given square images."""
    n, h, w = images.shape
    c=1

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length: int, d: int) -> torch.Tensor:
    """Generates positional embeddings for a given sequence length and embedding dimension."""
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        """Initializes a transformer encoder block specified in the ViT paper.

        Args:
            hidden_d: The hidden dimensionality of token embeddings
            n_heads: The number of attention heads configured within the MHA module
            mlp_ratio: The ratio of the hidden MLP hidden layers to hidden layers within the MHA module
        """
        super().__init__()
        
        # Note: hidden_d is the same value as num_hidden from the ViT class 
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        # Layer defintions
        self.norm1 = nn.LayerNorm(hidden_d)
        self.multi_head_attention = MultiHeadAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d, hidden_d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer Encoder block with residual connections.
        
        Args:
            x: Input tensor of shape (batch_size, num_tokens, hidden_d)
            
        Returns:
            torch.Tensor: Output tensor of the same shape after applying multi-head attention, 
            normalization, and MLP.
        """
        # TODO: Define the foward pass of the Transformer Encoder block as illustrated in 
        #       Figure 4 of the spec.
        # NOTE: Don't forget about the residual connections!
    
        norm1 = self.norm1(x)
        mha = self.multi_head_attention(norm1)
        residual1 = x + mha
        norm2 = self.norm2(residual1)
        mlp = self.mlp(norm2)
        residual2 = residual1 + mlp
        return residual2



class MultiHeadAttention(nn.Module):
    def __init__(self, num_features: int, num_heads: int) -> None:
        """Multi-Head Attention mechanism to compute attention over patches using multiple heads.

        Args:
            num_features: Total number of features in the input sequence (patch) embeddings.
            num_heads: Number of attention heads to use in the multi-head attention.
        """
        super().__init__()

        # Note: num_features is the same value as num_hidden from the ViT class 
        # num_hidden: Number of hidden dimensions in the patch embeddings.

        self.num_features = num_features
        self.num_heads = num_heads
        
        # query_size is the dimension of each atention head's representation. By splitting input 
        # features evenly between heads, we maintain the efficiency of single-head attention 
        # while also allowing the model to attend to multiple representational subspaces at once.
        query_size = int(num_features / num_heads)

        # Note: nn.ModuleLists(list) taskes a python list of layers as its parameters. The object at 
        # the i'th index of the list passed to nn.ModuleList should correspond to the i'th attention 
        # head's K,Q, or V respective learned linear mapping

        # Some MHA implementations split q, k, and v matrices between attention heads after multiplying
        # inputs by Wq, Wk, and Wv. However this implementation has separate Wq, Wk, and Wv matrices mapping 
        # for each attention head (going from the sequence representation to the reduced dimensions of each 
        # head). Keep in mind that these two implementations are computationally equivalent.

        # Here the nn.Linear class is learning a fully connected layer of size query_size which is a linear
        # combination of all elements in num_features

        q_modList_input = [nn.Linear(num_features, query_size) for _ in range(num_heads)]
        self.Q_mappers = nn.ModuleList(q_modList_input)

        k_modList_input = [nn.Linear(num_features, query_size) for _ in range(num_heads)]
        self.K_mappers = nn.ModuleList(k_modList_input)

        v_modList_input = [nn.Linear(num_features, query_size) for _ in range(num_heads)]
        self.V_mappers = nn.ModuleList(v_modList_input)

        self.c_proj = nn.Linear(num_features, num_features)

        self.query_size = query_size
        self.scale_factor = sqrt(query_size)
        self.softmax = nn.Softmax(dim=-1)

        # for param in self.named_parameters():
        #     if param[1].requires_grad:
        #         print(param[0], param[1].shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Multi-Head Attention

        Args:
            x: Input tensor of shape (N, num_tokens, num_features).
               Each sequence represents a sequence of patch embeddings.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention, 
            the same shape as inputted.
        """
        result = []

        # Note: we turned each image into a sequence of 16 dimensional "tokens" for our model

        # Loop through the batch of patch embedding sequences
        for sequence in x:
            # Each element in seq_result should be a single attention head's attention values
            seq_result = []
            for head in range(self.num_heads):
                W_k = self.K_mappers[head]
                W_q = self.Q_mappers[head]
                W_v = self.V_mappers[head]

                # Get the given head's k,q,and v representations
                k = W_k(sequence) # (N, d) @ (d, d/H) -> (N, d/H)
                q = W_q(sequence) # (N, d) @ (d, d/H) -> (N, d/H)
                v = W_v(sequence) # (N, d) @ (d, d/H) -> (N, d/H)

                # Perform scaled dot product self attention, refer to the formula in the spec
                attention = self.softmax(q @ k.T / self.scale_factor)
                attention = attention @ v

                # Log the current attention head's attention values
                seq_result.append(attention)

            # For the current sequence (patched image) being processed, combine each attention 
            # head's attention values columnwise
            projected_sequence = self.c_proj(torch.hstack(seq_result))
            result.append(projected_sequence)

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class ViT(nn.Module):
    def __init__(
        self,
        num_patches: int, 
        num_blocks: int,
        num_hidden: int,
        num_heads: int,
        num_classes: int = 2,
        chw_shape: tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        """Vision Transformer (ViT) model that processes an image by dividing it into patches,
        applying transformer encoders, and classifying the image using an MLP head.

        Args:
            num_patches: Number of patches to divide the image into along each dimension.
            num_blocks: Number of Transformer encoder blocks.
            num_hidden: Number of hidden dimensions in the patch embeddings.
            num_heads: Number of attention heads in the multi-head attention mechanism.
            num_classes: Number of output classes for classification.
            chw_shape: Shape of the input image in (channels, height, width).
        """
        super().__init__()

        self.chw = chw_shape
        self.num_patches = num_patches

        # Tip: What would the size of a single patch be given the width/height 
        # of an image and the number of patches? While the final patch size should be 2D,
        # it may be easier to consider each dimension separately as a starting point.
        self.patch_size = (self.chw[1] / num_patches, self.chw[2] / num_patches)
        self.embedding_d = num_hidden
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # 1) Patch Tokenizer
        # flattened_patch_d should hold the number of pixels in a single patch, 
        # don't forget a patch is created with pixels across all img chanels
        self.flattened_patch_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])

        # Create a linear layer to embed each patch token
        # Note: embedding_d is the same value as num_hidden from the ViT constructor
        # print(f"Flattened patch size: {self.flattened_patch_d}")
        # print(f"Embedding size: {self.embedding_d}")
        self.patch_to_token = nn.Linear(self.flattened_patch_d, self.embedding_d)

        # 2) Learnable classifiation token
        # Use nn.Parameter to create a learnable classification token of shape (1, self.embedding_d)
        self.cls_token = nn.Parameter(torch.rand(1, self.embedding_d))
        
        # 3) Positional embedding
        # Since this implementation of ViT uses a fixed sinusodal positional embedding, we freeze
        # a parameter module based on the embedding generated by get_positional_embeddings
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.num_patches ** 2 + 1, self.embedding_d).clone().detach())
        self.pos_embed.requires_grad = False

        # 4) Transformer encoder blocks
        # Create of list with the number of transformer blocks specified by num_blocks
        transformer_block_list = [TransformerEncoder(num_hidden, num_heads) for _ in range(num_blocks)]
        self.transformer_blocks = nn.ModuleList(transformer_block_list)

        # 5) Classification MLP
        # A final MLP is utilized to map the representation learned in the classification token to
        # output classes. Note that the nn.Linear class represents a fully connected layer without activation.
        # In other words, each element in the output tensor is a linear combination of input values and learned
        # weights. 

        self.mlp = nn.Linear(self.embedding_d, num_classes)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer (ViT). N is the number of images in a batch

        Args:
            X: Input batch of images, tensor of shape (N, channels, height, width).

        Returns:
            Tensor: Classification output of shape (batch_size, num_classes).
        """
        B, H, W = X.shape

        # Patch images
        patches = patchify(X,self.num_patches)


        # TODO: Get linear projection of each patch to a token (hint: the necessary layers might already be defined!)
        embedded_patches = self.patch_to_token(patches)

        # Add the classification (sometimes called 'cls') token to the tokenized_patches
        all_tokens = torch.stack([torch.vstack((self.cls_token, embedded_patches[i])) for i in range(len(embedded_patches))])
        
        # Add the positional embedding to the token sequence
        pos_embed = self.pos_embed.repeat(B, 1, 1)
        all_tokens = all_tokens + pos_embed

        # TODO: run the positionaly embedded tokens through all transformer blocks 
        # stored in self.transformer_blocks
        for block in self.transformer_blocks:
            all_tokens = block(all_tokens) 

        # Extract the classification token and put through mlp
        class_token = all_tokens[:, 0]
        output_logits = self.mlp(class_token)
        return output_logits

class ImageStandardizer:
    """Standardize a batch of images to mean 0 and variance 1.

    The standardization should be applied separately to each channel.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """

    def __init__(self) -> None:
        """Initialize mean and standard deviations to None."""
        self.image_mean = None
        self.image_std = None

    def fit(self, X: npt.NDArray) -> None:
        """Calculate per-channel mean and standard deviation from dataset X."""
        # TODO: Complete this function
        self.image_mean = np.mean(X, axis=(0, 1, 2))
        self.image_std = np.std(X, axis=(0, 1, 2))
    
    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """Return standardized dataset given dataset X."""
        # TODO: Complete this function
        shift_mean = X - self.image_mean
        shift_std = shift_mean / self.image_std
        return shift_std
        

class ConvModel(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: define each layer

        #28 x 28
        self.conv1 =  nn.Conv2d(
            in_channels=1,       # 1 Input channels
            out_channels=6,     # 16 filters
            kernel_size=5,       # 5x5 kernel
            stride=1,            # Stride of 2
            padding=2            # Padding of 2 to preserve spatial dimensions
        )
        #16 x 28 x 28

        self.pool = nn.MaxPool2d(
            kernel_size=2,        # 2x2 pooling
            stride=2             # Stride of 2
        )
        #16 x 14 x 14
        self.conv2 = nn.Conv2d(
            in_channels=6,       # 16 Input channels
            out_channels=36,     # 64 filters
            kernel_size=7,       # 5x5 kernel
            stride=1,            # Stride of 2
            padding=1            # Padding of 2 to preserve spatial diensions
        )
        #36 x 14 x 14
        
        self.conv3 = nn.Conv2d(
            in_channels=36,       # 64 Input channels
            out_channels=12,     # 64 filters
            kernel_size=3,       # 5x5 kernel
            stride=2,            # Stride of 2
            padding=2            # Padding of 2 to preserve spatial diensions
        )

        self.fc_1 = nn.Linear(192, 80)
        self.fc_2 = nn.Linear(80, 10)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, mean=0.0, std=1/sqrt(conv.kernel_size[0] * conv.kernel_size[1] * conv.in_channels))
            nn.init.constant_(conv.bias, 0.0)

        # TODO: initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=1/sqrt(self.fc_1.in_features))
        nn.init.normal_(self.fc_2.weight, mean=0.0, std=1/sqrt(self.fc_2.in_features))
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.constant_(self.fc_2.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N, C, H, W = x.shape


        # TODO: forward pass
        c1 = F.relu(self.conv1(x))

        p1 = self.pool(c1)
        c2 = F.relu(self.conv2(p1))
        p2 = self.pool(c2)
        c3 = F.relu(self.conv3(p2))
        p3 = self.pool(c3)
        # if True:
        #     print("Input shape: ",x.shape)
        #     print("C1 shape: ",c1.shape)
        #     print("P1 shape: ",p1.shape)
        #     print("C2 shape: ",c2.shape)
        #     print("P2 shape: ",p2.shape)
        #     print("C3 shape: ",c3.shape)
        #     print("P3 shape: ",p3.shape)
        flatten = torch.flatten(c3, start_dim=1)
        fc = self.fc_1(flatten)
        fc = F.relu(fc)
        fc = self.fc_2(fc)

        return fc