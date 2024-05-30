import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_VPT import VisionTransformerVPT
from timm.models.vision_transformer import PatchEmbed


class VitB16_32(nn.Module):
    def __init__(self, base_model_name: str, img_size: int, patch_size: int, num_classes: int = 1000, pretrained: bool = False, interpolate: bool = False, vpt: bool = True):
        """Visual Prompt Tuning based on ViT-B/16 Transformer

        Args:
            base_model_name (str): _description_
            img_size (int): _description_
            patch_size (int): _description_
            num_classes (int, optional): _description_. Defaults to 1000.
            pretrained (bool, optional): _description_. Defaults to False.
            interpolate (bool, optional): _description_. Defaults to False.
            vpt (bool, optional): _description_. Defaults to True.
        """
        super(VitB16_32, self).__init__()

        if vpt:
            base_vit_vpt = VisionTransformerVPT(base_model_name = base_model_name,
                                                pretrained = pretrained,
                                                VPT_Project_dim = -1,
                                                VPT_Prompt_Token_Num = 10,
                                                VPT_type = "Shallow",
                                                VPT_Deep_Shared = "False",
                                                VPT_Dropout = 0.1,
                                                VPT_Initiation = 'random',
                                                VPT_Location = 'prepend',
                                                VPT_Num_Deep_Layers = None,
                                                VPT_Reverse_Deep = False,
                                                VPT_VIT_Pool_Type = 'original',
                                                VPT_Forward_Deep_Noexpand = 'False',
                                                num_classes=1000, # or however many you need
                                                embed_dim=768, # Embedding dimension
                                                depth=12,  # Depth, number of transformer blocks
                                                num_heads=12, # Number of attention heads
                                                img_size=224, # Input image size
                                                patch_size=16,
                                                in_chans=3,  # Number of input channels (for RGB images this is 3)
                                                mlp_ratio=4.,  # MLP ratio
                                                qkv_bias=True,  # Include bias for QKV if True
                                            ) # Size of the patche
            
            # base_vit_vpt = vision_transformer_vpt_base_patch16_224(pretrained = True, img_size=224, patch_size=16,
            #                                                         num_classes = 1000, VPT_Prompt_Token_Num=1,
            #                                                         base_model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k")
            self.base_model = base_vit_vpt
        else:
            # load the base model without any weights, just a place holder
            base_vit = timm.create_model(model_name=base_model_name, pretrained=pretrained, num_classes=num_classes)
            
            self.base_model = base_vit

        # saving the 16x16 patch + pos embedding for interpolation
        self.orig_patch_embed = self.base_model.patch_embed
        self.orig_pos_embed = self.base_model.pos_embed

        # get the total patches and then init the new patch embeds
        self.num_patches = (img_size // patch_size) ** 2 # create a tuple (x,x)
        self.patch_size = patch_size

        embed_args = {} 
        # init the new patch embedding wieght
        self.base_model.patch_embed = PatchEmbed( img_size=img_size,
                                                 patch_size=patch_size,
                                                 in_chans=3,
                                                 embed_dim=768,
                                                 bias=not False,  # disable bias if pre-norm is used (e.g. CLIP)
                                                 dynamic_img_pad=False,
                                                 **embed_args)

        # now get the new pos embed
        # pos_embed_shape = (1, self.num_patches + 1, self.base_model.embed_dim)
        #nn.Parameter(torch.randn(pos_embed_shape))

        embed_len = self.num_patches if self.base_model.no_embed_class else self.num_patches + self.base_model.num_prefix_tokens
        self.base_model.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.base_model.embed_dim) * .02)

        # init the pos embed and patch embed in he init
        nn.init.kaiming_uniform_(self.base_model.pos_embed)

        if interpolate:
            # interpolate the weights from 16x16 to 32x32
            # resize the patch + pos embeddings
            resized_patch_embed_weights = F.interpolate(self.orig_patch_embed.proj.weight, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            resized_pos_embed = self.interpolate_pos_embed()

            self.base_model.patch_embed.proj.weight = torch.nn.Parameter(resized_patch_embed_weights) # convert back to parameter
            self.base_model.patch_embed.proj.bias = torch.nn.Parameter(self.orig_patch_embed.proj.bias.data)
            self.base_model.pos_embed = torch.nn.Parameter(resized_pos_embed)

        # only for validation without VPT addition
        if vpt is False and pretrained is True:
            assert isinstance(self.base_model, VisionTransformer)
            self._init_pretrained_weights(base_model_name, num_classes)

    def _init_pretrained_weights(self, base_model_name: str, num_classes: int):
        pretrained_weights = timm.create_model(model_name=base_model_name, pretrained=True, num_classes=num_classes).state_dict()
        current_state_dict = self.base_model.state_dict()

        # Filter out the weights that we don't want to copy
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k not in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']}

        # Update current state dict with filtered pretrained weights
        current_state_dict.update(pretrained_weights)

        # Load the updated state dict
        self.base_model.load_state_dict(current_state_dict)

    def interpolate_pos_embed(self):
        """interpolate the position embedding with the new size
        taken from https://github.com/mlfoundations/open_clip/blob/73fa7f03a33da53653f61841eb6d69aef161e521/src/open_clip/pos_embed.py#L75

        Args:
            pos_embed (_type_): _description_

        Returns:
            _type_: _description_
        """
        original_embedding_size = self.orig_pos_embed.shape[-1] # original patch embeds
        new_num_patches = self.num_patches # new num patches / old was 196 ; new will be 49

        new_num_extra_tokens = self.base_model.pos_embed.shape[-2] - new_num_patches # new num extra patches
        
        # height (== width) for the checkpoint position embedding
        orig_size = int((self.orig_pos_embed.shape[-2] - new_num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(new_num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = self.orig_pos_embed[:, :new_num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = self.orig_pos_embed[:, new_num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, original_embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            return new_pos_embed

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_model.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.base_model.pos_embed

        to_cat = []
        if self.base_model.cls_token is not None:
            to_cat.append(self.base_model.cls_token.expand(x.shape[0], -1, -1))
        if self.base_model.reg_token is not None:
            to_cat.append(self.base_model.reg_token.expand(x.shape[0], -1, -1))

        if self.base_model.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.base_model.pos_drop(x)
    # def forward(self, x):
    #     # Forward pass through the modified base model
    #     # This assumes the base model's forward method is equipped to handle the input correctly
    #     # after the architectural adjustments made during initialization.
    #     return self.base_model(x)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model.patch_embed(x)
        x = self._pos_embed(x)
        x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)
        if self.base_model.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.base_model.blocks, x)
        else:
            x = self.base_model.blocks(x)
        x = self.base_model.norm(x)
        return x
    
    def forward_features_prompts(self, x):
        # Apply parent class's patch embedding process
        embeddings = self.forward_features(x)

        # Process and prepend VPT embeddings
        x = self.base_model._incorporate_prompt(embeddings)

        return x


    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.base_model.attn_pool is not None:
            x = self.base_model.attn_pool(x)
        elif self.base_model.global_pool == 'avg':
            x = x[:, self.base_model.num_prefix_tokens:].mean(dim=1)
        elif self.base_model.global_pool:
            x = x[:, 0]  # class token
        x = self.base_model.fc_norm(x)
        x = self.base_model.head_drop(x)
        return x if pre_logits else self.base_model.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.base_model._freeze_all_parameters()
        self.base_model._unfreeze_vpt_parameters()

        x = self.forward_features_prompts(x) # encapsulates forward features
        x = self.forward_head(x)
        return x
    


    
# # Interpolate the position embeddings
# def interpolate_pos_embed(pos_embed, orig_num_patches, new_num_patches, model):
#     """interpolate the positional embeddings

#     Args:
#         pos_embed (_type_): _description_
#         orig_num_patches (_type_): _description_
#         new_num_patches (_type_): _description_
#         model (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     original_embedding_size = pos_embed.shape[-1]
#     new_num_extra_tokens = model.visual.positional_embedding.shape[-2] - new_num_patches
    
#     # orig_size = int((orig_num_patches) ** 0.5)
#     # new_size = int(new_num_patches ** 0.5)

#     # height (== width) for the checkpoint position embedding
#     orig_size = int((pos_embed.shape[-2] - new_num_extra_tokens) ** 0.5)
#     # height (== width) for the new position embedding
#     new_size = int(new_num_patches ** 0.5)
#     # class_token and dist_token are kept unchanged

#     print("Total tokens:", total_tokens)
#     print("Original grid size:", orig_size)
#     print("New grid size:", new_size)
#     print("Extra tokens:", num_extra_tokens)

#     if orig_size != new_size:
#         print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
#         extra_tokens = pos_embed[:, :new_num_extra_tokens]
#         pos_tokens = pos_embed[:, new_num_extra_tokens:]
#         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, original_embedding_size).permute(0, 3, 1, 2)
#         pos_tokens = torch.nn.functional.interpolate(
#             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
#         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#         return new_pos_embed
#     else:
#         return pos_embed


def interpolate_pos_embed(orig_pos_embed, num_patches, model_pos_embed):
    """Interpolate the position embedding with the new size based on new number of patches.

    Args:
        orig_pos_embed (torch.Tensor): Original position embedding tensor.
        num_patches (int): Number of patches after changing the patch size.
        model_pos_embed (torch.Tensor): Model's position embedding to extract dimensions for extra tokens.

    Returns:
        torch.Tensor: New position embedding tensor with interpolated positions.
    """
    original_embedding_size = orig_pos_embed.shape[-1]  # Dimension of each embedding
    new_num_patches = num_patches  # New number of patches (e.g., from 196 to 49)
    
    # Calculate the number of extra tokens based on the total length of the position embedding and the number of patches
    total_tokens = model_pos_embed.shape[-2]  # Total tokens in the current model's pos_embed
    new_num_extra_tokens = total_tokens - new_num_patches  # Calculate new extra tokens

    # Calculate sizes based on patches (original and new)
    orig_size = int((orig_pos_embed.shape[-2] - new_num_extra_tokens) ** 0.5)  # Original grid size
    new_size = int(new_num_patches ** 0.5)  # New grid size

    print("Total tokens:", total_tokens)
    print("Original grid size:", orig_size)
    print("New grid size:", new_size)
    print("Extra tokens:", new_num_extra_tokens)

    # Interpolate only if the grid sizes have changed
    if orig_size != new_size:
        print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
        extra_tokens = orig_pos_embed[:, :new_num_extra_tokens]  # Extract extra tokens
        pos_tokens = orig_pos_embed[:, new_num_extra_tokens:]  # Extract positional tokens

        # Reshape positional tokens to a 4D tensor for interpolation
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, original_embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Concatenate extra tokens back with interpolated positional tokens
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        return new_pos_embed

    # Return original embeddings if no interpolation is needed
    return orig_pos_embed
