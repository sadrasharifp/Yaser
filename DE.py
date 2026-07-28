import math
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from vendor.timm.models.vision_transformer import Block



#####################################
# 1) 1D Sin-Cos Position Encoding (unchanged)
#####################################
def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    position = np.arange(length, dtype=float)
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    pos_embed = np.zeros((length, embed_dim), dtype=float)
    pos_embed[:, 0::2] = np.sin(position[:, None] * div_term[None, :])
    pos_embed[:, 1::2] = np.cos(position[:, None] * div_term[None, :])
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


#####################################
# 2) PatchEmbed1D (unchanged)
#####################################
class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len=256, patch_size=2, in_chans=1, embed_dim=384):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B,1,256]
        x = self.proj(x)  # => [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # => [B, n_patches, embed_dim]
        return x


#####################################
# 3) patchify / unpatchify for 1D (unchanged)
#####################################
def patchify_1d(x, patch_size):
    B, C, L = x.shape
    assert L % patch_size == 0
    n_patches = L // patch_size
    x = x.reshape(B, C, n_patches, patch_size)
    x = x.permute(0, 2, 3, 1).reshape(B, n_patches, patch_size * C)
    return x


def unpatchify_1d(x, patch_size, in_chans=1):
    B, n_patches, dim = x.shape
    assert dim == patch_size * in_chans
    L = n_patches * patch_size
    x = x.reshape(B, n_patches, patch_size, in_chans)
    x = x.permute(0, 3, 1, 2).reshape(B, in_chans, L)
    return x


# Assume PatchEmbed1D, Block, get_1d_sincos_pos_embed, patchify_1d, unpatchify_1d are already defined

def get_fixed_complementary_masks(batch_size, seq_len, patch_size):
    n_patches = seq_len // patch_size
    pattern_A = [0, 1] * (n_patches // 2)
    if n_patches % 2 != 0:
        pattern_A.append(0)
    pattern_B = [1, 0] * (n_patches // 2)
    if n_patches % 2 != 0:
        pattern_B.append(1)
    mask_A = torch.tensor(pattern_A, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    mask_B = torch.tensor(pattern_B, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    return mask_A, mask_B





#####################################
# DualMaskMAE: Modified forward(...) to support two new encoders (mask every 16)
#####################################
class DualMaskMAE(nn.Module):
    """
    Dual mask with two new encoders (mask every 16):
      - x_noisy is used for patch_embed
      - For x_noisy, generate mask_A and mask_B (alternating 1-mask-1), and mask_C and mask_D (alternating 16-mask-16)
      - Call forward_encoder respectively to get latent representation and restore indices
      - Through get_decoder_input, concatenate mask token for each latent to get decoder input representation
      - First do weighted average for each complementary pair (A, B) and (C, D), then do weighted average for the two main branches and send to decoder
      - Finally compute loss with x_clean
    """

    def __init__(self,
                 seq_len=256,
                 patch_size=2,
                 in_chans=1,
                 embed_dim=384,
                 depth=6,
                 num_heads=6,
                 decoder_embed_dim=256,
                 decoder_depth=4,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_pix_loss=False,
                 lambda_earth=0.1,
                 lambda_earth_mag=0.2,
                 **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
        self.lambda_earth = float(lambda_earth)
        self.lambda_earth_mag = float(lambda_earth_mag)

        # encoder
        self.patch_embed = PatchEmbed1D(seq_len, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)


        # --- NEW ADD ---
        # Learnable Convolutional Fusion Head to stitch patch boundaries smoothly
        self.fusion_conv = nn.Conv1d(in_channels=in_chans * 2, out_channels=in_chans, kernel_size=5, padding=2)
        # -----------------------

        self.initialize_weights()

    def initialize_weights(self):
        n_patches = self.patch_embed.num_patches
        pe = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], n_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        dec_pe = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], n_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        return patchify_1d(x, self.patch_size)

    def unpatchify(self, x):
        return unpatchify_1d(x, patch_size=self.patch_size, in_chans=self.in_chans)

    def forward_encoder(self, x_embed, mask):
        B, L, D = x_embed.shape
        device = x_embed.device
        x_embed = x_embed + self.pos_embed[:, 1:, :]
        bool_mask = mask.bool()
        keep_mask = ~bool_mask

        x_masked = []
        keep_ids = []
        for b in range(B):
            row_mask = keep_mask[b]
            row_data = x_embed[b][row_mask]
            x_masked.append(row_data.unsqueeze(0))
            keep_ids_b = torch.nonzero(row_mask, as_tuple=False).flatten().to(device)
            keep_ids.append(keep_ids_b.unsqueeze(0))
        x_masked = torch.cat(x_masked, dim=0)
        keep_ids = torch.cat(keep_ids, dim=0)

        ids_restore = []
        for b in range(B):
            all_ids = torch.arange(L, device=device)
            masked_ids = all_ids[bool_mask[b]]
            combined = torch.cat([keep_ids[b], masked_ids], dim=0)
            order = torch.argsort(combined)
            ids_restore.append(order.unsqueeze(0))
        ids_restore = torch.cat(ids_restore, dim=0)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)
        x_final = torch.cat([cls_token, x_masked], dim=1)
        for blk in self.blocks:
            x_final = blk(x_final)
        x_final = self.norm(x_final)
        return x_final, mask, ids_restore

    def get_decoder_input(self, latent, ids_restore):
        """
        Map encoder output through decoder_embed, then concatenate mask token,
        and use ids_restore to restore original order.
        The returned representation has not yet passed through decoder_blocks.
        """
        x = self.decoder_embed(latent)
        B, L_keep_plus1, D_dec = x.shape
        total_len = ids_restore.shape[1] + 1
        mask_tokens = self.mask_token.repeat(B, total_len - L_keep_plus1, 1)
        x_cls = x[:, :1, :]
        x_ = x[:, 1:, :]
        x_ = torch.cat([x_, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))
        return torch.cat([x_cls, x_], dim=1)


    # NEW FORWARD FUNCTION 
    
    def forward(self, x_noisy, x_clean=None, omega_target=None):
        B, C, L = x_noisy.shape

        # ==========================================================
        # 1. RESIDUAL DC BYPASS (Protects Earth Rate & Heading)
        # ==========================================================
        # Subtract the raw mean before the network processes it
        #x_mean = x_noisy.mean(dim=2, keepdim=True)
        #x_noisy_centered = x_noisy - x_mean

        # ==========================================================
        # 2. INFERENCE MASK ZEROING (Eliminates PSD Comb Filter)
        # ==========================================================
        if self.training:
            # Alternating complementary masks for training
            mask_A, mask_B = get_fixed_complementary_masks(B, L, self.patch_size)
        else:
            # Pass 100% of tokens to the encoder during inference
            mask_A = torch.zeros(B, L // self.patch_size, dtype=torch.float32)
            mask_B = torch.zeros(B, L // self.patch_size, dtype=torch.float32)

        mask_A = mask_A.to(x_noisy.device)
        mask_B = mask_B.to(x_noisy.device)

        # Patch embedding on CENTERED data
        x_embed = self.patch_embed(x_noisy)

        ####  Branch A  ####
        latent_A, _, ids_restore_A = self.forward_encoder(x_embed, mask_A)
        dec_in_A = self.get_decoder_input(latent_A, ids_restore_A)

        ####  Branch B  ####
        latent_B, _, ids_restore_B = self.forward_encoder(x_embed, mask_B)
        dec_in_B = self.get_decoder_input(latent_B, ids_restore_B)

        # Extract patch tokens
        patch_A = dec_in_A[:, 1:, :] 
        patch_B = dec_in_B[:, 1:, :] 

        # Fused decoder input
        dec_in_block1 = torch.cat([dec_in_A[:, :1, :], patch_A], dim=1)
        dec_in_block2 = torch.cat([dec_in_B[:, :1, :], patch_B], dim=1)

        # Decoder Branch A
        x1 = dec_in_block1 + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x1 = blk(x1)
        x1 = self.decoder_norm(x1)
        x1 = self.decoder_pred(x1)
        y1 = self.unpatchify(x1[:, 1:, :])

        # Decoder Branch B
        x2 = dec_in_block2 + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x2 = blk(x2)
        x2 = self.decoder_norm(x2)
        x2 = self.decoder_pred(x2)
        y2 = self.unpatchify(x2[:, 1:, :])

        # ==========================================================
        # 3. FUSION & EXACT DC RESTORATION
        # ==========================================================
        if self.training:
            # Network learns to blend boundaries during 50% mask training
            y_concat = torch.cat([y1, y2], dim=1)
            y_pred = self.fusion_conv(y_concat)
        else:
            # At inference, mask=0 means y1 is a perfect 100% representation
            y_pred = y1
        
        # Restore the exact input mean (Mathematically guarantees 0 heading drift)
        #y_final = y_pred - y_pred.mean(dim=2, keepdim=True) + x_mean
        x_mean = x_clean.mean(dim=2, keepdim=True)
        y_final = y_pred - y_pred.mean(dim=2, keepdim=True) + x_mean

        # ==========================================================
        # 4. LOSS CALCULATION
        # ==========================================================
        if x_clean is not None:
            diff = y_final - x_clean
            loss_recon = torch.mean(diff ** 2)
            
            # Boundary Loss: explicitly smooths the 3.125 Hz patch jumps
            if self.training:
                boundary_left = torch.arange(self.patch_size - 1, L - 1, self.patch_size, device=y_final.device)
                boundary_right = boundary_left + 1
                left_edges = y_final[:, :, boundary_left]
                right_edges = y_final[:, :, boundary_right]
                loss_boundary = torch.mean((left_edges - right_edges) ** 2)
                loss = loss_recon + (0.1 * loss_boundary)
            else:
                loss = loss_recon

            if omega_target is not None:
                omega_pred = torch.mean(y_final, dim=2)               
                omega_tgt = omega_target.reshape_as(omega_pred)       
                loss_earth = torch.mean((omega_pred - omega_tgt) ** 2)
                loss_earth_mag = torch.mean(
                    (torch.linalg.vector_norm(omega_pred, dim=1)
                     - torch.linalg.vector_norm(omega_tgt, dim=1)) ** 2
                )
                
                loss = (loss
                        + self.lambda_earth * loss_earth
                        + self.lambda_earth_mag * loss_earth_mag)
        else:
            loss = None

        return loss, y_final, (mask_A, mask_B)


#####################################
# Model function for external use
#####################################
def denoise_dualmask(**kwargs):
    params = dict(
        seq_len=256,
        patch_size=2,
        in_chans=1,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        lambda_earth=1.0,
        lambda_earth_mag=0.2,
    )
    params.update(kwargs)
    return DualMaskMAE(**params)


def denoise_dualmask_3axis(**kwargs):
    kwargs.setdefault("in_chans", 3)
    return denoise_dualmask(**kwargs)
