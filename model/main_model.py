import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchSelect(nn.Module):

    """
    Patch-level region selection module.

    Args:
        embed_dim: Dimension of patch token features.
        hidden_dim: Hidden dimension of the MLP-based patch scorer.
        sparse_ratio: Ratio of patches selected for editable regions.
    """
    def __init__(self, embed_dim=768, hidden_dim=384, sparse_ratio=0.6):
        super().__init__()
        self.sparse_ratio = sparse_ratio

        self.score_predict = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    
        self.weights = nn.Parameter(torch.tensor([0.3, 0.4, 0.3]))

    def forward(self, patch_tokens, text_global, image_global):
        B, N, C = patch_tokens.shape

        # learnable patch importance 
        a_pi = self.score_predict(patch_tokens).squeeze(-1)

        patch_tokens_norm = F.normalize(patch_tokens, dim=-1)
        text_global_norm = F.normalize(text_global, dim=-1).unsqueeze(1)
        image_global_norm = F.normalize(image_global, dim=-1).unsqueeze(1)


        # text-patch relevance 
        a_ri = torch.sum(patch_tokens_norm * text_global_norm, dim=-1)

        # image structural saliency
        a_si = torch.sum(patch_tokens_norm * image_global_norm, dim=-1)

        def normalize_score(score):
            min_val = score.min(dim=1, keepdim=True)[0]
            max_val = score.max(dim=1, keepdim=True)[0]
            return (score - min_val) / (max_val - min_val + 1e-6)

        a_ri = normalize_score(a_ri)
        a_si = normalize_score(a_si)

        weights = F.softplus(self.weights)
        weights = weights / (weights.norm(p=2) + 1e-6)
        w_pi, w_ri, w_si = weights.unbind(0)

        final_score = w_pi * a_pi + w_ri * a_ri + w_si * a_si

        tau = 0.5
        soft_mask = F.gumbel_softmax(final_score / tau, dim=1)

        k = int(N * self.sparse_ratio)
        _, topk_indices = torch.topk(final_score, k=k, dim=1)
        hard_mask = torch.zeros_like(final_score).scatter(1, topk_indices, 1.0)

        # STE for hard mask selection with soft gradient propagation
        score_mask = hard_mask + (soft_mask - soft_mask.detach())

        H = W = int(N ** 0.5)
        return score_mask.view(B, H, W)


class RGN(nn.Module):
    """
    Region Guidance Network for patch-based text-driven image editing.

    Args:
        image_encoder: Frozen visual backbone (e.g., DINO ViT).
        clip_extractor: Frozen CLIP feature extractor.
        diffusion_fn: Diffusion-based image editing function.
    """

    def __init__(
        self,
        image_encoder,
        clip_extractor,
        diffusion_fn,
        embed_dim=768,
        token_sparse_ratio=0.6,
        loss_alpha=1.0,
        loss_beta=1.0,
        loss_gamma=1.0,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.image_encoder = image_encoder
        self.clip_extractor = clip_extractor
        self.diffusion_fn = diffusion_fn

        self.text_proj = nn.Linear(512, embed_dim)

        # patch-level region selection module for predicting editable patch masks
        self.score_predict = PatchSelect(
            embed_dim=embed_dim,
            hidden_dim=embed_dim // 2,
            sparse_ratio=token_sparse_ratio,
        )
        self.alpha = loss_alpha
        self.beta = loss_beta
        self.gamma = loss_gamma



    # Generate patch-level editing mask from image and text features
    def get_patch_mask(self, imgs, text_embs):
        """
        imgs: input images
        text_embs: CLIP text embeddings
        """

        # Extract patch tokens from the frozen image encoder
        feats = self.image_encoder(imgs)         
        patch_tokens = feats[:, 1:, :]           

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)

        text_global = F.normalize(self.text_proj(text_embs.float()), dim=-1)
        if text_global.dim() == 1:
            text_global = text_global.unsqueeze(0)
        if text_global.size(0) == 1 and B > 1:
            text_global = text_global.expand(B, -1)

        image_global = F.normalize(patch_tokens.mean(dim=1), dim=-1)

        mask = self.score_predict(patch_tokens, text_global, image_global)
        return mask.view(B, H, W)

 

    def generate_result(self, imgs, mask, prompts):
        """
        Diffusion-based image editing with patch masks.
        """
        return self.diffusion_fn(imgs, mask, prompts)



    # Directional CLIP loss for aligning visual edit directions with text directions
    def calculate_clip_dir_loss(self, inputs, outputs, target_embeddings, src_embeddings):
        n_embeddings = np.random.randint(1, min(len(src_embeddings), len(target_embeddings)) + 1)
        idx = torch.randint(min(len(src_embeddings), len(target_embeddings)), (n_embeddings,))
        src_emb = src_embeddings[idx]
        tgt_emb = target_embeddings[idx]
        target_dirs = tgt_emb - src_emb

        loss = 0.0
        for in_img, out_img in zip(inputs, outputs):
            in_e = self.clip_extractor.get_image_embedding(in_img.unsqueeze(0))
            out_e = self.clip_extractor.get_image_embedding(out_img.unsqueeze(0))
            for target_dir in target_dirs:
                loss += 1 - F.cosine_similarity(out_e - in_e, target_dir.unsqueeze(0)).mean()

        return loss / (len(outputs) * len(target_dirs))

    
    # CLIP guidance loss for semantic alignment
    # between edited images and target prompts
    def calculate_clip_loss(self, outputs, target_embeddings):
        loss = 0.0
        for img in outputs:
            img_e = self.clip_extractor.get_image_embedding(img.unsqueeze(0))
            for target_emb in target_embeddings:
                loss += 1 - F.cosine_similarity(img_e, target_emb.unsqueeze(0)).mean()
        return loss / (len(outputs) * len(target_embeddings))


    # structure-preservation loss based on clip feature self-similarity
    def calculate_structure_loss(self, outputs, inputs):
        loss = 0.0
        for inp, out in zip(inputs, outputs):
            with torch.no_grad():
                target_sim = self.clip_extractor.get_self_sim(inp.unsqueeze(0))
            current_sim = self.clip_extractor.get_self_sim(out.unsqueeze(0))
            loss += F.mse_loss(current_sim, target_sim)
        return loss / len(outputs)


    # Composite editing objective combining semantic alignment and structure preservation
    def get_loss(self, source_imgs, results, e_prompt, o_prompt):
        text_emb = self.clip_extractor.get_text_embedding(e_prompt)
        src_emb = self.clip_extractor.get_text_embedding(o_prompt)

        loss_clip = self.calculate_clip_loss(results, text_emb)
        loss_dir = self.calculate_clip_dir_loss(source_imgs, results, text_emb, src_emb)
        loss_struct = self.calculate_structure_loss(results, source_imgs)

        total = (
            self.alpha * loss_clip
            + self.beta * loss_dir
            + self.gamma * loss_struct
        )

        return total, loss_clip, loss_dir, loss_struct
