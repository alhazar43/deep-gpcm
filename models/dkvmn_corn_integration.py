"""
DKVMN-CORN Integration

This file implements the integration of the CORN layer with the DKVMN architecture,
maintaining the core memory and IRT parameter extraction functionalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .corn_layer import CornLayer

class DeepGpcmCornModel(nn.Module):
    """
    Deep-GPCM model with CORN integration.
    """

    def __init__(self, base_gpcm_model):
        super(DeepGpcmCornModel, self).__init__()

        # Copy attributes from the base model
        self.n_questions = base_gpcm_model.n_questions
        self.n_cats = base_gpcm_model.n_cats
        self.memory_size = base_gpcm_model.memory_size
        self.key_dim = base_gpcm_model.key_dim
        self.value_dim = base_gpcm_model.value_dim
        self.final_fc_dim = base_gpcm_model.final_fc_dim
        self.ability_scale = base_gpcm_model.ability_scale
        self.use_discrimination = base_gpcm_model.use_discrimination
        self.embedding_strategy = base_gpcm_model.embedding_strategy

        # Copy modules from the base model
        self.q_embed = base_gpcm_model.q_embed
        self.gpcm_value_embed = base_gpcm_model.gpcm_value_embed
        self.memory = base_gpcm_model.memory
        self.init_value_memory = base_gpcm_model.init_value_memory
        self.summary_network = base_gpcm_model.summary_network
        self.student_ability_network = base_gpcm_model.student_ability_network
        self.question_threshold_network = base_gpcm_model.question_threshold_network
        self.discrimination_network = base_gpcm_model.discrimination_network

        # Add CORN layer
        corn_input_dim = self.final_fc_dim + self.key_dim
        self.corn_layer = CornLayer(corn_input_dim, self.n_cats)

        self.gpcm_probability = base_gpcm_model.gpcm_probability

    def forward(self, q_data, r_data, target_mask=None, use_corn=True):
        batch_size, seq_len = q_data.shape
        q_one_hot = F.one_hot(q_data, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]

        self.memory.init_value_memory(batch_size, self.init_value_memory)

        student_abilities, item_thresholds, discrimination_params, predictions, all_logits = [], [], [], [], []

        for t in range(seq_len):
            q_t = q_data[:, t]
            r_t = r_data[:, t]
            q_embed_t = self.q_embed(q_t)
            q_one_hot_t = q_one_hot[:, t:t+1, :]
            r_t_unsqueezed = r_t.unsqueeze(1)

            from .model import ordered_embedding, unordered_embedding, linear_decay_embedding, adjacent_weighted_embedding
            if self.embedding_strategy == 'ordered':
                gpcm_embed_t = ordered_embedding(q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats)
            elif self.embedding_strategy == 'unordered':
                gpcm_embed_t = unordered_embedding(q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats)
            elif self.embedding_strategy == 'linear_decay':
                gpcm_embed_t = linear_decay_embedding(q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats)
            elif self.embedding_strategy == 'adjacent_weighted':
                gpcm_embed_t = adjacent_weighted_embedding(q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats)
            else:
                raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")
            
            gpcm_embed_t = gpcm_embed_t.squeeze(1)
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)

            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)

            theta_t = self.student_ability_network(summary_vector).squeeze(-1) * self.ability_scale
            betas_t = self.question_threshold_network(q_embed_t)
            discrim_input = torch.cat([summary_vector, q_embed_t], dim=-1)
            alpha_t = self.discrimination_network(discrim_input).squeeze(-1)

            if use_corn:
                prob_t, logits_t = self.corn_layer(discrim_input)
                all_logits.append(logits_t)
            else:
                theta_expanded = theta_t.unsqueeze(1)
                alpha_expanded = alpha_t.unsqueeze(1)
                betas_expanded = betas_t.unsqueeze(1)
                prob_t = self.gpcm_probability(theta_expanded, alpha_expanded, betas_expanded).squeeze(1)

            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            predictions.append(prob_t)

            if t < seq_len - 1:
                self.memory.write(correlation_weight, value_embed_t)

        student_abilities = torch.stack(student_abilities, dim=1)
        item_thresholds = torch.stack(item_thresholds, dim=1)
        discrimination_params = torch.stack(discrimination_params, dim=1)
        predictions = torch.stack(predictions, dim=1)
        all_logits = torch.stack(all_logits, dim=1) if use_corn else None

        return student_abilities, item_thresholds, discrimination_params, predictions, all_logits

def create_corn_enhanced_model(original_model):
    """
    Factory function to create a CORN-enhanced version of the DeepGpcmModel.
    """
    corn_model = DeepGpcmCornModel(original_model)
    corn_model.load_state_dict(original_model.state_dict(), strict=False)
    return corn_model
