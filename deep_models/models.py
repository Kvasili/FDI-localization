
'''
This module defines the LSTM Autoencoder model with feature attention for time series data. It includes the following components:
- TimeseriesDataset: A custom PyTorch Dataset class for loading time series data.
- FeatureAttention: A module that computes attention weights for each feature based on the encoder's bottleneck representation.

'''


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TimeseriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return torch.tensor(self.sequences[index], dtype=torch.float)


# Define the LSTM Autoencoder model

# ---- Feature Attention Module ---- #


class FeatureAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + 1, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch, seq_len, input_dim]
        # decoder_hidden: [batch, hidden_dim]

        batch, seq_len, input_dim = encoder_outputs.size()
        # Average encoder_outputs over time (seq_len) to get feature vector: shape [batch, input_dim]
        encoder_feature_avg = encoder_outputs.mean(dim=1)  # [batch, input_dim]

        # Expand decoder_hidden to [batch, input_dim, hidden_dim]
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).repeat(
            1, input_dim, 1)  # [batch, input_dim, hidden_dim]

        # Concatenate along last dim: feature vector (as 1D) + decoder hidden (hidden_dim)
        # So first unsqueeze encoder_feature_avg to [batch, input_dim, 1]
        # [batch, input_dim, hidden_dim+1]
        concat = torch.cat(
            [encoder_feature_avg.unsqueeze(2), decoder_hidden_exp], dim=2)

        # [batch, input_dim, hidden_dim]
        energy = torch.tanh(self.attn(concat))
        attn_weights = torch.softmax(
            self.v(energy).squeeze(-1), dim=1)  # [batch, input_dim]

        return attn_weights


# # ---- Feature Attention Over Time Module ---- #
class FeatureAttentionOverTime(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + 1, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch, seq_len, input_dim]
        # decoder_hidden: [batch, hidden_dim]

        batch, seq_len, input_dim = encoder_outputs.size()

        # For each time step separately
        attn_weights_time = []

        for t in range(seq_len):
            # Get feature vector at time t: [batch, input_dim]
            encoder_feature_t = encoder_outputs[:, t, :]  # [batch, input_dim]

            # Expand decoder hidden: [batch, input_dim, hidden_dim]
            decoder_hidden_exp = decoder_hidden.unsqueeze(
                1).repeat(1, input_dim, 1)

            # Concatenate: [batch, input_dim, hidden_dim+1]
            concat = torch.cat(
                [encoder_feature_t.unsqueeze(2), decoder_hidden_exp], dim=2)

            # [batch, input_dim, hidden_dim]
            energy = torch.tanh(self.attn(concat))
            attn_weights = torch.softmax(
                self.v(energy).squeeze(-1), dim=1)  # [batch, input_dim]

            attn_weights_time.append(attn_weights)

        # Stack over time: [batch, seq_len, input_dim]
        attn_matrix = torch.stack(attn_weights_time, dim=1)

        return attn_matrix  # attention over time and features


# # ---- Modified Autoencoder with Feature Attention ---- #
class AttentionLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.encoder_lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.bottleneck = nn.LSTM(64, 32, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(32, 64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.output_layer = nn.Linear(128, input_dim)

        # self.feature_attention = FeatureAttention(
        #     input_dim=input_dim, hidden_dim=32)

        # New Feature Attention Over Time
        self.feature_attention_over_time = FeatureAttentionOverTime(
            input_dim=input_dim, hidden_dim=32
        )

        self.feature_attention = FeatureAttention(
            input_dim=input_dim, hidden_dim=32)

    def forward(self, x):
        # Encoder
        out, _ = self.encoder_lstm1(x)
        out, _ = self.encoder_lstm2(out)
        encoder_outputs, (h, _) = self.bottleneck(out)

        # Bottleneck representation (final hidden state)
        h_bottleneck = h[-1]  # shape: [batch, 32]

        # Feature Attention
        attn_weights = self.feature_attention(
            x, h_bottleneck)  # [batch, input_dim]
        attn_applied = x * attn_weights.unsqueeze(1)  # weight input features

        # Feature Attention over time
        attn_matrix = self.feature_attention_over_time(
            x, h_bottleneck)  # [batch, seq_len, input_dim]

        # Apply attention to input features, element-wise
        # attn_applied_time = x * attn_matrix  # [batch, seq_len, input_dim]

        # Combine both attentions (e.g., element-wise multiply or add)
        # combined_attention = attn_applied * \
        #     attn_applied_time  # or use another combination

        # # Use attn_applied or attn_applied_time as input to encoder LSTM1
        # out, _ = self.encoder_lstm1(combined_attention)
        # out, _ = self.encoder_lstm2(out)
        # encoder_outputs, (h, _) = self.bottleneck(out)
        # h_bottleneck = h[-1]

        # Decoder input
        repeated = h_bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm1(repeated)
        out, _ = self.decoder_lstm2(out)
        out = self.output_layer(out)

        return out,  attn_weights, attn_matrix
