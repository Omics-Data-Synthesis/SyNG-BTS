# -*- coding: utf-8 -*-

# %%
import torch
import torch.nn as nn

# %%


class AE(nn.Module):
    def __init__(self, num_features):

        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(int(num_features), int(256)),
            nn.ReLU(True),
            nn.Linear(int(256), int(128)),
            nn.ReLU(True),
            nn.Linear(int(128), int(64)),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(64), int(128)),
            nn.ReLU(True),
            nn.Linear(int(128), int(256)),
            nn.ReLU(True),
            nn.Linear(int(256), int(num_features)),
            nn.ReLU(True),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# %%


class VAE(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(int(num_features), int(256)),
            nn.ReLU(True),
            nn.Linear(int(256), int(128)),
            nn.ReLU(True),
            nn.Linear(int(128), int(64)),
            nn.ReLU(True),
        )

        self.z_mean = torch.nn.Linear(64, 32)
        self.z_log_var = torch.nn.Linear(64, 32)

        self.decoder = nn.Sequential(
            nn.Linear(int(32), int(64)),
            nn.ReLU(True),
            nn.Linear(int(64), int(128)),
            nn.ReLU(True),
            nn.Linear(int(128), int(256)),
            nn.ReLU(True),
            nn.Linear(int(256), int(num_features)),
            nn.ReLU(True),
        )

    def reparameterize(self, z_mu, z_log_var, deterministic=False):
        if deterministic:
            return z_mu
        else:
            eps = torch.randn(z_mu.size(0), z_mu.size(1))
            # .to(z_mu.get_device())
            z = z_mu + eps * torch.exp(z_log_var / 2.0)
            return z

    def encoding_fn(self, x, deterministic=False):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var, deterministic)
        return encoded

    def forward(self, x, deterministic=False):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var, deterministic)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


# %%
class CVAE(nn.Module):
    def __init__(self, num_features, num_classes, wide_encoder=False):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.wide_encoder = wide_encoder

        # encoder input: features + label(dim = 1)
        # wide_encoder=True: suitable for high-dimensional data like RNA, use wider encoder
        # wide_encoder=False: suitable for miRNA data, keep original structure
        if wide_encoder:
            # RNA: wider encoder (512 -> 128 -> 64)
            enc_hidden = (512, 256, 128, 64)
        else:
            # miRNA: standard encoder (256 -> 128 -> 64)
            enc_hidden = (256, 128, 64)

        enc_layers = []
        in_dim = num_features + 1
        for h in enc_hidden:
            enc_layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(True),
            ])
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        bottleneck_dim = enc_hidden[-1]
        self.z_mean = nn.Linear(bottleneck_dim, 32)
        self.z_log_var = nn.Linear(bottleneck_dim, 32)

        # decoder: symmetric to encoder
        dec_hidden = tuple(reversed(enc_hidden))
        dec_layers = []
        in_dim = 32 + 1
        for h in dec_hidden:
            dec_layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(True),
            ])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, num_features))
        dec_layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, z_mu, z_log_var, deterministic=False):
        if deterministic:
            return z_mu
        else:
            eps = torch.randn_like(z_mu)
            z = z_mu + eps * torch.exp(z_log_var / 2.0)
            return z

    def encoding_fn(self, x, y, deterministic=False):
        x = self.encoder(torch.cat((x, y), dim=1))
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var, deterministic)
        return z_mean, z_log_var, encoded

    def decoding_fn(self, encoded, y):
        encoded = torch.cat((encoded, y), dim=1)
        decoded = self.decoder(encoded)
        return decoded

    def forward(self, x, y, deterministic=False):
        z_mean, z_log_var, encoded = self.encoding_fn(x, y, deterministic)
        decoded = self.decoding_fn(encoded, y)
        return encoded, z_mean, z_log_var, decoded


# %%
class GAN(torch.nn.Module):
    def __init__(self, num_features, latent_dim=32):
        super().__init__()

        self.num_features = num_features

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_features),
            nn.ReLU(inplace=True),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def generator_forward(self, z):  # z is input low dimension noise
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits
