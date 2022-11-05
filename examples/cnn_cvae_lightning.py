import pytorch_lightning as pl
import torch
import torch.distributions as td
from torch import nn

import rlutils.pytorch as rlu


class BetaVAE(pl.LightningModule):
    def __init__(self, latent_dim, beta=1.):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.prior = self._make_prior()

    def _make_encoder(self) -> nn.Module:
        raise NotImplementedError

    def _make_decoder(self) -> nn.Module:
        raise NotImplementedError

    def _make_prior(self):
        return td.Independent(td.Normal(loc=torch.zeros(size=[self.latent_dim], dtype=torch.float32),
                                        scale=torch.ones(size=[self.latent_dim], dtype=torch.float32)),
                              reinterpreted_batch_ndims=1)

    def forward(self, inputs):
        posterior = self.encoder(inputs)
        encode_sample = posterior.sample()
        out = self.decoder(encode_sample)
        log_likelihood = out.log_prob(inputs)  # (None,)
        kl_divergence = td.kl_divergence(posterior, self.prior)
        # print(f'Shape of nll: {log_likelihood.shape}, kld: {kl_divergence.shape}')
        return -log_likelihood, kl_divergence

    def training_step(self, batch):
        nll, kld = self(batch)
        loss = nll + kld * self.beta
        loss = torch.mean(loss, dim=0)
        return loss


class ConditionalBetaVAE(BetaVAE):
    def forward(self, inputs):
        x, cond = inputs
        posterior = self.encoder(x, cond)
        encode_sample = posterior.sample()
        out = self.decoder(encode_sample, cond)
        log_likelihood = out.log_prob(inputs)  # (None,)
        kl_divergence = td.kl_divergence(posterior, self.prior)
        # print(f'Shape of nll: {log_likelihood.shape}, kld: {kl_divergence.shape}')
        return -log_likelihood, kl_divergence


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes, image_size):
        super(CNNEncoder, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.conv_layers = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dense_layers = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(self.latent_dim * 2),
            rlu.nn.LambdaLayer(function=lambda t: rlu.distributions.make_independent_normal_from_params(t))
        )

        # build model
        self(torch.rand(size=[1] + list(image_size)), torch.randint(self.num_classes, size=[1], dtype=torch.int64))

    def forward(self, x, cond):
        embedding = torch.nn.functional.one_hot(cond, num_classes=self.num_classes)
        x = self.conv_layers(x)
        x = torch.cat((x, embedding), dim=-1)
        out = self.dense_layers(x)
        return out


class ConvolutionalCVAE(ConditionalBetaVAE):
    def __init__(self, image_size, num_classes, latent_dim=5):
        self.image_size = image_size
        self.channel, self.height, self.width = image_size
        assert self.height % 4 == 0 and self.width % 4 == 0
        self.num_classes = num_classes
        super(ConvolutionalCVAE, self).__init__(latent_dim=latent_dim, beta=1.0)

    def _make_encoder(self) -> nn.Module:
        return CNNEncoder(self.latent_dim, self.num_classes, self.image_size)

    def _make_decoder(self) -> nn.Module:
        model = nn.Sequential(
            rlu.nn.LambdaLayer(function=lambda x, cond: torch.cat([x, torch.nn.functional.one_hot(cond)], dim=-1)),
            nn.LazyLinear(16 * self.width // 4 * self.height // 4),
            nn.ReLU(),
            rlu.nn.LambdaLayer(function=lambda x: torch.reshape(x, shape=[16, self.height // 4, self.width // 4])),
            nn.LazyConvTranspose2d(out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConvTranspose2d(out_channels=self.channel, kernel_size=3, stride=2, padding=1),
            rlu.nn.LambdaLayer(function=lambda t: td.Independent(td.Bernoulli(logits=t), reinterpreted_batch_ndims=3))
        )
        model(torch.rand(size=[1, self.latent_dim]), torch.randint(self.num_classes, size=[1], dtype=torch.int64))
        return model


if __name__ == '__main__':
    pass
