import os.path

import pytorch_lightning as pl
import torch
import torch.distributions as td
import torchvision.datasets
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
        self.prior_mean = nn.Parameter(data=torch.zeros(size=[self.latent_dim], dtype=torch.float32),
                                       requires_grad=False)
        self.prior_std = nn.Parameter(data=torch.ones(size=[self.latent_dim], dtype=torch.float32),
                                      requires_grad=False)
        return td.Independent(td.Normal(loc=self.prior_mean, scale=self.prior_std), reinterpreted_batch_ndims=1)

    def sample(self, batch_size):
        z = self.prior.sample(sample_shape=batch_size)  # (None, latent_dim)
        out = self.decoder(z)

    def forward(self, inputs):
        posterior = self.encoder(inputs)
        encode_sample = posterior.sample()
        out = self.decoder(encode_sample)
        log_likelihood = out.log_prob(inputs)  # (None,)
        kl_divergence = td.kl_divergence(posterior, self.prior)
        return -log_likelihood, kl_divergence

    def training_step(self, batch):
        nll, kld = self(batch)
        loss = nll + kld * self.beta
        loss = torch.mean(loss, dim=0)
        return loss

    def validation_step(self, batch, batch_idx):
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
        log_likelihood = out.log_prob(x)  # (None,)
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
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=1),
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


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes, image_size):
        super(CNNDecoder, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.channel, self.height, self.width = image_size
        self.model = nn.Sequential(
            nn.LazyLinear(16 * self.width // 4 * self.height // 4),
            nn.ReLU(),
            rlu.nn.LambdaLayer(function=lambda x: torch.reshape(x, shape=[-1, 16, self.height // 4, self.width // 4])),
            nn.LazyConvTranspose2d(out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConvTranspose2d(out_channels=self.channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            rlu.distributions.IndependentNormalWithFixedVar(var_shape=self.image_size, reinterpreted_batch_ndims=3)
        )

        self(torch.rand(size=[1, self.latent_dim]), torch.randint(self.num_classes, size=[1], dtype=torch.int64))

    def forward(self, x, cond):
        x = torch.cat([x, torch.nn.functional.one_hot(cond, num_classes=self.num_classes)], dim=-1)
        return self.model(x)


class ConvolutionalCVAE(ConditionalBetaVAE):
    def __init__(self, image_size, num_classes, latent_dim=5):
        self.image_size = image_size
        self.channel, self.height, self.width = image_size
        assert self.height % 4 == 0 and self.width % 4 == 0
        self.num_classes = num_classes
        super(ConvolutionalCVAE, self).__init__(latent_dim=latent_dim, beta=1.0)

    def _make_encoder(self) -> nn.Module:
        model = CNNEncoder(self.latent_dim, self.num_classes, self.image_size)
        return model

    def _make_decoder(self) -> nn.Module:
        model = CNNDecoder(self.latent_dim, self.num_classes, self.image_size)
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == '__main__':
    import torch.utils.data

    batch_size = 256
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=0.5, std=0.5)]
    )

    root = os.path.expanduser('~/pytorch_dataset')
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    image_size = tuple(train_dataset[0][0].shape)
    num_classes = 10
    model = ConvolutionalCVAE(image_size=image_size, num_classes=num_classes, latent_dim=5)

    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=1,
                         enable_model_summary=False, enable_checkpointing=False)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
