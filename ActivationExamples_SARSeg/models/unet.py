import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier


"""
U-Net
"""
class CustomDecoderSkipConn(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super().__init__()

        # Upsampling blocks with transposed convolution + ConvBlock
        def up_block(in_ch, out_ch, scale_factor):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up1 = up_block(in_channels, decoder_channels[0], scale_factor=8/4)     # 4 → 8
        self.up2 = up_block(decoder_channels[0] + 1024, decoder_channels[1], scale_factor=15/8)  # Concatenate with encoder output (8 → 15)
        self.up3 = up_block(decoder_channels[1] + 512, decoder_channels[2], scale_factor=30/15)  # 15 → 30
        self.up4 = up_block(decoder_channels[2] + 256, decoder_channels[3], scale_factor=60/30)  # 30 → 60
        self.up5 = up_block(decoder_channels[3] + 64, decoder_channels[4], scale_factor=120/60)  # 60 → 120

    def forward(self, x, skip_connections):
        # print("skip_connections shape:", [f.shape for f in skip_connections])
        x = self.up1(x)
        x = torch.cat([x, skip_connections[4]], dim=1)  # Concatenate with the 4th encoder output
        x = self.up2(x)
        x = torch.cat([x, skip_connections[3]], dim=1)  # Concatenate with the 3rd encoder output
        x = self.up3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)  # Concatenate with the 2nd encoder output
        x = self.up4(x)
        x = torch.cat([x, skip_connections[1]], dim=1)  # Concatenate with the 1st encoder output
        x = self.up5(x)
        return x


class CustomUnetSkipConn(smp.Unet):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights=None,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels: int = 2,
        classes: int = 20,
        activation="softmax",
    ):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights,
                         in_channels=in_channels, classes=classes, activation=activation)

        self.decoder = CustomDecoderSkipConn(
            in_channels=self.encoder.out_channels[-1], decoder_channels=decoder_channels
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        print("CustomUnet initialized with encoder channels:", self.encoder.out_channels)

    def forward(self, x):
        features = self.encoder(x)  # encoder features
        x = features[-1]  # Last encoder layer
        skip_connections = features[:-1]

        # Pass feature map
        x = self.decoder(x, skip_connections)
        x = self.segmentation_head(x)
        return x


def create_base_model_skipconn(backbone="resnet50", weights=None, in_channel=2, num_classes=20):
    model = CustomUnetSkipConn(
        encoder_name=backbone,
        encoder_weights=weights,
        in_channels=in_channel,
        classes=num_classes,
        activation="softmax",
    )
    return model


def load_from_checkpoint_skipconn(checkpoint_path, num_classes=20):
    model = create_base_model_skipconn(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
    return model


def load_base_with_bigearth_pretrained_skipconn(num_classes=20):
    # Pretrained model
    model = create_base_model_skipconn(num_classes=num_classes)
    model_bigearth_classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.1.1"
    )
    pretrained_weights = model_bigearth_classifier.state_dict()

    # Mapping from the pretrained to untrained model
    key_mapping = {
        pretrained_key: pretrained_key.replace("model.vision_encoder", "encoder")
        for pretrained_key in pretrained_weights.keys()
        if pretrained_key.startswith("model.vision_encoder")
    }

    # Map weights
    mapped_state_dict = {
        untrained_key: pretrained_weights[pretrained_key]
        for pretrained_key, untrained_key in key_mapping.items()
    }
    # Load model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    return model
