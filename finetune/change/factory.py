import re
import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F
from src.model import Encoder

class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Define convolutional + max-pooling blocks for each scale (UNet-style encoder)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2, resulting in 1/4 resolution
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2, resulting in 1/8 resolution
        )
        
    def forward(self, x):
        feature1 = self.block1(x)  # 1/2 resolution
        feature2 = self.block2(feature1)  # 1/4 resolution
        feature3 = self.block3(feature2)  # 1/8 resolution
        return [feature1, feature2, feature3]

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_att(x)
        x = x * avg_out

        # Spatial Attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_att(spatial_att)
        x = x * spatial_att
        
        return x

class UNetDecoderWithAttentionSkipConnections(nn.Module):
    def __init__(self, encoder_dim, num_classes):
        super().__init__()
        
        # Define attention fusion modules for each skip connection
        self.att_fusion1 = AttentionFusion((encoder_dim // 2) + 256)
        self.att_fusion2 = AttentionFusion((encoder_dim // 4) + 128)
        self.att_fusion3 = AttentionFusion((encoder_dim // 8) + 64)
        
        # Conv layers after attention fusion
        self.conv1 = nn.Conv2d((encoder_dim // 2) + 256, encoder_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d((encoder_dim // 4) + 128, encoder_dim // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((encoder_dim // 8) + 64, encoder_dim // 8, kernel_size=3, padding=1)
        
        # Final output convolution
        self.final_conv = nn.Conv2d(encoder_dim // 8, num_classes, kernel_size=1)
        
        # Transposed convolutions for upsampling
        self.upsample1 = nn.ConvTranspose2d(encoder_dim, encoder_dim // 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(encoder_dim // 2, encoder_dim // 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(encoder_dim // 4, encoder_dim // 8, kernel_size=2, stride=2)

    def forward(self, encoder_features, unet_features):
        # First upsampling and attention fusion
        x = self.upsample1(encoder_features[-1])  # Upsample the encoder feature map
        unet_features[2] = F.interpolate(unet_features[2], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, unet_features[2]], dim=1)
        x = self.att_fusion1(x)  # Apply attention-based fusion
        x = self.conv1(x)

        # Second upsampling and attention fusion
        x = self.upsample2(x)
        unet_features[1] = F.interpolate(unet_features[1], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, unet_features[1]], dim=1)
        x = self.att_fusion2(x)  # Apply attention-based fusion
        x = self.conv2(x)

        # Third upsampling and attention fusion
        x = self.upsample3(x)
        unet_features[0] = F.interpolate(unet_features[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, unet_features[0]], dim=1)
        x = self.att_fusion3(x)  # Apply attention-based fusion
        x = self.conv3(x)

        # Final convolution to output logits
        x = self.final_conv(x)
        
        return x

class SegmentEncoder(Encoder):
    def __init__(self, mask_ratio, patch_size, shuffle, dim, depth, heads, dim_head, mlp_ratio, feature_maps, ckpt_path=None):
        super().__init__(mask_ratio, patch_size, shuffle, dim, depth, heads, dim_head, mlp_ratio)
        self.feature_maps = feature_maps

        # Define FPN layers for multi-scale feature extraction
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fpn3 = nn.Identity()
        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fpn5 = nn.Identity()

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
                
        if ckpt_path is not None:
            self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }
            self.load_state_dict(new_state_dict, strict=False)
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False

    def forward(self, datacube):
        # Access elements in `datacube` dictionary and handle them appropriately
        cube = datacube["pixels"]
        B, C, H, W = cube.shape
        time = datacube.get("time", None)
        latlon = datacube.get("latlon", None)
        gsd = datacube.get("gsd", None)
        waves = datacube.get("waves", None)

        # Convert to patches and add encodings
        patches, waves_encoded = self.to_patch_embed(cube, waves)
        patches = self.add_encodings(patches, time, latlon, gsd)
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)
        patches = torch.cat((cls_tokens, patches), dim=1)

        features = []
        for idx, (attn, ff) in enumerate(self.transformer.layers):
            patches = attn(patches) + patches
            patches = ff(patches) + patches
            if idx in self.feature_maps:
                _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // 8, W=W // 8)
                features.append(_cube)
        patches = self.transformer.norm(patches)
        _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // 8, W=W // 8)
        features.append(_cube)

        # Apply FPN layers with max pooling
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return features

class Segmentor(nn.Module):
    def __init__(self, num_classes, feature_maps, ckpt_path):
        super().__init__()
        self.encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=768,
            depth=12,
            heads=12,
            dim_head=64,
            mlp_ratio=4.0,
            feature_maps=feature_maps,
            ckpt_path=ckpt_path,
        )
        
        self.unet_encoder = UNetEncoder(10)


        # Upsampling with skip connections
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=False),
                nn.Conv2d(2432 if i == 0 else 2816 if i == 2 else 2560, 512, kernel_size=1),  # Adjust channel count based on index
                nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=256),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for i in range(4)
        ] + [
            nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(2560, 512, kernel_size=1),  # Adjusted channel count for the last block
                nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=256),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        ])

        #self.fusion = nn.Conv2d(256 * 5, 256, kernel_size=1)
        # Adjusted fusion layer with the correct input channels
        self.fusion = nn.Conv2d(768, 256, kernel_size=1)
        self.seg_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.decoder = UNetDecoderWithAttentionSkipConnections(encoder_dim=self.encoder.dim, num_classes=num_classes)
    def create_upsample_layer(self, input_channels):
        # Define a new upsample layer with dynamic input channel size
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(input_channels, 512, kernel_size=1),  # Adjust channels dynamically
            nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU()
    )
    def forward(self, before_datacube, after_datacube):
        original_height, original_width = before_datacube["pixels"].shape[2:]  # Target resolution

        # Extract pixel tensors from datacubes
        before_pixels = before_datacube["pixels"]
        after_pixels = after_datacube["pixels"]

        # Encode before and after features
        before_features = self.encoder(before_datacube)
        after_features = self.encoder(after_datacube)
        
        # Use pixel tensors as input to UNetEncoder
        unet_features_before = self.unet_encoder(before_pixels)
        unet_features_after = self.unet_encoder(after_pixels)

        combined_features = []
        
        for i, (before_feat, after_feat, unet_before, unet_after) in enumerate(
            zip(before_features, after_features, unet_features_before, unet_features_after)):
            # Resize unet_before and unet_after to match the spatial dimensions of before_feat
            unet_before = F.interpolate(unet_before, size=before_feat.shape[2:], mode='bilinear', align_corners=False)
            unet_after = F.interpolate(unet_after, size=after_feat.shape[2:], mode='bilinear', align_corners=False)

            # Compute change map and concatenate features
            change_map = before_feat - after_feat
            combined = torch.cat([before_feat, after_feat, change_map, unet_before, unet_after], dim=1)
            
            # Apply upsampling
            combined = self.upsamples[i](combined)
        
            # Upsample each feature to the original resolution
            combined = F.interpolate(combined, size=(original_height, original_width), mode='bilinear', align_corners=False)

            combined_features.append(combined)

        # Fuse combined features and apply segmentation head
        fused = torch.cat(combined_features, dim=1)

        fused = self.fusion(fused)
        
        logits = self.seg_head(fused)

        return logits 
