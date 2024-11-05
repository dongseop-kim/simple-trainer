# test_models.py
import pytest
import torch
import torch.nn as nn

from strainer.models.feature_decoder import build_feature_decoder
from strainer.models.feature_decoder.decoder_registry import (
    BaseFeatureDecoder, ConvolutionalMultiScaleDecoder, IdentityFeatureDecoder,
    MultiScaleFeatureDecoder)
from strainer.models.feature_extractor import build_feature_extractor


class TestFeatureExtractor:
    """Test cases for feature extractor (encoder) functionality."""

    @pytest.fixture
    def sample_image(self):
        return torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=width=224

    def test_resnet_feature_extractor(self, sample_image):
        """Test ResNet feature extractor creation and forward pass."""
        model = build_feature_extractor(model_name='resnet50', pretrained=False, features_only=True,
                                        out_indices=[1, 2, 3, 4])

        # Get outputs
        features = model(sample_image)

        # Basic assertions
        assert isinstance(features, list), "Features should be a list"
        assert len(features) == 4, "ResNet should return 4 feature levels"

        # Check feature dimensions
        expected_channels = [256, 512, 1024, 2048]  # ResNet50 channels
        expected_scales = [4, 8, 16, 32]  # ResNet50 strides

        for feat, exp_c, exp_s in zip(features, expected_channels, expected_scales):
            print(feat.shape)
            assert feat.shape[0] == 2, "Batch size should be preserved"
            assert feat.shape[1] == exp_c, f"Expected {exp_c} channels"
            assert feat.shape[2] == 224 // exp_s, "Feature spatial size mismatch"
            assert feat.shape[3] == 224 // exp_s, "Feature spatial size mismatch"

    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        with pytest.raises(ValueError, match="not found in timm registry"):
            build_feature_extractor("nonexistent_model")


class TestDecoder:
    """Test cases for decoder functionality."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample multi-scale features."""
        return [torch.randn(2, 256, 56, 56),   # 1/4 scale
                torch.randn(2, 512, 28, 28),   # 1/8 scale
                torch.randn(2, 1024, 14, 14),  # 1/16 scale
                ]

    def test_identity_decoder(self, sample_features):
        """Test IdentityFeatureDecoder functionality."""
        decoder = IdentityFeatureDecoder(input_channels=[256, 512, 1024], input_scales=[4, 8, 16])
        output = decoder(sample_features)  # Should return the input unchanged

        assert isinstance(output, list), "Output should be a list"
        assert len(output) == len(sample_features), "Output length should match input"
        for out, inp in zip(output, sample_features):
            assert torch.equal(out, inp), "Identity decoder should not modify input"

    def test_multi_scale_decoder(self, sample_features):
        """Test MultiScaleFeatureDecoder functionality."""
        decoder = MultiScaleFeatureDecoder(input_channels=[256, 512, 1024], input_scales=[4, 8, 16])
        output = decoder(sample_features)

        assert isinstance(output, list), "All Decoders should return a list"
        assert len(output) == 1, "MultiScaleFeatureDecoder should return a single tensor"
        output = output[0]
        assert output.shape[0] == 2, "Batch size should be preserved"
        assert output.shape[1] == sum([256, 512, 1024]), "Channels should be concatenated"
        assert output.shape[2] == 56, "Output should have largest spatial dimensions"
        assert output.shape[3] == 56, "Output should have largest spatial dimensions"

        decoder = MultiScaleFeatureDecoder(input_channels=[256, 512, 1024], input_scales=[4, 8, 16], output_scales=8)
        output = decoder(sample_features)[0]
        assert output.shape[2] == 28, "Output should have given spatial dimensions"
        assert output.shape[3] == 28, "Output should have given spatial dimensions"

    def test_conv_multi_scale_decoder(self, sample_features):
        """Test ConvolutionalMultiScaleDecoder functionality."""
        output_channels = 256
        decoder = ConvolutionalMultiScaleDecoder(input_channels=[256, 512, 1024], input_scales=[4, 8, 16],
                                                 output_channels=output_channels)
        output = decoder(sample_features)[0]

        # Check output
        assert isinstance(output, torch.Tensor), "Output should be a single tensor"
        assert output.shape[0] == 2, "Batch size should be preserved"
        assert output.shape[1] == output_channels, "Output channels should match configuration"
        assert output.shape[2] == 56, "Output should have largest spatial dimensions"
        assert output.shape[3] == 56, "Output should have largest spatial dimensions"

    def test_invalid_decoder_config(self):
        """Test error handling for invalid decoder configurations."""
        with pytest.raises(ValueError, match="Channel/scale length mismatch"):
            MultiScaleFeatureDecoder(input_channels=[256, 512],  # 2 channels
                                     input_scales=[4, 8, 16]     # 3 scales
                                     )

    @pytest.mark.parametrize('decoder_name', ['identity', 'multi_scale_fusion', 'convolutional_multi_scale_fusion'])
    def test_decoder_builder(self, decoder_name, sample_features):
        """Test decoder builder with different configurations."""
        decoder = build_feature_decoder(name=decoder_name, input_channels=[256, 512, 1024], input_scales=[4, 8, 16],
                                        output_channels=256  # Only used for conv_multi_scale
                                        )
        # Verify decoder type
        expected_types = {'identity': IdentityFeatureDecoder,
                          'multi_scale_fusion': MultiScaleFeatureDecoder,
                          'convolutional_multi_scale_fusion': ConvolutionalMultiScaleDecoder
                          }
        assert isinstance(decoder, expected_types[decoder_name])
        # Test forward pass
        output = decoder(sample_features)
        assert output is not None, "Decoder should produce output"


@pytest.mark.integration
class TestEncoderDecoderIntegration:
    """Integration tests for encoder-decoder pipeline."""

    @pytest.fixture
    def setup_models(self):
        """Setup encoder and decoder for integration tests."""
        encoder = build_feature_extractor(model_name='resnet50', pretrained=False,
                                          features_only=True, out_indices=[1, 2, 3, 4])
        decoder = ConvolutionalMultiScaleDecoder(input_channels=[256, 512, 1024, 2048],
                                                 input_scales=[4, 8, 16, 32],
                                                 output_channels=256)
        return encoder, decoder

    def test_full_pipeline(self, setup_models):
        """Test complete encoder-decoder pipeline."""
        encoder, decoder = setup_models
        input_image = torch.randn(2, 3, 224, 224)
        # Run full pipeline
        features = encoder(input_image)
        output = decoder(features)[0]  # Only one output tensor
        # Verify output
        assert output.shape == (2, 256, 56, 56), "Unexpected output shape"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"

    def test_gradient_flow(self, setup_models):
        """Test gradient flow through encoder-decoder pipeline."""
        encoder, decoder = setup_models

        input_image = torch.randn(2, 3, 224, 224)
        input_image.requires_grad = True

        # Forward pass
        features = encoder(input_image)
        output = decoder(features)
