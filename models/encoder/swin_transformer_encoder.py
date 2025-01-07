import numpy as np 

import torch 
import torch.nn as nn
from transformers import  AutoModelForImageClassification



class SwinTransformerMultiImageInput(nn.Module):
    def __init__(self, model_name='naver-ai/swin_rope_mixed_base_patch4_window7_224', pretrained=False, num_input_images=1):
        super(SwinTransformerMultiImageInput, self).__init__()
        # self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

        # Modify the first convolution layer to accept multiple input images
        self.model.swin.embeddings.patch_embeddings.projection = nn.Conv2d(
            num_input_images * 3, self.model.swin.embeddings.patch_embeddings.projection.out_channels,
            kernel_size=self.model.swin.embeddings.patch_embeddings.projection.kernel_size,
            stride=self.model.swin.embeddings.patch_embeddings.projection.stride,
            padding=self.model.swin.embeddings.patch_embeddings.projection.padding,
            bias=self.model.swin.embeddings.patch_embeddings.projection.bias is not None
        )

        self.encoder0 = self.model.swin.encoder.layers[0]
        self.encoder1 = self.model.swin.encoder.layers[1]
        self.encoder2 = self.model.swin.encoder.layers[2]
        self.encoder3 = self.model.swin.encoder.layers[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print(f'Conv2d: {m.weight.shape}')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                print(f'LayerNorm: {m.weight.shape}')
                
        if pretrained:
            loaded = self.model.state_dict()
            loaded['swin.embeddings.patch_embeddings.projection.weight'] = torch.cat(
                [loaded['swin.embeddings.patch_embeddings.projection.weight']] * num_input_images, 1) / num_input_images
            self.model.load_state_dict(loaded)
            
        
def swin_transformer_encoder(model_name='naver-ai/swin_rope_mixed_base_patch4_window7_224', pretrained=True, num_input_images=1):
    """Constructs a Swin Transformer model.
    Args:
        model_name (str): Name of the Swin Transformer model.
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        num_input_images (int): Number of frames stacked as input.
    """
    model = SwinTransformerMultiImageInput(model_name=model_name, pretrained=pretrained, num_input_images=num_input_images)
    return model

class RoPE_SwinViT_Encoder(nn.Module) : 
    def __init__(self, model_name='naver-ai/swin_rope_mixed_base_patch4_window7_224', pretrained=False, num_input_images=1) : 
        super(RoPE_SwinViT_Encoder, self).__init__()

        self.num_ch_enc = np.array([96,192,384,768,768])

        if num_input_images > 1:
            self.encoder = swin_transformer_encoder(model_name=model_name, pretrained=pretrained, num_input_images=num_input_images)
        else : 
            self.encoder = AutoModelForImageClassification.from_pretrained(model_name)
    
    def forward(self, input_image) : # input_image : [B, 4, 320,448]
        encoder = self.encoder
        features = []
        x = (input_image - 0.45) / 0.225

        features.append(encoder.swin.embeddings(x))
        features.append(encoder.swin.encoder.layers[0](features[-1][0],features[-1][1]))
        features.append(encoder.swin.encoder.layers[1](features[-1][0],features[-1][2][2:]))
        features.append(encoder.swin.encoder.layers[2](features[-1][0],features[-1][2][2:]))
        features.append(encoder.swin.encoder.layers[3](features[-1][0],features[-1][2][2:]))
        
        return features 