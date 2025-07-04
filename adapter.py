import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj

# AdapterCLIP作为主网络
class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels

        #前6层的图像残差适配器
        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        
        #异常分割投影层
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        #异常分类投影层
        det_proj = SimpleProj(1024, 768, relu)
        
        #图像投影和适配器词典
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )
        
        #前3层的文本残差适配器和文本投影层
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )
        self._init_weights_()

    #权重初始化
    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #原始CLIP的图像前向传播
    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    #加入适配器的CLIP的图像前向传播
    def forward(self, x):
        # print("x_input_size",x.shape) #2x3x518x518 (batch_size=2,channel=3.height=518.width=518)BCHW
        #将图片使用conv1变成token
        x = self.image_encoder.conv1(x)
        # print("x_tokenized_size",x.shape) #2x1024x37x37 (batch_size=2,channel=1024,height=37,width=37)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print("x_reshaped_size",x.shape) #2x1024x1369  (batch_size=2,channel=1024,seq_len=1369) BCL
        x = x.permute(0, 2, 1)
        # print("x_permuted1_size",x.shape) #1x1369x1024
        
        
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        
        # print("x_position_size",x.shape) #2x1370x1024

        x = x.permute(1, 0, 2)
        # print("x_permuted2_size",x.shape) #1370x2x1024

        tokens = []
        for i in range(24):
            #经过CLIP的全部层进行特征提取
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            # print("x_transformer_resbolock",x.shape) #1370x2x1024
            #前6层进行残差适配
            if i < self.image_adapt_until:
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                #i_w作为残差参数，适配器残差连接
                x = self.i_w * adapt_out + (1 - self.i_w) * x
                # print("x_resbolock",x.shape) #1370x2x1024
            if i + 1 in self.levels:
                #如果是6,12,18,24层，则输出图像特征
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        # print("x_permuted3_size",x.shape) #2x1369x1024
        
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        seg_features = [F.normalize(t, dim=-1) for t in seg_tokens]
        det_feature = self.image_adapter["det_proj"](tokens[-1])
        # det_token = F.normalize(det_feature, dim=-1).mean(1)
        # print("seg_features.shape",seg_featuress[0].shape) #批量x1369x768
        # print(" det_feature.shape", det_feature.shape) #批量x768
        return seg_features, det_feature

    #加入适配器的CLIP的文本前向传播
    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype 
        )  # [batch_size, n_ctx, d_model]
        # print("text_x_input",x.shape) #6x77x768 77表示文本最大长度
        
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print("text_x_permuted1",x.shape) #77x6x768
        
        
        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print("text_x_permuted2",x.shape) #6x77x768
        
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # print("text_x_ln_final",x.shape) #6x77x768
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        # x = (
            # x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            # @ self.clipmodel.text_projection
        # )
        # print("text_x_output",x.shape) #10x768
        
        return x