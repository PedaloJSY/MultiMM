import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import ResNetModel
from transformers import AutoImageProcessor, ResNetForImageClassification, ViTImageProcessor, ViTModel
from transformers import BertModel, BertPreTrainedModel, BertLayer, XLMRobertaModel
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet50
from transformers import logging

logging.set_verbosity_warning()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class MultimodalModel(torch.nn.Module):
    def __init__(self, TextModel, ImageModel):
        torch.nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.vit = ViTModel.from_pretrained('./vit')

        self.image_pool = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.3),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.Tanh()
        )

        self.classifier_text = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

        self.classifier_image = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

        self.classifier_all = nn.Linear(in_features=768 * 3, out_features=1)


    def forward(self, image_input = None, text_input = None, senti_input = None):
        if (image_input is not None) and (text_input is not None) and (senti_input is not None):

            """提取文本特征"""
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state[:, 0, :]

            """提取情感特征"""
            senti_features = self.bert(**senti_input)
            senti_hidden_state = senti_features.last_hidden_state[:, 0, :]

            """提取图像特征"""
            image_features = self.vit(**image_input).last_hidden_state
            image_hidden_state, _ = image_features.max(1)

            """拼接文本和图像，拼接得到共同特征"""
            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state, senti_hidden_state], 1)

            """利用拼接向量进行分类"""
            out = self.classifier_all(image_text_hidden_state).squeeze(1)
            out = torch.sigmoid(out)
            return out
        
        elif image_input is None:
            """text only"""
            assert(text_input is not None)

            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state[:, 0, :]

            # out = self.classifier_single(text_hidden_state).squeeze(1)
            out = self.classifier_text(text_hidden_state).squeeze(1)
            out = torch.sigmoid(out)
            return out


        elif text_input is None:
            """image only"""
            assert(image_input is not None)

            image_features = self.vit(**image_input).last_hidden_state

            image_pooled_output, _ = image_features.max(1)

            out = self.classifier_image(image_pooled_output).squeeze(1)
            out = torch.sigmoid(out)
            return out
