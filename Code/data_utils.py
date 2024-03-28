import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
import re
import csv
import json
from functools import partial
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, ViTImageProcessor, ViTModel
from transformers import BertTokenizer, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
ImageFile.LOAD_TRUNCATED_IMAGES = True

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def get_image(data_path, language, image_name):
    if language == 'en':
        file_path_image = os.path.join(data_path, 'imgs_EN_ads', image_name)
    else:
        file_path_image = os.path.join(data_path, 'imgs_ZH_ads', image_name)

    return file_path_image


def get_data_list(data_path, language) -> (list, list):
    """
    读取训练和测试数据，分别返回训练集和测试集
    """

    train_data_list = []
    val_data_list = []
    test_data_list = []

    if language == 'en':
        train_label_path = os.path.join(data_path, 'EN_train.csv')
        val_label_path = os.path.join(data_path, 'EN_dev.csv')
        test_label_path = os.path.join(data_path, 'EN_test.csv')
        with open(os.path.join(data_path, 'EN_Senti.json'), 'r') as jsonfile:
            senti = json.load(jsonfile)
        encoding_type = 'utf-8'
        null_text = "text is empty"
    elif language == 'zh':
        train_label_path = os.path.join(data_path, 'ZH_train.csv')
        val_label_path = os.path.join(data_path, 'ZH_dev.csv')
        test_label_path = os.path.join(data_path, 'ZH_test.csv')
        with open(os.path.join(data_path, 'ZH_Senti.json'), 'r') as jsonfile:
            senti = json.load(jsonfile)
        encoding_type = 'gbk'
        null_text = "文本为空"


    for file in ['train', 'val', 'test']:
        if file == 'train':
            label_path = train_label_path
        elif file == 'val':
            label_path = val_label_path
        elif file == 'test':
            label_path = test_label_path

        with open(label_path, 'r', encoding=encoding_type) as f:
            f.readline()  
            f_csv = csv.reader(f)
            for row in f_csv:
                data_dict = {}
                data_dict['guid'] = str(row[1])
                if row[2]:
                    data_dict['text'] = row[2]
                else:
                    data_dict['text'] = null_text
                if language == 'en':
                    data_dict['senti'] = "the sentiment is " + senti[str(row[1])]['senti']
                else:
                    data_dict['senti'] = "情感为" + senti[str(row[1])]['senti']

                data_dict['tag'] = int(row[3])

                data_dict['image'] = get_image(data_path, language, str(row[1]))

                if file == 'train':
                    train_data_list.append(data_dict)
                elif file == 'val':
                    val_data_list.append(data_dict)
                elif file == 'test':
                    test_data_list.append(data_dict)

    return train_data_list, val_data_list, test_data_list

def clean_text(text: bytes):
    try:
        decode = text.decode(encoding='utf-8')
    except:
        try:
            decode = text.decode(encoding='GBK')
        except:
            try:
                decode = text.decode(encoding='gb18030')
            except:
                decode = str(text)
    return decode


def data_preprocess(train_data_list, val_data_list, test_data_list):
    """
    数据预处理，清洗文本数据
    """
    for data in train_data_list:
        data['text'] = clean_text(data['text'])

    for data in val_data_list:
        data['text'] = clean_text(data['text'])

    for data in test_data_list:
        data['text'] = clean_text(data['text'])

    return train_data_list, val_data_list, test_data_list

def collate_fn(data_list):
    guid = [data['guid'] for data in data_list]
    tag = [data['tag'] for data in data_list]
    image = [Image.open(data['image']).convert('RGB') for data in data_list]
    image = feature_extractor(image, return_tensors="pt")
    text = [data['text'] for data in data_list]
    text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=30)
    senti = [data['senti'] for data in data_list]
    senti = tokenizer(senti, return_tensors="pt", padding=True, truncation=True, max_length=10)

    return guid, torch.FloatTensor(tag), image, text, senti


def get_data_loader(train_data_list, val_data_list, test_data_list) -> (DataLoader, DataLoader, DataLoader):

    train_data_loader = DataLoader(
        dataset=train_data_list,
        collate_fn=collate_fn,
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )

    valid_data_loader = DataLoader(
        dataset=val_data_list,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    test_data_loader = DataLoader(
        dataset=test_data_list,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    return train_data_loader, valid_data_loader, test_data_loader


def calc_metrics(target, pred):

    accuracy = accuracy_score(target, pred)
    precision_w = precision_score(target, pred, average='weighted')
    recall_w = recall_score(target, pred, average='weighted')
    f1_w = f1_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return accuracy, precision_w, recall_w, f1_w, precision, recall, f1


def calc_metrics_binary(target, pred):
    """
    计算评估指标， 分别为准确率、 精确率、 召回率、 F1-score
    """
    accuracy = accuracy_score(target, pred)
    # binary
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    # weighted
    weight_precision = precision_score(target, pred, average='weighted')
    weight_recall = recall_score(target, pred, average='weighted')
    weight_f1 = f1_score(target, pred, average='weighted')

    return accuracy, precision, recall, f1, weight_precision, weight_recall, weight_f1

