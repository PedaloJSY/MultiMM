import os
import re
import csv
import json
import pandas as pd
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "./llava"
data_path = './data'


def get_image(data_path, language, image_name):
    if language == 'en':
        file_path_image = os.path.join(data_path, 'imgs_EN_ads', image_name)
    elif language == 'zh':
        file_path_image = os.path.join(data_path, 'imgs_ZH_ads', image_name)
    return file_path_image


for language in ['en', 'zh']:
    if language == 'en':
        data_path = os.path.join(data_path, 'EN.csv')
        null_text = "text is empty"
        encoding_type = 'utf-8'
    elif language == 'zh':
        data_path = os.path.join(data_path, 'ZH.csv')
        encoding_type = 'gbk'
        null_text = "文本为空"

    with open(data_path, 'r', encoding=encoding_type) as f:
            f.readline()  # 跳过首行
            f_csv = csv.reader(f)
            if language == 'en':
                data = {}
                count = 0
                for row in f_csv:
                    count += 1
                    if count > 10:
                        break
                    image_id = str(row[1])
                    if row[2]:
                        text = re.sub(r'[0-9]|[^a-zA-Z\s]', '', row[2])
                    else:
                        text = null_text
                    image_path = get_image(data_path, language, image_id)
                    args = type('Args', (), {
                        "model_path": model_path,
                        "model_base": None,
                        "model_name": get_model_name_from_path(model_path),
                        "query": f"Text:{text} Please analyze the sentiment of this text and image.(just return positive, neutral, or negative)",
                        "conv_mode": None,
                        "image_file": f"{image_path}",
                        "sep": ",",
                        "temperature": 0.2,
                        "top_p": None,
                        "num_beams": 1,
                        "max_new_tokens": 100,
                    })()
                    senti = eval_model(args)
                    data[image_id] = {"senti":senti}
                    print(image_id, senti)
                with open(os.path.join(data_path, "EN_Senti.json"), "w", encoding=encoding_type) as f:
                    json.dump(data, f, indent=4)
            else:
                data = {}
                count = 0
                for row in f_csv[:10]:
                    count += 1
                    if count > 10:
                        break
                    image_id = str(row[1])
                    if row[2]:
                        text = row[2]
                    else:
                        text = null_text
                    image_path = get_image(data_path, language, image_id)
                    args = type('Args', (), {
                        "model_path": model_path,
                        "model_base": None,
                        "model_name": get_model_name_from_path(model_path),
                        "query": f"Text:{text} 请分析这段文本和图像的情感倾向.(只用回答积极、中性或消极)",
                        "conv_mode": None,
                        "image_file": f"{image_path}",
                        "sep": ",",
                        "temperature": 0.2,
                        "top_p": None,
                        "num_beams": 1,
                        "max_new_tokens": 100,
                    })()
                    senti = eval_model(args)
                    data[image_id] = {"senti": senti}
                with open(os.path.join(data_path, "ZH_Senti.json"), "w", encoding=encoding_type) as f:
                    json.dump(data, f, indent=4)

