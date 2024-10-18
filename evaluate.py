# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.bio_schema = self.valid_data.dataset.bio_schema
        self.attribute_schema = self.valid_data.dataset.attribute_schema
        self.text_data = self.valid_data.dataset.text_data
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"object_acc":0, "attribute_acc": 0, "value_acc": 0, "full_match_acc":0}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            text_data = self.text_data[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id = batch_data[0]
            with torch.no_grad():
                attribute_pred, bio_pred = self.model(input_id) #不输入labels，使用模型当前参数进行预测
                self.write_stats(attribute_pred, bio_pred, text_data)
        self.show_stats()
        return

    def write_stats(self, attribute_pred, bio_pred, text_data):
        attribute_pred_labels = torch.argmax(attribute_pred, dim=-1)
        bio_pred_labels = torch.argmax(bio_pred, dim=-1)
        for attribute_p, bio_p, info in zip(attribute_pred_labels, bio_pred_labels, text_data):
            context, object, attribute, value = info
            bio_p = bio_p.cpu().detach().tolist()
            pred_object, pred_value = self.decode(bio_p, context)
            pred_attribute = self.index_to_label[int(attribute_p)]
            self.stats_dict["object_acc"] += int(pred_object == object)
            self.stats_dict["attribute_acc"] += int(pred_attribute == attribute)
            self.stats_dict["value_acc"] += int(pred_value == value)
            if pred_value == value and pred_attribute == attribute and pred_object == object:
                self.stats_dict["full_match_acc"] += 1
        return
    
    def decode(self, pred_label, context):
        """
        将预测的标签转化为实体的字符串，用于与原先的实体进行比较
        """
        pred_label = "".join([str(i) for i in pred_label])  # 把预测出来的label整合成字符串
        pred_obj = self.seek_pattern("01*", pred_label, context)
        pred_value = self.seek_pattern("23*", pred_label, context)
        return pred_obj, pred_value


    def seek_pattern(self, pattern, pred_label, context):
        """
        使用 span() 方法有助于处理和操作字符串数据，尤其是在文本解析和数据提取的场景中。
        """
        pred_entity = re.search(pattern, pred_label)
        if pred_entity:
            start, end = pred_entity.span()  # 输出匹配的起始和结束位置
            pred_entity = context[start:end]
        else:
            pred_entity = ""
        return pred_entity


    # 打印结果
    def show_stats(self):
        for key, value in self.stats_dict.items():
            self.logger.info("%s : %s " %(key, value / len(self.text_data)))
        self.logger.info("--------------------")
        return

