import os.path
import json
import random

from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
from PIL import Image
import numpy as np

class VQALoader(Dataset):
    """
    This class manages the Dataloading.
    """

    def __init__(self,
                 imgFolder,
                 images_file,
                 questions_file,
                 answers_file,
                 tokenizer,
                 image_processor,
                 Dataset,
                 train=True,
                 ratio_images_to_use=0.1,
                 selected_answers=None,
                 sequence_length=40,
                 transform=None,
                 label=None):

        self.train = train
        self.imgFolder = imgFolder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        LR_number_outputs = 9
        HR_number_outputs = 94
        self.Dataset = Dataset
        self.transform = transform
        self.dict = {}

        # sequence length of the tokens
        self.sequence_length = sequence_length

        # loading the json files for the question, answers and images
        print("Loading JSONs...")
        with open(questions_file) as json_data:
            questionsJSON = json.load(json_data)

        with open(answers_file) as json_data:
            answersJSON = json.load(json_data)

        with open(images_file) as json_data:
            imagesJSON = json.load(json_data)
        print("Done.")

        # todo:活动的图片
        # select only the active images
        images = [img['id'] for img in imagesJSON['images'] if img['active']]

        # select the requested amount of images
        # images = images[:int(len(images) * ratio_images_to_use)]
        self.img_ids = images
        if self.Dataset == 'LR':
            self.images = np.empty((len(images), 256, 256, 3))
        else:
            self.images = np.empty((len(images), 512, 512, 3))

        print("Construction of the Dataset")
        print("+++++++++++++++++++++++++++++++++++++train", train)
        # when training we construct the answer set
        if train:
            print("--------train----------------")
            # this dict will store as keys the answers and the values are the frequencies they occur 0 yes ,1 no ,2 0...
            self.freq_dict = {}
            all_anser = {}
            for i, image in enumerate(tqdm(images)):

                # select the questionids, aligned to the image
                for questionid in imagesJSON['images'][image]['questions_ids']:

                    # select question with the id
                    question = questionsJSON['questions'][questionid]

                    # get the answer str with the answer id from the question
                    answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']
                    if answer_str not in all_anser:
                        all_anser[answer_str] = 1
                    # group the counting answers
                    if self.Dataset == 'LR':
                        if answer_str.isdigit():
                            num = int(answer_str)
                            if num > 0 and num <= 10:
                                answer_str = "between 0 and 10"
                            if num > 10 and num <= 100:
                                answer_str = "between 10 and 100"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100 and 1000"
                            if num > 1000:
                                answer_str = "more than 1000"
                    else:
                        if 'm2' in answer_str:
                            num = int(answer_str[:-2])
                            if num > 0 and num <= 10:
                                answer_str = "between 0m2 and 10m2"
                            if num > 10 and num <= 100:
                                answer_str = "between 10m2 and 100m2"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100m2 and 1000m2"
                            if num > 1000:
                                answer_str = "more than 1000m2"

                    # update the dictionary
                    if answer_str not in self.freq_dict:
                        self.freq_dict[answer_str] = 1
                    else:
                        self.freq_dict[answer_str] += 1

            # sort the dictionary by the most common
            # so that position 0 contains the most frequent word
            self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

            self.selected_answers = []
            self.non_selected_answers = []

            # store the total number of selected answers
            coverage = 0

            # store the total number of answers
            total_answers = 0

            # this loop creates the list with selected answers
            # with respect to the number of outputs
            for i, key in enumerate(self.freq_dict):

                # select the answer if not have enough
                if self.Dataset == 'LR':
                    if i < LR_number_outputs:

                        # append the answer string
                        self.selected_answers.append(key[0])

                        # add the frequency of the answer
                        coverage += key[1]

                    # if we have enough answers we append them to non_selected_answers
                    else:
                        self.non_selected_answers.append(key[0])

                    # add all frequencies occuring
                    total_answers += key[1]
                else:
                    if i < HR_number_outputs:

                        # append the answer string
                        self.selected_answers.append(key[0])

                        # add the frequency of the answer
                        coverage += key[1]

                    # if we have enough answers we append them to non_selected_answers
                    else:
                        self.non_selected_answers.append(key[0])

                    # add all frequencies occuring
                    total_answers += key[1]
            print(self.selected_answers)
            # print(all_anser)

        else:
            '''['yes', 'no', 'between 10 and 100', 'between 0 and 10', '0', 'between 100 and 1000', 'more than 1000', 'rural', 'urban']'''
            '''  main ['yes', 'no', 'between 0 and 10', '0', 'between 10 and 100', 'between 100 and 1000', 'more than 1000', 'rural', 'urban']'''
            print("-------------else ----")

            select_answer = ['yes', 'no', 'between 0 and 10', '0', 'between 10 and 100', 'between 100 and 1000',
                             'more than 1000', 'rural', 'urban']

            # HR = select_answer=['no', 'yes', '0', '0m2', '1', 'more than 1000m2', '2', 'between 100m2 and 1000m2', '3', '4', '5', 'between 10m2 and 100m2', '6', '7', '8', '9', '10', '11', '12', '13', '14', 'between 0m2 and 10m2', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '32', '31', '33', '35', '36', '34', '41', '40', '37', '38', '43', '39', '47', '56', '44', '42', '49']
            # HR = select_answer=['no', 'yes', '0', '0m2', '1', 'more than 1000m2', '2', 'between 100m2 and 1000m2', '3', '4', '5', 'between 10m2 and 100m2', '6', '7', '8', '9', '10', '11', '12', '13', '14', 'between 0m2 and 10m2', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '32', '31', '33', '35', '36', '34', '41', '40', '37', '38', '43', '39', '47', '56', '44', '42', '49']
            self.selected_answers = select_answer

        # list for storing the image-question-answer pairs
        self.images_questions_answers = []

        # we go through all img ids
        for i, image in enumerate(tqdm(images)):

            img = io.imread(os.path.join(imgFolder, str(image) + '.tif'))
            self.images[i, :, :, :] = img

            # use img id to get the question id for corresponding to the img
            for questionid in imagesJSON['images'][image]['questions_ids']:

                # question id gives the dict of the question
                question = questionsJSON['questions'][questionid]

                # get the question str and the question type (e.g. Y/N)
                question_str = question["question"]
                type_str = question["type"]

                # get the answer str with the answer id from the question
                answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']

                # group the counting answers
                if self.Dataset == 'LR':
                    if answer_str.isdigit():
                        num = int(answer_str)
                        if num > 0 and num <= 10:
                            answer_str = "between 0 and 10"
                        if num > 10 and num <= 100:
                            answer_str = "between 10 and 100"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100 and 1000"
                        if num > 1000:
                            answer_str = "more than 1000"
                else:
                    if 'm2' in answer_str:
                        num = int(answer_str[:-2])
                        if num > 0 and num <= 10:
                            answer_str = "between 0m2 and 10m2"
                        if num > 10 and num <= 100:
                            answer_str = "between 10m2 and 100m2"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100m2 and 1000m2"
                        if num > 1000:
                            answer_str = "more than 1000m2"

                answer = self.selected_answers.index(answer_str)


                if label is not None:
                    if type_str == label:
                        self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])
                else:
                    self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])

        print("Done.")

    def __len__(self):
        # return the number of image-question-answer pairs, which are selected
        return len(self.images_questions_answers)

    '''-------------------问答逻辑，将答案归类为 0-8个选项，分别对应----------------------------'''

    '''---------------------------------------------------------------------------------------'''

    def __getitem__(self, idx):

        # 加载当前索引的数据   data ['what is ...',answer_id:0,  i :382,answer_type:'comp',answer:'yes']    between
        data = self.images_questions_answers[idx]
        # print(data)
        # 处理语言特征 生成： input_ids(1,40),attention_mask(1,40)

        language_feats = self.tokenizer(
            data[0],
            return_tensors='pt',
            padding='max_length',
            max_length=self.sequence_length
        )
        # todo : 这里用来加提示

        new_question = data[0] + ' ' + data[3]

        tishi_language_feats = self.tokenizer(
            new_question,
            return_tensors='pt',
            padding='max_length',
            max_length=self.sequence_length

        )
        # print(new_question)
        # print(tishi_language_feats)
        # todo:-------------------------------------------------------------------

        # 处理图像数据 w,h,c 256 256 3
        img = self.images[data[2], :, :, :]
        # if img.dtype != np.uint8:
        #     img = (img * 255).astype(np.uint8)  # 转换为 uint8 类型
        #
        # image = Image.fromarray(img)
        # image.save("/home/pengfei/mamba/assets/demo.tif")  # 保存为 TIFF 格式
        # print(self.images[data[2]])
        # exit()

        # todo:--------------------no_enhance--------------------------
        #
        if self.train :
            if random.random() < .5:
                img = np.flip(img, axis=0)
            if random.random() < .5:
                img = np.flip(img, axis=1)
            if random.random() < .5:
                img = np.rot90(img, k=1)
            if random.random() < .5:
                img = np.rot90(img, k=3)

        # todo:----------------------------------------------

        # image = Image.fromarray(img)

        # 使用图像处理器处理图像
        # imgT = self.image_processor.image_processor(images=img, return_tensors="pt")

        # todo:--------------------------------------------

        # imgT = self.image_processor(img, return_tensors="pt")
        # img 为256*256*3 格式化
        if self.transform is not None:
            img_unaugment = Image.fromarray(np.uint8(img)).convert('RGB')
            img_augment = self.transform(img_unaugment)
            imgT = self.image_processor(img_augment, return_tensors="pt", do_resize=True)
        else:
            imgT = self.image_processor(img, return_tensors="pt")
        # todo:--------------------------------------------

        # 获取答案索引
        answer_idx = data[1]

        # todo :这里
        # 构造labels（二维张量）
        # 在训练模式下返回标签
        # print(data[4])
        # labels = self.tokenizer.encode(data[4], add_special_tokens=True)
        data4 = data[4]
        question_type_text = data[3]
        # todo:--------------promote-01-------------------
        # 构建一个结构化的提示
        # prompt_text = f"Question type is {question_type_text}"  # 例如 "Question type: quantity"
        # select_answer = ['yes', 'no', 'between 0 and 10', '0', 'between 10 and 100', 'between 100 and 1000',
        #                               'more than 1000', 'rural', 'urban']
        # todo:--------------promote-02-------------------
        if data[1] < 2:
            # type_pro = data[0]
            type_pro = data[3]+' answer yes or no?'
        elif data[1] > 1 and data[1] < 7:
            type_pro = data[3]+' answer between 0 and 10 ,  0 ,  between 10 and 100 ,  between 100 and 1000 more than 1000 ?'
            # type_pro = ' answer 2,3,4,5,6'
        else:
            type_pro = data[3]+" answer rural or urban ?"
        # todo:--------------promote--------------------

        labels = self.tokenizer(data[4], padding='max_length', truncation=True, max_length=9, return_tensors="pt")
        # labels = self.tokenizer(data[3], padding='max_length', truncation=True, max_length=9, return_tensors="pt")
        # print(labels)

        # if self.train:
        #     labels = torch.tensor([answer_idx] * self.sequence_length)  # 将answer_idx重复填充到sequence_length长度
        #     labels = labels.unsqueeze(0)  # 添加batch维度，变为 (1, sequence_length)
        # else:
        #     labels = torch.tensor([answer_idx] * self.sequence_length)  # 对于推理时，也是如此
        #     labels = labels.unsqueeze(0)
        label = labels['input_ids']
        label = label.squeeze(0)
        label_attention_mask = labels['attention_mask']
        # TODO:labels有问题：！！！！
        # 返回的字典
        if self.train:
            return {
                "pixel_values": imgT['pixel_values'][0],  # 图像的特征
                "input_ids": language_feats['input_ids'][0],  # 语言特征
                "attention_mask": language_feats['attention_mask'][0],  # 语言的attention mask
                "labels": label,
                "label_attention_mask": label_attention_mask,
                "question_type": data[3],  # 问题类型
                "answer": data[1],  # 可选：文本答案grop的下标
                "tishi_language_feats": tishi_language_feats['input_ids'][0],
                "tishi_attention_mask": tishi_language_feats['attention_mask'][0],
            }
        else:
            #  pixel_values, input_ids, token_type_ids, attention_mask, answer, question_type, img_id, question, answer_str
            return {
                "pixel_values": imgT['pixel_values'][0],  # 图像的特征
                "input_ids": language_feats['input_ids'][0],  # 语言特征
                "attention_mask": language_feats['attention_mask'][0],  # 语言的 attention mask
                "labels": label,  # 答案的标签
                "image_id": data[2],  # 图像 ID
                "question": data[0],  # 问题文本
                "answer_str": data[4],  # 答案字符串
                "question_type": data[3],  # 问题类型
                "answer": data[1],  # 可选：文本答案（字符串）
                "label_attention_mask": label_attention_mask,
                "tishi_language_feats": tishi_language_feats['input_ids'][0],
                "tishi_attention_mask": tishi_language_feats['attention_mask'][0],

            }
# [8, 0, 5, 0, 1, 1, 0, 1, 0, 4, 5, 6, 3, 6, 4, 0, 0, 0, 0, 0, 1, 0, 1, 4,
#         0, 1, 1, 1, 0, 2, 1, 4, 4, 0, 6, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3,
#         1, 1, 4, 2, 2, 0, 3, 1, 0, 1, 5, 3, 0, 0, 1, 0]
