import os.path
import json

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
                 ratio_images_to_use=1,
                 selected_answers=None,
                 sequence_length=40,
                 transform=None,
                 label=None):

        self.train = train
        self.imgFolder = imgFolder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.Dataset = Dataset
        self.freq_dict = get_freq_dic()
        self.transform = transform

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

        # select only the active images
        images = [img['id'] for img in imagesJSON['images'] if img['active']]

        # select the requested amount of images
        images = images[:int(len(images) * ratio_images_to_use)]
        self.img_ids = images
        if self.Dataset == 'LR':
            self.images = np.empty((len(images), 256, 256, 3))
            # self.images = np.empty((len(images), 512, 512, 3))
        else:
            # self.images = np.empty((len(images), 256, 256, 3))
            self.images = np.empty((len(images), 512, 512, 3))

        print("Construction of the Dataset")

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

                answer = self.freq_dict.index(answer_str)
                if label is not None:
                    if type_str == label:
                        self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])
                else:
                    self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])

        print("Done.")

    def __len__(self):
        # return the number of image-question-answer pairs, which are selected
        return len(self.images_questions_answers)

    def __getitem__(self, idx):
        # load the features of the index
        data = self.images_questions_answers[idx]

        language_feats = self.tokenizer(data[0], return_tensors='pt', padding='max_length',
                                        max_length=self.sequence_length)
        img = self.images[data[2], :, :, :]

        if self.transform is not None:
            img_unaugment = Image.fromarray(np.uint8(img)).convert('RGB')
            img_augment = self.transform(img_unaugment)
            imgT = self.image_processor(img_augment, return_tensors="pt")
        else:
            imgT = self.image_processor(img, return_tensors="pt")
        '''
    pixel_values, input_ids, token_type_ids, attention_mask, answer = batch
    '''
        labels = self.tokenizer(data[4], padding='max_length', truncation=True, max_length=9, return_tensors="pt")
        label = labels['input_ids']
        label = label.squeeze(0)
        label_attention_mask = labels['attention_mask']
        if self.train:
            return {
                "pixel_values": imgT['pixel_values'][0],  # 图像的特征
                "input_ids": language_feats['input_ids'][0],  # 语言特征
                "attention_mask": language_feats['attention_mask'][0],  # 语言的attention mask
                "question_type": data[3],  # 问题类型
                # "token_type_ids": language_feats['token_type_ids'][0],
                "answer": data[1],  # 可选：文本答案grop的下标
                "question_type": data[3],
                "img_id": data[2],
                "question": data[0],
                "answer_str": data[-1],
                "labels": label,
                "label_attention_mask": label_attention_mask,

            }
            # return imgT['pixel_values'][0], language_feats['input_ids'][0], language_feats['token_type_ids'][0], \
            #        language_feats['attention_mask'][0], data[1]
        else:
            return {
                "pixel_values": imgT['pixel_values'][0],  # 图像的特征
                "input_ids": language_feats['input_ids'][0],  # 语言特征
                "attention_mask": language_feats['attention_mask'][0],  # 语言的attention mask
                "question_type": data[3],  # 问题类型
                # "token_type_ids": language_feats['token_type_ids'][0],
                "answer": data[1],  # 可选：文本答案grop的下标
                "question_type": data[3],
                "img_id": data[2],
                "question": data[0],
                "answer_str": data[-1],
                "labels": label,
                "label_attention_mask": label_attention_mask,

            }
            # return imgT['pixel_values'][0], language_feats['input_ids'][0], language_feats['token_type_ids'][0], \
            #        language_feats['attention_mask'][0], data[1], \
            #        data[3], data[2], data[0], data[-1]


def get_freq_dic():
    data_path = '/home/pengfei/DataSet/RSVQA_HR'
    questions_file = os.path.join(data_path, 'USGSquestions.json')
    answers_file = os.path.join(data_path, 'USGSanswers.json')
    images_file = os.path.join(data_path, 'USGSimages.json')

    print("Get freq_dic...")
    with open(questions_file) as json_data:
        questionsJSON = json.load(json_data)

    with open(answers_file) as json_data:
        answersJSON = json.load(json_data)

    with open(images_file) as json_data:
        imagesJSON = json.load(json_data)

    images = [img['id'] for img in imagesJSON['images']]

    freq_dict = {}

    for i, image in enumerate(tqdm(images)):

        # select the questionids, aligned to the image
        for questionid in imagesJSON['images'][image]['questions_ids']:

            # select question with the id
            question = questionsJSON['questions'][questionid]

            # get the answer str with the answer id from the question
            answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']
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
            if answer_str not in freq_dict:
                freq_dict[answer_str] = 1
            else:
                freq_dict[answer_str] += 1

    print("Done.")

    list_ip = []

    for j in range(len(freq_dict)):
        list_ip = list(freq_dict.keys())

    return list_ip
