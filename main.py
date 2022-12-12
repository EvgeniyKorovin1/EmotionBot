
import lock_token as tk

import requests
import telebot
import face_recognition
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt 

from tensorflow import keras
from PIL import Image, ImageDraw


def telegram_bot(token):
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id,"Привет! Я Эмо-Бот! Не то что ты подумал - я в эмоциях разбираюсь! Отправь мне фото и я скажу кто и что там испытывает!")

    @bot.message_handler(content_types=["photo"]) #Функция, которая реагирует на получение фото
    def photo_work(message):

        fileID = message.photo[-1].file_id   
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)
        with open("images\image_scan_faces.jpg", 'wb') as new_file:#фото загрузилось в папку с указанным именем
            new_file.write(downloaded_file)

        #дальше работа с полученной картинкой 
        image = face_recognition.load_image_file("images\image_scan_faces.jpg")
        face_locations = face_recognition.face_locations(image)

        #print(face_locations)
        print(f"Я нашел аж {len(face_locations)} мордашек на твоей фотке!")
        bot.send_message(message.chat.id,"Я обвел все лица которые нашел! Вот они - смотри!")

        pil_image = Image.fromarray(image)
        draw1 = ImageDraw.Draw(pil_image)

        for(top, right, bottom, left) in face_locations:
            draw1.rectangle(((left, top),(right, bottom)), outline = (255,0,0), width= 4)
        del draw1

        pil_image.save("images\image_faces_square.jpg")
        with open('images\image_faces_square.jpg', 'rb') as photo_square:
            bot.send_photo(message.chat.id, photo_square) #Отправляет фото с обводкой в чат

        bot.send_message(message.chat.id,"А теперь каждое в отдельности!")

        count = 1
        for(top, right, bottom, left) in face_locations:
            face_img = image[top:bottom, left:right]
            pil_face_image = Image.fromarray(face_img)
            pil_face_image.save(f"images\image_face{count}.jpg")

            with open(f"images\image_face{count}.jpg", 'rb') as photo_face:
                bot.send_photo(message.chat.id, photo_face) #Отправляет фото головушки в чат

            img = cv2.imread(f"images\image_face{count}.jpg", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48,48))
            cv2.imwrite(f"images_gray\image_face_go{count}.jpg", img)

            img_gr = keras.utils.load_img(f"images_gray\image_face_go{count}.jpg", color_mode='grayscale')
            img_array = keras.utils.img_to_array(img_gr)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array.shape

            predict = model.predict(img_array)

            for i in range(len(predict[0])):
                if predict[0][i] >= 0.51:
                    bot.send_message(message.chat.id,f"Я думаю, что человек на фотографии - {emotions[i]}!")
                    break
            count += 1

    bot.infinity_polling()

if __name__ == '__main__':
    model = keras.models.load_model('S:\PythonProjects\EmotionBot\model\model.h5')
    model.inputs
    emotions = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
    telegram_bot(tk.token)