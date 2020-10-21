# -*- coding: UTF-8 -*-
import sys
import cv2
import os
import configparser
import numpy as np
from PIL import Image

# 引数を取得
args = sys.argv
id = args[1]

# 設定ファイル読み込み
inifile = configparser.ConfigParser()
inifile.read('./config.ini', 'UTF-8')

# 学習画像データ枚数取得変数初期化
sample_cnt = 0

# 学習画像データ保存領域パス情報
learnPath = inifile.get('file-dir', 'learnPath').format(args[1])
# 学習用した結果を.ymlファイルの保存先
ymlPath = inifile.get('file-dir', 'ymlPath').format(args[1])
# 元画像格納先
imgPath = inifile.get('file-dir', 'imgPath').format(args[1])

#######################################################################
# 顔検出を認識する　カスケードファイルは「haarcascade_frontalface_alt2.xml」 #
#######################################################################
face_detector = cv2.CascadeClassifier(inifile.get('file-dir', 'cascadeFace'))

#######################################################
# 学習画像用データから顔認証データymlファイル作成するメソッド  #
#######################################################
def image_learning_make_Labels():

    # リスト保存用変数
    face_list=[]
    ids_list=[]

    # Local Binary Patterns Histogram(LBPH)アルゴリズム　インスタンス
    recognizer = cv2.face_LBPHFaceRecognizer.create()

    # 学習画像ファイルパスを全て取得
    imagePaths = [os.path.join(learnPath,f) for f in os.listdir(learnPath)]

    # 学習画像ファイル分ループ
    for imagePath in imagePaths:
        # グレースケールに変換
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        # UseriDが入っているファイル名からUserID番号として取得
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # 物体認識（顔認識）の実行
        faces = face_detector.detectMultiScale(img_numpy)

        # 認識した顔認識情報を保存
        for (x,y,w,h) in faces:
            face_list.append(img_numpy[y:y+h,x:x+w])
            ids_list.append(id)

    print ("\n Training Start ...")
    ##############################
    # 学習スタート                 #
    ##############################
    recognizer.train(face_list, np.array(ids_list))

    #####################################
    # 学習用した結果を.ymlファイルに保存する  #
    #####################################
    recognizer.save(ymlPath + "/trainer.yml")

    # 学習した顔種類を標準出力
    print("\n User {0} trained. Program end".format(len(np.unique(ids_list))))

#####################################
# ディレクトリがなかったら作るメソッド
#####################################
def dir_check(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)

#####################################
# 顔認証したい人物の通し番号を入力させる
#####################################
User_id = args[1]
print("\n Learn Image Get Start ............")

####################################
#  学習用画像データ取得と保存
####################################
# 各ディレクトリ作成
dir_check(learnPath)
dir_check(ymlPath)
# 学習用顔データを取得する
imagePaths = [os.path.join(imgPath, f) for f in os.listdir(imgPath)]

# 学習用画像分処理
for imagePath in imagePaths:
    # 画像を読み込む
    img = cv2.imread(imagePath)
    # 画像をグレースケールに変換する
    image_pil = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # NumPyの配列に格納
    gray = np.array(image_pil, 'uint8')
    # Haar-like特徴分類器で顔を検知
    faces = face_detector.detectMultiScale(gray)

    # 学習用画像データを作成
    for (x,y,w,h) in faces:
        # 顔部分を切り取り
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        sample_cnt += 1
        # 画像ファイル名にUSERIDを付与して保存
        cv2.imwrite(learnPath + "/User.".format(args[1]) + str(User_id) + '.' + str(sample_cnt) + ".jpg", image_pil[y:y+h,x:x+w])

print("\n Learn Image Get End ")
########################
# 学習ファイル作成
########################
image_learning_make_Labels()
