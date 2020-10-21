# -*- coding: UTF-8 -*-
import logging.config

# ログ設定ファイルからログ設定を読み込み
logging.config.fileConfig('/var/www/html/MyWeb/FaceMOD/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger()

logger.debug('face_auth.py 呼び出し')

try:
    import sys, codecs, io
    import cv2
    import numpy as np
    import os
    import time
    import json
    import configparser

    from PIL import Image

    # 最小Windowサイズを定義
    minW = 64
    minH = 48

    # 設定ファイル読み込み
    inifile = configparser.ConfigParser()
    inifile.read('/var/www/html/MyWeb/FaceMOD/config.ini', 'UTF-8')

    # 学習した結果を.ymlファイルの保存先
    ymlPath = inifile.get('file-dir', 'ymlPath')

    # 顔認証で使用するxmlをパラメーターとして物体認識（顔認識）のインスタンス生成
    cascadePath = inifile.get('file-dir', 'cascadeFace')
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # 顔認証で使用するxmlをパラメーターとして物体認識（目認識）のインスタンス生成
    eye_cascadePath = inifile.get('file-dir', 'cascadeEye')
    eye_cascade = cv2.CascadeClassifier(eye_cascadePath)

    # 学習した際のUserIDと人物の名前を変換するための配列（ここではUserID = 05まで　USERIDを増やしたい人はここを増やす）
    names = json.loads(inifile.get('person-list', 'personList'))
    logger.debug(names)
    # 結果格納用リスト
    resultList = {}

    # グレースケールに変換
    img = Image.open("/var/www/html/MyWeb/storage/app/public/images/person.jpeg")
    PIL_img = img.convert('L')
    gray = np.array(PIL_img,'uint8')
    # 顔検出
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 3,
        minSize = (int(minW), int(minH)),
       )

    # 顔の検出ができたかチェック
    if len(faces) == 0:
        logger.debug("顔検出できませんでした。")
    else:
        logger.debug("顔検出できました。")

        for i in range(len(names)):
            # Local Binary Patterns Histogram(LBPH)アルゴリズム　インスタンス
            recognizer=cv2.face_LBPHFaceRecognizer.create()
            # 学習した顔認証ファイルを読み出しする
            ymlPath = ymlPath.encode('utf-8')
            trainer_file = (ymlPath + "/trainer.yml").format(i)
            logger.debug(str(i + 1) + "人目の学習ファイル：" + trainer_file)
            recognizer.read(trainer_file)

            # 顔検出した人物認証のためのループ
            for(x,y,w,h) in faces:

                # 顔の上半分を検出対象範囲とする
                eyes_gray = gray[y : y + int(h/2), x : x + w]
                ################
                # 目検出        #
                ################
                logger.debug("目を検出します。")
                eyes = eye_cascade.detectMultiScale(
                    eyes_gray,
                    scaleFactor=1.11,
                    minNeighbors=3,
                    minSize=(8, 8))

                #################################################
                # 学習した顔から推論をかける。戻値が認識した番号と信頼度 #
                #################################################
                logger.debug("顔を推論にかけます。")
                id ,confidence = recognizer.predict(gray[y:y+h,x:x+w])

                ##################################################################
                # confidece（信頼度）を40%100%とする（ちょっと信頼度を低めからしている） #
                ##################################################################
                if round(100 - confidence) > 50:
                    # USER IDから名前を取得
                    id = names[i]
                    id = id.encode('utf-8')
                    confidence = round(100 - confidence)
                    # 結果を格納する
                    resultList[id] = confidence
                else:
                    id = "unknown"
                    confidence = round(100 - confidence)

                logger.debug("対象：" + id)
                logger.debug("信頼度：" + str(confidence))


    # Do a bit of cleanup
    cv2.destroyAllWindows()

    # 一番近しい人物を返す
    if len(resultList) != 0:
        return_kv = max(resultList.items(), key = lambda x : x[1])
        print(return_kv[0])
        print(return_kv[1])
        logger.debug("該当あり")
    else:
        print("unknown")
        print("別の写真を選択して下さい。")
        logger.debug("該当なし")
except Exception as e:
    logger.debug("エラーあり")
    logger.debug(sys.exc_info)
    import traceback
    logger.debug(traceback.print_exc())
    logger.debug(str(e))
