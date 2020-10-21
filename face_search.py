import cv2

# 顔が写っている画像
img = cv2.imread("./img/test.jpeg")
#Haar-like特徴を用いたブースティングされた分類器のカスケード　顔用
face_cascade = cv2.CascadeClassifier('/Users/tomiyamasatoshi/.pyenv/versions/3.8.1/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#目
eye_cascade = cv2.CascadeClassifier('/Users/tomiyamasatoshi/.pyenv/versions/3.8.1/lib/python3.8/site-packages/cv2/data/haarcascade_eye.xml')
#グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 顔を検知
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
    # 検知した顔を矩形で囲む
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 顔画像（グレースケール）
    roi_gray = gray[y:y+h, x:x+w]
    # 顔ｇ増（カラースケール）
    roi_color = img[y:y+h, x:x+w]
    # 顔の中から目を検知
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # 検知した目を矩形で囲む
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# 画像保存
cv2.imwrite("./img/test1_face.jpg", img)
