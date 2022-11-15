import cv2
from imutils import paths
import os

# data_dirs = ["CRIC_data/SCC", "CRIC_data/ASC-H", "CRIC_data/HSIL", "CRIC_data/NILM"]
data_dirs = ["CRIC_data/ASC-H"]

for data_dir in data_dirs:
    print("[INFO] processing '{}'...".format(data_dir))
    # grab all image paths in the current data_dir
    p = os.path.sep.join(['./', data_dir])
    imagePaths = list(paths.list_images(p))
    print(imagePaths)

    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        img_hz = cv2.flip(img, 1)
        img_vt = cv2.flip(img, 0)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), 90, 1)
        img_rt90 = cv2.warpAffine(img, M, (w, h))
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), 270, 1)
        img_rt270 = cv2.warpAffine(img, M, (w, h))

        img_name = imagePath.split('/')[-1]
        img_name = img_name.split('.')[0]
        img_name_hz = img_name + '_hz.png'
        p = os.path.join(data_dir, img_name_hz)
        cv2.imwrite(p, img_hz)
        
        img_name_vt = img_name + '_vt.png'
        p = os.path.join(data_dir, img_name_vt)
        cv2.imwrite(p, img_vt)

        img_name_rt90 = img_name + '_rt90.png'
        p = os.path.join(data_dir, img_name_rt90)
        cv2.imwrite(p, img_rt90)

        img_name_rt270 = img_name + '_rt270.png'
        p = os.path.join(data_dir, img_name_rt270)
        cv2.imwrite(p, img_rt270)