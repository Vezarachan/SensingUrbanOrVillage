import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import get_file_path_by_name, IMAGE_SEARCH_URL, IMAGE_URL
import shutil
import requests
import datetime


def log_transform(image, c):
    output_image = c * np.log(1.0 + image)
    output_image = np.uint8(output_image + 0.5)
    return output_image


def is_bright(image, dim=10, threshold=0.5):
    image = cv2.resize(image, (dim, dim))
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    L = L / np.max(L)
    return np.mean(L) > threshold


def brightness(image, dim=10):
    image = cv2.resize(image, (dim, dim))
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    L = L / np.max(L)
    return L.mean()


def preprocess(image_name, image_path):
    image = cv2.imread(image_path)
    if not is_bright(image, threshold=0.4):
        output = log_transform(image, 25)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    else:
        output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave(f'D:/Research/datasets/UrbanOrRural/PreprocessedImages/{image_name}', output)


def preprocess_all(images_dir):
    image_paths = get_file_path_by_name(images_dir)
    for image_path in image_paths:
        name, path_ = image_path
        print(name)
        preprocess(name, path_)


def move_low_light_images(image_name, image_path):
    image = cv2.imread(image_path)
    if not is_bright(image, threshold=0.3):
        shutil.copyfile(image_path, f'D:/Research/datasets/UrbanOrRural/ReImages/{image_name}')


def move_low_light_images_all(images_dir):
    image_paths = get_file_path_by_name(images_dir)
    for image_path in image_paths:
        name, path_ = image_path
        print(name)
        move_low_light_images(name, path_)


def re_download_image_with_low_light(image_name, image_path):
    fid, x, y = image_name.replace('.jpg', '').split(',')
    x = float(x)
    y = float(y)
    min_lon = x - 0.0003
    max_lon = x + 0.0003
    min_lat = y - 0.0003
    max_lat = y + 0.0003
    bbox = '{0:.8f},{1:.8f},{2:.8f},{3:.8f}'.format(min_lon, min_lat, max_lon, max_lat)
    url = IMAGE_SEARCH_URL.format(bbox)
    headers = {'Content-Type': 'application/json',
               'Authorization': f'OAuth MLY|8053761161317252|0fe0c0f8a37c7b4c2f8976fb82a40e00'}
    res = requests.get(url, headers=headers, timeout=30)
    print(res.status_code)
    if res.status_code == 200:
        if len(res.json()['data']) > 0:
            content = res.json()
            data = content['data']
            index = 1
            near_noon = 30
            best_hour = -1
            image_data = None
            for info in data:
                fid_ = info['id']
                image_res = requests.get(IMAGE_URL.format(fid_))
                if image_res.status_code == 200:
                    content = image_res.json()
                    # print(content)
                    if 'thumb_2048_url' in content.keys():
                        image_url = content['thumb_2048_url']
                    elif 'thumb_original_url' in content.keys():
                        image_url = content['thumb_original_url']
                    elif 'thumb_1024_url' in content.keys():
                        image_url = content['thumb_1024_url']
                    else:
                        continue
                    captured_at = content['captured_at']
                    dt = datetime.datetime.fromtimestamp(captured_at / 1000.0)
                    print(dt.hour)

                    hour = dt.hour

                    image_res = requests.get(image_url, stream=True).raw
                    image = np.asarray(bytearray(image_res.read()), dtype=np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if abs(12 - hour) <= near_noon:
                        best_hour = hour
                        image_data = image


                    # #
                    # bright = brightness(image)
                    # if bright > high_brightness:
                    #     high_brightness = bright
                    #     image_data = image
                index += 1
            print(best_hour)
            cv2.imwrite(f'D:/Research/datasets/UrbanOrRural/ReImages_/{fid},{x},{y}.jpg', image_data)


def re_download_image_with_low_light_all(images_dir):
    image_paths = get_file_path_by_name(images_dir)
    for image_path in image_paths:
        name, path_ = image_path
        print(name)
        re_download_image_with_low_light(name, path_)


if __name__ == '__main__':
    # preprocess_all('D:/Research/datasets/UrbanOrRural/ReImages')
    # move_low_light_images_all('D:/Research/datasets/UrbanOrRural/Images')
    # img = cv2.imread('./Datasets/Images/114978547919914,11.567815,48.1605.jpg')
    # print(is_bright(img))
    # output = log_transform(img, 40)
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # plt.imshow(output)
    # plt.show()
    re_download_image_with_low_light_all('D:/Research/datasets/UrbanOrRural/ReImages')

