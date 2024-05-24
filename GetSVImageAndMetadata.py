import os
import time
import requests
import pandas as pd
import numpy as np
from utils import FILE_PATH, IMAGE_SEARCH_URL, IMAGE_URL

IMAGE_NAME = '{0}/{1},{2},{3}.jpg'
IMAGE_METADATA = '{0}/{1}/{2}.csv'


def count_detected_objects(detected_object_list):
    all_objects: list = list(map(lambda x: x['value'], detected_object_list))
    if len(detected_object_list) == 0:
        return ''
    all_objects_str = ['{0}:{1}'.format(d, all_objects.count(d)) for d in set(all_objects)]
    return ';'.join(all_objects_str)


def save_sv_images_with_metadata(file_path, samples_file, output_file):
    print('Category: {0}'.format(samples_file))
    print('{0}/{1}.csv'.format(file_path, samples_file))
    image_info = pd.read_csv('{0}/{1}.csv'.format(file_path, samples_file),
                             header=None,
                             delimiter=',').astype({0: str, 1: np.float64, 2: np.float64})
    print(image_info)
    for index, row in image_info.iterrows():
        image_id = row[0]
        x = row[1]
        y = row[2]
        print('Download Image Id: {0}'.format(image_id))
        res = requests.get(IMAGE_URL.format(image_id))
        if res.status_code == 200:
            content = res.json()
            print(content)
            if 'thumb_2048_url' in content.keys():
                image_url = content['thumb_2048_url']
            elif 'thumb_original_url' in content.keys():
                image_url = content['thumb_original_url']
            elif 'thumb_1024_url' in content.keys():
                image_url = content['thumb_1024_url']
            else:
                continue
            captured_at = content['captured_at']
            if 'detections' in content.keys():
                detections = content['detections']['data']
                detected_objects_str = count_detected_objects(detections)
            else:
                detected_objects_str = ''

            img_data = requests.get(image_url).content
            with open(IMAGE_NAME.format(file_path, image_id, x, y), 'wb') as handler:
                handler.write(img_data)

            # with open(IMAGE_METADATA.format(file_path, samples_file, output_file), 'a') as metadata:
            #     metadata.write('{0},{1},{2},{3},{4}\n'.format(image_id, x, y, captured_at, detected_objects_str))


def re_save_images_wrong(filepath):
    files = os.listdir(filepath)
    for file in files:
        fid, x, y = file.replace('.jpg', '').split(',')
        res = requests.get(IMAGE_URL.format(fid))
        if res.status_code == 200:
            content = res.json()
            print(content)
            if 'thumb_2048_url' in content.keys():
                image_url = content['thumb_2048_url']
            elif 'thumb_original_url' in content.keys():
                image_url = content['thumb_original_url']
            elif 'thumb_1024_url' in content.keys():
                image_url = content['thumb_1024_url']
            else:
                continue

            img_data = requests.get(image_url).content
            with open(f'D:/Research/datasets/UrbanOrRural/ReImages_/{fid},{x},{y}.jpg', 'wb') as handler:
                handler.write(img_data)


def re_save_images_wrong_multi(filepath):
    files = os.listdir(filepath)
    for file in files:
        fid, x, y = file.replace('.jpg', '').split(',')
        x = float(x)
        y = float(y)
        min_lon = x - 0.0003
        max_lon = x + 0.0003
        min_lat = y - 0.0003
        max_lat = y + 0.0003
        bbox = '{0:.8f},{1:.8f},{2:.8f},{3:.8f}'.format(min_lon, min_lat, max_lon, max_lat)
        url = IMAGE_SEARCH_URL.format(bbox)
        print(url)
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'OAuth MLY|8053761161317252|0fe0c0f8a37c7b4c2f8976fb82a40e00'}
        res = requests.get(url, headers=headers, timeout=30)
        if res.status_code == 200:
            if len(res.json()['data']) > 0:
                content = res.json()
                data = content['data']
                index = 1
                for info in data:
                    fid_ = info['id']
                    image_res = requests.get(IMAGE_URL.format(fid_))
                    if image_res.status_code == 200:
                        content = image_res.json()
                        print(content)
                        if 'thumb_2048_url' in content.keys():
                            image_url = content['thumb_2048_url']
                        elif 'thumb_original_url' in content.keys():
                            image_url = content['thumb_original_url']
                        elif 'thumb_1024_url' in content.keys():
                            image_url = content['thumb_1024_url']
                        else:
                            continue

                        img_data = requests.get(image_url).content
                        with open(f'D:/Research/datasets/UrbanOrRural/ReImages_/{fid},{x},{y}-{index}.jpg', 'wb') as handler:
                            handler.write(img_data)
                    index += 1


def save_sv_images_with_metadata_for_all():
    for i in range(198):
        samples_file = 'MunichSamplePointsLocal_{0}'.format(i)
        output_file = 'MunichMetadataLocal_{0}'.format(i)
        if len(os.listdir('{0}/{1}'.format(FILE_PATH, samples_file))) == 0:
            continue
        save_sv_images_with_metadata(FILE_PATH, samples_file, output_file)
        time.sleep(60)


def save_sv_images_with_meta_data():
    i = 0
    image_ids = pd.read_csv('{0}/{1}.csv'.format(FILE_PATH, 'VillageNearMunichBuffer5to15kmSamplePoints')).astype({'x': np.float64, 'y': np.float64, 'id': str})
    print(len(image_ids))
    for index, row in image_ids.iterrows():
        i += 1
        # if i < 68550:
        #     print(i)
        #     continue
        x = row['x']
        y = row['y']
        image_id = row['id']
        print('Downloading {0} images: {1}'.format(i, image_id))
        res = requests.get(IMAGE_URL.format(image_id))
        print(res.json())
        if res.status_code == 200:
            content = res.json()
            print(content)
            if 'thumb_2048_url' in content.keys():
                image_url = content['thumb_2048_url']
            elif 'thumb_original_url' in content.keys():
                image_url = content['thumb_original_url']
            elif 'thumb_1024_url' in content.keys():
                image_url = content['thumb_1024_url']
            else:
                continue
            captured_at = content['captured_at']
            if 'detections' in content.keys():
                detections = content['detections']['data']
                detected_objects_str = count_detected_objects(detections)
            else:
                detected_objects_str = ''

            is_pano = content['is_pano']

            img_data = requests.get(image_url).content
            with open('{0}/{1},{2},{3}.jpg'.format(FILE_PATH, image_id, x, y), 'wb') as handler:
                handler.write(img_data)

            # with open('{0}/{1}.csv'.format(FILE_PATH, 'metadata'), 'a') as metadata:
            #     metadata.write('{0},{1},{2},{3},{4},{5}\n'.format(image_id, x, y, captured_at, is_pano, detected_objects_str))
