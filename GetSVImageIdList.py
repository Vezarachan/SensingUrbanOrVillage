import os
import time
import requests
import geojson
from utils import FILE_PATH, IMAGE_SEARCH_URL


def get_image_url_list_locally(samples_file):

    with open(f'./Datasets/SamplePoints/Stuttgart/{samples_file}.geojson', 'r') as f:
        sample_points = geojson.load(f)
    # with open('./Datasets/SamplePoints/Munich/{0}.geojson'.format(samples_file), 'r') as f:
    #     sample_points = geojson.load(f)

    # os.mkdir('{0}/{1}'.format(FILE_PATH, samples_file))

    image_ids = list()

    for i, feature in enumerate(sample_points['features']):
        print(feature)
        if i < 23816:
            continue
        coord = feature['geometry']['coordinates']
        x, y = coord
        min_lon = x - 0.0004
        max_lon = x + 0.0004
        min_lat = y - 0.0004
        max_lat = y + 0.0004
        bbox = '{0:.8f},{1:.8f},{2:.8f},{3:.8f}'.format(min_lon, min_lat, max_lon, max_lat)
        url = IMAGE_SEARCH_URL.format(bbox)
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'OAuth MLY|8053761161317252|0fe0c0f8a37c7b4c2f8976fb82a40e00'}
        res = requests.get(url, headers=headers, timeout=30)
        print(res.json())
        if res.status_code == 200:
            if len(res.json()['data']) > 0:
                content = res.json()
                data = content['data']
                image_info = data[0]
                image_id = image_info['id']
                image_ids.append({
                    'image_id': image_id,
                    'coordinate': coord,
                })
                # print(image_info['id'])
                # with open('{0}/{1}/{2}.csv'.format(FILE_PATH, samples_file, samples_file), 'a') as f:
                #     f.write('{0},{1:.8f},{2:.8f}\n'.format(image_id, x, y))
                with open('{0}/{1}.csv'.format(FILE_PATH, samples_file), 'a') as f:
                    f.write('{0},{1:.8f},{2:.8f}\n'.format(image_id, x, y))
    return len(image_ids)


def get_image_url_list_all():
    number_all = 0
    for i in range(1):
        samples_file = 'MunichSamplePointsLocal_{0}'.format(i)
        image_number = get_image_url_list_locally(samples_file)
        number_all += image_number
        print('{0}: {1}'.format(samples_file, image_number))
        time.sleep(25)
    return number_all

