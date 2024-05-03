import os


# FILE_PATH = './Datasets/Images'
FILE_PATH = 'D:/Research/datasets/UrbanOrRural/VillageImages'


def get_file_path_by_name(file_dir):
    file_paths = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                file_paths.append((file, '/'.join((root, file))))
    return file_paths


IMAGE_SEARCH_URL = 'https://graph.mapillary.com/images?fields=id&bbox={0}&limit=5'
IMAGE_URL = 'https://graph.mapillary.com/{0}?access_token=MLY|8053761161317252|0fe0c0f8a37c7b4c2f8976fb82a40e00&fields=id,computed_geometry,detections.value,is_pano,captured_at,thumb_2048_url,thumb_original_url,thumb_1024_url'
