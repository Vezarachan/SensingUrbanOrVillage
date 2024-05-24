from GetSVImageIdList import get_image_url_list_all, get_image_url_list_locally
from GetSVImageAndMetadata import save_sv_images_with_meta_data, save_sv_images_with_metadata
from utils import FILE_PATH

if __name__ == '__main__':
    # get_image_url_list_locally('StuttgartSamplePoints4326')
    save_sv_images_with_metadata('D:/Research/datasets/UrbanOrRural/Stuttgart/Images', 'StuttgartSamplePoints4326', '')
