import osmnx as ox
import osmnx.io as oxio
import fiona
import shapely

area_boundary = fiona.open('./Datasets/Urban/Stuttgart.shp')
# area_boundary = fiona.open('D:/Research/datasets/BayernPOIs/Oberbayern/Oberbayern.shp')


# coordinates = next(iter(area_boundary)).geometry['coordinates']
#
# print(len(coordinates))

# area_polygon = shapely.MultiPolygon([coordinates])


def download_road_network_graph_for_a_place(place_boundary, filename):
    coordinates = next(iter(place_boundary)).geometry['coordinates']
    area_polygon = shapely.MultiPolygon([coordinates])
    area_road_network = ox.graph_from_polygon(polygon=area_polygon, network_type='drive', simplify=True)
    oxio.save_graph_geopackage(area_road_network,
                               filepath=filename)


def download_road_network_graph_for_multiple_places(places_boundaries):
    for i, polygon in enumerate(places_boundaries):
        print(f'Downloading road network for {i + 1} out of {len(places_boundaries)}')
        area_polygon = shapely.Polygon(polygon.geometry['coordinates'][0])
        try:
            area_road_network = ox.graph_from_polygon(polygon=area_polygon, network_type='drive', simplify=True)
            oxio.save_graph_geopackage(area_road_network,
                                       filepath=f'D:/Research/datasets/BayernPOIs/RoadNetworks/VillagesInOberbayernBuffer_{i}.gpkg')
        finally:
            continue


if __name__ == '__main__':
    # download_road_network_graph_for_multiple_places(area_boundary)
    download_road_network_graph_for_a_place(area_boundary, './Datasets/Urban/stuttgart_road_network')
