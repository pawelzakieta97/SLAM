import numpy as np
import cv2
from matplotlib import pyplot as plt

from transformations import translation, rotation_x, rotation_y, rotation_z


class Environment:
    def __init__(self, height_map: np.array, rgb=None):
        self.height_map = height_map
        if rgb is None:
            rgb = height_map# * 0 + 255
            rgb = np.repeat(rgb[:,:,np.newaxis], 3, axis=2)
        self.rgb = rgb
        self.point_cloud = []

        self.gradient_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=5)
        self.gradient_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=5)
        for x in range(height_map.shape[1]):
            for y in range(height_map.shape[0]):
                color = rgb[y, x, :]
                self.point_cloud.append(np.array([y, x, height_map[y, x], color[0], color[1], color[2]]))

    def get_pc(self):
        return np.array(self.point_cloud)

    def get_rgbd_view(self, camera: np.array, res=(500, 500), f=250, show=True, filename=None):
        camera_inv = np.linalg.inv(camera)
        points = self.get_pc()
        point_coords = points[:, :3]
        point_coords = np.c_[point_coords, np.ones(point_coords.shape[0])]
        point_coords = (camera_inv.dot(point_coords.T)).T
        points[:, :3] = point_coords[:, :3]
        depths = points[:, 2]
        points = points[depths>1]
        depths = points[:, 2]
        image_coords = points[:,:2]# /depths[:, None] * f
        result_image = np.zeros((res[0], res[1], 4), dtype=np.uint8)
        result_image[:,:,3] = 255
        for idx in range(image_coords.shape[0]):
            pixel_y = int(points[idx, 1]/points[idx, 2] * f)
            pixel_x = int(points[idx, 0]/points[idx, 2] * f)
            pixel_x += res[1]//2
            pixel_y += res[0]//2

            if pixel_x >= res[1] or pixel_x<0 or pixel_y >= res[0] or pixel_y<0:
                continue
            if depths[idx]<result_image[pixel_y, pixel_x, 3]:
                result_image[pixel_y, pixel_x, 3] = min(depths[idx], 255)
                result_image[pixel_y, pixel_x, :3] = points[idx, 3:]
        size = (35, 35)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)
        result_image[:,:,3] = cv2.erode(result_image[:,:,3], kernel)
        if show:
            cv2.imshow('nic', result_image[:, :, 3])
            cv2.waitKey(0)
        if filename is not None:
            cv2.imwrite(filename+'_depth.png', result_image[:, :, 3])
        return result_image

    def update_height_map(self, res=(500, 500), show=True, filename=None):
        points = self.get_pc()
        heights = points[:, 2]
        result_image = np.zeros((res[0], res[1]), dtype=np.uint8)
        for idx in range(points.shape[0]):
            pixel_y = int(points[idx, 1])
            pixel_x = int(points[idx, 0])
            pixel_x += res[1] // 2
            pixel_y += res[0] // 2

            if pixel_x >= res[1] or pixel_x < 0 or pixel_y >= res[0] or pixel_y < 0:
                continue
            if heights[idx] > result_image[pixel_y, pixel_x]:
                result_image[pixel_y, pixel_x] = min(heights[idx], 255)
        result_image = result_image * 255 / result_image.max()
        size = (3, 3)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)
        result_image = cv2.dilate(result_image, kernel)
        if show:
            cv2.imshow('nic', result_image)
            cv2.waitKey(0)
        if filename is not None:
            cv2.imwrite(filename + '_height.png', result_image)
        self.height_map = result_image

    def get_lidar_reading(self, x, y, resolution=100, show=True):
        readings = []
        beam = self.height_map*0
        beam_len = self.height_map.shape[0]
        vis = self.height_map.copy()
        for alfa in np.linspace(0, np.pi*2, resolution):
            start = (x + self.height_map.shape[0]//2, y+self.height_map.shape[1]//2)
            end = (int(start[0] + np.cos(alfa)*beam_len), int(start[1] + np.sin(alfa)*beam_len))
            beam = beam * 0
            cv2.line(beam, start, end, (1, 1, 1), 1)
            intersections = np.multiply(beam, self.height_map)
            coords = np.nonzero(intersections)
            lens = np.sqrt(np.power(coords[0] - start[1], 2) + np.power(coords[1] - start[0], 2))
            cv2.line(vis, start, (int(start[0] + np.cos(alfa)*lens.min()), int(start[1] + np.sin(alfa)*lens.min())), (255, 255, 255), 1)
            readings.append(lens.min())
            pass
        if show:
            cv2.imshow('niccc', vis)
            cv2.waitKey(0)
        return readings

class Box:
    def __init__(self, position, length, width, height, color):
        self.position = position
        self.length = length
        self.width = width
        self.height = height
        self.color = color

    def get_pc(self, density=1):
        points = []
        for h in np.linspace(0, self.height, int(self.height * density)):
            for x in np.linspace(0, self.length, int(self.length * density)):
                points.append(
                    np.array([self.position[0] + x, self.position[1], h, self.color[0], self.color[1], self.color[2]]))
                points.append(np.array(
                    [self.position[0] + x, self.position[1] + self.width, h, self.color[0], self.color[1],
                     self.color[2]]))

            for y in np.linspace(0, self.length, int(self.width * density)):
                points.append(
                    np.array([self.position[0], self.position[1] + y, h, self.color[0], self.color[1], self.color[2]]))
                points.append(np.array(
                    [self.position[0] + self.length, self.position[1] + y, h, self.color[0], self.color[1],
                     self.color[2]]))

        return points

class Plane:
    def __init__(self, position, length, width, color):
        self.position = position
        self.length = length
        self.width = width
        self.color = color

    def get_pc(self, density=1):
        points = []
        for y in np.linspace(0, self.length, int(self.width * density)):
            for x in np.linspace(0, self.length, int(self.length * density)):
                points.append(
                    np.array([self.position[0] + x, self.position[1] + y, 0, self.color[0], self.color[1], self.color[2]]))

        return points

def store_pc(filename, pc):
    with open(filename, 'w+') as f:
        for idx in range(pc.shape[0]):
            point = pc[idx, :]
            f.write(',\t'.join([str(x) for x in point])+'\n')

def get_camera_transformation(x, y, z, yaw, pitch):
    camera = rotation_x(-np.pi/2 + pitch)
    camera = rotation_z(yaw).dot(camera)
    camera = translation(np.array([x, y, z])).dot(camera)
    return camera

if __name__ == '__main__':

    objects = []
    objects.append(Box(np.array([0, 0]), 100, 100, 20, (0,0,255)))
    objects.append(Plane(np.array([0,0]), 100, 100, (128,128,128)))
    objects.append(Box(np.array([20, 20]), 10, 10, 10, (0,0,255)))
    objects.append(Box(np.array([70, 20]), 10, 10, 10, (0, 0, 255)))
    objects.append(Box(np.array([70, 70]), 10, 10, 10, (0, 0, 255)))
    objects.append(Box(np.array([20, 70]), 10, 10, 10, (0, 0, 255)))
    points = []
    for o in objects:
        points = points + o.get_pc(1)

    hm = cv2.imread('height.png')
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    env = Environment(hm)
    env.point_cloud = np.array(points)
    env.update_height_map(filename='map')
    points = env.get_pc()

    # UZYWANIE
    # ZAPISANIE CHMURY PUNKTOW MAPY
    store_pc('pc.txt', env.get_pc())

    # ZAPISANIE ODCZYTÓW Z LIDARU
    with open('lidar.txt', 'w+') as l:
        l.write('\n'.join(str(reading) for reading in env.get_lidar_reading(50, 50)))

    # WYGENEROWANIE TEORETYCZNEGO WIDOKU Z KAMERY RGBD
    camera = get_camera_transformation(50, 50, 15, yaw=0.5, pitch=-0.2)
    rgbd = env.get_rgbd_view(camera, show=False, filename='out')

    # POMOCNICZE WYŚWIETLENIE CHMURY PUNKTÓW AKTUALNEJ MAPY
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2]/255, cmap='Greens')
    plt.show()
    pass
