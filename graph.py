import cv2
import numpy as np
import matplotlib.pyplot as plt


class Graph():

    CLAHE = None

    @staticmethod
    def to_grayscale(image: np.ndarray):
        '''
        Changes to grayscale.
        :param image: 3-channel RGB image.
        :return: 1-channel grayscale image.
        '''
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def to_3channel_binary(image: np.ndarray):
        '''
        Changes 1-channel binary image to 3-channel binary image.
        :param image: 1-channel binary image.
        :return: 3-channel binary image.
        '''
        int_image = image.astype(dtype=np.uint8)
        color_binary = np.dstack((int_image, int_image, int_image)) * 255
        return color_binary


    @staticmethod
    def to_hls(image: np.ndarray, color_numb: int = 2, thresh=(0, 255)):
        '''
        Method changes image to HLS color map and calculates binary reshold in one color from HLS color map.
        :param image: RGB 3-channel image.
        :param color_numb: Channel number from HLS map to be tresholded. 0 for Hue; 1 for Lightness; 2 for Saturation. Default value is Saturation.
        :param thresh: Treshold level. 2 values tuple. 1st value is lower boundary. 2nd value is upper boundary. Default value is (0, 255).
        :return: Binary (values 0..1) 1-channel image.
        '''
        # 1) Convert to HLS color space
        hls_color = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls = hls_color[:, :, color_numb]
        # 2) Apply a threshold to the S channel
        binary_output = np.zeros_like(hls)
        binary_output[(hls >= thresh[0]) & (hls <= thresh[1])] = 1

        # 3) Return a binary image of threshold result
        return binary_output

    @staticmethod
    def to_rgb(image: np.ndarray, color_numb: int = 0, thresh=(0, 255)):
        '''
        Method takes image in RGB color map and calculates binary reshold in one color from RGB color map.
        :param image: RGB 3-channel image.
        :param color_numb: Channel number from RGB map to be tresholded. 0 for Red; 1 for Green; 2 for Blue. Default value is Red.
        :param thresh: Treshold level. 2 values tuple. 1st value is lower boundary. 2nd value is upper boundary. Default value is (0, 255).
        :return: Binary (values 0..1) 1-channel image.
        '''
        rgb = image[:, :, color_numb]
        binary_output = np.zeros_like(rgb)
        binary_output[(rgb >= thresh[0]) & (rgb <= thresh[1])] = 1
        return binary_output


    @staticmethod
    def histogram(image: np.ndarray):
        # 1 channel only
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        sizey, sizex = image.shape[0], image.shape[1]
        bottom_half = image[sizey // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    def adjust_brightness(image: np.ndarray):
        hls_color = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        # sizey, sizex = hls_color.shape[0], hls_color.shape[1]
        light_channel = np.array(hls_color[:, :, 1], dtype=np.uint8)
        sat_channel = np.array(hls_color[:, :, 2], dtype=np.uint8)
        hue_channel = np.array(hls_color[:, :, 0], dtype=np.uint8)

        # create a CLAHE object (Arguments are optional).
        if (Graph.CLAHE is None):
            Graph.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        light_channel = Graph.CLAHE.apply(light_channel)
        # sat_channel = Graph.CLAHE.apply(sat_channel)
        # hue_channel = Graph.CLAHE.apply(hue_channel)

        stacked = np.dstack((hue_channel, light_channel, sat_channel))
        result = cv2.cvtColor(stacked, cv2.COLOR_HLS2RGB)

        return result

