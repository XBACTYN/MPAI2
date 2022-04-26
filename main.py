import json

import numpy as np
from scipy import signal
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt

#change paths
settings = {
    'filepath': 'C:\\Users\\user\\Desktop\\MPAI2\\09_lena2.tif',
    'path_to_save_result': "C:\\Users\\user\\Desktop\\MPAI2\\results"
}


MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0
#BORDER_PROCESSING_PARAMETER = 20


WINDOW_HORIZONTAL = np.array([[-1, 1]]) #для обыч град

#для обыч град
WINDOW_VERTICAL = np.array([[-1],
                            [1]])

#Прюитт
WINDOW_PREWITT_S1 = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]]) * (1/6)

#Прюитт
WINDOW_PREWITT_S2 = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]]) * (1/6)

#Солгасованный лапласиан
WINDOW_LAPLACIAN_AGREEMENT_METHOD = np.array([[2, -1, 2],
                                              [-1, -4, -1],
                                              [2, -1, 2]]) * (1/3)

#Вертикально-гориз лаплас
WINDOW_LAPLACIAN = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

#Диагональный лаплас
WINDOW_LAPLACIAN_MUTUALLY_PERPENDICULAR = np.array([[1, 0, 1],
                                                    [0, -4, 0],
                                                    [1, 0, 1]]) * (1/2)

#суммарный из верт-гор + диаг лаплас
WINDOW_LAPLACIAN_OF_SUM_APPROXIMATIONS = np.array([[1, 1, 1],
                                                   [1, -8, 1],
                                                   [1, 1, 1]]) * (1/3)


def border_processing_function(element_value, border):#пороговая обработка поэлементно
    if element_value < border:
        return MIN_BRIGHTNESS_VALUE
    else:
        return MAX_BRIGHTNESS_VALUE


def border_processing(img_as_arrays, border): #применение ко всей картинке пороговой обработки
    vector_img = np.vectorize(border_processing_function)
    new_img = vector_img(img_as_arrays, border)
    return new_img


def gradient_module(matrix_u, matrix_v): #засовываем частные производные, выч модуль градиента.
    sum_of_squares = np.square(matrix_u) + np.square(matrix_v)
    return np.sqrt(sum_of_squares)


def laplacian_agreement_method(alpha, beta): #Лапласиан с согласованием для поверхностей 2го порядка
    return (alpha * 2 + beta * 2).astype(int)


def window_processing(matrix, window): #готовая фунция. 2мерная свертка
    return signal.convolve2d(matrix, window, boundary='symm', mode='same').astype(int)


def create_figure_of_gradient_method(src_img, deriv_horiz, grad_matrix, deriv_vert, img_border,window_title): #deriv-horiz- част производ по горизонтали.
    fig = plt.figure(window_title,figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("Source image")
    imshow(src_img, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(2, 3, 2)
    plt.title("Derivative horizontal")
    imshow(deriv_horiz, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(2, 3, 3)
    plt.title("Gradient evaluation")
    imshow(grad_matrix, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(2, 3, 4)
    plt.title("Border processing")
    imshow(img_border, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(2, 3, 5)
    plt.title("Derivative vertical")
    imshow(deriv_vert, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(2, 3, 6)
    plt.title("Histogram of gradient evaluation")
    plt.xlabel('Brightness values')
    plt.ylabel('Pixels quantity')
    create_wb_histogram_plot(grad_matrix)
    return fig


def create_figure_of_laplacian_method(src_img, laplacian, img_border,window_title):
    fig = plt.figure(window_title,figsize=(20, 10))
    fig.add_subplot(2, 2, 1)
    plt.title("Source image")
    imshow(src_img, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 2)
    plt.title("Laplacian evaluation")
    imshow(laplacian, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 3)
    plt.title("Border processing")
    imshow(img_border, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 4)
    plt.title("Histogram of laplacian evaluation")
    plt.xlabel('Brightness values')
    plt.ylabel('Pixels quantity')
    create_wb_histogram_plot(laplacian)
    return fig


def create_wb_histogram_plot(img_as_arrays):
    hist, bins = np.histogram(img_as_arrays.flatten(), 256, [0, 256])
    plt.plot(bins[:-1], hist, color='blue', linestyle='-', linewidth=1)


def open_image_as_arrays(filepath):
    return imread(filepath)


def save_image(img, directory):
    imsave(directory, img)


def save_current_parameters_to_file(filepath):
    json.dump(settings, open(filepath, 'w'))


#start main
image_filepath = settings['filepath']
#path_to_save = settings['path_to_save_result']
img_as_array = open_image_as_arrays(image_filepath)


#gradient
img_derivative_horizontal = window_processing(img_as_array, WINDOW_HORIZONTAL)
img_derivative_vertical = window_processing(img_as_array, WINDOW_VERTICAL)
img_gradient = gradient_module(img_derivative_horizontal, img_derivative_vertical)

#порог 20 экспериментально для лены.
create_figure_of_gradient_method(img_as_array, np.abs(img_derivative_horizontal), img_gradient, np.abs(img_derivative_vertical), border_processing(img_gradient, 20),'Градиент обычный')
plt.tight_layout()

#show()

#prewitt
img_prewitt_s1 = window_processing(img_as_array, WINDOW_PREWITT_S1)
img_prewitt_s2 = window_processing(img_as_array, WINDOW_PREWITT_S2)
img_gradient_prewitt = gradient_module(img_prewitt_s1, img_prewitt_s2)

create_figure_of_gradient_method(img_as_array, np.abs(img_prewitt_s1), img_gradient_prewitt, np.abs(img_prewitt_s2), border_processing(img_gradient_prewitt, 20),'Прюитт')
plt.tight_layout()
#show()

#laplacian_agreement
img_laplacian_agreement = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN_AGREEMENT_METHOD))
create_figure_of_laplacian_method(img_as_array, img_laplacian_agreement, border_processing(img_laplacian_agreement, 20),'Лапласиан согласованный')
plt.tight_layout()

#show()

#laplacian simple
img_laplacian = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN))
create_figure_of_laplacian_method(img_as_array, img_laplacian, border_processing(img_laplacian, 35),'Лапласиан вертикально-горизонтальный ')
plt.tight_layout()

#show()

#laplacian x
img_laplacian_mutually_perpendicular = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN_MUTUALLY_PERPENDICULAR))
create_figure_of_laplacian_method(img_as_array, img_laplacian_mutually_perpendicular, border_processing(img_laplacian_mutually_perpendicular, 25),'Лапласиан диагональный')
plt.tight_layout()

#show()


#laplacians sums
img_laplacian_of_sum_approximations = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN_OF_SUM_APPROXIMATIONS))
create_figure_of_laplacian_method(img_as_array, img_laplacian_of_sum_approximations, border_processing(img_laplacian_of_sum_approximations, 30),'Лапласиан суммарный')
plt.tight_layout()


show()




