import numpy as np
import cv2
import matplotlib.pyplot as plt

# Считаем среднее смещение
def calculate_avg(bias):
    bias_x_avg = 0
    bias_y_avg = 0

    for b in bias:
        bias_x_avg += b[0]
        bias_y_avg += b[1]

    bias_x_avg /= len(bias)
    bias_y_avg /= len(bias)
    return bias_x_avg, bias_y_avg


# считаем среднее смещение
def calculate_bias(coordinates):
    bias = []

    i = 0
    for coor in coordinates:
        x1 = coor[0][0]
        y1 = coor[0][1]
        x2 = coor[1][0]
        y2 = coor[1][1]
        bias.append([x2 - x1, y2 - y1])

    bias_x_avg, bias_y_avg = calculate_avg(bias)

    return bias_x_avg, bias_y_avg

# рисуем результирующее изображение, на котором отмечаем смещение
def plot_create_image(data, title, fileName):
    fig, ax1 = plt.subplots(1, 1)

    im = ax1.imshow(data, interpolation='nearest')

    fg_color = 'black'
    bg_color = 'white'

    # IMSHOW
    # устанавилваем цвет заголовка
    ax1.set_title(title, color=fg_color)

    ax1.patch.set_facecolor(bg_color)

    # установить цвет меток
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)

    fig.patch.set_facecolor(bg_color)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300)

    plt.show()
    # plt.savefig('save/to/pic.png', dpi=200, facecolor=bg_color)


# отрисовываем и сохраняем совмещенное итоговое изображение
def createImages(img1, img2, coordinates, resultPathName):
    key_points_1 = []
    key_points_2 = []
    allignment_points = []

    i = 0
    for coor in coordinates:
        x1 = coor[0][0]
        y1 = coor[0][1]
        x2 = coor[1][0]
        y2 = coor[1][1]
        key_points_1.append(cv2.KeyPoint(x1, y1, 0))
        key_points_2.append(cv2.KeyPoint(x2, y2, 0))
        allignment_points.append([cv2.DMatch(i, i, 0, 57)])
        i += 1

    # рисуем результат, совмещаем точки первого и второго изображения
    img_result = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, allignment_points, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # считаем смещение
    bias_x_middle, bias_y_middle = calculate_bias(coordinates)

    # заголовок результирующего изображения
    title = "X: " + str(round(bias_x_middle, 5)) + \
            " Y: " + str(round(bias_y_middle, 5))

    # рисуем и сохраняем результат
    plot_create_image(img_result, title, resultPathName)

def loadImage(image_name):
    img1 = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)  # first image
    cap1 = cv2.VideoCapture(image_name)
    return img1, cap1


def main():

    image1_name = 'images/1.jpg'
    image2_name = 'images/2.jpg'

    img1, cap1 = loadImage(image1_name)
    img2, cap2 = loadImage(image2_name)


    # считываем первого изображение
    ret, old_frame = cap1.read()
    old_frame = img1
    # переводим изображение в грейСкейл (черно-белое)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # считываем второе изображение
    ret, frame = cap2.read()
    # переводим изображение в грейСкейл (черно-белое)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # параметры углового детектора Ши-Томаси, который
    # ищет особые точки по которым далее будет рассчитываться оптичискей поток Лукаса-Канаде
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # находит N сильнейших углов на изображении
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


    # Параметры алгоритма Лукааса - Канаде
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Расчитываем оптический поток методом Лукаса-Канаде
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Выбираем координаты хороших точек на первом изображении
    good_old = p0[st == 1]

    # Выбираем координаты хороших точек на втором изображении
    good_new = p1[st == 1]

    # отображаем смещение изображения
    draw_points = []  # точки по которым рисуем
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        draw_points.append([[x1, y1], [x2, y2]])
        print('(' + str(x1) + ' ; ' + str(y1) + ')' + ' - ' + '(' + str(x2) + ' ; ' + str(y2) + ')')

    #отрисовываем и сохраняем совмещенное итоговое изображение
    createImages(img1, img2, draw_points, "results/1_2.jpg")



if __name__ == "__main__":
    main()
