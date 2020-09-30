import cv2
import matplotlib.pyplot as plt


def plot_create_image(data, title, fileName):
    fig, ax1 = plt.subplots(1, 1)

    im = ax1.imshow(data, interpolation='nearest')

    fg_color = 'black'
    bg_color = 'white'

    # IMSHOW
    # set title plus title color
    ax1.set_title(title, color=fg_color)

    # set figure facecolor
    ax1.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)

    fig.patch.set_facecolor(bg_color)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300)
    plt.show()
    # plt.savefig('save/to/pic.png', dpi=200, facecolor=bg_color)


def var_2(img1_info, img2_info):
    img1 = img1_info[0]
    img2 = img2_info[0]
    img1_name = img1_info[1]
    img2_name = img2_info[1]

    sift = cv2.SIFT_create()

    key_points_1, des1 = sift.detectAndCompute(img1, None)
    key_points_2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_points.append([m])

    coordinate_pairs = []
    for x in good_points:
        # print("queryIdx = " + str(x[0].queryIdx) + "   " "trainIdx = " + str(x[0].trainIdx ))
        coordinate_pairs.append([x[0].queryIdx, x[0].trainIdx])

    bias = []

    for c in coordinate_pairs:
        first = c[0]
        second = c[1]
        point_1 = key_points_1[first]
        point_2 = key_points_2[second]
        point_1_pt = point_1.pt
        point_2_pt = point_2.pt
        bias.append([point_1_pt[0] - point_2_pt[0], point_1_pt[1] - point_2_pt[1]])

    bias_x_middle = 0
    bias_y_middle = 0
    for b in bias:
        bias_x_middle += b[0]
        bias_y_middle += b[1]
    bias_x_middle /= len(bias)
    bias_y_middle /= len(bias)
    kp_1_test = [cv2.KeyPoint(20, 20, 0), cv2.KeyPoint(30, 30, 0)]
    kp_2_test = [cv2.KeyPoint(25, 25, 0), cv2.KeyPoint(35, 35, 0)]
    good_test = [[cv2.DMatch(0, 0, 0, 57)], [cv2.DMatch(1, 1, 0, 25)]]
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp_1_test, img2, kp_2_test, good_test, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()
    title = "X: " + str(round(bias_x_middle, 5)) + \
            " Y: " + str(round(bias_y_middle, 5))
    plot_create_image(img3, title, "results/" + img1_name + "_" + img2_name + ".jpg")
    return img3, [bias_x_middle, bias_y_middle]


img1 = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)  # first image
img2 = cv2.imread('images/2.jpg', cv2.IMREAD_GRAYSCALE)  # second image

result, bias = var_2([img1, '1'], [img2, '2'])
cv2.imwrite("result.jpg", result)
