import cv2
import numpy as np
import scipy.cluster.hierarchy as hcluster
import math


def crop_image(img, xmin, ymin, xmax, ymax):
    """
    crops image given the (x, y) of the top left coordinates
    and the length (l) and width (w) of the rectangle to be cropped
    if l and w exceed the bottom/right side of the original image,
    then it will crop up to the bottom/right image borders
    """
    ret = img[ymin:ymax, xmin:xmax]
    return ret


def increase_contrast(img):
    """
    increases contrast of an image by **** amount?
    """
    contrast = 50
    ret = img.copy()

    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    
    ret = cv2.addWeighted(ret, alpha_c, ret, 0, gamma_c)

    return ret

def detect_ink_blocks(img):
    """
    returns list of bounding boxes for identified ink clusters
    [(xmin, ymin, xmax, ymax)]
    """
    # THRESHOLDING --> binary image
    img_copy = img.copy()
    ret,binary = cv2.threshold(img_copy,70,255,0)

    # IDENTIFY INKED COORDINATES
    ink_coords = []
    for row in range(0, len(binary), 2):
        for col in range(0, len(binary[row]), 2):
            if binary[row][col] < 100: ink_coords.append([row,col])

    data = np.array(ink_coords)

    # CLUSTERING
    ink_density = get_ink_density(img)
    cluster_thresh = get_clustering_threshold(ink_density)
    # clusters is a list whose entries = cluster number, index corresponding
    # to the same index in data / ink_coords
    clusters = hcluster.fclusterdata(data, cluster_thresh, criterion="distance")

    # IDENTIFY BOUNDING BOXES OF CLUSTERS
    coord_by_cluster = []
    for i in range(max(clusters)):
        coord_by_cluster.append([])
    # populate list: index = cluster number
    # entries = coordinates in that cluster
    # iterate through all coordinates
    for ind in range(len(ink_coords)):
        coord = ink_coords[ind]
        coord = [coord[1], coord[0]]
        cluster_num = clusters[ind]-1

        coord_by_cluster[cluster_num].append(coord)
        
    
    cluster_blocks = [] 
    # find min x, min y, max x, max y of each cluster
    for cluster in coord_by_cluster:
        minx = math.inf
        miny = math.inf
        maxx = 0
        maxy = 0

        for coord in cluster:
            x = coord[0]
            y = coord[1]

            if x < minx: minx = x
            elif x > maxx: maxx = x

            if y < miny: miny = y
            elif y > maxy: maxy = y

        cluster_blocks.append((minx, miny, maxx, maxy))

    return coord_by_cluster, cluster_blocks


def get_ink_density(img):

    # THRESHOLDING --> binary image
    img_copy = img.copy()
    ret,binary = cv2.threshold(img_copy,70,255,0)

    ink = 0
    total = 0
    for row in range(0, len(binary), 2):
        for col in range(0, len(binary[row]), 2):
            total += 1
            if binary[row][col] < 100: ink += 1
    
    return ink/total

def get_clustering_threshold(ink_density):
    # lower ink density (~0.015) does well with a higher clustering 
    # threshold (>75)
    # higher ink density (>= ~0.05) does well with lower clustering
    # threshold (20< thresh <25)

    # clustering threshold should be between 20 and 80
    # ink density is generally between 0.01 and 0.07
    if ink_density <= 0.017:
        return 75
    if ink_density <= 0.025:
        return 60
    if ink_density <= 0.03:
        return 50
    if ink_density <= 0.04:
        return 35
    else: return 25

