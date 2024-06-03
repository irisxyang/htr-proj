import io
import os
import json
import cv2
import math
import numpy as np

import image_processing

from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse

credential_path = '/Users/irisxyang/Desktop/term_project/venv/htr-422302-227236128117.json'

def run_extraction_multiple_pages(pages):
    all_transcripts = []
    all_sketches = []

    for page in pages:
        transcript, sketches = run_extraction(page)
        all_transcripts.append(transcript)
        all_sketches.append(sketches)

    return all_transcripts, all_sketches

def run_extraction(image_cv2):
    processed_image = process_image(image_cv2)
    response = run_text_detection(processed_image, credential_path)
    word_box_dict, block_score_dict, block_text_dict, blocks = get_bounding_boxes(response)
    transcript = response['text']

    # identify bounding boxes for ink blocks
    coord_in_block, ink_blocks = image_processing.detect_ink_blocks(processed_image)

    final_text_blocks, final_sketch_blocks = refine_blocks(blocks, ink_blocks, coord_in_block, word_box_dict, block_score_dict)

    sketches = []
    num = 0
    for sketch_block in final_sketch_blocks:
        xmin = sketch_block[0] - 10
        ymin = sketch_block[1] - 10
        xmax = sketch_block[2] + 10
        ymax = sketch_block[3] + 10

        cropped_sketch = image_processing.crop_image(image_cv2, xmin, ymin, xmax, ymax)

        save_folder = '/Users/irisxyang/Desktop/term_project/extracted'
        file_name = 'extracted_page'
        cur_file = file_name + "1_sketch" + str(num) + ".png"
        new_image_path = os.path.join(save_folder, cur_file)
        cv2.imwrite(new_image_path, cropped_sketch)
        sketches.append(cropped_sketch)
        num += 1
    
    return transcript, sketches


def process_image(image_cv2):
    """
    returns processed version of cv2 image (also in cv2 format)
    """

    processed_image = image_processing.increase_contrast(image_cv2)

    return processed_image

def run_text_detection(image_cv2, credential_path):
    """
    runs text detection on the image
    returns dictionary representation of the full text annotations

    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    client = vision.ImageAnnotatorClient()

    success, encoded_image = cv2.imencode('.jpg', image_cv2)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    # convert --> json --> dict output
    response_json = AnnotateImageResponse.to_json(response)
    response_output = json.loads(response_json)

    return response_output['fullTextAnnotation']


def get_bounding_boxes(response):
    """
    returns
        word dict: key=text, val=bounding box
        block score dict: key=bounding box, val=confidence
        block text dict: key=bounding box, val=text
        x val dict: key=x-coord, val=(text, y-coord)
        y val dict: key=y-coord, val=(text, x-coord)
    """

    word_box_dict = {}
    block_score_dict = {}
    block_text_dict = {}
    blocks = []

    for page in response['pages']:
        for block in page['blocks']:
            confidence = block['confidence']

            block_text = ''
            for paragraph in block['paragraphs']:
                for word in paragraph['words']:
                    word_box = calculate_box(word['boundingBox']['vertices'])
                    text = ''.join([symbol['text'] for symbol in word['symbols']])

                    word_box_dict[word_box] = text

                    block_text = ' '.join(join_punctuation([block_text, text]))
            
            block_box = calculate_box(block['boundingBox']['vertices'])

            block_score_dict[block_box] = confidence
            block_text_dict[block_box] = block_text
            blocks.append(block_box)
    
    return word_box_dict, block_score_dict, block_text_dict, blocks

def calculate_box(bounding_box):
    minx = math.inf
    maxx = 0
    miny = math.inf
    maxy = 0

    for coords in bounding_box:
        x = coords['x']
        y = coords['y']

        if x < minx: minx = x
        elif x > maxx: maxx = x

        if y < miny: miny = y
        elif y > maxy: maxy = y

    return (minx, miny, maxx, maxy)

def join_punctuation(seq, characters='.,;?!'):
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current
    
def refine_blocks(text_blocks, ink_blocks, coords_in_block, word_box_dict, block_score_dict):

    # 1. combine text bounding boxes 
    merged_text_blocks = []
    is_overlap = [False] * len(text_blocks)
    for i in range(len(text_blocks)):
        box1 = text_blocks[i]
        for j in range(i+1, len(text_blocks)):
            box2 = text_blocks[j]
            percent_overlap = overlap(box1, box2)
            # if overlapped enough, merge the two blocks into 1
            if percent_overlap > 0.75:
                box1_area = get_box_area(box1)
                box2_area = get_box_area(box2)
                box1_weight = box1_area / (box1_area + box2_area)
                box2_weight = box2_area / (box1_area + box2_area)

                merged_box = merge_boxes(box1, box2)
                merged_text_blocks.append(merged_box)

                block_score_dict[merged_box] = box1_weight*block_score_dict[box1] + box2_weight*block_score_dict[box2]

                is_overlap[i] = True
                is_overlap[j] = True
        # if we have no overlap, add original text box to final text block list
        if is_overlap[i] == False:
            merged_text_blocks.append(box1)

    # get average text block size
    avg_text_block_size = 0
    for block in merged_text_blocks:
        avg_text_block_size += get_box_area(block)
    avg_text_block_size = avg_text_block_size/len(merged_text_blocks)

    # get average word size
    avg_word_size = 0
    for word_box in word_box_dict.keys():
        avg_word_size += get_box_area(word_box)
    avg_word_size = avg_word_size/len(word_box_dict.keys())

    # 2. iterate through all ink blocks and categorize as
    # either text or sketch
        # if text: vast majority of ink is included in a text block
        # if sketch: majority of ink is not included in text block
    final_sketch_blocks = []
    for ink_block in range(len(ink_blocks)):
        block_points = coords_in_block[ink_block]
        # remove all negligible ink block areas

        ink_block_area = get_box_area(ink_blocks[ink_block])
        if ink_block_area < 0.5* avg_word_size:
            # consider < 0.10*avg_text_block_size
            continue

        # iterate through all coordinates in an ink block
        num_coords_in_block = 0
        
        for coord in block_points:
            for text_block in merged_text_blocks:
                if check_coord_in_box(coord, text_block):
                    num_coords_in_block += 1
        # get percentage of ink coordinates in block that are a part of 
        # any text block
        percent_coords_in_block = num_coords_in_block / len(block_points)

        # if over 80% of coordinates are in some text block
        # consider that text
        if percent_coords_in_block > 0.96:
            continue
        # else consider it part of a sketch, add corresponding bounding
        # box to the final sketch
        else:
            final_sketch_blocks.append(ink_blocks[ink_block])

    # get avg text block confidence
    avg_text_block_confidence = 0
    for block in block_score_dict.keys():
        avg_text_block_confidence += block_score_dict[block]
    avg_text_block_confidence = avg_text_block_confidence / len(block_score_dict.keys())

    # 3. iterate through final sketch blocks, and any text that is > x% overlapped with
    # the final sketch is not considered a text block
    final_text_blocks = []
    for sketch_block in final_sketch_blocks:
        for text_block in merged_text_blocks:
            percent_overlap = overlap(sketch_block, text_block)

            if percent_overlap > 0.5 or block_score_dict[text_block] < 0.5*avg_text_block_confidence:
                continue
            else:
                final_text_blocks.append(text_block)
    
    return final_text_blocks, final_sketch_blocks


def merge_boxes(box1, box2):
    new_xmin = min(box1[0], box2[0])
    new_ymin = min(box1[1], box2[1])
    new_xmax = max(box1[2], box2[2])
    new_ymax = max(box1[3], box2[3])
    
    return (new_xmin, new_ymin, new_xmax, new_ymax)


def check_coord_in_box(coord, box):
    xcoord = coord[0]
    ycoord = coord[1]

    box_xmin = box[0]
    box_ymin = box[1]
    box_xmax = box[2]
    box_ymax = box[3]

    if (xcoord >= box_xmin and xcoord <= box_xmax) and (ycoord >= box_ymin and ycoord <= box_ymax):
        return True
    else: return False

def get_box_area(box):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    xlen = xmax-xmin
    ylen = ymax-ymin
    return (xlen*ylen)

def overlap(box1, box2):
    """
    checks if 2 boxes are overlapping (by over x% ?)
    if not, return -1
    if yes, return a float (0,1] that indicates the max %
    overlap between the 2 boxes
    """
    xmin1 = box1[0]
    ymin1 = box1[1]
    xmax1 = box1[2]
    ymax1 = box1[3]

    xmin2 = box2[0]
    ymin2 = box2[1]
    xmax2 = box2[2]
    ymax2 = box2[3]

    #change in x = smallest of the largest x
    # - largest of smallest x
    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)

    overlap_area = dx*dy
    box1_area = get_box_area(box1)
    box2_area = get_box_area(box2)

    # check that dx is nonneg, else we might have neg dx and neg dy
    # which would yield nonneg overlap_area
    if overlap_area > 0 and (dx >= 0):
        # return the max % overlap of the 2 boxes
        # i.e. the percentage overlap for the box
        # whose area is more covered by the overlap 
        return min((overlap_area)/(min(box1_area, box2_area)), 1)
    else: return -1

