import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
import time


class BlockDetector:
    def __init__(self, img_size, mode='height', small_block_size=25, large_block_size=38, max_block_num=6, depth_margin=2):
        if type(mode) is str:
            self.mode = mode
        elif type(mode) is tuple:
            self.mode = mode[0]
            self.depth_limit = mode[1]
        self.max_block_num = max_block_num
        self.img_size = img_size

        self.boarder_pos = np.array([[197, 116], [1083, 720]])
        self.base_pos = np.array([[539, 380], [725, 720]])
        self.positive_mask_pos = np.array([[0, 520], [1280, 720]])
        self.upper_area_mask_pos =  np.array([[0, 0], [1280, 520]])
        self.neg_left_mask_pos = np.array([[487, 543], [1280, 720]])
        self.neg_right_mask_pos = np.array([[0, 543], [725, 720]])
        self.area_masks = [self.positive_mask_pos, self.neg_left_mask_pos, self.neg_right_mask_pos]

        self.working_space = self.gen_working_space()
        self.current_working_space = self.working_space[0]
        self.small_block_size = small_block_size
        self.large_block_size = large_block_size
        self.depth_margin = depth_margin

        self.mask_all_blocks = None
        self.mask_hierarchy = None
        self.top_centers = None

        self.img_raw_depth = None  # H x W
        self.img_raw_rgb = None  # H x W x C

        # cv2.namedWindow('first_contours')
        # cv2.namedWindow('final contours')

        self.offset_kx = 0.025
        self.offset_ky = 0.035

        cv2.namedWindow('final_contours_0_0')
        cv2.namedWindow('final_contours_0_2')
        cv2.namedWindow('final_contours_0_4')
        cv2.namedWindow('final_contours_0_6')
        cv2.namedWindow('final_contours_0_8')
        cv2.namedWindow('final_contours_2_0')
        cv2.namedWindow('final_contours_2_4')
        cv2.namedWindow('final_contours_2_6')
        cv2.namedWindow('final_contours_2_8')
        cv2.namedWindow('final_contours_4_0')
        cv2.namedWindow('final_contours_4_2')
        cv2.namedWindow('final_contours_4_4')
        cv2.namedWindow('final_contours_4_6')
        cv2.namedWindow('final_contours_4_8')
        cv2.namedWindow('final_contours_6_0')
        cv2.namedWindow('final_contours_6_2')
        cv2.namedWindow('final_contours_6_4')
        cv2.namedWindow('final_contours_6_6')
        cv2.namedWindow('final_contours_6_8')
        cv2.namedWindow('final_contours_8_0')
        cv2.namedWindow('final_contours_8_2')
        cv2.namedWindow('final_contours_8_4')
        cv2.namedWindow('final_contours_8_6')
        cv2.namedWindow('final_contours_8_8')

        
    def gen_depth_mask(self, img_depth, depth_lower, depth_upper):
        """
        depth_img:  Image Obj
        """
        thresholded_depth = cv2.inRange(img_depth, depth_lower, depth_upper)
        # print(thresholded_depth.shape)
        # print(self.working_space)
        # print(self.current_working_space)
        depth_masked = cv2.bitwise_and(thresholded_depth, self.current_working_space[1])
        # depth_masked = cv2.bitwise_and(thresholded_depth, self.working_space[2][1])

        return depth_masked

    def gen_working_space(self):
        # TODO Find the exact working space in terms of our situation
        working_space_pos = np.zeros(self.img_size, dtype=np.uint8)
        cv2.rectangle(working_space_pos, tuple(self.boarder_pos[0]), tuple(self.boarder_pos[1]), 255, cv2.FILLED)
        cv2.rectangle(working_space_pos, tuple(self.base_pos[0]), tuple(self.base_pos[1]), 0, cv2.FILLED)
        cv2.rectangle(working_space_pos, tuple(self.positive_mask_pos[0]), tuple(self.positive_mask_pos[1]), 0, cv2.FILLED)

        working_space_neg_left = np.zeros(self.img_size, dtype=np.uint8)
        cv2.rectangle(working_space_neg_left, tuple(self.boarder_pos[0]), tuple(self.boarder_pos[1]), 255, cv2.FILLED)
        cv2.rectangle(working_space_neg_left, tuple(self.base_pos[0]), tuple(self.base_pos[1]), 0, cv2.FILLED)
        cv2.rectangle(working_space_neg_left, tuple(self.neg_left_mask_pos[0]), tuple(self.neg_left_mask_pos[1]), 0, cv2.FILLED)
        cv2.rectangle(working_space_neg_left, tuple(self.upper_area_mask_pos[0]), tuple(self.upper_area_mask_pos[1]), 0, cv2.FILLED)

        working_space_neg_right = np.zeros(self.img_size, dtype=np.uint8)
        cv2.rectangle(working_space_neg_right, tuple(self.boarder_pos[0]), tuple(self.boarder_pos[1]), 255, cv2.FILLED)
        cv2.rectangle(working_space_neg_right, tuple(self.base_pos[0]), tuple(self.base_pos[1]), 0, cv2.FILLED)
        cv2.rectangle(working_space_neg_right, tuple(self.neg_right_mask_pos[0]), tuple(self.neg_right_mask_pos[1]), 0, cv2.FILLED)
        cv2.rectangle(working_space_neg_right, tuple(self.upper_area_mask_pos[0]), tuple(self.upper_area_mask_pos[1]), 0, cv2.FILLED)

        return [(0, working_space_pos), (1, working_space_neg_left), (2, working_space_neg_right)]

    @staticmethod
    def find_gradient(img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        x_kernel = np.array([[1, -1]])
        y_kernel = np.array([[1], [-1]])
        Ix = scipy.ndimage.convolve(img, x_kernel)
        Iy = scipy.ndimage.convolve(img, y_kernel)
        gradient = np.array([Ix, Iy])

        return gradient
        # return Ix, Iy

    def find_top_centers(self, mask):
        if cv2.__version__.startswith('4'):
            contours_depth, _ = cv2.findContours(self.mask_all_blocks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours_depth, _ = cv2.findContours(self.mask_all_blocks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_depth = self._filter_contours(contours_depth)

        centers = []
        for cnt in contours_depth:
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))
            # print(box)
            # cv2.drawContours(self.img_raw_rgb, [box], 0, (255, 0, 0), 1)
            img_x, img_y = rect[0]
            img_x, img_y = int(img_x), int(img_y)
            img_z = self.img_raw_depth[img_y][img_x]
            centers.append((img_x, img_y, img_z))

        # cv2.imshow('rectangle', self.img_raw_rgb)
        # cv2.waitKey(0)

        return centers

    def find_centers_hierarchy(self, img_depth, upper_margin, lower_margin, window_name):
        # time_begin = time.time()
        if self.mode == 'depth':
            img_depth = self.depth_limit - img_depth

        # gen masks for all blocks
        self.img_raw_depth = img_depth
        lower_depth_all = 0 - self.depth_margin
        upper_depth_all = self.max_block_num * self.large_block_size + self.depth_margin

        # print(img_depth)
        # time_mask_all_blocks_start = time.time()
        self.mask_all_blocks = self.gen_depth_mask(img_depth, lower_depth_all, upper_depth_all)
        # time_mask_all_blocks_end = time.time()

        # print('time_mask_all_blocks_end:', time_mask_all_blocks_end - time_mask_all_blocks_start)

        
        # time_find_all_contours_start = time.time()
        # find contours of all blocks
        if cv2.__version__.startswith('4'):
            contours_depth_all, _ = cv2.findContours(self.mask_all_blocks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours_depth_all, _ = cv2.findContours(self.mask_all_blocks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # time_find_all_contours_end = time.time()
        # print('time_find_all_contours_start', time_find_all_contours_end - time_find_all_contours_start)
        
        # print(len(contours_depth_all))
        c = np.zeros(self.img_size)

        # print(type(self.img_raw_rgb))
        # tmp_rgb = self.img_raw_rgb.copy()
        # cv2.drawContours(tmp_rgb, contours_depth_all, -1, color=(255, 0, 0))
        # cv2.imshow('first_contours', tmp_rgb)
        
        

        contours_depth_all = self._filter_contours(contours_depth_all)
        # print(contours_depth_all)

        # cv2.drawContours(c, contours_depth_all, -1, (0, 255, 255), 3)
        # cv2.imshow('c', c)
        # cv2.waitKey(0) 

        # find the bounding-box for each contour
        top_rects = []
        top_centers = []
        top_contours = []

        # time_cnt_for_loop_begin = time.time()
        for cnt in contours_depth_all:
            max_height, filled_pixels = self.find_max_height(cnt)
            # lower_bound = int(max_height - self.large_block_size)
            # upper_bound = int(max_height + self.depth_margin)
            lower_bound = int(max_height - lower_margin)
            upper_bound = int(max_height + upper_margin)

            # print('max height:', max_height)
            # print('lower:', lower_bound)
            # print('upper:', upper_bound)
            # print('_'*20)
            if lower_bound < 0:
                lower_bound = 0

            if self.mode == 'depth':
                lower_depth_all = self.depth_limit - upper_depth_all
                upper_depth_all = self.depth_limit - lower_depth_all

            cnt_mask = np.zeros(self.img_size, dtype=np.uint8)
            cnt_mask[np.array(filled_pixels).T[0], np.array(filled_pixels).T[1]] = 255

            top_block_mask = self.gen_depth_mask(self.img_raw_depth, lower_bound, upper_bound)
            top_block_mask = top_block_mask * cnt_mask

            if cv2.__version__.startswith('4'):
                top_cnt, _ = cv2.findContours(top_block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            else:
                _, top_cnt, _ = cv2.findContours(top_block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            rect = cv2.minAreaRect(top_cnt[0])
            center = np.int0(rect[0])

            top_centers.append(self.compensate_offset_center(center))
            top_rects.append(self.compensate_offset_rect(rect))
            top_contours.append(top_cnt[0].squeeze())
        # time_cnt_for_loop_end = time.time()
        # print('time_cnt_for_loop', time_cnt_for_loop_end - time_cnt_for_loop_begin)

        tmp_rgb_final = self.img_raw_rgb.copy()
        try:
            cv2.drawContours(tmp_rgb_final, top_contours, -1, color=(255, 0, 0))
        except cv2.error:
            # print(len(top_contours))
            for rect in top_rects:
                if rect[1] == (0,0):
                    print(rect)
                print(top_rects)
        cv2.imshow(window_name, tmp_rgb_final)

        # def on_trackbar(upper_val):
        #     alpha = val / alpha_slider_max
        #     beta = ( 1.0 - alpha )
        #     cv2.drawContours(tmp_rgb_final, top_contours, -1, color=(255, 0, 0))
        #     cv.imshow('final_contours', tmp_rgb_final)



        return top_contours, top_rects, top_centers

    def compensate_offset_center(self, center):
        offset_x = int(np.abs(center[0]- 640) * self.offset_kx)
        offset_y = int(np.abs(center[1]- 360) * self.offset_ky)
        
        if center[0] > 640:
            new_x = center[0] + offset_x
        else:
            new_x = center[0] - offset_x
        
        if new_x < 0:
            new_x = int(0)
        elif new_x >=1280:
            new_x = 1279

        if center[1] > 360:
            new_y = center[1] + offset_y
        else:
            new_y = center[1] - offset_y
        
        if new_y < 0:
            new_y = int(0)
        elif new_y >= 720:
            new_y = 719

        return new_x, new_y

    def compensate_offset_rect(self, rect):
        new_rect = list(rect)
        new_rect[0] = self.compensate_offset_center(rect[0])

        return tuple(new_rect)

    def find_max_height(self, cnt):
        # Find the max height of blocks, given ONE contour
        filled_pixels = self.find_pxl_inside_cnt(cnt)
        heights = []
        for x, y in filled_pixels:
            height = self.img_raw_depth[x][y]
            heights.append(height)

        max_height = np.argmax(np.bincount(sorted(heights, reverse=True)))

        return max_height, filled_pixels

    def find_pxl_inside_cnt(self, cnt):
        """
        cnt: [ [], [], [], ... [] ]
        """
        canvas = np.zeros(self.img_size)
        cv2.drawContours(canvas, [cnt], -1, color=(255, 255, 255), thickness=-1)

        # cv2.imshow('canvas', canvas)
        # cv2.waitKey(0)
        pixels = np.where(canvas != 0)
        # num_pixels = len(pixels[0])
        # filled_pixels = []
        # for i in range(num_pixels):
        #     x, y = pixels[0][i], pixels[1][i]
        #     filled_pixels.append((x, y))

        tmp = np.array([pixels[0], pixels[1]])
        filled_pixels = tmp.T

        return filled_pixels

    def set_img_rgb(self, img_rgb):
        self.img_raw_rgb = img_rgb

    @staticmethod
    def _filter_contours(contours, thresh_area=200, thresh_pos=None):
        contours_filtered = []
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area >= thresh_area:
                contours_filtered.append(cnt)

        return contours_filtered


def main():
    img_depth_path = "/home/student_am/armlab_opencv_examples/depth_blocks.png"
    img_rgb_path = "/home/student_am/armlab_opencv_examples/image_blocks.png"

    img_depth = cv2.imread(img_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # W x H
    img_rgb = cv2.imread(img_rgb_path)  # W x H x C

    detector = BlockDetector(img_size=img_depth.shape, mode=('depth', 1205))
    detector.set_img_rgb(img_rgb)

    rects = detector.find_centers_hierarchy(img_depth)

    return


if __name__ == '__main__':
    main()
