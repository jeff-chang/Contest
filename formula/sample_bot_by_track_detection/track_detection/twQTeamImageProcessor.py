#!env python
#
# TW Site Q-Team self-driving Bot
#
# Revision:      v1.0
# Released Date: Sep 12, 2018
#

import cv2
import numpy as np
from datetime import datetime

class twQTeamImageProcessor(object):

    def __init__(self, track_mode=0):

        self.m_img_width    = 320
        self.m_img_height   = 240
        self.m_track_mode   = track_mode
        self.m_debug_mode   = False
        self.m_crop_ratio   = 0.55
        self.m_cnt_reverse  = 0
        self.m_line_results = None

    def showImage(self, img, name="image", scale=1.0):

        if scale and scale != 1.0:
            newsize = (int(img.shape[1]*scale), int(img.shape[0]*scale))
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_LINEAR)

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)

    def setTrackMode(self, track_mdoe):

        # mode 0 represents |bW|r|b|r|g|b|g|bW|
        # mode 1 represents |bW|g|b|g|r|b|r|bW|
        # mode 2 represents |bW|r|b|r|yW|
        # mode 3 represents |yW|g|b|g|bW|
        # mode 4 represents |bW|g|b|g|yW|
        # mode 5 represents |yW|g|b|g|bW|
        self.m_track_mode = track_mdoe

    def showCurrentTrack(self, line_idx):
        # mode 0 represents |bW|r|b|r|g|b|g|bW|
        if self.m_track_mode == 0:
            if line_idx == 0:
                print('On track |BlackWall|Red|Blue|')
            elif line_idx == 1:
                print('On track |Red|Blue|Red|')
            elif line_idx == 2:
                print('On track |Blue|Red|Green|')
            elif line_idx == 3:
                print('On track |Red|Green|Blue|')
            elif line_idx == 4:
                print('On track |Green|Blue|Green|')
            elif line_idx == 5:
                print('On track |Blue|Green|BlackWall|')
        # mode 1 represents |bW|g|b|g|r|b|r|bW|
        elif self.m_track_mode == 1:
            if line_idx == 0:
                print('On track |BlackWall|Green|Blue|')
            elif line_idx == 1:
                print('On track |Green|Blue|Green|')
            elif line_idx == 2:
                print('On track |Blue|Green|Red|')
            elif line_idx == 3:
                print('On track |Green|Red|Blue|')
            elif line_idx == 4:
                print('On track |Red|Blue|Red|')
            elif line_idx == 5:
                print('On track |Blue|Red|BlackWall|')

    def cropImage(self, img):

        crop_ratios = (self.m_crop_ratio , 1.0)
        crop_slice  = slice(*(int(cnt * img.shape[0]) for cnt in crop_ratios))
        crop_img    = img[crop_slice, :, :]

        return crop_img

    def brightenImage(self, img):

        max_pixel = img.max()
        max_pixel = max_pixel if max_pixel != 0 else 255

        brighten_img = img * (255 / max_pixel)
        brighten_img = np.clip(brighten_img, 0, 255)
        brighten_img = np.array(brighten_img, dtype=np.uint8)

        return brighten_img

    def flattenImage(self, img):

        bChn_img, gChn_img, rChn_img = cv2.split(img)
        max_pixel = np.maximum(np.maximum(bChn_img, gChn_img), rChn_img)
        b_filter  = (bChn_img == max_pixel) & (bChn_img >= 120) & (gChn_img < 150) & (rChn_img < 150)
        g_filter  = (gChn_img == max_pixel) & (gChn_img >= 120) & (bChn_img < 150) & (rChn_img < 150)
        r_filter  = (rChn_img == max_pixel) & (rChn_img >= 120) & (bChn_img < 150) & (gChn_img < 150)
        y_filter  = ((bChn_img >= 128) & (gChn_img >= 128) & (rChn_img < 100))
        bChn_img[y_filter]            = 255
        gChn_img[y_filter]            = 255
        rChn_img[np.invert(y_filter)] = 0
        rChn_img[r_filter], rChn_img[np.invert(r_filter)] = 255, 0
        bChn_img[b_filter], bChn_img[np.invert(b_filter)] = 255, 0
        gChn_img[g_filter], gChn_img[np.invert(g_filter)] = 255, 0
        flat_img = cv2.merge([bChn_img, gChn_img, rChn_img])

        return flat_img

    def getExtWallImage(self, img):

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])
        bWall_img   = cv2.inRange(img, lower_black, upper_black)

        return bWall_img

    def getForkWallImage(self, bWall_img, flat_img):

        bChn_img, gChn_img, rChn_img = cv2.split(flat_img)
        yWall_img = np.invert(bChn_img | gChn_img | rChn_img | bWall_img)

        return yWall_img

    def getWallImage(self, bWall_img, yWall_img):

        allWall_img = (bWall_img | yWall_img)

        return allWall_img

    def getEdgePoint(self, line):

        line = line > 0
        pos_on  = []
        pos_off = []
        pos_chg = np.where(line[:-1] != line[1:])[0]
        pos_chg = np.asarray(pos_chg)

        thrd_linked = 20
        idx_remove = []
        for cnt in range(0, int(len(pos_chg)/ 2), 2):
            if abs(pos_chg[cnt] - pos_chg[cnt+1]) < thrd_linked:
                idx_remove.append(cnt)
                idx_remove.append(cnt+1)
        pos_chg = np.delete(pos_chg, idx_remove)

        thrd_boundary = 5
        idx_remove = []
        for cnt in pos_chg:
            if cnt < thrd_boundary or cnt >= self.m_img_width-thrd_boundary:
                idx_remove.append(cnt)
        pos_chg = np.delete(pos_chg, idx_remove)

        for cnt in pos_chg:
            if line[cnt] == True:
                pos_off.append(cnt)
            else:
                pos_on.append(cnt)

        return pos_on, pos_off

    def getEdgeLine(self, loc_y, posEdgePointLeft, posEdgePointRight, line_edges):

        idx_remove_left = []
        idx_remove_right = []
        min_gap = 5
        max_gap = self.m_img_width + self.m_img_height
        for cnt1 in range(0, len(posEdgePointLeft)):
            dist_len = max_gap
            dist_idx = -1
            for cnt2 in range(0, len(posEdgePointRight)):
                if abs(posEdgePointLeft[cnt1] - posEdgePointRight[cnt2]) < dist_len:
                    dist_len = abs(posEdgePointLeft[cnt1] - posEdgePointRight[cnt2])
                    dist_idx = cnt2

            if dist_len < min_gap:
                idx_remove_left.append(cnt1)
                idx_remove_right.append(dist_idx)
                loc_x = int((posEdgePointLeft[cnt1] + posEdgePointRight[dist_idx]) / 2)
                line_edges.append((loc_x, loc_y))
                continue

        new_posEdgePointLeft = np.delete(posEdgePointLeft, idx_remove_left)
        new_posEdgePointRight = np.delete(posEdgePointRight, idx_remove_right)

        return new_posEdgePointLeft, new_posEdgePointRight

    def plotEdgeLine(self, img, posEdgeLines, color):

        shift_y = int(self.m_img_height * self.m_crop_ratio)
        for cnt in range(0, len(posEdgeLines)-1):
            loc_sta = (int(posEdgeLines[cnt][0]), int(posEdgeLines[cnt][1] + shift_y))
            loc_end = (int(posEdgeLines[cnt+1][0]), int(posEdgeLines[cnt+1][1] + shift_y))
            cv2.line(img, loc_sta, loc_end, color, 2)

    def doEdgeDetection(self, img, proc_img):

        line_edges = [[],[],[],[],[],[],[]]
        line_rev   = [[],[],[]]
        for loc_y in range(0, int(self.m_img_height*(1-self.m_crop_ratio))):
            bTrack_posOn, bTrack_posOff = self.getEdgePoint(proc_img[0][loc_y, :])
            gTrack_posOn, gTrack_posOff = self.getEdgePoint(proc_img[1][loc_y, :])
            rTrack_posOn, rTrack_posOff = self.getEdgePoint(proc_img[2][loc_y, :])
            b_Wall_posOn, b_Wall_posOff = self.getEdgePoint(proc_img[3][loc_y, :])
            y_Wall_posOn, y_Wall_posOff = self.getEdgePoint(proc_img[4][loc_y, :])
            bgWall_posOn, bgWall_posOff = self.getEdgePoint(proc_img[5][loc_y, :])

            if self.m_track_mode == 0:
                bgWall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, gTrack_posOn, line_rev[0])
                gTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, rTrack_posOn, line_rev[1])
                rTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bgWall_posOn, line_rev[2])
                bgWall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, rTrack_posOn, line_edges[0])
                rTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bTrack_posOn, line_edges[1])
                bTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, rTrack_posOn, line_edges[2])
                rTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, gTrack_posOn, line_edges[3])
                gTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bTrack_posOn, line_edges[4])
                bTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, gTrack_posOn, line_edges[5])
                gTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bgWall_posOn, line_edges[6])
            elif self.m_track_mode == 1:
                bgWall_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, rTrack_posOn, line_rev[0])
                rTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, gTrack_posOn, line_rev[1])
                gTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bgWall_posOn, line_rev[2])
                bgWall_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bgWall_posOff, gTrack_posOn, line_edges[0])
                gTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, bTrack_posOn, line_edges[1])
                bTrack_posOff, gTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, gTrack_posOn, line_edges[2])
                gTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, gTrack_posOff, rTrack_posOn, line_edges[3])
                rTrack_posOff, bTrack_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bTrack_posOn, line_edges[4])
                bTrack_posOff, rTrack_posOn = self.getEdgeLine(loc_y, bTrack_posOff, rTrack_posOn, line_edges[5])
                rTrack_posOff, bgWall_posOn = self.getEdgeLine(loc_y, rTrack_posOff, bgWall_posOn, line_edges[6])

        # plot
        if self.m_debug_mode == True:
            color = (255, 255, 255)
            for cnt in range(0, len(line_edges)):
                line = line_edges[cnt]
                self.plotEdgeLine(img, line, color)
            self.showImage(img, name='draw_img', scale=1.2)

        line_results = []
        line_results.append(line_edges)
        line_results.append(line_rev)

        return line_results

    def processImage(self, img):

        crop_img    = self.cropImage(img)
        # correct_img = self.brightenImage(crop_img)
        correct_img = crop_img
        flat_img    = self.flattenImage(correct_img)
        bWall_img   = self.getExtWallImage(correct_img)
        yWall_img   = self.getForkWallImage(bWall_img, flat_img)
        allWall_img = self.getWallImage(bWall_img, yWall_img)

        bChn_img, gChn_img, rChn_img = cv2.split(flat_img)

        proc_img = []
        proc_img.append(bChn_img)
        proc_img.append(gChn_img)
        proc_img.append(rChn_img)
        proc_img.append(bWall_img)
        proc_img.append(yWall_img)
        proc_img.append(allWall_img)

        # edge detection
        self.m_line_results = self.doEdgeDetection(img, proc_img)

    def isReverseTrack(self):

        bIsReverse = False
        line_rev = self.m_line_results[1]

        for line in line_rev:
            if len(line) >= 5:
                self.m_cnt_reverse += 1
                bIsReverse = True

        return bIsReverse

    def currentTrackIndex(self):

        line_edges = self.m_line_results[0]
        idx_max    = -1
        len_max    = -1
        idx_second = -1
        len_second = -1
        for cnt in range(0, len(line_edges)):
            if len(line_edges[cnt]) > len_max:
                idx_second = idx_max
                len_second = len_max
                idx_max = cnt
                len_max = len(line_edges[cnt])
            elif len(line_edges[cnt]) > len_second:
                idx_second = cnt
                len_second = len(line_edges[cnt])

        current_track_idx = -1
        if abs(idx_max - idx_second) == 1 and len_second > 0 and (float(len_max)/len_second) <= 2.5:
            current_track_idx = min(idx_max, idx_second)

        return current_track_idx
