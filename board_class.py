import numpy as np
import math
import pygame as pg


class CheckerBoard():
    def __init__(self, board, bline, xbline, w_size, pad, sep):
        self.board = board
        self.bline = bline
        self.xbline = xbline
        self.w_size = w_size
        self.pad = pad
        self.sep = sep

    def copy(self):
        return CheckerBoard(
            np.copy(self.board),
            self.bline,
            self.xbline,
            self.w_size,
            self.pad,
            self.sep
        )

    def update_board(self, row, col, color):
        self.board[row, col] = color

    def draw_dash_line(self, surface, color, start, end, width=1, dash_length=4):
        x1, y1 = start
        x2, y2 = end
        dl = dash_length

        if (x1 == x2):
            ycoords = [y for y in range(y1, y2, dl if y1 < y2 else -dl)]
            xcoords = [x1] * len(ycoords)
        elif (y1 == y2):
            xcoords = [x for x in range(x1, x2, dl if x1 < x2 else -dl)]
            ycoords = [y1] * len(xcoords)
        else:
            a = abs(x2 - x1)
            b = abs(y2 - y1)
            c = round(math.sqrt(a ** 2 + b ** 2))
            dx = dl * a / c
            dy = dl * b / c

            xcoords = [x for x in np.arange(x1, x2, dx if x1 < x2 else -dx)]
            ycoords = [y for y in np.arange(y1, y2, dy if y1 < y2 else -dy)]

        next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
        last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
        for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
            start = (round(x1), round(y1))
            end = (round(x2), round(y2))
            pg.draw.line(surface, color, start, end, width)

    def draw_board(self):

        surface = pg.display.set_mode((self.w_size, self.w_size))
        pg.display.set_caption("Gomuku (a.k.a Five-in-a-Row)")

        color_line = [0, 0, 0]
        color_board = [241, 196, 15]

        surface.fill(color_board)

        for i in range(0, self.xbline):
            self.draw_dash_line(surface, color_line, [self.pad, self.pad + i * self.sep],
                           [self.w_size - self.pad, self.pad + i * self.sep])
            self.draw_dash_line(surface, color_line, [self.pad + i * self.sep, self.pad],
                           [self.pad + i * self.sep, self.w_size - self.pad])

        for i in range(0, self.bline):
            pg.draw.line(surface, color_line, [self.pad + 4 * self.sep, self.pad + (i + 4) * self.sep],
                         [self.w_size - self.pad - 4 * self.sep, self.pad + (i + 4) * self.sep], 4)
            pg.draw.line(surface, color_line, [self.pad + (i + 4) * self.sep, self.pad + 4 * self.sep],
                         [self.pad + (i + 4) * self.sep, self.w_size - self.pad - 4 * self.sep], 4)

        pg.display.update()

        return surface

    def draw_stone(self, surface, pos, color=0):
        color_black = [0, 0, 0]
        color_dark_gray = [75, 75, 75]
        color_white = [255, 255, 255]
        color_light_gray = [235, 235, 235]

        matx = pos[0] + 4 + self.bline * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).flatten()
        matx1 = np.logical_and(matx >= 0, matx < self.xbline)
        maty = pos[1] + 4 + self.bline * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).T.flatten()
        maty1 = np.logical_and(maty >= 0, maty < self.xbline)
        mat = np.logical_and(np.logical_and(matx1, maty1),
                             np.array([[True, True, True], [True, False, True], [True, True, True]]).flatten())

        if color == 1:
            pg.draw.circle(surface, color_black, [self.pad + (pos[0] + 4) * self.sep, self.pad + (pos[1] + 4) * self.sep], 15, 0)
            for f, x, y in zip(mat, matx, maty):
                if f:
                    pg.draw.circle(surface, color_dark_gray, [self.pad + x * self.sep, self.pad + y * self.sep], 15, 0)

        elif color == -1:
            pg.draw.circle(surface, color_white, [self.pad + (pos[0] + 4) * self.sep, self.pad + (pos[1] + 4) * self.sep], 15, 0)
            for f, x, y in zip(mat, matx, maty):
                if f:
                    pg.draw.circle(surface, color_light_gray, [self.pad + x * self.sep, self.pad + y * self.sep], 15, 0)

        pg.display.update()

    def check_winner(self):
        """我改了这里-wkw"""
        # 输入的是11*11的board，要判断19*19的xboard
        xboard = np.zeros((self.xbline,self.xbline), dtype=int)
        xboard[4:self.xbline-4,4:self.xbline-4] = self.board

        for i in range(self.bline):
            for j in range(self.bline):
                matx = i + 4 + self.bline*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).flatten()
                matx1 = np.logical_and(matx >= 0, matx < self.xbline)
                maty = j + 4 + self.bline*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).T.flatten()
                maty1 = np.logical_and(maty >= 0, maty < self.xbline)
                mat = np.logical_and(np.logical_and(matx1, maty1), np.array([[True, True, True], [True, False, True], [True, True, True]]).flatten())

                for f, x, y in zip(mat, matx, maty):
                    if f:
                        xboard[x,y] = self.board[i,j]

        # Check rows
        for i in range(self.xbline):
            for j in range(self.xbline-5+1):
                if np.sum(xboard[i, j:j + 5]) == 5 or np.sum(xboard[i, j:j + 5]) == -5:
                    return xboard[i, j]

        for i in range(self.xbline-5+1):
            for j in range(self.xbline):

                if np.sum(xboard[i:i + 5, j]) == 5 or np.sum(xboard[i:i + 5, j]) == -5:
                    return xboard[i, j]
        # Check diagonals
        for i in range(self.xbline-5+1):
            for j in range(self.xbline-5+1):
                if xboard[i][j] == xboard[i + 1][j + 1] == xboard[i + 2][j + 2] == xboard[i + 3][j + 3] \
                        == xboard[i + 4][j + 4] != 0:
                    return self.board[i][j]
                if xboard[i][j + 4] == xboard[i + 1][j + 3] == xboard[i + 2][j + 2] == xboard[i + 3][
                    j + 1] == xboard[i + 4][j] != 0:
                    return xboard[i][j + 4]
        # Check draw
        if all([stone != 0 for row in xboard for stone in row]):
            return 2

        return 0

    def print_winner(self, surface, winner=0):
        if winner == 2:
            msg = "Draw! So White wins"
            color = [170, 170, 170]
        elif winner == 1:
            msg = "Black wins!"
            color = [0, 0, 0]
        elif winner == -1:
            msg = 'White wins!'
            color = [255, 255, 255]
        else:
            return

        font = pg.font.Font('freesansbold.ttf', 32)
        text = font.render(msg, True, color)
        textRect = text.get_rect()
        textRect.topleft = (0, 0)
        surface.blit(text, textRect)
        pg.display.update()





