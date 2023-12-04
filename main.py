import numpy as np
import pygame as pg
from board_class import *
from mcts_class_5 import *

if __name__ == '__main__':

    player_is_black = True
    bline = 11
    xbline = bline + 8
    w_size = 720
    pad = 36
    sep = int((w_size - pad * 2) / (xbline - 1))
    board = np.zeros((bline, bline), dtype=int) #这里是小棋盘

    pg.init()
    checkerboard = CheckerBoard(board, bline, xbline, w_size, pad, sep)
    surface = checkerboard.draw_board()
    running = True
    gameover = False

    while running:

        for event in pg.event.get():  # A for loop to process all the events initialized by the player

            if event.type == pg.QUIT:  # terminate if player closes the game window
                running = False

            if event.type == pg.MOUSEBUTTONDOWN and not gameover:  # detect whether the player is clicking in the window

                (x, y) = event.pos  # check if the clicked position is on the 11x11 center grid
                if (x > pad + 3.75 * sep) and (x < w_size - pad - 3.75 * sep) and (y > pad + 3.75 * sep) and (
                        y < w_size - pad - 3.75 * sep):
                    row = round((x - pad) / sep - 4)
                    col = round((y - pad) / sep - 4)

                    if checkerboard.board[row, col] == 0:  # update the board matrix if that position has not been occupied
                        color = 1 if player_is_black else -1
                        checkerboard.update_board(row, col, color) #这里就不用加4了
                        checkerboard.draw_stone(surface, [row, col], color)

                    winner = checkerboard.check_winner()
                    # winner = check_game_over(checkerboard.board)
                    if winner != 0:
                        checkerboard.print_winner(surface, winner)
                        gameover = True

                    if not gameover:
                        color = 1 if player_is_black else -1
                        robot_color = -1 if player_is_black else 1
                        last_move = [row, col]
                        node = Node(checkerboard, last_move, color) #Node 增加了一个传入参数->上一步的位置
                        mcts = MCTS()
                        move = mcts.get_next_move(node) #check一下这个move的坐标是大棋盘还是小棋盘

                        checkerboard.update_board(move[0], move[1], robot_color)
                        checkerboard.draw_stone(surface, move, robot_color)

                    winner = checkerboard.check_winner()
                    if winner != 0:
                        checkerboard.print_winner(surface, winner)
                        gameover = True
                        break

    pg.quit()
