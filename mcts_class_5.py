import numpy as np
import random
import time
from scipy import signal

class Node:

    def __init__(self, checkerboard, last_move, color, parent=None, children=None, value=0, n_visits=0):

        self.checkerboard = checkerboard  # 棋盘
        self.color = color  # 棋子颜色
        self.last_move = last_move #a list

        self.parent = parent
        if children is None:
            children = []
        self.children = children

        self.value = value  # 胜率
        self.n_visits = n_visits  # 访问次数

        self.good = [
            # 应该优先中间是1或者-1的位置放每个level的前面
            # 一颗棋子的情况，
            # [1, 3, 0, 0, 0]并不一定表示一行的形式，也可能是斜对角线的形式
            [[0, 0, 1, 3, 0], [0, 3, 1, 0, 0], [0, 3, -1, 0, 0], [0, 0, -1, 3, 0],  # 优先找这列
             [1, 3, 0, 0, 0], [0, 1, 3, 0, 0], [0, 0, 0, 1, 3], [0, 0, 0, 3, 1],
             [3, 1, 0, 0, 0], [0, 0, 3, 1, 0], [3, -1, 0, 0, 0], [0, 0, 3, -1, 0],
             [-1, 3, 0, 0, 0], [0, -1, 3, 0, 0], [0, 0, 0, -1, 3], [0, 0, 0, 3, -1]],
            # 二颗棋子的情况
            [[3, 1, 1, 0, 0], [0, 1, 1, 3, 0], [0, 0, 1, 1, 3], [0, 3, 1, 1, 0],
             [3, -1, -1, 0, 0], [0, -1, -1, 3, 0], [0, 0, -1, -1, 3], [0, 3, -1, -1, 0],  # 优先找这两列中间是1或者-1的
             [1, 1, 3, 0, 0], [0, 1, 3, 1, 0], [0, 0, 3, 1, 1],
             [-1, -1, 3, 0, 0], [0, 0, 3, -1, -1], [0, -1, 3, -1, 0]],
            # 三颗棋子的情况 #思考要只放置最好的可能情况还是这种[-1, -1, 3, 0, -1]也放上，先放上
            [[1, 1, 1, 3, 0], [3, 1, 1, 1, 0], [1, 3, 1, 1, 0], [0, 3, 1, 1, 1], [0, 1, 1, 3, 1],
             [-1, -1, -1, 3, 0], [3, -1, -1, -1, 0], [-1, 3, -1, -1, 0], [0, 3, -1, -1, -1], [0, -1, -1, 3, -1],
             [1, 1, 3, 1, 0], [0, 1, 3, 1, 1], [-1, -1, 3, -1, 0], [0, -1, 3, -1, -1],
             [1, 1, 3, 0, 1], [1, 0, 3, 1, 1], [-1, -1, 3, 0, -1], [-1, 0, 3, -1, -1]],
            # 四颗棋子情况
            [[1, 1, 1, 1, 3], [3, 1, 1, 1, 1], [1, 1, 1, 3, 1], [1, 3, 1, 1, 1],
             [-1, -1, -1, -1, 3], [3, -1, -1, -1, -1], [-1, 3, -1, -1, -1], [-1, -1, -1, 3, -1],
             [1, 1, 3, 1, 1], [-1, -1, 3, -1, -1]]
        ]

    def copy(self):
        copied_node = Node(self.checkerboard.copy(), self.last_move, self.color, None, None, 0, 0)

        return copied_node

    def is_terminal_node(self):
        # t = self.checkerboard.check_winner()
        return self.checkerboard.check_winner() != 0

    def get_legal_moves(self):
        """
        找以落子为中心3*3内的空位
        """
        bline = self.checkerboard.bline
        pos = self.last_move
        row, col = pos[0], pos[1]
        row_lim = [max(row - 1, 0), min(row + 2, bline)]
        col_lim = [max(col - 1, 0), min(col + 2, bline)]

        legal_moves = []
        for i in range(row_lim[0], row_lim[1]):
            for j in range(col_lim[0], col_lim[1]):
                if self.checkerboard.board[i, j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def is_fully_expanded(self):

        n_moves = len(self.get_legal_moves())
        n_children = len(self.children)
        return n_children == n_moves

    def add_child(self, move):

        new_checkerboard = self.checkerboard.copy()

        new_color = -1 if self.color == 1 else 1
        new_pos = move
        new_checkerboard.update_board(new_pos[0], new_pos[1], new_color)
        new_node = Node(checkerboard=new_checkerboard, last_move=new_pos, color=new_color, parent=self)
        # 所以每次增加一个节点之后，新的节点对应的checkerboard的局面是更新之后的局面，这就意味着我们不能在selection的时候从根节点开始找？
        # 不应该从这个结点开始的子节点找起，应该从根节点开始找起
        self.children.append(new_node)
        return new_node


    def update_value(self, result):
        n_win = self.value * self.n_visits
        if result == self.color:
            n_win += 1
        elif result == 2: #这行应该需要加上，万一结果是平局呢。
            n_win += 2
        self.n_visits += 1
        self.value = n_win / self.n_visits


    def check_good_move(self, row, col, level, dx, dy):
        """
        这个函数的作用在于，调用了之后虽然不返回但是要更新self.optimal_moves.
        从棋盘的（row行，col列）这个点开始检查，
        每一层级level代表了优先度，检查的时候从最后一个层级开始检查，逐次向前，level=0,1,2,3
        优先最后一个层级的，因为说明要赢了或者要输了，不同层级对应的score设置不一样
        dx和dy分别是下一步的方向，dx,dy可能的取值都是0或者1，没必要同时取0，
        返回有没有找到一个好的下一步移动策略，注意要在检查的过程中更新一些参数。
        """
        board = self.checkerboard.copy().board
        nrow, ncol = -1, -1 #用于放置更新后的下一步位置的行和列index
        # check = 1
        self.optimal_moves = []
        for s in self.good[level]: #s是situation
            check = 1
            for i in range(5): #(dx,dy)是检查的方向, 没匹配到情况就是check=Flase
                ##self.checkerboard.bline = 11
                if row + i * dx in range(0,11) and col + i * dy in range(0,11): #目前设定的是在棋盘种可下棋区域里测，搜索范围没有包括torus
                    # print(f"s[i]={s[i]}, board[row + i * dx, col + i * dy]={board[row + i * dx, col + i * dy]}")
                    if s[i] == 3 and board[row + i * dx, col + i * dy] == 0:
                        nrow, ncol = row + i * dx, col + i * dy
                    elif s[i] != board[row + i * dx, col + i * dy]: #如果s[i]!=3,两者就应该相等才算匹配
                        check = 0
                        break
            if check != 0: #如果check在经历了一次五个连续棋子位置的比较之后check != 0，则把check加一并记录下匹配的最好位置
                self.optimal_moves.append((nrow, ncol))
                # check += 1
            if len(self.optimal_moves) > 1: #设定如果self.optimal_moves有2个可以匹配的位置就不用寻找了
                break


    def get_all_available_moves(self):
        """
            找以棋盘里所有的空位
        """
        legal_moves = []
        for i in range(0, 11):
            for j in range(0, 11):
                if self.checkerboard.board[i, j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def choose_best_move(self):
        #dx,dy可能的取值都是0或者1，-1，没必要同时取0
        bline = self.checkerboard.bline
        board = self.checkerboard.board
        lrow, lcol = self.last_move

        best_row, best_col = -1,-1
        max_level = -1
        #因为是在当前落点的5*5区域搜索，所以八个方向都要包括进来
        direction = [(0, 1), (1, 1), (1, 0), (1, -1), (-1,0), (0,-1), (-1,1), (-1,-1)]
        #(dx,dy) 右, 右下，下，左下, 左, 上，右上，左上,
        # start_row = self.get_first_chessrow()
        # print("start_row:",start_row)
        # find = False
        #其实仔细想想，我在5*5的区域搜索，每个位置都会往八个方向搜索，范围也蛮大的了
        row_range = [max(lrow-2,0),min(lrow+3,11)]
        col_range = [max(lcol-2,0),min(lcol+3,11)]
        best_move_list = []
        for i in range(row_range[0], row_range[1]): #5*5区域搜索
            # if find:
            #     break
            for j in range(col_range[0],col_range[1]):
                # if find:
                #     break
                # print("\n搜索开始的坐标(i,j)：",i,j)
                for level in range(3,-1,-1): #按照3， 2， 1， 0的顺序检查
                    # if find:
                    #     break
                    # print("level:",level)
                    if level >= max_level:
                        for d in direction: #检查方向分别是：下，右下，左下，右
                            self.check_good_move(i, j, level, *d)
                            print("self.optimal_moves:",self.optimal_moves)
                            if len(self.optimal_moves) > 0: #当这个list里面有最好的下一步下棋位置的时候
                                # find = True
                                max_level = level
                                best_move_list.append([self.optimal_moves,level]) #把不同方向找到的self.optimal_moves放在一起
                                # best_row, best_col = random.choice(self.optimal_moves)

        # print("max_level: ",max_level)
        if max_level >= 0:
            bmoves = [bmove for bmove, l in best_move_list if l == max_level]
            best = [m for bmove_list in bmoves for m in bmove_list]
            best_move = random.choice(best)

        else: #如果上面的策略没找到的话
            moves = self.get_legal_moves() #有可能会传入空列表报错
            best_move = random.choice(moves) #先随便下3*3的区域内的空点
            if len(moves) == 0:
                board = self.checkerboard.board
                kernel = np.ones((3, 3))
                score = signal.convolve2d(abs(board), kernel, mode="same")
                top_scores = sorted(score.flatten(), reverse=True)
                for top_score in top_scores:
                    xy = np.where(score == top_score)
                    moves = list(zip(xy[0], xy[1]))
                    best_moves = [move for move in moves if board[move] == 0]
                    if best_moves:
                        break

                best_move = random.choice(best_moves)

        return best_move[0], best_move[1]


    def get_first_chessrow(self):
        """
            遍历棋盘，从左上角开始，每行寻找，找到出现第一个棋子（黑棋或者白棋）的行
        """
        board = self.checkerboard.board
        for i in range(0, self.checkerboard.bline):  # 注意：这里下棋的时候只能下中间的board区域
            for j in range(0, self.checkerboard.bline):
                if board[i, j] == 1 or board[i,j] == -1:
                    line_idx = i
                    return line_idx  # 返回值是行index



class MCTS:

    def __init__(self, n_searches=40):
        self.n_searches = n_searches
        self.C = 1

    def calculate_UCB(self, node):
        if node.n_visits == 0: #我们根据UCB公式，未被访问过的节点其值是正无穷，也就是说，是一定会被选择到。
            return np.inf
        if node.parent:
            parent_n_visits = node.parent.n_visits
        else:
            parent_n_visits = self.n_searches #如果当前结点没有父节点，说明当前结点是根节点的时候
        return node.value + self.C * np.sqrt(np.log(parent_n_visits) / node.n_visits)

    def selection(self, node): #selection应该从根节点（当前结点）开始寻找

        if node.is_terminal_node():
            return node

        if not node.is_fully_expanded():
            return node

        UCBs = []
        children = node.children
        for child_node in children:
            UCB = self.calculate_UCB(child_node)
            UCBs.append(UCB)
        # idx = UCBs.index(max(UCBs))
        max_UCB = max(UCBs)
        indices = [i for i, x in enumerate(UCBs) if x == max_UCB]
        idx = random.choice(indices) #这里修改了代码，使得在一样最大值ucb的index中随机选择一个
        best_child = children[idx]
        return self.selection(best_child)

    def expansion(self, node):
        moves = node.get_legal_moves()
        children = node.children #这是一个list
        children_moves = []
        for child_node in children:
            child_move = child_node.last_move
            children_moves.append(child_move)
        for move in moves:
            if move not in children_moves:
                new_node = node.add_child(move) #返回了第一个选定结点的下一步可能走的棋子的结点
                return new_node #只会在expansion里面真正add child

    def simulation(self, node):  #simulation里面不会add child，只需要得到最终结果是哪方赢了

        t = time.time()
        node_copy = node.copy()
        # board = node_copy.checkerboard.board
        color = node_copy.color
        best_move = node_copy.choose_best_move()  # 最开始找的那个move就确定下来，这里需要随机，但随机的可能性不多，所以之后的n_searches可以设很小
        node_copy.checkerboard.update_board(best_move[0], best_move[1], color)
        node_copy.last_move = best_move #注意一定要更新参数
        color = 1 if color == -1 else -1
        node_copy.color = color

        while node_copy.checkerboard.check_winner() == 0:

            board = node_copy.checkerboard.board
            kernel = np.ones((3, 3))
            score = signal.convolve2d(abs(board), kernel, mode="same")
            top_scores = sorted(score.flatten(), reverse=True)
            for top_score in top_scores:
                xy = np.where(score == top_score)
                moves = list(zip(xy[0], xy[1]))
                best_moves = [move for move in moves if board[move] == 0]
                if best_moves:
                    break

            best_move = random.choice(best_moves)

            color = 1 if node_copy.color == -1 else -1
            node_copy.checkerboard.update_board(best_move[0], best_move[1], color)
            node_copy.last_move = best_move
            node_copy.color = color
        d = time.time() - t

        return node_copy.checkerboard.check_winner(), d

    def backprogation(self, node, result):
        while node.parent is not None:
            node.update_value(result)
            node = node.parent
        node.update_value(result)

    def search(self, root):
        """
        一个search就是一个完整的模拟过程，root对应最新落子的位置
        """
        T = []
        start = time.time()
        for _ in range(self.n_searches):
            selected_node = self.selection(root)
            if selected_node.is_terminal_node():
                result = selected_node.checkerboard.check_winner()
                self.backprogation(selected_node, result)
            else:
                new_node = self.expansion(selected_node)
                result, t = self.simulation(new_node)
                self.backprogation(new_node, result)
                T.append(t)
        print(time.time() - start)
        print("Simulation:", np.sum(T))
    def get_next_move(self, node):
        """
        决定接下来走哪一步
        """
        self.search(node)
        children = node.children
        values = [child_node.value for child_node in children]
        max_values = max(values) #选择当前结点的孩子结点的最大值
        print("Winning rate:", max_values)
        # idx = values.index(max_values)
        indices = [i for i, k in enumerate(values) if k == max_values]
        idx = random.choice(indices)  # 这里修改了代码，随机选择一个
        best_node = children[idx]
        move = best_node.last_move
        return move
# 我有一个问题，MCST在selection的时候是从当前结点开始寻找最大UCB的点还是从根节点开始呀？
# 我一方面觉得从根节点开始用递归去找很合理，一方面又觉得根节点是没有
#现在觉得：当前的位置就是根节点，每次都是，因为棋盘格局每次都更新