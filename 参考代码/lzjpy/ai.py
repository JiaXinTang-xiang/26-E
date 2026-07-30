"""
Minimax AI 三子棋决策 (从 放/chess.py 移植)
支持执黑(1)/执白(2)双向
"""

SIZE = 3


def check_win(board):
    """检查胜负, 返回: 1=黑赢, 2=白赢, 0=平局, -1=未结束"""
    for player in [1, 2]:
        for i in range(SIZE):
            if all(board[i][j] == player for j in range(SIZE)):
                return player
            if all(board[j][i] == player for j in range(SIZE)):
                return player
        if all(board[i][i] == player for i in range(SIZE)):
            return player
        if all(board[i][SIZE - 1 - i] == player for i in range(SIZE)):
            return player

    if all(board[i][j] != 0 for i in range(SIZE) for j in range(SIZE)):
        return 0
    return -1


def _minimax(board, depth, is_maximizing, computer, player, alpha=float('-inf'), beta=float('inf')):
    winner = check_win(board)
    if winner == computer:
        return 10 - depth
    if winner == player:
        return depth - 10
    if winner == 0:
        return 0

    if is_maximizing:
        best = float('-inf')
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == 0:
                    board[i][j] = computer
                    score = _minimax(board, depth + 1, False, computer, player, alpha, beta)
                    board[i][j] = 0
                    best = max(best, score)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        return best
        return best
    else:
        best = float('inf')
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == 0:
                    board[i][j] = player
                    score = _minimax(board, depth + 1, True, computer, player, alpha, beta)
                    board[i][j] = 0
                    best = min(best, score)
                    beta = min(beta, best)
                    if beta <= alpha:
                        return best
        return best


def find_best_move(board, side):
    """
    找最佳落子位置.
    side: 1=黑棋, 2=白棋
    返回: (row, col) 或 None
    """
    computer = side
    player = 3 - side  # 1→2, 2→1

    # 空棋盘走中心
    if all(board[i][j] == 0 for i in range(SIZE) for j in range(SIZE)):
        return 1, 1

    best_score = float('-inf')
    best_move = None

    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] == 0:
                board[i][j] = computer
                score = _minimax(board, 0, False, computer, player)
                board[i][j] = 0
                if score > best_score:
                    best_score = score
                    best_move = (i, j)

    return best_move
