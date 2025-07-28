import chess

class Board:
    def __init__(self, fen=None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.history = []

    def to_fen(self):
        return self.board.fen()

def make_move(board_state, move_uci):
    new_board = Board(board_state.to_fen())
    new_board.history = list(board_state.history)
    new_board.history.append(board_state.to_fen())

    move = chess.Move.from_uci(move_uci)
    if move in new_board.board.legal_moves:
        new_board.board.push(move)
        return new_board
    else:
        raise ValueError(f"Illegal move: {move_uci}")

def undo_move(board_state):
    if not board_state.history:
        raise ValueError("No history to undo.")
    previous_fen = board_state.history[-1]
    new_board = Board(previous_fen)
    new_board.history = board_state.history[:-1]
    return new_board

def is_legal(board_state, move_uci):
    move = chess.Move.from_uci(move_uci)
    return move in board_state.board.legal_moves

def is_terminal(board_state):
    if board_state.board.is_checkmate():
        return 'checkmate'
    if board_state.board.is_stalemate():
        return 'stalemate'
    if board_state.board.is_insufficient_material():
        return 'draw_insufficient_material'
    if board_state.board.can_claim_fifty_moves():
        return 'draw_50'
    if board_state.board.can_claim_threefold_repetition():
        return 'draw_3rep'
    return None
