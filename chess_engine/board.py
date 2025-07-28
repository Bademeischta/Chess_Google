import chess
from typing import Optional, List

class IllegalMoveError(Exception):
    """Raised when an illegal move is attempted."""
    pass

class Board:
    """
    Represents the state of a chess game.

    This class encapsulates a `chess.Board` object from the `python-chess`
    library and manages the history of moves.
    """
    def __init__(self, fen: Optional[str] = None):
        """
        Initializes the board.

        Args:
            fen: An optional FEN string to initialize the board state.
                 If None, the board is set to the starting position.
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.history: List[str] = []

    def to_fen(self) -> str:
        """Returns the FEN representation of the current board state."""
        return self.board.fen()

def make_move(board_state: Board, move_uci: str) -> Board:
    """
    Executes a move and returns the new board state.

    Args:
        board_state: The current board state.
        move_uci: The move in UCI format (e.g., "e2e4").

    Returns:
        A new Board object with the move applied.

    Raises:
        IllegalMoveError: If the move is not legal.
    """
    new_board = Board(board_state.to_fen())
    new_board.history = list(board_state.history)
    new_board.history.append(board_state.to_fen())

    try:
        move = chess.Move.from_uci(move_uci)
        if move in new_board.board.legal_moves:
            new_board.board.push(move)
            return new_board
        else:
            raise IllegalMoveError(f"Illegal move: {move_uci}")
    except ValueError:
        raise IllegalMoveError(f"Invalid UCI move string: {move_uci}")

def undo_move(board_state: Board) -> Board:
    """
    Undoes the last move and returns the previous board state.

    Args:
        board_state: The current board state.

    Returns:
        A new Board object representing the state before the last move.

    Raises:
        ValueError: If there is no history to undo.
    """
    if not board_state.history:
        raise ValueError("No history to undo.")
    previous_fen = board_state.history[-1]
    new_board = Board(previous_fen)
    new_board.history = board_state.history[:-1]
    return new_board

def is_legal(board_state: Board, move_uci: str) -> bool:
    """Checks if a move is legal in the given board state."""
    try:
        move = chess.Move.from_uci(move_uci)
        return move in board_state.board.legal_moves
    except ValueError:
        return False

def is_terminal(board_state: Board) -> Optional[str]:
    """
    Checks if the game is over.

    Returns:
        A string describing the terminal state (e.g., 'checkmate', 'stalemate'),
        or None if the game is not over.
    """
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
