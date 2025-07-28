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
            self._board = chess.Board(fen)
        else:
            self._board = chess.Board()
        self.history: List[str] = []

    def to_fen(self) -> str:
        """Returns the FEN representation of the current board state."""
        return self._board.fen()

    def is_legal(self, move_uci: str) -> bool:
        """Checks if a move is legal in the given board state."""
        try:
            move = chess.Move.from_uci(move_uci)
            return move in self._board.legal_moves
        except ValueError:
            return False

    def make_move(self, move_uci: str) -> 'Board':
        """
        Executes a move and returns a new board state.

        Args:
            move_uci: The move in UCI format (e.g., "e2e4").

        Returns:
            A new Board object with the move applied.

        Raises:
            IllegalMoveError: If the move is not legal.
        """
        if not self.is_legal(move_uci):
            raise IllegalMoveError(f"Illegal move: {move_uci}")

        new_board = Board(self.to_fen())
        new_board.history = list(self.history)
        new_board.history.append(self.to_fen())

        move = chess.Move.from_uci(move_uci)
        new_board._board.push(move)
        return new_board

    def undo_move(self) -> 'Board':
        """
        Undoes the last move and returns the previous board state.

        Returns:
            A new Board object representing the state before the last move.

        Raises:
            ValueError: If there is no history to undo.
        """
        if not self.history:
            raise ValueError("No history to undo.")
        previous_fen = self.history[-1]
        new_board = Board(previous_fen)
        new_board.history = self.history[:-1]
        return new_board

    def is_terminal(self) -> Optional[str]:
        """
        Checks if the game is over.

        Returns:
            A string describing the terminal state (e.g., 'checkmate', 'stalemate'),
            or None if the game is not over.
        """
        if self._board.is_checkmate():
            return 'checkmate'
        if self._board.is_stalemate():
            return 'stalemate'
        if self._board.is_insufficient_material():
            return 'draw_insufficient_material'
        if self._board.can_claim_fifty_moves():
            return 'draw_50'
        if self._board.can_claim_threefold_repetition():
            return 'draw_3rep'
        return None
