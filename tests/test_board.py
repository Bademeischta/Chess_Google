import unittest
import chess
from chess_engine.board import Board, IllegalMoveError

class TestBoard(unittest.TestCase):

    def test_make_move(self):
        board = Board()
        new_board = board.make_move("e2e4")
        self.assertNotEqual(board.to_fen(), new_board.to_fen())
        self.assertEqual(new_board._board.peek(), chess.Move.from_uci("e2e4"))

    def test_illegal_move(self):
        board = Board()
        with self.assertRaises(IllegalMoveError):
            board.make_move("e2e5")

    def test_undo_move(self):
        board = Board()
        fen_before = board.to_fen()
        new_board = board.make_move("e2e4")
        undone_board = new_board.undo_move()
        self.assertEqual(undone_board.to_fen(), fen_before)

    def test_is_legal(self):
        board = Board()
        self.assertTrue(board.is_legal("e2e4"))
        self.assertFalse(board.is_legal("e2e5"))

    def test_is_terminal_checkmate(self):
        # Fool's Mate
        board = Board(fen="rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        new_board = board.make_move("d8h4")
        self.assertEqual(new_board.is_terminal(), 'checkmate')

    def test_is_terminal_stalemate(self):
        board = Board(fen="k7/8/8/8/8/8/8/8 w - - 0 1")
        self.assertEqual(board.is_terminal(), 'stalemate')

    def test_fen_parsing(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = Board(fen=fen)
        self.assertEqual(board.to_fen(), fen)

    def test_regression_make_undo(self):
        board = Board()
        initial_fen = board.to_fen()
        board = board.make_move("g1f3")
        board = board.make_move("g8f6")
        board = board.undo_move()
        board = board.undo_move()
        self.assertEqual(board.to_fen(), initial_fen)

    def test_promotion(self):
        board = Board(fen="8/P7/8/k7/8/8/8/K7 w - - 0 1")
        # Promote to queen
        new_board = board.make_move("a7a8q")
        self.assertEqual(new_board._board.piece_at(chess.A8).symbol(), 'Q')

    def test_castling_kingside(self):
        board = Board(fen="r3k2r/pppp1ppp/8/8/8/8/PPPP1PPP/R3K2R w KQkq - 0 1")
        new_board = board.make_move("e1g1")
        self.assertEqual(new_board._board.piece_at(chess.G1).symbol(), 'K')
        self.assertEqual(new_board._board.piece_at(chess.F1).symbol(), 'R')

    def test_castling_queenside(self):
        board = Board(fen="r3k2r/pppp1ppp/8/8/8/8/PPPP1PPP/R3K2R w KQkq - 0 1")
        new_board = board.make_move("e1c1")
        self.assertEqual(new_board._board.piece_at(chess.C1).symbol(), 'K')
        self.assertEqual(new_board._board.piece_at(chess.D1).symbol(), 'R')

    def test_en_passant(self):
        board = Board()
        board = board.make_move("e2e4")
        board = board.make_move("a7a6")
        board = board.make_move("e4e5")
        board = board.make_move("d7d5")
        new_board = board.make_move("e5d6")
        self.assertIsNone(new_board._board.piece_at(chess.D5))


if __name__ == '__main__':
    unittest.main()
