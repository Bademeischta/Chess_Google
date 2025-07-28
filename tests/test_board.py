import unittest
import chess
from chess_engine.board import Board, make_move, undo_move, is_legal, is_terminal

class TestBoard(unittest.TestCase):

    def test_make_move(self):
        board = Board()
        new_board = make_move(board, "e2e4")
        self.assertNotEqual(board.to_fen(), new_board.to_fen())
        self.assertEqual(new_board.board.peek(), chess.Move.from_uci("e2e4"))

    def test_illegal_move(self):
        board = Board()
        with self.assertRaises(ValueError):
            make_move(board, "e2e5")

    def test_undo_move(self):
        board = Board()
        fen_before = board.to_fen()
        new_board = make_move(board, "e2e4")
        undone_board = undo_move(new_board)
        self.assertEqual(undone_board.to_fen(), fen_before)

    def test_is_legal(self):
        board = Board()
        self.assertTrue(is_legal(board, "e2e4"))
        self.assertFalse(is_legal(board, "e2e5"))

    def test_is_terminal_checkmate(self):
        # Fool's Mate
        board = Board(fen="rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        new_board = make_move(board, "d8h4")
        self.assertEqual(is_terminal(new_board), 'checkmate')

    def test_is_terminal_stalemate(self):
        board = Board(fen="k7/8/8/8/8/8/8/8 w - - 0 1")
        self.assertEqual(is_terminal(board), 'stalemate')


    def test_fen_parsing(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = Board(fen=fen)
        self.assertEqual(board.to_fen(), fen)

    def test_regression_make_undo(self):
        board = Board()
        initial_fen = board.to_fen()
        moved_board = make_move(board, "g1f3")
        moved_board2 = make_move(moved_board, "g8f6")
        undone_board = undo_move(moved_board2)
        undone_board2 = undo_move(undone_board)
        self.assertEqual(undone_board2.to_fen(), initial_fen)

if __name__ == '__main__':
    unittest.main()
