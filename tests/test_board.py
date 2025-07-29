import pytest
from chess_engine.board import (
    parse_fen, to_fen, State, Move,
    is_legal, make_move, undo_move,
    get_terminal_status,
    IllegalMoveError, FenError, HistoryUnderflowError,
    Terminal, Color, PieceType, _get_piece_at
)

def square(file: str, rank: int) -> int:
    return (rank - 1) * 8 + ("abcdefgh".index(file))

# --- FEN Parsing & Serialization -------------------------------------------

def test_parse_and_roundtrip_fen_initial():
    fen = "rn1qkb1r/pp3ppp/2p1pn2/3p4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 1"
    state = parse_fen(fen)
    assert to_fen(state) == fen

def test_parse_fen_invalid():
    with pytest.raises(FenError):
        parse_fen("invalid fen string")

# --- Move Making & Undo -----------------------------------------------------

def test_make_move_and_undo_initial_e2e4():
    state = parse_fen("startpos")
    move = Move(from_sq=square('e', 2), to_sq=square('e', 4), promotion=None)
    assert is_legal(state, move) is True
    new_state = make_move(state, move)
    # nach Zug: side_to_move gewechselt und halfmove_clock ggf. zurückgesetzt
    assert new_state.side_to_move == Color.BLACK
    restored = undo_move(new_state)
    assert restored == state

def test_illegal_move_raises():
    state = parse_fen("startpos")
    bad = Move(from_sq=0, to_sq=1, promotion=None)
    with pytest.raises(IllegalMoveError):
        make_move(state, bad)

def test_undo_move_underflow():
    state = parse_fen("startpos")
    with pytest.raises(HistoryUnderflowError):
        undo_move(state)

# --- Sonderregeln ------------------------------------------------------------

def test_castling_rights_and_moves():
    state = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    kingside = Move(from_sq=4, to_sq=6, promotion=None)
    queenside = Move(from_sq=4, to_sq=2, promotion=None)
    assert is_legal(state, kingside)
    assert is_legal(state, queenside)

def test_en_passant():
    # Weiß hat e2-e4 gespielt → Schwarz kann d4xd3 en passant ziehen
    state = parse_fen("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 2")
    ep_move = Move(from_sq=27, to_sq=20, promotion=None)  # d4 → e3
    assert is_legal(state, ep_move)
    new_state = make_move(state, ep_move)
    # Bauer auf e4 muss verschwunden sein
    assert _get_piece_at(new_state, square('e', 4)) is None

def test_50_move_rule_draw():
    state = parse_fen("8/8/8/8/8/8/8/8 w - - 100 50")
    assert get_terminal_status(state) == Terminal.DRAW_50_MOVES

def test_threefold_repetition():
    base = parse_fen("startpos")
    # dreimalige Wiederholung
    s1 = make_move(base, Move(from_sq=square('g', 1), to_sq=square('f', 3), promotion=None))
    s2 = make_move(s1, Move(from_sq=square('g', 8), to_sq=square('f', 6), promotion=None))
    s3 = make_move(s2, Move(from_sq=square('f', 3), to_sq=square('g', 1), promotion=None))
    s4 = make_move(s3, Move(from_sq=square('f', 6), to_sq=square('g', 8), promotion=None))
    s5 = make_move(s4, Move(from_sq=square('g', 1), to_sq=square('f', 3), promotion=None))
    s6 = make_move(s5, Move(from_sq=square('g', 8), to_sq=square('f', 6), promotion=None))
    s7 = make_move(s6, Move(from_sq=square('f', 3), to_sq=square('g', 1), promotion=None))
    s8 = make_move(s7, Move(from_sq=square('f', 6), to_sq=square('g', 8), promotion=None))
    assert get_terminal_status(s8) == Terminal.DRAW_THREEFOLD_REP

def test_insufficient_material():
    state = parse_fen("8/8/8/8/8/8/8/Kk6 w - - 0 1")
    assert get_terminal_status(state) == Terminal.DRAW_INSUFFICIENT
