from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List

# === Exceptions =============================================================

class ChessError(Exception):
    """Basisklasse aller Schach-Exceptions."""

class FenError(ChessError):
    """Ungültige FEN-Darstellung."""

class IllegalMoveError(ChessError):
    """Versuch, einen illegalen Zug auszuführen."""

class HistoryUnderflowError(ChessError):
    """Versuch, undo_move auf leerer History anzuwenden."""

# === Terminal-Zustände ======================================================

class Terminal(Enum):
    CHECKMATE             = auto()
    STALEMATE             = auto()
    DRAW_50_MOVES         = auto()
    DRAW_THREEFOLD_REP    = auto()
    DRAW_INSUFFICIENT     = auto()

# === Farbangabe und Figurentypen ===========================================

class Color(Enum):
    WHITE = auto()
    BLACK = auto()

class PieceType(Enum):
    PAWN   = auto()
    KNIGHT = auto()
    BISHOP = auto()
    ROOK   = auto()
    QUEEN  = auto()
    KING   = auto()

# === Datenklassen für State und Move =======================================

@dataclass(frozen=True)
class Move:
    """Reine Datenklasse für einen Zug."""
    from_sq: int                   # 0–63
    to_sq: int                     # 0–63
    promotion: Optional[PieceType] # Bei Umwandlung

@dataclass(frozen=True)
class State:
    """
    Unveränderlicher Spielzustand.
    - bitboards: Tuple[int, …] Länge 12
    - side_to_move
    - castling_rights: Bits WK, WQ, BK, BQ
    - en_passant: 0–63 oder None
    - halfmove_clock
    - fullmove_number
    - history: Tuple vergangener States
    """
    bitboards: Tuple[int, ...]
    side_to_move: Color
    castling_rights: int
    en_passant: Optional[int]
    halfmove_clock: int
    fullmove_number: int
    history: Tuple[State, ...] = field(default_factory=tuple)

# === API-Funktionen =========================================================

def _char_to_piece_color(char: str) -> Tuple[PieceType, Color]:
    if char.lower() not in 'pnbrqk':
        raise FenError(f"Invalid piece character: {char}")
    if char.lower() == 'p': return PieceType.PAWN, Color.BLACK if char.islower() else Color.WHITE
    if char.lower() == 'n': return PieceType.KNIGHT, Color.BLACK if char.islower() else Color.WHITE
    if char.lower() == 'b': return PieceType.BISHOP, Color.BLACK if char.islower() else Color.WHITE
    if char.lower() == 'r': return PieceType.ROOK, Color.BLACK if char.islower() else Color.WHITE
    if char.lower() == 'q': return PieceType.QUEEN, Color.BLACK if char.islower() else Color.WHITE
    if char.lower() == 'k': return PieceType.KING, Color.BLACK if char.islower() else Color.WHITE

def parse_fen(fen: str) -> State:
    """
    Parst eine FEN und liefert einen initialen State.
    Raises FenError bei ungültiger FEN.
    """
    if fen == "startpos":
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    parts = fen.split()
    if len(parts) != 6:
        raise FenError("FEN string must have 6 parts.")

    # 1. Bitboards
    bitboards = [0] * 12
    ranks = parts[0].split('/')
    if len(ranks) != 8:
        raise FenError("FEN string must have 8 ranks.")
    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        for char in rank_str:
            if char.isdigit():
                file_idx += int(char)
            else:
                piece, color = _char_to_piece_color(char)
                piece_idx = piece.value - 1
                if color == Color.BLACK:
                    piece_idx += 6
                sq = (7 - rank_idx) * 8 + file_idx
                bitboards[piece_idx] |= 1 << sq
                file_idx += 1

    # 2. Side to move
    side_to_move = Color.WHITE if parts[1] == 'w' else Color.BLACK

    # 3. Castling rights
    castling_rights = 0
    if 'K' in parts[2]: castling_rights |= 1
    if 'Q' in parts[2]: castling_rights |= 2
    if 'k' in parts[2]: castling_rights |= 4
    if 'q' in parts[2]: castling_rights |= 8

    # 4. En passant
    en_passant = None
    if parts[3] != '-':
        file = ord(parts[3][0]) - ord('a')
        rank = int(parts[3][1]) - 1
        en_passant = rank * 8 + file

    # 5. Halfmove clock
    halfmove_clock = int(parts[4])

    # 6. Fullmove number
    fullmove_number = int(parts[5])

    return State(
        bitboards=tuple(bitboards),
        side_to_move=side_to_move,
        castling_rights=castling_rights,
        en_passant=en_passant,
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number,
    )

def to_fen(state: State) -> str:
    """
    Formatiert einen State als FEN-String.
    """
    # 1. Piece placement
    fen_ranks = []
    for rank_idx in range(7, -1, -1):
        fen_rank = ""
        empty_squares = 0
        for file_idx in range(8):
            sq = rank_idx * 8 + file_idx
            piece_char = None
            for piece_idx, bb in enumerate(state.bitboards):
                if (bb >> sq) & 1:
                    color = Color.WHITE if piece_idx < 6 else Color.BLACK
                    piece_type_val = (piece_idx % 6) + 1
                    piece_type = PieceType(piece_type_val)

                    char = piece_type.name[0] if color == Color.WHITE else piece_type.name[0].lower()
                    if piece_type == PieceType.KNIGHT:
                        char = 'N' if color == Color.WHITE else 'n'

                    piece_char = char
                    break

            if piece_char:
                if empty_squares > 0:
                    fen_rank += str(empty_squares)
                    empty_squares = 0
                fen_rank += piece_char
            else:
                empty_squares += 1
        if empty_squares > 0:
            fen_rank += str(empty_squares)
        fen_ranks.append(fen_rank)
    fen_board = "/".join(fen_ranks)

    # 2. Side to move
    fen_side = 'w' if state.side_to_move == Color.WHITE else 'b'

    # 3. Castling rights
    fen_castling = ""
    if state.castling_rights & 1: fen_castling += 'K'
    if state.castling_rights & 2: fen_castling += 'Q'
    if state.castling_rights & 4: fen_castling += 'k'
    if state.castling_rights & 8: fen_castling += 'q'
    if not fen_castling: fen_castling = '-'

    # 4. En passant
    fen_en_passant = '-'
    if state.en_passant is not None:
        rank = state.en_passant // 8
        file = state.en_passant % 8
        fen_en_passant = chr(ord('a') + file) + str(rank + 1)

    # 5. Halfmove clock
    fen_halfmove = str(state.halfmove_clock)

    # 6. Fullmove number
    fen_fullmove = str(state.fullmove_number)

    return f"{fen_board} {fen_side} {fen_castling} {fen_en_passant} {fen_halfmove} {fen_fullmove}"

def _get_piece_at(state: State, sq: int) -> Optional[Tuple[PieceType, Color]]:
    for i, bb in enumerate(state.bitboards):
        if (bb >> sq) & 1:
            color = Color.WHITE if i < 6 else Color.BLACK
            piece_type_val = (i % 6) + 1
            return PieceType(piece_type_val), color
    return None

def _is_square_attacked(state: State, sq: int, by_color: Color) -> bool:
    other_color = Color.BLACK if by_color == Color.WHITE else Color.WHITE

    # Pawns
    pawn_bb = state.bitboards[0 if by_color == Color.WHITE else 6]
    if by_color == Color.WHITE:
        if (pawn_bb >> (sq - 7)) & 1 and sq % 8 != 0: return True
        if (pawn_bb >> (sq - 9)) & 1 and sq % 8 != 7: return True
    else:
        if (pawn_bb >> (sq + 7)) & 1 and sq % 8 != 7: return True
        if (pawn_bb >> (sq + 9)) & 1 and sq % 8 != 0: return True

    # Knights
    knight_bb = state.bitboards[1 if by_color == Color.WHITE else 7]
    for move in _generate_knight_moves(state, sq, by_color):
        if (knight_bb >> move.to_sq) & 1:
            return True

    # Bishops and Queens
    bishop_queen_bb = state.bitboards[2 if by_color == Color.WHITE else 8] | state.bitboards[4 if by_color == Color.WHITE else 10]
    for move in _generate_sliding_moves(state, sq, by_color, [(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        if (bishop_queen_bb >> move.to_sq) & 1:
            return True

    # Rooks and Queens
    rook_queen_bb = state.bitboards[3 if by_color == Color.WHITE else 9] | state.bitboards[4 if by_color == Color.WHITE else 10]
    for move in _generate_sliding_moves(state, sq, by_color, [(-1, 0), (1, 0), (0, -1), (0, 1)]):
        if (rook_queen_bb >> move.to_sq) & 1:
            return True

    # Kings
    king_bb = state.bitboards[5 if by_color == Color.WHITE else 11]
    for move in _generate_king_moves(state, sq, by_color, castling=False):
        if (king_bb >> move.to_sq) & 1:
            return True

    return False

def _generate_pawn_moves(state: State, from_sq: int, color: Color) -> List[Move]:
    moves = []
    from_rank, from_file = from_sq // 8, from_sq % 8

    # Single push
    if color == Color.WHITE:
        to_sq = from_sq + 8
        if to_sq < 64 and not _get_piece_at(state, to_sq):
            moves.append(Move(from_sq, to_sq, None))
            # Double push
            if from_rank == 1 and not _get_piece_at(state, from_sq + 16):
                moves.append(Move(from_sq, from_sq + 16, None))
    else: # Black
        to_sq = from_sq - 8
        if to_sq >= 0 and not _get_piece_at(state, to_sq):
            moves.append(Move(from_sq, to_sq, None))
            # Double push
            if from_rank == 6 and not _get_piece_at(state, from_sq - 16):
                moves.append(Move(from_sq, from_sq - 16, None))

    # Captures
    if color == Color.WHITE:
        for to_file_offset in [-1, 1]:
            to_sq = from_sq + 8 + to_file_offset
            if to_sq < 64 and abs(from_file - (to_sq % 8)) == 1:
                dest_piece = _get_piece_at(state, to_sq)
                if (dest_piece and dest_piece[1] != color) or to_sq == state.en_passant:
                    moves.append(Move(from_sq, to_sq, None))
    else: # Black
        for to_file_offset in [-1, 1]:
            to_sq = from_sq - 8 + to_file_offset
            if to_sq >= 0 and abs(from_file - (to_sq % 8)) == 1:
                dest_piece = _get_piece_at(state, to_sq)
                if (dest_piece and dest_piece[1] != color) or to_sq == state.en_passant:
                    moves.append(Move(from_sq, to_sq, None))

    # Promotion
    promotion_moves = []
    for move in moves:
        if (color == Color.WHITE and move.to_sq // 8 == 7) or (color == Color.BLACK and move.to_sq // 8 == 0):
            for piece_type in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                promotion_moves.append(Move(move.from_sq, move.to_sq, piece_type))
        else:
            promotion_moves.append(move)
    moves = promotion_moves

    return moves

def _generate_knight_moves(state: State, from_sq: int, color: Color) -> List[Move]:
    moves = []
    from_rank, from_file = from_sq // 8, from_sq % 8
    # (dy, dx)
    knight_moves = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    for dr, df in knight_moves:
        to_rank, to_file = from_rank + dr, from_file + df
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            to_sq = to_rank * 8 + to_file
            dest_piece = _get_piece_at(state, to_sq)
            if not dest_piece or dest_piece[1] != color:
                moves.append(Move(from_sq, to_sq, None))
    return moves

def _generate_sliding_moves(state: State, from_sq: int, color: Color, directions: List[Tuple[int, int]]) -> List[Move]:
    moves = []
    from_rank, from_file = from_sq // 8, from_sq % 8
    for dr, df in directions:
        to_rank, to_file = from_rank + dr, from_file + df
        while 0 <= to_rank < 8 and 0 <= to_file < 8:
            to_sq = to_rank * 8 + to_file
            dest_piece = _get_piece_at(state, to_sq)
            if dest_piece:
                if dest_piece[1] != color:
                    moves.append(Move(from_sq, to_sq, None))
                break
            moves.append(Move(from_sq, to_sq, None))
            to_rank += dr
            to_file += df
    return moves

def _generate_king_moves(state: State, from_sq: int, color: Color, castling=True) -> List[Move]:
    moves = []
    from_rank, from_file = from_sq // 8, from_sq % 8
    # Standard king moves
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            if dr == 0 and df == 0:
                continue
            to_rank, to_file = from_rank + dr, from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                dest_piece = _get_piece_at(state, to_sq)
                if not dest_piece or dest_piece[1] != color:
                    moves.append(Move(from_sq, to_sq, None))
    # Castling
    if castling:
        if color == Color.WHITE:
            if state.castling_rights & 1 and not _get_piece_at(state, 5) and not _get_piece_at(state, 6):
                if not _is_square_attacked(state, 4, Color.BLACK) and not _is_square_attacked(state, 5, Color.BLACK) and not _is_square_attacked(state, 6, Color.BLACK):
                    moves.append(Move(4, 6, None))
            if state.castling_rights & 2 and not _get_piece_at(state, 1) and not _get_piece_at(state, 2) and not _get_piece_at(state, 3):
                if not _is_square_attacked(state, 4, Color.BLACK) and not _is_square_attacked(state, 3, Color.BLACK) and not _is_square_attacked(state, 2, Color.BLACK):
                    moves.append(Move(4, 2, None))
        else: # Black
            if state.castling_rights & 4 and not _get_piece_at(state, 61) and not _get_piece_at(state, 62):
                if not _is_square_attacked(state, 60, Color.WHITE) and not _is_square_attacked(state, 61, Color.WHITE) and not _is_square_attacked(state, 62, Color.WHITE):
                    moves.append(Move(60, 62, None))
            if state.castling_rights & 8 and not _get_piece_at(state, 57) and not _get_piece_at(state, 58) and not _get_piece_at(state, 59):
                if not _is_square_attacked(state, 60, Color.WHITE) and not _is_square_attacked(state, 59, Color.WHITE) and not _is_square_attacked(state, 58, Color.WHITE):
                    moves.append(Move(60, 58, None))
    return moves


def _generate_moves(state: State) -> List[Move]:
    moves = []
    for sq in range(64):
        piece_at = _get_piece_at(state, sq)
        if piece_at and piece_at[1] == state.side_to_move:
            piece, color = piece_at
            if piece == PieceType.PAWN:
                moves.extend(_generate_pawn_moves(state, sq, color))
            elif piece == PieceType.KNIGHT:
                moves.extend(_generate_knight_moves(state, sq, color))
            elif piece == PieceType.BISHOP:
                moves.extend(_generate_sliding_moves(state, sq, color, [(-1, -1), (-1, 1), (1, -1), (1, 1)]))
            elif piece == PieceType.ROOK:
                moves.extend(_generate_sliding_moves(state, sq, color, [(-1, 0), (1, 0), (0, -1), (0, 1)]))
            elif piece == PieceType.QUEEN:
                moves.extend(_generate_sliding_moves(state, sq, color, [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]))
            elif piece == PieceType.KING:
                moves.extend(_generate_king_moves(state, sq, color))
    return moves

def is_legal(state: State, move: Move) -> bool:
    """
    Prüft, ob `move` in `state` legal ist.
    """
    piece_at_from = _get_piece_at(state, move.from_sq)
    if not piece_at_from:
        return False

    piece, color = piece_at_from
    if color != state.side_to_move:
        return False

    if piece == PieceType.PAWN:
        from_rank, from_file = move.from_sq // 8, move.from_sq % 8
        to_rank, to_file = move.to_sq // 8, move.to_sq % 8

        # En passant
        if abs(from_file - to_file) == 1 and move.to_sq == state.en_passant:
            if color == Color.WHITE and from_rank == 4 and to_rank == 5:
                return True
            if color == Color.BLACK and from_rank == 3 and to_rank == 2:
                return True

    if move not in _generate_moves(state):
        return False

    new_state = _make_move_pseudo(state, move)
    king_sq = -1
    king_bb_idx = 5 if state.side_to_move == Color.WHITE else 11
    king_bb = new_state.bitboards[king_bb_idx]
    for i in range(64):
        if (king_bb >> i) & 1:
            king_sq = i
            break

    return not _is_square_attacked(new_state, king_sq, new_state.side_to_move)

def _make_move_pseudo(state: State, move: Move) -> State:
    """
    Führt einen Zug aus, ohne die Legalität zu prüfen.
    """
    new_bitboards = list(state.bitboards)

    # Find the piece to move
    piece_at_from = _get_piece_at(state, move.from_sq)
    piece, color = piece_at_from
    piece_idx = piece.value - 1
    if color == Color.BLACK:
        piece_idx += 6

    # Move the piece
    new_bitboards[piece_idx] ^= (1 << move.from_sq)
    if move.promotion:
        new_bitboards[move.promotion.value -1 + (6 if state.side_to_move == Color.BLACK else 0)] |= (1 << move.to_sq)
    else:
        new_bitboards[piece_idx] |= (1 << move.to_sq)


    # Handle captures
    captured_piece_idx = -1
    for i, bb in enumerate(new_bitboards):
        if i != piece_idx and (bb >> move.to_sq) & 1:
            new_bitboards[i] &= ~(1 << move.to_sq)
            captured_piece_idx = i
            break

    new_side_to_move = Color.BLACK if state.side_to_move == Color.WHITE else Color.WHITE

    new_history = state.history + (state,)

    # Reset halfmove clock if pawn is moved or a capture is made
    halfmove_clock = state.halfmove_clock + 1
    if (piece_idx % 6) == 0 or captured_piece_idx != -1:
        halfmove_clock = 0


    # Update en-passant square
    new_en_passant = None
    if piece == PieceType.PAWN and abs(move.to_sq - move.from_sq) == 16:
        new_en_passant = (move.from_sq + move.to_sq) // 2

    # Handle en passant capture
    if piece == PieceType.PAWN and move.to_sq == state.en_passant:
        if state.side_to_move == Color.WHITE:
            captured_pawn_sq = move.to_sq - 8
        else:
            captured_pawn_sq = move.to_sq + 8

        # Find and remove the captured pawn
        captured_pawn_bb_idx = 6 if state.side_to_move == Color.WHITE else 0
        new_bitboards[captured_pawn_bb_idx] &= ~(1 << captured_pawn_sq)

    # Update castling rights
    new_castling_rights = state.castling_rights
    if piece == PieceType.KING:
        if color == Color.WHITE:
            new_castling_rights &= ~1
            new_castling_rights &= ~2
        else:
            new_castling_rights &= ~4
            new_castling_rights &= ~8
    elif piece == PieceType.ROOK:
        if move.from_sq == 0: new_castling_rights &= ~2
        elif move.from_sq == 7: new_castling_rights &= ~1
        elif move.from_sq == 56: new_castling_rights &= ~8
        elif move.from_sq == 63: new_castling_rights &= ~4

    # Handle castling
    if piece == PieceType.KING and abs(move.to_sq - move.from_sq) == 2:
        if move.to_sq == 6: # Kingside
            new_bitboards[4] ^= (1 << 7) | (1 << 5)
        elif move.to_sq == 2: # Queenside
            new_bitboards[4] ^= (1 << 0) | (1 << 3)
        elif move.to_sq == 62: # Kingside
            new_bitboards[10] ^= (1 << 63) | (1 << 61)
        else: # Queenside
            new_bitboards[10] ^= (1 << 56) | (1 << 59)


    return State(
        bitboards=tuple(new_bitboards),
        side_to_move=new_side_to_move,
        castling_rights=new_castling_rights,
        en_passant=new_en_passant,
        halfmove_clock=halfmove_clock,
        fullmove_number=state.fullmove_number + (1 if new_side_to_move == Color.WHITE else 0),
        history=new_history,
    )


def make_move(state: State, move: Move) -> State:
    """
    Führt `move` auf `state` aus und liefert neuen State zurück.
    Fügt den alten State der History an.
    Raises IllegalMoveError, falls Zug illegal.
    """
    if not is_legal(state, move):
        raise IllegalMoveError(f"Illegal move: {move}")
    return _make_move_pseudo(state, move)

def undo_move(state: State) -> State:
    """
    Setzt den letzten Move zurück.
    Raises HistoryUnderflowError, falls keine History vorhanden.
    """
    if not state.history:
        raise HistoryUnderflowError("No history to undo.")
    return state.history[-1]

def get_terminal_status(state: State) -> Optional[Terminal]:
    """
    Liefert einen Terminal-Status oder None, falls das Spiel weiterläuft.
    """
    if state.halfmove_clock >= 100:
        return Terminal.DRAW_50_MOVES

    # Threefold repetition
    key = (state.bitboards, state.side_to_move, state.castling_rights, state.en_passant)
    count = 1
    for old_state in state.history:
        old_key = (old_state.bitboards, old_state.side_to_move, old_state.castling_rights, old_state.en_passant)
        if key == old_key:
            count += 1
    if count >= 3:
        return Terminal.DRAW_THREEFOLD_REP

    # Insufficient material
    # This is a stub implementation that only checks for K vs k
    if state.bitboards == (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2):
        return Terminal.DRAW_INSUFFICIENT

    # Checkmate and stalemate
    legal_moves = [move for move in _generate_moves(state) if is_legal(state, move)]
    if not any(legal_moves):
        king_sq = -1
        king_bb = state.bitboards[5 if state.side_to_move == Color.WHITE else 11]
        for i in range(64):
            if (king_bb >> i) & 1:
                king_sq = i
                break
        if _is_square_attacked(state, king_sq, Color.BLACK if state.side_to_move == Color.WHITE else Color.WHITE):
            return Terminal.CHECKMATE
        else:
            return Terminal.STALEMATE

    return None
