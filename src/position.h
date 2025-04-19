/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <string>

#include "bitboard.h"
#include "types.h"

namespace Stockfish {

class TranspositionTable;

// StateInfo struct stores information needed to restore a Position object to
// its previous state when we retract a move. Whenever a move is made on the
// board (by calling Position::do_move), a StateInfo object must be passed.

class StateInfo {
   public:
    // FEN string input/output
    StateInfo&  set(const std::string& fenStr, bool isChess960);
    StateInfo&  set(const std::string& code, Color c);
    std::string fen() const;

    // Position representation
    Bitboard pieces(PieceType pt = ALL_PIECES) const;
    template<typename... PieceTypes>
    Bitboard pieces(PieceType pt, PieceTypes... pts) const;
    Bitboard pieces(Color c) const;
    template<typename... PieceTypes>
    Bitboard pieces(Color c, PieceTypes... pts) const;
    Piece    piece_on(Square s) const;
    Square   ep_square() const;
    bool     empty(Square s) const;
    template<PieceType Pt>
    int count(Color c) const;
    template<PieceType Pt>
    int count() const;
    template<PieceType Pt>
    Square square(Color c) const;

    // Castling
    CastlingRights castling_rights(Color c) const;
    bool           can_castle(CastlingRights cr) const;
    bool           castling_impeded(CastlingRights cr) const;
    Square         castling_rook_square(CastlingRights cr) const;

    // Checking
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    Bitboard check_squares(PieceType pt) const;
    Bitboard pinners(Color c) const;

    // Attacks to/from a given square
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    bool     attackers_to_exist(Square s, Bitboard occupied, Color c) const;
    void     update_slider_blockers(Color c);
    template<PieceType Pt>
    Bitboard attacks_by(Color c) const;

    // Properties of moves
    bool  legal(Move m) const;
    bool  pseudo_legal(const Move m) const;
    bool  capture(Move m) const;
    bool  capture_stage(Move m) const;
    bool  gives_check(Move m) const;
    Piece moved_piece(Move m) const;
    Piece captured_piece() const;

    // Doing and undoing moves
    void       do_move(Move m, const TranspositionTable* tt);
    DirtyPiece do_move(Move m, bool givesCheck, const TranspositionTable* tt);
    void       do_null_move(const TranspositionTable& tt);

    // Static Exchange Evaluation
    bool see_ge(Move m, int threshold = 0) const;

    // Accessing hash keys
    Key zobrist_key() const;
    Key key() const;
    Key material_key() const;
    Key pawn_key() const;
    Key minor_piece_key() const;
    Key non_pawn_key(Color c) const;

    // Other properties of the position
    Color side_to_move() const;
    int   game_ply() const;
    bool  is_chess960() const;
    int   rule50_count() const;
    bool  is_50mr_draw() const;
    int   plies_from_null() const;
    Value non_pawn_material(Color c) const;
    Value non_pawn_material() const;

    // Position consistency check, for debugging
    bool pos_is_ok() const;
    void flip();

    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);

   private:
    // Initialization helpers (used while setting up a position)
    void set_castling_right(Color c, Square rfrom);
    void set_state();
    void set_check_info();

    // Other helpers
    void move_piece(Square from, Square to);
    template<bool Do>
    void do_castling(Color             us,
                     Square            from,
                     Square&           to,
                     Square&           rfrom,
                     Square&           rto,
                     DirtyPiece* const dp = nullptr);
    template<bool AfterMove>
    Key adjust_key50(Key k) const;

    // Copied when making a move
    Piece    board[SQUARE_NB];
    Bitboard byTypeBB[PIECE_TYPE_NB];
    Bitboard byColorBB[COLOR_NB];
    int      pieceCount[PIECE_NB];
    int      castlingRightsMask[SQUARE_NB];
    Square   castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard castlingPath[CASTLING_RIGHT_NB];
    Key      materialKey;
    Key      pawnKey;
    Key      minorPieceKey;
    Key      nonPawnKey[COLOR_NB];
    Value    nonPawnMaterial[COLOR_NB];
    int      castlingRights;
    int      rule50;
    int      pliesFromNull;
    Square   epSquare;
    int      gamePly;
    Color    sideToMove;
    bool     chess960;

    // Not copied when making a move (will be recomputed anyhow)
    Key      zobristKey;
    Bitboard checkersBB;
    Bitboard blockersForKing[COLOR_NB];
    Bitboard pinnersByColor[COLOR_NB];
    Bitboard checkSquares[PIECE_TYPE_NB];
    Piece    capturedPiece;
};

std::ostream& operator<<(std::ostream& os, const StateInfo& pos);

inline Color StateInfo::side_to_move() const { return sideToMove; }

inline Piece StateInfo::piece_on(Square s) const {
    assert(is_ok(s));
    return board[s];
}

inline bool StateInfo::empty(Square s) const { return piece_on(s) == NO_PIECE; }

inline Piece StateInfo::moved_piece(Move m) const { return piece_on(m.from_sq()); }

inline Bitboard StateInfo::pieces(PieceType pt) const { return byTypeBB[pt]; }

template<typename... PieceTypes>
inline Bitboard StateInfo::pieces(PieceType pt, PieceTypes... pts) const {
    return pieces(pt) | pieces(pts...);
}

inline Bitboard StateInfo::pieces(Color c) const { return byColorBB[c]; }

template<typename... PieceTypes>
inline Bitboard StateInfo::pieces(Color c, PieceTypes... pts) const {
    return pieces(c) & pieces(pts...);
}

template<PieceType Pt>
inline int StateInfo::count(Color c) const {
    return pieceCount[make_piece(c, Pt)];
}

template<PieceType Pt>
inline int StateInfo::count() const {
    return count<Pt>(WHITE) + count<Pt>(BLACK);
}

template<PieceType Pt>
inline Square StateInfo::square(Color c) const {
    assert(count<Pt>(c) == 1);
    return lsb(pieces(c, Pt));
}

inline Square StateInfo::ep_square() const { return epSquare; }

inline bool StateInfo::can_castle(CastlingRights cr) const { return castlingRights & cr; }

inline CastlingRights StateInfo::castling_rights(Color c) const {
    return c & CastlingRights(castlingRights);
}

inline bool StateInfo::castling_impeded(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return pieces() & castlingPath[cr];
}

inline Square StateInfo::castling_rook_square(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return castlingRookSquare[cr];
}

inline Bitboard StateInfo::attackers_to(Square s) const { return attackers_to(s, pieces()); }

template<PieceType Pt>
inline Bitboard StateInfo::attacks_by(Color c) const {

    if constexpr (Pt == PAWN)
        return c == WHITE ? pawn_attacks_bb<WHITE>(pieces(WHITE, PAWN))
                          : pawn_attacks_bb<BLACK>(pieces(BLACK, PAWN));
    else
    {
        Bitboard threats   = 0;
        Bitboard attackers = pieces(c, Pt);
        while (attackers)
            threats |= attacks_bb<Pt>(pop_lsb(attackers), pieces());
        return threats;
    }
}

inline Bitboard StateInfo::checkers() const { return checkersBB; }

inline Bitboard StateInfo::blockers_for_king(Color c) const { return blockersForKing[c]; }

inline Bitboard StateInfo::pinners(Color c) const { return pinnersByColor[c]; }

inline Bitboard StateInfo::check_squares(PieceType pt) const { return checkSquares[pt]; }

inline Key StateInfo::zobrist_key() const { return zobristKey; }

inline Key StateInfo::key() const { return adjust_key50<false>(zobrist_key()); }

template<bool AfterMove>
inline Key StateInfo::adjust_key50(Key k) const {
    return rule50 < 14 - AfterMove ? k : k ^ make_key((rule50 - (14 - AfterMove)) / 8);
}

inline Key StateInfo::pawn_key() const { return pawnKey; }

inline Key StateInfo::material_key() const { return materialKey; }

inline Key StateInfo::minor_piece_key() const { return minorPieceKey; }

inline Key StateInfo::non_pawn_key(Color c) const { return nonPawnKey[c]; }

inline Value StateInfo::non_pawn_material(Color c) const { return nonPawnMaterial[c]; }

inline Value StateInfo::non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
}

inline int StateInfo::game_ply() const { return gamePly; }

inline int StateInfo::rule50_count() const { return rule50; }

inline int StateInfo::plies_from_null() const { return pliesFromNull; }

inline bool StateInfo::is_chess960() const { return chess960; }

inline bool StateInfo::capture(Move m) const {
    assert(m.is_ok());
    return (!empty(m.to_sq()) && m.type_of() != CASTLING) || m.type_of() == EN_PASSANT;
}

// Returns true if a move is generated from the capture stage, having also
// queen promotions covered, i.e. consistency with the capture stage move
// generation is needed to avoid the generation of duplicate moves.
inline bool StateInfo::capture_stage(Move m) const {
    assert(m.is_ok());
    return capture(m) || m.promotion_type() == QUEEN;
}

inline Piece StateInfo::captured_piece() const { return capturedPiece; }

inline void StateInfo::put_piece(Piece pc, Square s) {

    board[s] = pc;
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= s;
    byColorBB[color_of(pc)] |= s;
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]++;
}

inline void StateInfo::remove_piece(Square s) {

    Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]--;
}

inline void StateInfo::move_piece(Square from, Square to) {

    Piece    pc     = board[from];
    Bitboard fromTo = from | to;
    byTypeBB[ALL_PIECES] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to]   = pc;
}

inline void StateInfo::do_move(Move m, const TranspositionTable* tt = nullptr) {
    do_move(m, gives_check(m), tt);
}


// A list to keep track of the position states along the setup moves (from the
// start position to the position just before the search starts). Needed by
// 'draw by repetition' detection. Use a std::deque because pointers to
// elements are not invalidated upon list resizing.
using StateListPtr = std::unique_ptr<std::deque<StateInfo>>;


// Position class stores information regarding the board representation as
// pieces, side to move, hash keys, castling info, etc. Important methods are
// do_move() and undo_move(), used by the search to update node info when
// traversing the search tree.
class Position {
   public:
    static void init();

    Position()                           = default;
    Position(const Position&)            = default;
    Position& operator=(const Position&) = default;

    // FEN string input/output
    Position&   set(const std::string& fenStr, bool isChess960);
    Position&   set(const std::string& code, Color c);
    std::string fen() const;

    // Position representation
    Bitboard pieces(PieceType pt = ALL_PIECES) const;
    template<typename... PieceTypes>
    Bitboard pieces(PieceType pt, PieceTypes... pts) const;
    Bitboard pieces(Color c) const;
    template<typename... PieceTypes>
    Bitboard pieces(Color c, PieceTypes... pts) const;
    Piece    piece_on(Square s) const;
    Square   ep_square() const;
    bool     empty(Square s) const;
    template<PieceType Pt>
    int count(Color c) const;
    template<PieceType Pt>
    int count() const;
    template<PieceType Pt>
    Square square(Color c) const;

    // Castling
    CastlingRights castling_rights(Color c) const;
    bool           can_castle(CastlingRights cr) const;
    bool           castling_impeded(CastlingRights cr) const;
    Square         castling_rook_square(CastlingRights cr) const;

    // Checking
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    Bitboard check_squares(PieceType pt) const;
    Bitboard pinners(Color c) const;

    // Attacks to/from a given square
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    bool     attackers_to_exist(Square s, Bitboard occupied, Color c) const;
    template<PieceType Pt>
    Bitboard attacks_by(Color c) const;

    // Properties of moves
    bool  legal(Move m) const;
    bool  pseudo_legal(const Move m) const;
    bool  capture(Move m) const;
    bool  capture_stage(Move m) const;
    bool  gives_check(Move m) const;
    Piece moved_piece(Move m) const;
    Piece captured_piece() const;

    // Doing and undoing moves
    void       do_move(Move m, const TranspositionTable* tt);
    DirtyPiece do_move(Move m, bool givesCheck, const TranspositionTable* tt);
    void       do_null_move(const TranspositionTable& tt);
    void       undo_move();

    // Static Exchange Evaluation
    bool see_ge(Move m, int threshold = 0) const;

    // Accessing hash keys
    Key key() const;
    Key material_key() const;
    Key pawn_key() const;
    Key minor_piece_key() const;
    Key non_pawn_key(Color c) const;

    // Other properties of the position
    Color side_to_move() const;
    int   game_ply() const;
    bool  is_chess960() const;
    bool  is_draw(int ply) const;
    bool  is_repetition(int ply) const;
    bool  upcoming_repetition(int ply) const;
    bool  has_repeated() const;
    int   rule50_count() const;
    Value non_pawn_material(Color c) const;
    Value non_pawn_material() const;

    // Position consistency check, for debugging
    bool pos_is_ok() const;
    void flip();

    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);

    const StateInfo& state() const;

   private:
    StateInfo& state();
    int        repetition() const;

    struct StateWithRepetition {
        StateWithRepetition(StateInfo& st) :
            state(st),
            repetition(0) {}

        StateInfo state;
        int       repetition;
    };

    // Data members
    std::vector<StateWithRepetition> sts;
};

std::ostream& operator<<(std::ostream& os, const Position& pos);

inline StateInfo& Position::state() {

    assert(sts.size() > 1);
    return sts.back().state;
}
inline const StateInfo& Position::state() const {

    assert(sts.size() > 1);
    return sts.back().state;
}

inline int Position::repetition() const { return sts.back().repetition; }

inline Color Position::side_to_move() const { return state().side_to_move(); }

inline Piece Position::piece_on(Square s) const { return state().piece_on(s); }

inline bool Position::empty(Square s) const { return state().empty(s); }

inline Piece Position::moved_piece(Move m) const { return state().moved_piece(m); }

inline Bitboard Position::pieces(PieceType pt) const { return state().pieces(pt); }

template<typename... PieceTypes>
inline Bitboard Position::pieces(PieceType pt, PieceTypes... pts) const {
    return state().pieces(pt, pts...);
}

inline Bitboard Position::pieces(Color c) const { return state().pieces(c); }

template<typename... PieceTypes>
inline Bitboard Position::pieces(Color c, PieceTypes... pts) const {
    return state().pieces(c, pts...);
}

template<PieceType Pt>
inline int Position::count(Color c) const {
    return state().count<Pt>(c);
}

template<PieceType Pt>
inline int Position::count() const {
    return state().count<Pt>();
}

template<PieceType Pt>
inline Square Position::square(Color c) const {
    return state().square<Pt>(c);
}

inline Square Position::ep_square() const { return state().ep_square(); }

inline bool Position::can_castle(CastlingRights cr) const { return state().can_castle(cr); }

inline CastlingRights Position::castling_rights(Color c) const {
    return state().castling_rights(c);
}

inline bool Position::castling_impeded(CastlingRights cr) const {
    return state().castling_impeded(cr);
}

inline Square Position::castling_rook_square(CastlingRights cr) const {
    return state().castling_rook_square(cr);
}

inline Bitboard Position::attackers_to(Square s) const { return state().attackers_to(s); }

template<PieceType Pt>
inline Bitboard Position::attacks_by(Color c) const {
    return state().attacks_by<Pt>(c);
}

inline Bitboard Position::checkers() const { return state().checkers(); }

inline Bitboard Position::blockers_for_king(Color c) const { return state().blockers_for_king(c); }

inline Bitboard Position::pinners(Color c) const { return state().pinners(c); }

inline Bitboard Position::check_squares(PieceType pt) const { return state().check_squares(pt); }

inline Key Position::key() const { return state().key(); }

inline Key Position::pawn_key() const { return state().pawn_key(); }

inline Key Position::material_key() const { return state().material_key(); }

inline Key Position::minor_piece_key() const { return state().minor_piece_key(); }

inline Key Position::non_pawn_key(Color c) const { return state().non_pawn_key(c); }

inline Value Position::non_pawn_material(Color c) const { return state().non_pawn_material(c); }

inline Value Position::non_pawn_material() const { return state().non_pawn_material(); }

inline int Position::game_ply() const { return state().game_ply(); }

inline int Position::rule50_count() const { return state().rule50_count(); }

inline bool Position::is_chess960() const { return state().is_chess960(); }

inline bool Position::capture(Move m) const { return state().capture(m); }

inline bool Position::capture_stage(Move m) const { return state().capture_stage(m); }

inline Piece Position::captured_piece() const { return state().captured_piece(); }

inline void Position::put_piece(Piece pc, Square s) { state().put_piece(pc, s); }

inline void Position::remove_piece(Square s) { state().remove_piece(s); }

inline void Position::do_move(Move m, const TranspositionTable* tt = nullptr) {
    state().do_move(m, tt);
}

}  // namespace Stockfish

#endif  // #ifndef POSITION_H_INCLUDED
