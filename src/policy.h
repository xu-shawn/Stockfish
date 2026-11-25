/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

#ifndef POLICY_H_INCLUDED
#define POLICY_H_INCLUDED

#include <array>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "movegen.h"
#include "types.h"

namespace Stockfish {

class Position;

namespace Policy {

// Move policy network. Each perspective has 16 king buckets x 12 relative piece
// types x 64 squares, plus four castling rights and eight en-passant files, in a
// shared 128-unit feature transformer. Four piece-count stacks score a move as
// a from-to row plus a promotion-category row.
constexpr int Features  = 12300;
constexpr int HalfWidth = 128;
constexpr int Width     = 2 * HalfWidth;
constexpr int Stacks    = 4;
constexpr int Outputs   = 64 * 64 + 5;

class Network {
   public:
    void load(const std::filesystem::path& rootDirectory, const std::string& file);
    bool is_loaded() const { return !current.empty(); }

    std::vector<i16> weights, biases, outputs;

   private:
    bool read(std::istream& stream);

    std::string current;
};

// Feature sums are computed only at nodes that ask for policy, updated from the
// nearest ancestor that did.
class Accumulator {
   public:
    Accumulator() :
        frames(MAX_PLY + 1) {}

    void reset() { ply = 0, lastFrame = -1; }
    void push() { ++ply; }
    void pop() {
        if (lastFrame == ply)
            lastFrame = parentFrame[ply];
        --ply;
    }

    void evaluate(
      const Network& network, const Position& pos, const Move* moves, float* logits, int count);

   private:
    struct Frame {
        std::array<Piece, SQUARE_NB>                     board;
        std::array<int, COLOR_NB>                        flip, bucket;
        std::array<std::array<bool, 12>, COLOR_NB>       auxiliary;
        std::array<std::array<i32, HalfWidth>, COLOR_NB> sums;
    };

    std::vector<Frame>           frames;
    std::array<int, MAX_PLY + 1> parentFrame;
    int                          ply = 0, lastFrame = -1;
};

// The good quiets after the TT move, within 2048 history points of the first
struct QuietCandidates {
    static constexpr int Limit = 8;

    void collect(ExtMove* begin, ExtMove* end, Move ttMove, int threshold);
    void promote() const;

    std::array<ExtMove*, Limit> slots;
    std::array<float, Limit>    logits;
    int                         count = 0;
};

// Per-thread cache of move logits, indexed by position key
class QuietCache {
   public:
    void clear();
    bool contains(Key key) const;
    int  get(Key                                       key,
             QuietCandidates&                          candidates,
             std::array<bool, QuietCandidates::Limit>& missing) const;
    void put(Key key, Move move, float logit);

   private:
    static constexpr int Size          = 256;
    static constexpr int MovesPerEntry = 16;

    struct Entry {
        Key                              key   = 0;
        int                              count = 0, replace = 0;
        std::array<Move, MovesPerEntry>  moves;
        std::array<float, MovesPerEntry> logits;
    };

    std::array<Entry, Size> entries;
};

// Inference work is bounded per searched node
inline u64 work_cost(int missing) { return u64(Width) * (8 + missing); }
inline u64 work_allowance(u64 nodes) { return 4096 * (1 + nodes / 256); }

}  // namespace Policy
}  // namespace Stockfish

#endif  // #ifndef POLICY_H_INCLUDED
