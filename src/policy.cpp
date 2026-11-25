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

#include "policy.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <istream>
#include <streambuf>

#include "bitboard.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/nnue_common.h"
#include "position.h"

#if !defined(UNIVERSAL_BINARY) && !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
    #define INCBIN_SILENCE_BITCODE_WARNING
    #include "incbin/incbin.h"
INCBIN(EmbeddedPolicy, EvalFileDefaultNamePolicy);
#else
const unsigned char gEmbeddedPolicyData[1] = {0x0};
const unsigned int  gEmbeddedPolicySize    = 1;
#endif

namespace Stockfish::Policy {

namespace {

i64 dot(const i16* hidden, const i16* weights) {
    i64 sum = 0;
    for (int j = 0; j < Width; j += 16)
    {
        // Sixteen products of [0, 4096] x int16 fit in int32
        i32 block = 0;
        for (int k = 0; k < 16; ++k)
            block += hidden[j + k] * weights[j + k];
        sum += block;
    }
    return sum;
}

template<int Sign>
void add_row(std::array<i32, HalfWidth>& sums, const i16* row) {
    for (int j = 0; j < HalfWidth; ++j)
        sums[j] += Sign * row[j];
}

}  // namespace

void Network::load(const std::filesystem::path& rootDirectory, const std::string& file) {
    current.clear();

    if (file == EvalFileDefaultNamePolicy)
    {
        class MemoryBuffer: public std::basic_streambuf<char> {
           public:
            MemoryBuffer(char* p, usize n) { setg(p, p, p + n); }
        };

        MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(gEmbeddedPolicyData)),
                            usize(gEmbeddedPolicySize));
        std::istream stream(&buffer);
        if (read(stream))
        {
            current = file;
            return;
        }
    }

    for (const auto& directory : {std::filesystem::path{}, rootDirectory})
    {
        std::ifstream stream(directory / path_from_utf8(file), std::ios::binary);
        if (read(stream))
        {
            current = file;
            return;
        }
    }
}

bool Network::read(std::istream& stream) {
    char magic[8] = {};
    stream.read(magic, 8);
    const u32 width    = Eval::NNUE::read_little_endian<u32>(stream);
    const u32 features = Eval::NNUE::read_little_endian<u32>(stream);

    if (!stream || std::memcmp(magic, "SFPOLI5", 8) || width != Width || features != Features)
        return false;

    weights.resize(Features * HalfWidth);
    biases.resize(HalfWidth);
    outputs.resize(Stacks * Outputs * Width);

    for (auto* values : {&weights, &biases, &outputs})
        Eval::NNUE::read_little_endian<i16>(stream, values->data(), values->size());

    return stream && stream.peek() == std::char_traits<char>::eof();
}

void Accumulator::evaluate(
  const Network& network, const Position& pos, const Move* moves, float* logits, int count) {

    assert(ply < int(frames.size()));

    Frame& f = frames[ply];

    auto index = [](const Frame& frame, Color c, Piece pc, Square s) {
        return frame.bucket[c] * 768 + ((color_of(pc) != c) * 6 + type_of(pc) - 1) * 64
             + (int(s) ^ frame.flip[c]);
    };

    auto refresh = [&](Color c) {
        std::copy(network.biases.begin(), network.biases.end(), f.sums[c].begin());
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            if (f.board[s] != NO_PIECE)
                add_row<1>(f.sums[c], &network.weights[index(f, c, f.board[s], s) * HalfWidth]);
        for (int a = 0; a < 12; ++a)
            if (f.auxiliary[c][a])
                add_row<1>(f.sums[c], &network.weights[(12288 + a) * HalfWidth]);
    };

    if (lastFrame != ply)
    {
        f.board     = pos.piece_array();
        f.auxiliary = {};
        for (Color c : {WHITE, BLACK})
        {
            int       king   = int(pos.square<KING>(c)) ^ (c == BLACK ? 56 : 0);
            const int mirror = king % 8 >= 4 ? 7 : 0;
            f.flip[c]        = (c == BLACK ? 56 : 0) ^ mirror;
            king ^= mirror;
            f.bucket[c] = (king / 8) * 2 + (king % 8) / 2;

            for (int opponent = 0; opponent < 2; ++opponent)
                for (int wing = 0; wing < 2; ++wing)
                    f.auxiliary[c][opponent * 2 + wing] = pos.can_castle(
                      CastlingRights(1 << ((c ^ opponent) * 2 + (wing ^ bool(mirror)))));

            if (pos.ep_square() != SQ_NONE)
                f.auxiliary[c][4 + (int(pos.ep_square()) ^ f.flip[c]) % 8] = true;
        }

        const int ancestor = lastFrame;
        Square    changed[SQUARE_NB];
        int       changedCount = 0;
        if (ancestor >= 0)
            for (Square s = SQ_A1; s <= SQ_H8; ++s)
                if (f.board[s] != frames[ancestor].board[s])
                    changed[changedCount++] = s;

        const int pieceCount = popcount(pos.pieces());
        for (Color c : {WHITE, BLACK})
        {
            if (ancestor < 0 || frames[ancestor].bucket[c] != f.bucket[c]
                || frames[ancestor].flip[c] != f.flip[c])
            {
                refresh(c);
                continue;
            }

            // Refresh when that adds fewer rows than updating
            const Frame& parent    = frames[ancestor];
            int          edits     = 0;
            int          freshRows = pieceCount;
            for (int i = 0; i < changedCount; ++i)
                edits += (parent.board[changed[i]] != NO_PIECE) + (f.board[changed[i]] != NO_PIECE);
            for (int a = 0; a < 12; ++a)
            {
                freshRows += f.auxiliary[c][a];
                edits += f.auxiliary[c][a] != parent.auxiliary[c][a];
            }

            if (edits > freshRows)
            {
                refresh(c);
                continue;
            }

            f.sums[c] = parent.sums[c];
            for (int i = 0; i < changedCount; ++i)
            {
                const Square s = changed[i];
                if (parent.board[s] != NO_PIECE)
                    add_row<-1>(f.sums[c],
                                &network.weights[index(parent, c, parent.board[s], s) * HalfWidth]);
                if (f.board[s] != NO_PIECE)
                    add_row<1>(f.sums[c], &network.weights[index(f, c, f.board[s], s) * HalfWidth]);
            }
            for (int a = 0; a < 12; ++a)
                if (f.auxiliary[c][a] != parent.auxiliary[c][a])
                {
                    if (f.auxiliary[c][a])
                        add_row<1>(f.sums[c], &network.weights[(12288 + a) * HalfWidth]);
                    else
                        add_row<-1>(f.sums[c], &network.weights[(12288 + a) * HalfWidth]);
                }
        }

        parentFrame[ply] = ancestor;
        lastFrame        = ply;
    }

    const Color us = pos.side_to_move();

    alignas(64) i16 hidden[Width];
    for (int p = 0; p < 2; ++p)
        for (int j = 0; j < HalfWidth; ++j)
            hidden[p * HalfWidth + j] = i16(std::clamp(f.sums[us ^ p][j], 0, 4096));

    const int  stack  = std::clamp((popcount(pos.pieces()) - 1) / 8, 0, Stacks - 1);
    const int  orient = (us == BLACK ? 56 : 0) ^ (file_of(pos.square<KING>(us)) >= FILE_E ? 7 : 0);
    const i16* rows   = &network.outputs[stack * Outputs * Width];

    for (int i = 0; i < count; ++i)
    {
        const Move m        = moves[i];
        const int  from     = int(m.from_sq()) ^ orient;
        const int  to       = int(m.to_sq()) ^ orient;
        const int  category = m.type_of() == PROMOTION ? m.promotion_type() - 1 : 0;

        const i64 sum = dot(hidden, rows + (from * 64 + to) * Width)
                      + dot(hidden, rows + (4096 + category) * Width);
        logits[i] = float(sum) / (4096.0f * 4096.0f);
    }
}

void QuietCandidates::collect(ExtMove* begin, ExtMove* end, Move ttMove, int threshold) {
    count = 0;
    for (ExtMove* it = begin; it != end && count < Limit; ++it)
    {
        if (it->value <= threshold)
            break;
        if (*it == ttMove)
            continue;
        if (count && slots[0]->value - it->value > 2048)
            break;
        slots[count++] = it;
    }
}

// Lowers each history score by up to 2048 for a lower logit than the best, then
// picks the first three by that score. All other moves keep their order.
void QuietCandidates::promote() const {
    if (count == 2)
    {
        if (logits[1] <= logits[0])
            return;
        const float leader =
          float(slots[0]->value) + 1024.0f * std::clamp(logits[0] - logits[1], -2.0f, 0.0f);
        if (float(slots[1]->value) > leader)
            std::swap(*slots[0], *slots[1]);
        return;
    }

    const float maximum = *std::max_element(logits.begin(), logits.begin() + count);

    std::array<float, Limit>   scores;
    std::array<ExtMove, Limit> original;
    std::array<int, Limit>     order;
    for (int i = 0; i < count; ++i)
    {
        original[i] = *slots[i];
        order[i]    = i;
        scores[i] =
          float(original[i].value) + 1024.0f * std::clamp(logits[i] - maximum, -2.0f, 0.0f);
    }

    bool changed = false;
    for (int i = 0; i < std::min(3, count); ++i)
    {
        int best = i;
        for (int j = i + 1; j < count; ++j)
            if (scores[order[j]] > scores[order[best]])
                best = j;
        if (best != i)
        {
            std::rotate(order.begin() + i, order.begin() + best, order.begin() + best + 1);
            changed = true;
        }
    }

    if (changed)
        for (int i = 0; i < count; ++i)
            *slots[i] = original[order[i]];
}

void QuietCache::clear() {
    for (auto& entry : entries)
        entry.count = entry.replace = 0;
}

bool QuietCache::contains(Key key) const {
    const Entry& entry = entries[key & (Size - 1)];
    return entry.count && entry.key == key;
}

// Fills the cached logits and returns the number of candidates still missing
int QuietCache::get(Key                                       key,
                    QuietCandidates&                          candidates,
                    std::array<bool, QuietCandidates::Limit>& missing) const {
    const Entry& entry   = entries[key & (Size - 1)];
    int          remains = 0;
    for (int i = 0; i < candidates.count; ++i)
    {
        missing[i] = true;
        if (entry.key == key)
            for (int j = 0; j < entry.count; ++j)
                if (entry.moves[j] == Move(*candidates.slots[i]))
                {
                    candidates.logits[i] = entry.logits[j];
                    missing[i]           = false;
                    break;
                }
        remains += missing[i];
    }
    return remains;
}

void QuietCache::put(Key key, Move move, float logit) {
    Entry& entry = entries[key & (Size - 1)];
    if (entry.key != key)
    {
        entry.key   = key;
        entry.count = entry.replace = 0;
    }

    for (int i = 0; i < entry.count; ++i)
        if (entry.moves[i] == move)
        {
            entry.logits[i] = logit;
            return;
        }

    const int slot     = entry.count < MovesPerEntry ? entry.count++ : entry.replace;
    entry.moves[slot]  = move;
    entry.logits[slot] = logit;
    entry.replace      = (slot + 1) % MovesPerEntry;
}

}  // namespace Stockfish::Policy
