#include "nnue_accumulator.h"

namespace Stockfish::Eval::NNUE {

void AccumulatorState::reset(const DirtyPiece& dirtyPiece) noexcept {
    m_dirtyPiece = dirtyPiece;
    m_accumulatorBig.computed.fill(false);
    m_accumulatorSmall.computed.fill(false);
}

}
