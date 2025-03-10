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

#include "nnue_accumulator.h"

#include "nnue_feature_transformer.h"

namespace Stockfish::Eval::NNUE {

void AccumulatorState::reset(const DirtyPiece& dp) noexcept {
    dirtyPiece = dp;
    accumulatorBig.computed.fill(false);
    accumulatorSmall.computed.fill(false);
}

void AccumulatorStack::push(const DirtyPiece& dirtyPiece) noexcept {
    assert(m_current_idx + 1 < m_accumulators.size());
    m_current_idx++;
    m_accumulators[m_current_idx].reset(dirtyPiece);
}

void AccumulatorStack::pop() noexcept { m_current_idx--; }


template<IndexType                                 TransformedFeatureDimensions,
         Color                                     Perspective,
         IncUpdateDirection                        Direction = FORWARD,
         Accumulator<TransformedFeatureDimensions> AccumulatorState::* accPtr>
void update_accumulator_incremental(
  const FeatureTransformer<TransformedFeatureDimensions, accPtr>& featureTransformer,
  const Square                                                    ksq,
  AccumulatorState&                                               target_state,
  const AccumulatorState&                                         computed) {
    [[maybe_unused]] constexpr bool Forward   = Direction == FORWARD;
    [[maybe_unused]] constexpr bool Backwards = Direction == BACKWARDS;

    assert(Forward != Backwards);

    assert((computed->*accPtr).computed[Perspective]);
    assert(!(target_state->*accPtr).computed[Perspective]);

    // The size must be enough to contain the largest possible update.
    // That might depend on the feature set and generally relies on the
    // feature set's update cost calculation to be correct and never allow
    // updates with more added/removed features than MaxActiveDimensions.
    // In this case, the maximum size of both feature addition and removal
    // is 2, since we are incrementally updating one move at a time.
    FeatureSet::IndexList removed, added;
    if constexpr (Forward)
        FeatureSet::append_changed_indices<Perspective>(ksq, computed.dirtyPiece, removed, added);
    else
        FeatureSet::append_changed_indices<Perspective>(ksq, computed.dirtyPiece, added, removed);

    if (removed.size() == 0 && added.size() == 0)
    {
        std::memcpy((target_state->*accPtr).accumulation[Perspective],
                    (computed->*accPtr).accumulation[Perspective],
                    TransformedFeatureDimensions * sizeof(BiasType));
        std::memcpy((target_state->*accPtr).psqtAccumulation[Perspective],
                    (computed->*accPtr).psqtAccumulation[Perspective],
                    PSQTBuckets * sizeof(PSQTWeightType));
    }
    else
    {
        assert(added.size() == 1 || added.size() == 2);
        assert(removed.size() == 1 || removed.size() == 2);

        if (Forward)
            assert(added.size() <= removed.size());
        else
            assert(removed.size() <= added.size());

#ifdef VECTOR
        auto* accIn =
          reinterpret_cast<const vec_t*>(&(computed->*accPtr).accumulation[Perspective][0]);
        auto* accOut =
          reinterpret_cast<vec_t*>(&(target_state->*accPtr).accumulation[Perspective][0]);

        const IndexType offsetA0 = TransformedFeatureDimensions * added[0];
        auto* columnA0 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetA0]);
        const IndexType offsetR0 = TransformedFeatureDimensions * removed[0];
        auto* columnR0 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetR0]);

        if ((Forward && removed.size() == 1) || (Backwards && added.size() == 1))
        {
            assert(added.size() == 1 && removed.size() == 1);
            for (IndexType i = 0;
                 i < TransformedFeatureDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                accOut[i] = vec_add_16(vec_sub_16(accIn[i], columnR0[i]), columnA0[i]);
        }
        else if (Forward && added.size() == 1)
        {
            assert(removed.size() == 2);
            const IndexType offsetR1 = TransformedFeatureDimensions * removed[1];
            auto* columnR1 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetR1]);

            for (IndexType i = 0;
                 i < TransformedFeatureDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                accOut[i] = vec_sub_16(vec_add_16(accIn[i], columnA0[i]),
                                       vec_add_16(columnR0[i], columnR1[i]));
        }
        else if (Backwards && removed.size() == 1)
        {
            assert(added.size() == 2);
            const IndexType offsetA1 = TransformedFeatureDimensions * added[1];
            auto* columnA1 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetA1]);

            for (IndexType i = 0;
                 i < TransformedFeatureDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                accOut[i] = vec_add_16(vec_add_16(accIn[i], columnA0[i]),
                                       vec_sub_16(columnA1[i], columnR0[i]));
        }
        else
        {
            assert(added.size() == 2 && removed.size() == 2);
            const IndexType offsetA1 = TransformedFeatureDimensions * added[1];
            auto* columnA1 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetA1]);
            const IndexType offsetR1 = TransformedFeatureDimensions * removed[1];
            auto* columnR1 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetR1]);

            for (IndexType i = 0;
                 i < TransformedFeatureDimensions * sizeof(WeightType) / sizeof(vec_t); ++i)
                accOut[i] = vec_add_16(accIn[i], vec_sub_16(vec_add_16(columnA0[i], columnA1[i]),
                                                            vec_add_16(columnR0[i], columnR1[i])));
        }

        auto* accPsqtIn = reinterpret_cast<const psqt_vec_t*>(
          &(computed->*accPtr).psqtAccumulation[Perspective][0]);
        auto* accPsqtOut =
          reinterpret_cast<psqt_vec_t*>(&(target_state->*accPtr).psqtAccumulation[Perspective][0]);

        const IndexType offsetPsqtA0 = PSQTBuckets * added[0];
        auto*           columnPsqtA0 =
          reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offsetPsqtA0]);
        const IndexType offsetPsqtR0 = PSQTBuckets * removed[0];
        auto*           columnPsqtR0 =
          reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offsetPsqtR0]);

        if ((Forward && removed.size() == 1)
            || (Backwards && added.size() == 1))  // added.size() == removed.size() == 1
        {
            for (std::size_t i = 0; i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t);
                 ++i)
                accPsqtOut[i] =
                  vec_add_psqt_32(vec_sub_psqt_32(accPsqtIn[i], columnPsqtR0[i]), columnPsqtA0[i]);
        }
        else if (Forward && added.size() == 1)
        {
            const IndexType offsetPsqtR1 = PSQTBuckets * removed[1];
            auto*           columnPsqtR1 =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offsetPsqtR1]);

            for (std::size_t i = 0; i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t);
                 ++i)
                accPsqtOut[i] = vec_sub_psqt_32(vec_add_psqt_32(accPsqtIn[i], columnPsqtA0[i]),
                                                vec_add_psqt_32(columnPsqtR0[i], columnPsqtR1[i]));
        }
        else if (Backwards && removed.size() == 1)
        {
            const IndexType offsetPsqtA1 = PSQTBuckets * added[1];
            auto*           columnPsqtA1 =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offsetPsqtA1]);

            for (std::size_t i = 0; i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t);
                 ++i)
                accPsqtOut[i] = vec_add_psqt_32(vec_add_psqt_32(accPsqtIn[i], columnPsqtA0[i]),
                                                vec_sub_psqt_32(columnPsqtA1[i], columnPsqtR0[i]));
        }
        else
        {
            const IndexType offsetPsqtA1 = PSQTBuckets * added[1];
            auto*           columnPsqtA1 =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offsetPsqtA1]);
            const IndexType offsetPsqtR1 = PSQTBuckets * removed[1];
            auto*           columnPsqtR1 =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offsetPsqtR1]);

            for (std::size_t i = 0; i < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t);
                 ++i)
                accPsqtOut[i] = vec_add_psqt_32(
                  accPsqtIn[i], vec_sub_psqt_32(vec_add_psqt_32(columnPsqtA0[i], columnPsqtA1[i]),
                                                vec_add_psqt_32(columnPsqtR0[i], columnPsqtR1[i])));
        }
#else
        std::memcpy((target_state->*accPtr).accumulation[Perspective],
                    (computed->*accPtr).accumulation[Perspective],
                    TransformedFeatureDimensions * sizeof(BiasType));
        std::memcpy((target_state->*accPtr).psqtAccumulation[Perspective],
                    (computed->*accPtr).psqtAccumulation[Perspective],
                    PSQTBuckets * sizeof(PSQTWeightType));

        // Difference calculation for the deactivated features
        for (const auto index : removed)
        {
            const IndexType offset = TransformedFeatureDimensions * index;
            for (IndexType i = 0; i < TransformedFeatureDimensions; ++i)
                (target_state->*accPtr).accumulation[Perspective][i] -=
                  featureTransformer.weights[offset + i];

            for (std::size_t i = 0; i < PSQTBuckets; ++i)
                (target_state->*accPtr).psqtAccumulation[Perspective][i] -=
                  featureTransformer.psqtWeights[index * PSQTBuckets + i];
        }

        // Difference calculation for the activated features
        for (const auto index : added)
        {
            const IndexType offset = TransformedFeatureDimensions * index;
            for (IndexType i = 0; i < TransformedFeatureDimensions; ++i)
                (target_state->*accPtr).accumulation[Perspective][i] +=
                  featureTransformer.weights[offset + i];

            for (std::size_t i = 0; i < PSQTBuckets; ++i)
                (target_state->*accPtr).psqtAccumulation[Perspective][i] +=
                  featureTransformer.psqtWeights[index * PSQTBuckets + i];
        }
#endif
    }

    (target_state->*accPtr).computed[Perspective] = true;
}

}
