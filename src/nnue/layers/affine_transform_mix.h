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

// Definition of layer AffineTransformMix of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_MIX_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_MIX_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "../nnue_common.h"

// Dense mix of a factorised first hidden layer: the sparse layer computes a few directions
// and this layer forms the hidden layer's pre-activations as linear combinations of them.
// Inputs are the sparse layer's int32 outputs, saturated to int16 after >> 1; weights are
// int16 with scale 512, so out = (sum(x16 * w) >> 8) + bias is in the input's scale.
// Weights are limited to |w| <= 1019 by the trainer, so the int32 sums cannot overflow.

namespace Stockfish::Eval::NNUE::Layers {

template<IndexType InDims, IndexType OutDims>
class AffineTransformMix {
   public:
    using InputType  = i32;
    using OutputType = i32;
    using BiasType   = OutputType;
    using WeightType = i16;

    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static_assert(InputDimensions % 2 == 0 && InputDimensions <= 32);
    static_assert(OutputDimensions % 16 == 0);

    using OutputBuffer = OutputType[OutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = 0x5A3C9E17u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        read_little_endian<WeightType>(stream, weights, OutputDimensions * InputDimensions);

        // SIMD layout: for input pair p and output i, the two weights of that pair
        for (IndexType i = 0; i < OutputDimensions; ++i)
            for (IndexType j = 0; j < InputDimensions; ++j)
            {
                pairWeights[((j / 2) * OutputDimensions + i) * 2 + j % 2] =
                  weights[i * InputDimensions + j];
                columnWeights[j * OutputDimensions + i] = weights[i * InputDimensions + j];
            }

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);
        write_little_endian<WeightType>(stream, weights, OutputDimensions * InputDimensions);

        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {
        alignas(CacheLineSize) i16 x[InputDimensions];
        for (IndexType j = 0; j < InputDimensions; ++j)
            x[j] = i16(std::clamp(input[j] >> 1, -32767, 32767));

#if defined(USE_AVX512)
        constexpr IndexType NumAccums = OutputDimensions / 16;
        __m512i             acc[NumAccums];
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = _mm512_setzero_si512();
        for (IndexType p = 0; p < InputDimensions / 2; ++p)
        {
            const __m512i xp =
              _mm512_set1_epi32(i32(u32(u16(x[2 * p])) | (u32(u16(x[2 * p + 1])) << 16)));
            const auto* w =
              reinterpret_cast<const __m512i*>(&pairWeights[p * OutputDimensions * 2]);
            for (IndexType k = 0; k < NumAccums; ++k)
    #if defined(USE_VNNI)
                acc[k] = _mm512_dpwssd_epi32(acc[k], xp, w[k]);
    #else
                acc[k] = _mm512_add_epi32(acc[k], _mm512_madd_epi16(xp, w[k]));
    #endif
        }
        for (IndexType k = 0; k < NumAccums; ++k)
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&output[k * 16]),
                                _mm512_add_epi32(_mm512_srai_epi32(acc[k], 8),
                                                 _mm512_load_si512(reinterpret_cast<const __m512i*>(
                                                   &biases[k * 16]))));
#elif defined(USE_AVX2)
        constexpr IndexType NumAccums = OutputDimensions / 8;
        __m256i             acc[NumAccums];
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = _mm256_setzero_si256();
        for (IndexType p = 0; p < InputDimensions / 2; ++p)
        {
            const __m256i xp =
              _mm256_set1_epi32(i32(u32(u16(x[2 * p])) | (u32(u16(x[2 * p + 1])) << 16)));
            const auto* w =
              reinterpret_cast<const __m256i*>(&pairWeights[p * OutputDimensions * 2]);
            for (IndexType k = 0; k < NumAccums; ++k)
                acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(xp, w[k]));
        }
        for (IndexType k = 0; k < NumAccums; ++k)
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[k * 8]),
                                _mm256_add_epi32(_mm256_srai_epi32(acc[k], 8),
                                                 _mm256_load_si256(reinterpret_cast<const __m256i*>(
                                                   &biases[k * 8]))));
#else
        i32 acc[OutputDimensions] = {};
        for (IndexType j = 0; j < InputDimensions; ++j)
            for (IndexType i = 0; i < OutputDimensions; ++i)
                acc[i] += i32(x[j]) * i32(columnWeights[j * OutputDimensions + i]);
        for (IndexType i = 0; i < OutputDimensions; ++i)
            output[i] = (acc[i] >> 8) + biases[i];
#endif
    }

   private:
    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * InputDimensions];
    alignas(CacheLineSize) WeightType pairWeights[OutputDimensions * InputDimensions];
    alignas(CacheLineSize) WeightType columnWeights[OutputDimensions * InputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_MIX_H_INCLUDED
