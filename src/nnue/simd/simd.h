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

#include <algorithm>
#include <utility>

#ifndef SIMD_H_INCLUDED

    #if defined(USE_AVX2)
        #include <immintrin.h>

    #elif defined(USE_SSE41)
        #include <smmintrin.h>

    #elif defined(USE_SSSE3)
        #include <tmmintrin.h>

    #elif defined(USE_SSE2)
        #include <emmintrin.h>

    #elif defined(USE_NEON)
        #include <arm_neon.h>
    #endif

    #define VECTOR

    #ifdef USE_AVX512
using vec_t      = __m512i;
using psqt_vec_t = __m256i;
        #define vec_load(a) _mm512_load_si512(a)
        #define vec_store(a, b) _mm512_store_si512(a, b)
        #define vec_add_16(a, b) _mm512_add_epi16(a, b)
        #define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
        #define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
        #define vec_zero() _mm512_setzero_epi32()
        #define vec_set_16(a) _mm512_set1_epi16(a)
        #define vec_max_16(a, b) _mm512_max_epi16(a, b)
        #define vec_min_16(a, b) _mm512_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
        // Inverse permuted at load time
        #define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
        #define vec_load_psqt(a) _mm256_load_si256(a)
        #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
        #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
        #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
        #define vec_zero_psqt() _mm256_setzero_si256()
        #define NumRegistersSIMD 16
        #define MaxChunkSize 64

    #elif USE_AVX2
using vec_t      = __m256i;
using psqt_vec_t = __m256i;
        #define vec_load(a) _mm256_load_si256(a)
        #define vec_store(a, b) _mm256_store_si256(a, b)
        #define vec_add_16(a, b) _mm256_add_epi16(a, b)
        #define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
        #define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
        #define vec_zero() _mm256_setzero_si256()
        #define vec_set_16(a) _mm256_set1_epi16(a)
        #define vec_max_16(a, b) _mm256_max_epi16(a, b)
        #define vec_min_16(a, b) _mm256_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
        // Inverse permuted at load time
        #define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
        #define vec_load_psqt(a) _mm256_load_si256(a)
        #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
        #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
        #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
        #define vec_zero_psqt() _mm256_setzero_si256()
        #define NumRegistersSIMD 16
        #define MaxChunkSize 32

    #elif USE_SSE2
using vec_t      = __m128i;
using psqt_vec_t = __m128i;
        #define vec_load(a) (*(a))
        #define vec_store(a, b) *(a) = (b)
        #define vec_add_16(a, b) _mm_add_epi16(a, b)
        #define vec_sub_16(a, b) _mm_sub_epi16(a, b)
        #define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
        #define vec_zero() _mm_setzero_si128()
        #define vec_set_16(a) _mm_set1_epi16(a)
        #define vec_max_16(a, b) _mm_max_epi16(a, b)
        #define vec_min_16(a, b) _mm_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm_slli_epi16(a, b)
        #define vec_packus_16(a, b) _mm_packus_epi16(a, b)
        #define vec_load_psqt(a) (*(a))
        #define vec_store_psqt(a, b) *(a) = (b)
        #define vec_add_psqt_32(a, b) _mm_add_epi32(a, b)
        #define vec_sub_psqt_32(a, b) _mm_sub_epi32(a, b)
        #define vec_zero_psqt() _mm_setzero_si128()
        #define NumRegistersSIMD (Is64Bit ? 16 : 8)
        #define MaxChunkSize 16

    #elif USE_NEON
using vec_t      = int16x8_t;
using psqt_vec_t = int32x4_t;
        #define vec_load(a) (*(a))
        #define vec_store(a, b) *(a) = (b)
        #define vec_add_16(a, b) vaddq_s16(a, b)
        #define vec_sub_16(a, b) vsubq_s16(a, b)
        #define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
        #define vec_zero() \
            vec_t { 0 }
        #define vec_set_16(a) vdupq_n_s16(a)
        #define vec_max_16(a, b) vmaxq_s16(a, b)
        #define vec_min_16(a, b) vminq_s16(a, b)
        #define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
        #define vec_packus_16(a, b) \
            reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
        #define vec_load_psqt(a) (*(a))
        #define vec_store_psqt(a, b) *(a) = (b)
        #define vec_add_psqt_32(a, b) vaddq_s32(a, b)
        #define vec_sub_psqt_32(a, b) vsubq_s32(a, b)
        #define vec_zero_psqt() \
            psqt_vec_t { 0 }
        #define NumRegistersSIMD 16
        #define MaxChunkSize 16

    #else
        #undef VECTOR

    #endif

    #define SIMD_H_INCLUDED
#endif


struct Vec16Wrapper {
#ifdef VECTOR
    using type = vec_t;
    static type add(const type& lhs, const type& rhs) { return vec_add_16(lhs, rhs); }
    static type sub(const type& lhs, const type& rhs) { return vec_sub_16(lhs, rhs); }
#else
    using type = BiasType;
    static type add(const type& lhs, const type& rhs) { return lhs + rhs; }
    static type sub(const type& lhs, const type& rhs) { return lhs - rhs; }
#endif
};

struct Vec32Wrapper {
#ifdef VECTOR
    using type = psqt_vec_t;
    static type add(const type& lhs, const type& rhs) { return vec_add_psqt_32(lhs, rhs); }
    static type sub(const type& lhs, const type& rhs) { return vec_sub_psqt_32(lhs, rhs); }
#else
    using type = PSQTWeightType;
    static type add(const type& lhs, const type& rhs) { return lhs + rhs; }
    static type sub(const type& lhs, const type& rhs) { return lhs - rhs; }
#endif
};

enum UpdateOperation {
    Add,
    Sub
};

template<typename VecWrapper,
         UpdateOperation... ops,
         std::enable_if_t<sizeof...(ops) == 0, bool> = true>
typename VecWrapper::type fused(const typename VecWrapper::type& in) {
    return in;
}

template<typename VecWrapper,
         UpdateOperation update_op,
         UpdateOperation... ops,
         typename T,
         typename... Ts,
         std::enable_if_t<is_all_same_v<typename VecWrapper::type, T, Ts...>, bool> = true,
         std::enable_if_t<sizeof...(ops) == sizeof...(Ts), bool>                    = true>
typename VecWrapper::type
fused(const typename VecWrapper::type& in, const T& operand, const Ts&... operands) {
    switch (update_op)
    {
    case Add :
        return fused<VecWrapper, ops...>(VecWrapper::add(in, operand), operands...);
    case Sub :
        return fused<VecWrapper, ops...>(VecWrapper::sub(in, operand), operands...);
    default :
        static_assert(update_op == Add || update_op == Sub,
                      "Only Add and Sub are currently supported.");
        return typename VecWrapper::type();
    }
}


#ifdef VECTOR
    // We use __m* types as template arguments, which causes GCC to emit warnings
    // about losing some attribute information. This is irrelevant to us as we
    // only take their size, so the following pragma are harmless.
    #if defined(__GNUC__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif

template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
static constexpr int BestRegisterCount() {
    constexpr std::size_t RegisterSize = sizeof(SIMDRegisterType);
    constexpr std::size_t LaneSize     = sizeof(LaneType);

    static_assert(RegisterSize >= LaneSize);
    static_assert(MaxRegisters <= NumRegistersSIMD);
    static_assert(MaxRegisters > 0);
    static_assert(NumRegistersSIMD > 0);
    static_assert(RegisterSize % LaneSize == 0);
    static_assert((NumLanes * LaneSize) % RegisterSize == 0);

    const int ideal = (NumLanes * LaneSize) / RegisterSize;
    if (ideal <= MaxRegisters)
        return ideal;

    // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
    for (int divisor = MaxRegisters; divisor > 1; --divisor)
        if (ideal % divisor == 0)
            return divisor;

    return 1;
}

    #if defined(__GNUC__)
        #pragma GCC diagnostic pop
    #endif
#endif
