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

#ifndef NNUE_SIMD_SSE2_H_INCLUDED
#define NNUE_SIMD_SSE2_H_INCLUDED

#include "simd_common.h"

#include <cstdint>
#include <immintrin.h>

namespace Stockfish::Eval::NNUE::SIMD {

template<>
struct VectorizedStorage<int8_t, 16> {
    using type = __m128i;
};

template<>
struct VectorizedStorage<int16_t, 8> {
    using type = __m128i;
};

template<>
inline Vectorized<int8_t, 16>::Vectorized(int8_t scalar) {
    data = _mm_set1_epi8(scalar);
}

template<>
inline void Vectorized<int8_t, 16>::store(int8_t* dest) const {
    _mm_storeu_epi8(dest, data);
}

template<>
inline void Vectorized<int8_t, 16>::storeu(int8_t* dest) const {
    _mm_storeu_epi8(dest, data);
}

template<>
inline Vectorized<int8_t, 16> Vectorized<int8_t, 16>::zero() {
    return _mm_setzero_si128();
}

template<>
inline Vectorized<int8_t, 16> Vectorized<int8_t, 16>::load(const int8_t* data) {
    return _mm_loadu_epi8(data);
}

template<>
inline Vectorized<int8_t, 16> Vectorized<int8_t, 16>::loadu(const int8_t* data) {
    return _mm_loadu_epi8(data);
}

template<>
inline Vectorized<int16_t, 8>::Vectorized(int16_t scalar) {
    data = _mm_set1_epi16(scalar);
}

template<>
inline void Vectorized<int16_t, 8>::store(int16_t* dest) const {
    _mm_storeu_epi16(dest, data);
}

template<>
inline void Vectorized<int16_t, 8>::storeu(int16_t* dest) const {
    _mm_storeu_epi16(dest, data);
}

template<>
inline Vectorized<int16_t, 8> Vectorized<int16_t, 8>::zero() {
    return _mm_setzero_si128();
}

template<>
inline Vectorized<int16_t, 8> Vectorized<int16_t, 8>::load(const int16_t* data) {
    return _mm_loadu_epi16(data);
}

template<>
inline Vectorized<int16_t, 8> Vectorized<int16_t, 8>::loadu(const int16_t* data) {
    return _mm_loadu_epi16(data);
}

template<>
inline Vectorized<int8_t, 16> add(const Vectorized<int8_t, 16> lhs,
                                  const Vectorized<int8_t, 16> rhs) {
    return _mm_add_epi8(lhs.data, rhs.data);
}

template<>
inline Vectorized<int8_t, 16> sub(const Vectorized<int8_t, 16> lhs,
                                  const Vectorized<int8_t, 16> rhs) {
    return _mm_sub_epi8(lhs.data, rhs.data);
}

template<>
inline Vectorized<int8_t, 16> max(const Vectorized<int8_t, 16> lhs,
                                  const Vectorized<int8_t, 16> rhs) {
    return _mm_max_epi8(lhs.data, rhs.data);
}

template<>
inline Vectorized<int8_t, 16> min(const Vectorized<int8_t, 16> lhs,
                                  const Vectorized<int8_t, 16> rhs) {
    return _mm_min_epi8(lhs.data, rhs.data);
}

template<>
inline Vectorized<int8_t, 16> packus(const Vectorized<int16_t, 8> a,
                                     const Vectorized<int16_t, 8> b) {
    return _mm_packus_epi16(a.data, b.data);
}

template<>
inline Vectorized<int16_t, 8> add(const Vectorized<int16_t, 8> lhs,
                                  const Vectorized<int16_t, 8> rhs) {
    return _mm_add_epi16(lhs.data, rhs.data);
}

template<>
inline Vectorized<int16_t, 8> sub(const Vectorized<int16_t, 8> lhs,
                                  const Vectorized<int16_t, 8> rhs) {
    return _mm_sub_epi16(lhs.data, rhs.data);
}

template<>
inline Vectorized<int16_t, 8> max(const Vectorized<int16_t, 8> lhs,
                                  const Vectorized<int16_t, 8> rhs) {
    return _mm_max_epi16(lhs.data, rhs.data);
}

template<>
inline Vectorized<int16_t, 8> min(const Vectorized<int16_t, 8> lhs,
                                  const Vectorized<int16_t, 8> rhs) {
    return _mm_min_epi16(lhs.data, rhs.data);
}

}

#endif
