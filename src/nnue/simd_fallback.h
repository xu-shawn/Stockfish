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

#ifndef NNUE_SIMD_FALLBACK_H_INCLUDED
#define NNUE_SIMD_FALLBACK_H_INCLUDED

#include "simd_common.h"

#include <array>

namespace Stockfish::Eval::NNUE::SIMD {

template<typename T, int Width>
struct VectorizedStoage {
    using type = std::array<T, Width>;
};

template<typename T, int Width>
Vectorized<T, Width>::Vectorized(const T scalar) {
    data.fill(scalar);
}

template<typename T, int Width>
void Vectorized<T, Width>::store(T* dest) const {
    std::copy(data.begin(), data.end(), dest);
}

template<typename T, int Width>
void Vectorized<T, Width>::storeu(T* dest) const {
    std::copy(data.begin(), data.end(), dest);
}

template<typename T, int Width>
Vectorized<T, Width> Vectorized<T, Width>::zero() {
    return Vectorized{0};
}

template<typename T, int Width>
Vectorized<T, Width> Vectorized<T, Width>::load(const T* data) {
    Vectorized result;
    std::copy(data, data + Width, result.data.begin());
    return result;
}

template<typename T, int Width>
Vectorized<T, Width> Vectorized<T, Width>::loadu(const T* data) {
    Vectorized result;
    std::copy(data, data + Width, result.data.begin());
    return result;
}

template<typename T, int Width>
Vectorized<T, Width> add(const Vectorized<T, Width> lhs, const Vectorized<T, Width> rhs) {
    Vectorized<T, Width> result;
    for (int i = 0; i < Width; i++)
        result.data[i] = lhs.data[i] + rhs.data[i];
    return result;
}

template<typename T, int Width>
Vectorized<T, Width> sub(const Vectorized<T, Width> lhs, const Vectorized<T, Width> rhs) {
    Vectorized<T, Width> result;
    for (int i = 0; i < Width; i++)
        result.data[i] = lhs.data[i] - rhs.data[i];
    return result;
}

template<typename T, int Width>
Vectorized<T, Width> max(const Vectorized<T, Width> lhs, const Vectorized<T, Width> rhs) {
    Vectorized<T, Width> result;
    for (int i = 0; i < Width; i++)
        result.data[i] = std::max(lhs.data[i], rhs.data[i]);
    return result;
}

template<typename T, int Width>
Vectorized<T, Width> min(const Vectorized<T, Width> lhs, const Vectorized<T, Width> rhs) {
    Vectorized<T, Width> result;
    for (int i = 0; i < Width; i++)
        result.data[i] = std::min(lhs.data[i], rhs.data[i]);
    return result;
}

template<typename T, int Width, typename U>
Vectorized<T, Width> packus(const Vectorized<U, Width / 2>, const Vectorized<U, Width / 2>) {
    static_assert(always_false_v<T>, "Unimplemented operation");
}

}

#endif
