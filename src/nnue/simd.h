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

#ifndef NNUE_SIMD_H_INCLUDED
#define NNUE_SIMD_H_INCLUDED

#include "simd_common.h"

#ifdef USE_AVX512
    #include "simd_sse2.h"
    #include "simd_avx2.h"
    #include "simd_avx512.h"
#elif USE_AVX2
    #include "simd_sse2.h"
    #include "simd_avx2.h"
#elif USE_SSE2
    #include "simd_sse2.h"
#elif USE_NEON
    #include "simd_neon.h"
#else
    #include "simd_fallback.h"
#endif

#endif
