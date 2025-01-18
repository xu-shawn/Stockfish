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

#include "timeman.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "misc.h"
#include "search.h"
#include "tune.h"
#include "ucioption.h"

namespace Stockfish {

int mtg_base     = 5051;
int ota_coeff    = 3128;
int ota_constant = 4354;

int opt_base  = 321160;
int opt_coeff = 321123;
int opt_max   = 508017;

int max_constant_constant = 339770;
int max_constant_coeff    = 303950;
int max_constant_min      = 294761;

int opt_scale_constant     = 121431;
int opt_scale_pow_base     = 294693;
int opt_scale_pow_exponent = 461073;
int opt_scale_max_coeff    = 213035;

int max_scale_maximum = 667704;
int max_scale_divisor = 119847;

int maximum_time_clamp_coeff = 825178;

TUNE(SetRange(1000, 10000), mtg_base);
TUNE(ota_coeff,
     ota_constant,
     opt_base,
     opt_coeff,
     opt_max,
     max_constant_constant,
     max_constant_coeff,
     max_constant_min,
     opt_scale_constant,
     opt_scale_pow_base,
     opt_scale_pow_exponent,
     opt_scale_max_coeff,
     max_scale_maximum,
     max_scale_divisor);
TUNE(SetRange(805000, 855000), maximum_time_clamp_coeff);

TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }

void TimeManagement::clear() {
    availableNodes = -1;  // When in 'nodes as time' mode
}

void TimeManagement::advance_nodes_time(std::int64_t nodes) {
    assert(useNodesTime);
    availableNodes = std::max(int64_t(0), availableNodes - nodes);
}

// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)
void TimeManagement::init(Search::LimitsType& limits,
                          Color               us,
                          int                 ply,
                          const OptionsMap&   options,
                          double&             originalTimeAdjust) {
    TimePoint npmsec = TimePoint(options["nodestime"]);

    // If we have no time, we don't need to fully initialize TM.
    // startTime is used by movetime and useNodesTime is used in elapsed calls.
    startTime    = limits.startTime;
    useNodesTime = npmsec != 0;

    if (limits.time[us] == 0)
        return;

    TimePoint moveOverhead = TimePoint(options["Move Overhead"]);

    // optScale is a percentage of available time to use for the current move.
    // maxScale is a multiplier applied to optimumTime.
    double optScale, maxScale;

    // If we have to play in 'nodes as time' mode, then convert from time
    // to nodes, and use resulting values in time management formulas.
    // WARNING: to avoid time losses, the given npmsec (nodes per millisecond)
    // must be much lower than the real engine speed.
    if (useNodesTime)
    {
        if (availableNodes == -1)                       // Only once at game start
            availableNodes = npmsec * limits.time[us];  // Time is in msec

        // Convert from milliseconds to nodes
        limits.time[us] = TimePoint(availableNodes);
        limits.inc[us] *= npmsec;
        limits.npmsec = npmsec;
        moveOverhead *= npmsec;
    }

    // These numbers are used where multiplications, divisions or comparisons
    // with constants are involved.
    const int64_t   scaleFactor = useNodesTime ? npmsec : 1;
    const TimePoint scaledTime  = limits.time[us] / scaleFactor;
    const TimePoint scaledInc   = limits.inc[us] / scaleFactor;

    // Maximum move horizon of 50 moves
    int centiMTG = limits.movestogo ? std::min(limits.movestogo, 50) * 100 : mtg_base;

    // If less than one second, gradually reduce mtg
    if (scaledTime < 1000 && double(centiMTG) / scaledInc > mtg_base / 1000.0)
    { centiMTG = scaledTime * mtg_base / 1000.0; }

    // Make sure timeLeft is > 0 since we may use it as a divisor
    TimePoint timeLeft =
      std::max(TimePoint(1),
               limits.time[us]
                 + (limits.inc[us] * (centiMTG - 100) - moveOverhead * (centiMTG + 200)) / 100);

    // x basetime (+ z increment)
    // If there is a healthy increment, timeLeft can exceed the actual available
    // game time for the current move, so also cap to a percentage of available game time.
    if (limits.movestogo == 0)
    {
        // Extra time according to timeLeft
        if (originalTimeAdjust < 0)
            originalTimeAdjust =
              ota_coeff / 10000.0 * std::log10(timeLeft) - ota_constant / 10000.0;

        // Calculate time constants based on current time left.
        double logTimeInSec = std::log10(scaledTime / 1000.0);
        double optConstant  = std::min(
           opt_base / 100000000.0 + opt_coeff / 1000000000.0 * logTimeInSec, opt_max / 100000000.0);
        double maxConstant =
          std::max(max_constant_constant / 100000.0 + max_constant_coeff * logTimeInSec / 100000.0,
                   max_constant_min / 100000.0);

        optScale = std::min(opt_scale_constant / 10000000.0
                              + std::pow(ply + opt_scale_pow_base / 100000.0,
                                         opt_scale_pow_exponent / 1000000.0)
                                  * optConstant,
                            opt_scale_max_coeff / 1000000.0 * limits.time[us] / timeLeft)
                 * originalTimeAdjust;

        maxScale =
          std::min(max_scale_maximum / 100000.0, maxConstant + ply / (max_scale_divisor / 10000.0));
    }

    // x moves in y seconds (+ z increment)
    else
    {
        optScale =
          std::min((0.88 + ply / 116.4) / (centiMTG / 100.0), 0.88 * limits.time[us] / timeLeft);
        maxScale = 1.3 + 0.11 * (centiMTG / 100.0);
    }

    // Limit the maximum possible time for this move
    optimumTime = TimePoint(optScale * timeLeft);
    maximumTime =
      TimePoint(std::min(maximum_time_clamp_coeff / 1000000.0 * limits.time[us] - moveOverhead,
                         maxScale * optimumTime))
      - 10;

    if (options["Ponder"])
        optimumTime += optimumTime / 4;
}

}  // namespace Stockfish
