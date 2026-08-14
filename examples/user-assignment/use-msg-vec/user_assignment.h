/*
 * Copyright (c) 2023 Huazhong University of Science and Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Muyuan Shen <muyuan_shen@hust.edu.cn>
 */

#ifndef USER_ASSIGNMENT_H
#define USER_ASSIGNMENT_H

#include <cstdint>

constexpr uint32_t MAX_ENBS = 4;

struct UserAssignmentEnv
{
    int32_t done;

    double timeSec;
    uint32_t ueId;
    uint32_t numUes;
    uint32_t numEnbs;

    double ux;
    double uy;
    double vx;
    double vy;

    // Debug/mobility-validation fields. Python RL code should not use these as features.
    double reward;
    double totalThroughputMbps;
    double loadStd;
    uint32_t handovers;
    uint32_t outages;
    uint32_t step;

    int32_t servingCell;
    int32_t targetBestCell;

    // Real ns-3 PHY SINR of the serving cell only [dB]. There is no per-cell
    // SINR array here: ns-3 only ever measures true SINR for the cell a UE is
    // actually attached to, so a per-cell array would mix one real value with
    // N-1 RSRP-derived proxies. Use rsrp[] below for a real, per-cell signal
    // comparison instead.
    double servingSinr;
    double load[MAX_ENBS];
    double capacity[MAX_ENBS];
    double rsrp[MAX_ENBS]; // per-cell RSRP [dBm], incl. non-serving cells
};

struct UserAssignmentAct
{
    int32_t predictedCell;
};

#endif // USER_ASSIGNMENT_H