/*
music-detect
Copyright (C) 2024 Alex <uni@vrsal.xyz>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include "inference.hpp"

#include "network.hpp"

int run_inference(float* frames, int count)
{
    return g_network->run(frames, count);
}

bool get_inference_result(int job_id, InferenceResult* result)
{
    return g_network->get_result(job_id, *result);
}
