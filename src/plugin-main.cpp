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

#include <obs-module.h>

#include "log.h"
#include "music-filter.hpp"
#include "network.hpp"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("music-detect", "en-US")

bool obs_module_load(void)
{
    binfo("loaded v%s, %s@%s, compile time: %s", PLUGIN_VERSION,
        GIT_COMMIT_HASH, GIT_BRANCH, BUILD_TIME);

    g_network = new Network;
    // don't stall obs startup
    std::thread([] { g_network->load(); }).detach();

    obs_register_source(&music_filter);
    return true;
}

void obs_module_unload(void)
{
    delete g_network;
    g_network = nullptr;
    binfo("plugin unloaded");
}

