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

#pragma once

#include <obs-module.h>

#define write_log(log_level, format, ...) \
    blog(log_level, "[music-detect] " format, ##__VA_ARGS__)

#if defined(_DEBUG)
#    define bdebug(format, ...) \
        write_log(              \
            LOG_INFO, format,   \
            ##__VA_ARGS__) // apparently debug level isn't logged by default, so we just use info level for debug builds
#else
#    define bdebug(format, ...) write_log(LOG_DEBUG, format, ##__VA_ARGS__)
#endif
#define binfo(format, ...) write_log(LOG_INFO, format, ##__VA_ARGS__)
#define bwarn(format, ...) write_log(LOG_WARNING, format, ##__VA_ARGS__)
#define berr(format, ...) write_log(LOG_ERROR, format, ##__VA_ARGS__)

