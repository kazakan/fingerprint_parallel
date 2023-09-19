#pragma once

#include <stdio.h>
#include <time.h>

#include <string>

#define LOG(format, arg...)                                              \
    do {                                                                 \
        time_t cur_time;                                                 \
                                                                         \
        time(&cur_time);                                                 \
                                                                         \
        struct tm* ptm = localtime(&cur_time);                           \
        std::string time_fmt = "[%02d:%02d:%02d] ";                      \
        std::string final_fmt = time_fmt + format + "\n";                \
        printf(final_fmt.data(), ptm->tm_hour, ptm->tm_min, ptm->tm_sec, \
               ##arg);                                                   \
    } while (0)

#define DLOG(format, arg...)                           \
    do {                                               \
        std::string debug_fmt = "[func %s, line %d] "; \
        std::string fmt = debug_fmt + format;    \
        LOG(fmt, __func__, __LINE__, ##arg);     \
    } while (0);
