/***
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2021-12-21 16:20:06
 * @LastEditors: Tianyun Xuan
 * @FilePath: src/innolog/innolog_base.h
 */

#ifndef OMNISENSE_INNOLOG_INNOLOG_BASE_H_
#define OMNISENSE_INNOLOG_INNOLOG_BASE_H_

enum InnoLogLevel {
    INNO_LOG_LEVEL_FATAL = 0,
    INNO_LOG_LEVEL_CRITICAL = 1,
    INNO_LOG_LEVEL_ERROR = 2,
    INNO_LOG_LEVEL_TEMP = 3,
    INNO_LOG_LEVEL_WARNING = 4,
    INNO_LOG_LEVEL_DEBUG = 5,
    INNO_LOG_LEVEL_INFO = 6,
    INNO_LOG_LEVEL_TRACE = 7,
    INNO_LOG_LEVEL_DETAIL = 8,
    INNO_LOG_LEVEL_MAX = 9,
};
typedef void (*InnoLogCallback)(void *ctx, enum InnoLogLevel level,
                                const char *header1, const char *header2,
                                const char *msg);
namespace innovusion {
namespace innolog {

int innolog_write(int fd, const char *h1, const char *h2, const char *body);
}  // namespace innolog
}  // namespace innovusion
#endif  // OMNISENSE_INNOLOG_INNOLOG_BASE_H_
