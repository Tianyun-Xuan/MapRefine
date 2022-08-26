/*** 
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2021-12-14 14:05:33
 * @LastEditors: Tianyun Xuan
 * @FilePath: /apollo/modules/omnisense/innolog/logger.h
 */

#ifndef OMNISENSE_INNOLOG_LOGGER_H_
#define OMNISENSE_INNOLOG_LOGGER_H_
#include "innolog.h"
#include <errno.h>
#include <string.h>


#ifndef ABORT
#define ABORT() abort()
#endif

#define ILOG innovusion::innolog::InnoLog::get_instance()

// internal interface
#define ILOG_LEVEL(_level, ...)                                                \
    do {                                                                       \
        if (ILOG->check_level(_level)) {                                       \
            innolog_collector(_level, __FILE__,                                \
            __LINE__, __VA_ARGS__);                                            \
        }                                                                      \
    } while (0)
#define ILOG_IF(_level, _condition, ...)                                       \
    do {                                                                       \
        if (_condition) {                                                      \
            ILOG_LEVEL(_level, __VA_ARGS__);                                   \
        }                                                                      \
    } while (0)

// external interface
#define inno_log_fatal(...)                                                    \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_FATAL, __VA_ARGS__);          \
    } while (0)
#define inno_log_critical(...)                                                 \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_CRITICAL, __VA_ARGS__);       \
    } while (0)
#define inno_log_error(...)                                                    \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_ERROR, __VA_ARGS__);          \
    } while (0)
#define inno_log_temp(...)                                                     \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_TEMP, __VA_ARGS__);           \
    } while (0)
#define inno_log_warning(...)                                                  \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_WARNING, __VA_ARGS__);        \
    } while (0)
#define inno_log_debug(...)                                                    \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_DEBUG, __VA_ARGS__);          \
    } while (0)
#define inno_log_info(...)                                                     \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_INFO, __VA_ARGS__);           \
    } while (0)

// debug interface
#ifdef NDEBUG
#define inno_log_trace(...) do {} while (0)
#else
#define inno_log_trace(...)                                                    \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_TRACE, __VA_ARGS__);          \
    } while (0)
#endif
#ifdef NDEBUG
#define inno_log_detail(...) do {} while (0)
#else
#define inno_log_detail(...)                                                   \
    do {                                                                       \
        ILOG_LEVEL(INNO_LOG_LEVEL_DETAIL, __VA_ARGS__);         \
    } while (0)
#endif
#ifdef NDEBUG
#define inno_log_assert(_condition, ...) do {} while (0)
#else
#define inno_log_assert(_condition, ...)                                       \
    do {                                                                       \
        if (!((_condition))) {                                                 \
            ILOG_LEVEL(INNO_LOG_LEVEL_FATAL,                    \
                "[ASSERT] Condition check failed : (" #_condition ")");        \
            inno_log_fatal(__VA_ARGS__);                                       \
            ABORT();                                                           \
        }                                                                      \
    } while (0)
#endif

// ILOG conditional interface
#define IFATAL_IF(_condition, ...)                                             \
    do {                                                                       \
        ILOG_IF(INNO_LOG_LEVEL_FATAL, _condition, __VA_ARGS__);        \
        ABORT();                                                               \
    } while (0)
#define IERROR_IF(_condition, ...)                                             \
    do {                                                                       \
        ILOG_IF(INNO_LOG_LEVEL_ERROR, _condition, __VA_ARGS__);        \
    } while (0)
#define IWARNING_IF(_condition, ...)                                           \
    do {                                                                       \
        ILOG_IF(INNO_LOG_LEVEL_WARNING, _condition, __VA_ARGS__);      \
    } while (0)
#define IINFO_IF(_condition, ...)                                              \
    do {                                                                       \
        ILOG_IF(INNO_LOG_LEVEL_INFO, _condition, __VA_ARGS__);         \
    } while (0)

// inno_log conditional interface
#define inno_log_panic(...)                                                    \
    do {                                                                       \
            inno_log_fatal(__VA_ARGS__);                                       \
            ABORT();                                                           \
        }                                                                      \
    } while (0)
#define inno_log_verify(_condition, ...)                                       \
    do {                                                                       \
        if (!((_condition))) {                                                 \
            ILOG_LEVEL(INNO_LOG_LEVEL_FATAL,                    \
                "[VERIFY] Condition check failed : (" #_condition ")");        \
                inno_log_fatal(__VA_ARGS__);                                   \
            ABORT();                                                           \
        }                                                                      \
    } while (0)
#define inno_log_NOT_INPLEMENTED()                                             \
    do {                                                                       \
        inno_log_panic("NOT IMPLEMENTED%s", "");                               \
    } while (0)

// with errno output
#define inno_log_error_errno(_fmt, ...)                                        \
    do {                                                                       \
        inno_log_error("strerror: '%s' " _fmt, strerror(errno), __VA_ARGS__);  \
    } while (0)
#define inno_log_warning_errno(_fmt, ...)                                      \
    do {                                                                       \
        inno_log_warning("strerror: '%s' " _fmt, strerror(errno), __VA_ARGS__);\
    } while (0)

#endif  // OMNISENSE_INNOLOG_LOGGER_H_
