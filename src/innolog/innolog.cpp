/***
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2022-04-14 10:11:07
 * @LastEditors: Tianyun Xuan
 * @FilePath: src/innolog/innolog.cpp
 */

#include "innolog.h"

#include <execinfo.h>
#include <stdarg.h>
#include <string.h>
#include <syscall.h>
#include <unistd.h>

#include <algorithm>
#include <string>

const char *innolog_header[] = {
    "[FATAL]", "[CRITI]", "[ERROR]", "[ TEMP]", "[ WARN]",
    "[DEBUG]", "[ INFO]", "[TRACE]", "[DETAL]", "[     ]",
};

/***
 * @description:
 * @param {enum InnoLogLevel} log level
 * @param {std::string} module module name
 * @param {char} *file __FILE__
 * @param {int} line __LINE__
 * @param {char} *fmt format string
 * @return {*} void
 */
void innolog_collector(enum InnoLogLevel level, const char *file, int line,
                       const char *fmt, ...) {
    va_list valist;
    va_start(valist, fmt);  // package up

    std::string module = KMODULE_NAME;
    if (!module.size()) {
        // search two ‘/’ in reverse direction
        std::string full_address = file;
        int end_ite = full_address.find_last_of('/');
        if (end_ite > 0) {
            int start_ite = full_address.find_last_of('/', end_ite - 1);
            if (start_ite >= 0) {
                module =
                    full_address.substr(start_ite + 1, end_ite - start_ite - 1);
            } else if (start_ite == -1) {
                module = full_address.substr(0, end_ite);
            }
        }
        if (module.empty()) module = full_address;
        transform(module.begin(), module.end(), module.begin(), ::toupper);
    }

    int ret = innovusion::innolog::InnoLog::get_instance()->log_operator(
        level, module, file, line, 0, fmt, valist);
    va_end(valist);
    // ret is the total lenth of over-sized log info, +1 for \0
    if (ret > 0) {
        va_start(valist, fmt);
        innovusion::innolog::InnoLog::get_instance()->log_operator(
            level, module, file, line, ret + 1, fmt, valist);
        va_end(valist);
    }
    return;
}

namespace innovusion {
namespace innolog {

InnoLog::InnoLog() {
    log_out_fd_ = 1;    // stdout
    log_error_fd_ = 2;  // stderr
    rotate_out_ = NULL;
    rotate_error_ = NULL;
    log_callback_ = NULL;
    log_callback_ctx_ = NULL;
    inno_log_level_ = INNO_LOG_LEVEL_INFO;
    module_gather_ = {};
    module_switch_ = true;
    pthread_mutex_init(&log_out_mutex_, NULL);
    pthread_mutex_init(&log_err_mutex_, NULL);
    std::string name = "/home/xavier/repos/MapRefine/data/log";
    rotate_out_ = new RotateLogFile(name.c_str(), 10, 50 * 1000 * 1000);
    rotate_error_ = new RotateLogFile((name + ".err").c_str(), 10, 500 * 1000);
}

InnoLog::~InnoLog() {
    pthread_mutex_destroy(&log_out_mutex_);
    pthread_mutex_destroy(&log_err_mutex_);
    if (rotate_out_) {
        delete rotate_out_;
        rotate_out_ = NULL;
    }
    if (rotate_error_) {
        delete rotate_error_;
        rotate_error_ = NULL;
    }
    if (log_callback_) {
        log_callback_ = nullptr;
    }
    if (log_callback_ctx_) {
        log_callback_ctx_ = nullptr;
    }
}
InnoLog *InnoLog::get_instance(void) {
    static InnoLog instance;
    return &instance;
}
RotateLogFile *InnoLog::get_rotate_out(void) { return rotate_out_; }
RotateLogFile *InnoLog::get_rotate_error(void) { return rotate_error_; }
bool InnoLog::check_level(enum InnoLogLevel level) {
    return inno_log_level_ >= level;
}
bool InnoLog::check_module(std::string module) {
    transform(module.begin(), module.end(), module.begin(), ::toupper);
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_gather_.find(module) != module_gather_.end()) {
        return module_gather_[module];
    } else {
        module_gather_.emplace(module, module_switch_);
        return module_switch_;
    }
}
bool InnoLog::check_switch() { return module_switch_; }
enum InnoLogLevel InnoLog::get_log_level() { return inno_log_level_; }
void InnoLog::set_log_level(int level) {
    if (level <= 9 && level >= 0) {
        inno_log_level_ = (enum InnoLogLevel)level;
    } else {
        perror("Error:Input loglevel illegal");
    }
}
void InnoLog::module_switch_on() {
    std::lock_guard<std::mutex> lock(mutex_);
    module_switch_ = true;
    for (auto &item : module_gather_) {
        item.second = true;
    }
}
void InnoLog::module_switch_off() {
    std::lock_guard<std::mutex> lock(mutex_);
    module_switch_ = false;
    for (auto &item : module_gather_) {
        item.second = false;
    }
}
void InnoLog::module_switch_on(std::initializer_list<std::string> module_list) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_list.size() > 0) {
        module_switch_ = true;
    }
    for (auto item : module_list) {
        transform(item.begin(), item.end(), item.begin(), ::toupper);
        if (module_gather_.find(item) != module_gather_.end()) {
            module_gather_[item] = true;
        } else {
            module_gather_.emplace(item, true);
        }
    }
}
void InnoLog::module_switch_off(
    std::initializer_list<std::string> module_list) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto item : module_list) {
        transform(item.begin(), item.end(), item.begin(), ::toupper);
        if (module_gather_.find(item) != module_gather_.end()) {
            module_gather_[item] = false;
        } else {
            module_gather_.emplace(item, false);
        }
    }
}

/***
 * @description: Initialize InnoLog
 * @remarque: Only use for initalization
 */
void InnoLog::set_logs(
    int out_fd, int error_fd,  // output
    const char *rotate_file_base_file, uint32_t rotate_file_number,
    uint64_t rotate_file_size_limit, const char *rotate_file_base_file_err,
    uint32_t rotate_file_number_err, uint64_t rotate_file_size_limit_err,
    InnoLogCallback log_callback, void *ctx, enum InnoLogLevel inno_log_level,
    std::unordered_map<std::string, bool> module_gather) {
    log_out_fd_ = out_fd;
    log_error_fd_ = error_fd;
    if (rotate_file_base_file && strlen(rotate_file_base_file) > 0) {
        rotate_error_ =
            new RotateLogFile(rotate_file_base_file_err, rotate_file_number_err,
                              rotate_file_size_limit_err);
        rotate_out_ = new RotateLogFile(
            rotate_file_base_file, rotate_file_number, rotate_file_size_limit);
    }
    log_callback_ = log_callback;
    log_callback_ctx_ = ctx;
    inno_log_level_ = inno_log_level;
    module_gather_ = module_gather;
}

void InnoLog::set_log_address(std::string addr) {
    if (rotate_out_) {
        delete rotate_out_;
    }
    if (rotate_error_) {
        delete rotate_error_;
    }
    rotate_out_ = new RotateLogFile(addr.c_str(), 10, 50 * 1000 * 1000);
    rotate_error_ = new RotateLogFile((addr + ".err").c_str(), 10, 500 * 1000);
}

/***
 * @description: Generate complet log message
 * @param {enum InnoLogLevel} level
 * @param {char} *file __FILE__
 * @param {int} line __LINE__
 * @param {int} malloc_size defaut value 0
 * @param {char} *fmt format string
 * @param {va_list} valist additional variables list
 * @return {*}
 */
int InnoLog::log_operator(enum InnoLogLevel level, std::string module,
                          const char *file, int line, int malloc_size,
                          const char *fmt, va_list valist) {
    char tbuffer[32];
    char tbuffer2[64];
    struct timespec spec;  // seconds + nanoseconds
    clock_gettime(CLOCK_REALTIME, &spec);
    time_t now_sec = spec.tv_sec;              // seconds
    int milli = spec.tv_nsec / (1000 * 1000);  // nanoseconds

    struct tm *tm_info;
    struct tm result_time;
    tm_info = localtime_r(&now_sec, &result_time);
    strftime(tbuffer, sizeof(tbuffer) - 1, "%Y-%m-%d %H:%M:%S", tm_info);
    tbuffer[sizeof(tbuffer) - 1] = 0;
    // ex : 2021-10-28 18:07:30.428
    snprintf(tbuffer2, sizeof(tbuffer2), "%s.%03d", tbuffer, milli);

    pid_t tid = syscall(SYS_gettid);
    static const size_t kMaxHeaderSize = 100;
    // log header with level and module
    // ex : [FATAL] [VLOG] 6560 logger/vlog_test.cpp:27
    char header[kMaxHeaderSize];
    header[0] = 0;
    snprintf(header, kMaxHeaderSize, "%s [%s] %u %s:%d", innolog_header[level],
             module.c_str(), tid, file, line);
    header[kMaxHeaderSize - 1] = 0;
    // log header without level and module
    char header_simple[kMaxHeaderSize];
    header_simple[0] = 0;
    snprintf(header_simple, kMaxHeaderSize, "%u %s:%d", tid, file, line);
    header_simple[kMaxHeaderSize - 1] = 0;

    char *buffer = NULL;
    char *buffer_alloc = NULL;
    static const ssize_t kMaxSize = 10000;
    char buffer_stack[kMaxSize];
    if (malloc_size) {
        buffer_alloc = reinterpret_cast<char *>(malloc(malloc_size));
        if (buffer_alloc) {
            buffer_alloc[0] = 0;
            vsnprintf(buffer_alloc, malloc_size, fmt, valist);
            buffer_alloc[malloc_size - 1] = 0;
            buffer = buffer_alloc;
        }
    }
    // if buffer is NULL (i.e. not use_big_malloc or not enough memory)
    if (buffer == NULL) {
        buffer_stack[0] = 0;
        int snp_size = vsnprintf(buffer_stack, kMaxSize, fmt, valist);
        // vsnprintf will add '\0' at the end
        buffer_stack[kMaxSize - 1] = 0;
        if (snp_size >= kMaxSize) {  // snp_size = total size >= KMaxSize
            if (malloc_size == 0) {
                return snp_size;
            }
        }
        buffer = buffer_stack;
    }

    int bl = strlen(buffer);
    if (bl > 0 && buffer[bl - 1] == '\n') {
        // remove unnecessary \n
        buffer[bl - 1] = 0;
    }

    {  // write to ternimal and file
        if (true) {
            // printout error
            if (level <= INNO_LOG_LEVEL_DEBUG) {
                if (log_error_fd_ >= 0) {
                    pthread_mutex_lock(&log_out_mutex_);
                    innolog_write(log_error_fd_, tbuffer2, header, buffer);
                    pthread_mutex_unlock(&log_out_mutex_);
                }
            } else {
                // printout non-error
                if (log_out_fd_ >= 0 && check_level(level)) {
                    pthread_mutex_lock(&log_err_mutex_);
                    innolog_write(log_out_fd_, tbuffer2, header, buffer);
                    pthread_mutex_unlock(&log_err_mutex_);
                }
            }
            // write non-error
            if (rotate_out_) {
                rotate_out_->write(tbuffer2, header, buffer);
            }
            // write error
            if (level <= INNO_LOG_LEVEL_DEBUG) {
                if (rotate_error_) {
                    rotate_error_->write(tbuffer2, header, buffer);
                }
            }
        }
    }

    // make log callback
    if (log_callback_) {
        log_callback_(log_callback_ctx_, (enum InnoLogLevel)level, tbuffer2,
                      header_simple, buffer);
    }
    if (buffer_alloc) {
        free(buffer_alloc);
    }
    return 0;
}
}  // namespace innolog
}  // namespace innovusion
