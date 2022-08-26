/***
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2022-04-14 10:07:55
 * @LastEditors: Tianyun Xuan
 * @FilePath: /calib/innolog/innolog.h
 */

#ifndef OMNISENSE_INNOLOG_INNOLOG_H_
#define OMNISENSE_INNOLOG_INNOLOG_H_

#include <signal.h>

#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "innolog_base.h"
#include "rotate_log_file.h"
static std::string KMODULE_NAME = "Removert";
void innolog_collector(enum InnoLogLevel, const char *file, int line,
                       const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));

namespace innovusion {
namespace innolog {

class InnoLog {
   private:
    InnoLog();
    ~InnoLog();

   public:
    InnoLog(InnoLog const &) = delete;
    void operator=(InnoLog const &) = delete;

    // static void setup_sig_handler();
    static InnoLog *get_instance(void);
    RotateLogFile *get_rotate_out(void);
    RotateLogFile *get_rotate_error(void);

    bool check_level(enum InnoLogLevel level);
    bool check_module(std::string module);
    bool check_switch();
    enum InnoLogLevel get_log_level();
    void set_log_level(int level);
    void module_switch_on();
    void module_switch_off();
    void module_switch_on(std::initializer_list<std::string> module_list);
    void module_switch_off(std::initializer_list<std::string> module_list);

   public:
    void set_logs(int out_fd, int error_fd, const char *rotate_file_base_file,
                  uint32_t rotate_file_number, uint64_t rotate_file_size_limit,
                  const char *rotate_file_base_file_err,
                  uint32_t rotate_file_number_err,
                  uint64_t rotate_file_size_limit_err,
                  InnoLogCallback log_callback, void *ctx,
                  enum InnoLogLevel inno_log_level,
                  std::unordered_map<std::string, bool> module_gather);
    void set_log_address(std::string addr);
    int log_operator(enum InnoLogLevel level, std::string module,
                     const char *file, int line, int malloc_size,
                     const char *fmt, va_list valist);

   private:
    int log_out_fd_;
    int log_error_fd_;
    RotateLogFile *rotate_out_;
    RotateLogFile *rotate_error_;
    InnoLogCallback log_callback_;
    void *log_callback_ctx_;
    enum InnoLogLevel inno_log_level_;
    std::unordered_map<std::string, bool> module_gather_;
    bool module_switch_;
    pthread_mutex_t log_out_mutex_;
    pthread_mutex_t log_err_mutex_;
    std::mutex mutex_;
};
}  // namespace innolog
}  // namespace innovusion
#endif  // OMNISENSE_INNOLOG_INNOLOG_H_
