/*** 
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2021-12-14 11:09:28
 * @LastEditors: Tianyun Xuan
 * @FilePath: /apollo/modules/omnisense/innolog/rotate_log_file.h
 */

#ifndef OMNISENSE_INNOLOG_ROTATE_LOG_FILE_H_
#define OMNISENSE_INNOLOG_ROTATE_LOG_FILE_H_
#include <string>

namespace innovusion {
namespace innolog {

class RotateLogFile {
 public:
  RotateLogFile(const char *rotate_file_base_file,
             uint32_t rotate_file_number,
             uint64_t rotate_file_size_limit);
  ~RotateLogFile();
  void write(const char *h1, const char *h2, const char *body);
  std::string get_log_address();

 private:
  int log_rotate_();

 private:
  pthread_mutex_t mutex_;
  int log_file_fd_;
  size_t current_file_size_;
  std::string rotate_file_base_file_;
  uint32_t rotate_file_number_;
  size_t rotate_file_size_limit_;
};

}  // namespace innolog
}  // namespace innovusion
#endif  // OMNISENSE_INNOLOG_ROTATE_LOG_FILE_H_


