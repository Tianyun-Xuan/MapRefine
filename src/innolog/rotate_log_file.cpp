/*** 
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2021-12-14 11:29:07
 * @LastEditors: Tianyun Xuan
 * @FilePath: /apollo/modules/omnisense/innolog/rotate_log_file.cpp
 */

#include "rotate_log_file.h"
#include "innolog_base.h"
#include <fcntl.h>
#include <unistd.h>
#include <string>

namespace innovusion {
namespace innolog {

RotateLogFile::RotateLogFile(const char *rotate_file_base_file,
            uint32_t rotate_file_number,
            uint64_t rotate_file_size_limit)
    : log_file_fd_(-1)
    , current_file_size_(0)
    , rotate_file_base_file_(rotate_file_base_file)
    , rotate_file_number_(rotate_file_number)
    , rotate_file_size_limit_(rotate_file_size_limit) {
  pthread_mutex_init(&mutex_, NULL);}

RotateLogFile::~RotateLogFile() {
    if (log_file_fd_ >= 0) {
      close(log_file_fd_);
      log_file_fd_ = -1;
    }
    pthread_mutex_destroy(&mutex_);
}

/*** 
 * @description: Lock and call innolog_write to write log to file
 * @param {char} *h1 log time info
 * @param {char} *h2 log header with level
 * @param {char} *body log body
 * @return {*}
 */
void RotateLogFile::write(const char *h1, const char *h2, const char *body) {
  // write to external files with size limit and rotate
  if (rotate_file_number_ > 0 &&
      rotate_file_size_limit_ > 0 &&
      rotate_file_base_file_.size() > 0) {
    pthread_mutex_lock(&mutex_);
    if (log_file_fd_ < 0) {
      log_rotate_();
    }
    if (log_file_fd_ >= 0) {
      int ret = innovusion::innolog::innolog_write(log_file_fd_, h1, h2, body);
      if (ret >= 0) {
        current_file_size_ += ret;
        if (current_file_size_ >= rotate_file_size_limit_) {
          close(log_file_fd_);
          log_file_fd_ = -1;
          current_file_size_ = 0;
        }
      }
    }
    pthread_mutex_unlock(&mutex_);
  }
  return;
}

std::string RotateLogFile::get_log_address() {
  return rotate_file_base_file_;
}

/*** 
 * @description: Rotate current log file
 * @param {*}  void
 * @return {*} 0 succeed; -1 setting error; -2 file error; -3 cursor error
 */
int RotateLogFile::log_rotate_() {
  if (rotate_file_base_file_.size() == 0) {
    perror("Rotate_file_base is not valide");
    return -1;
  }
  if (rotate_file_size_limit_  <= 0) {
    perror("Rotate_file_size_limit not positive");
    return -1;
  }
  if (rotate_file_number_ < 1) {
    perror("Rotate function disabled");
    return -1;
  }
  if (log_file_fd_ >= 0) {
    // still use the existing file
    perror("File rotate demanded while keeping using previous file");
    return 0;
  }
  int fd = open(rotate_file_base_file_.c_str(),
                O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) {
    perror("Failed to open log file");
    return -2;
  }
  // move the cursor to the end of file
  off_t sz = lseek(fd, 0, SEEK_END);
  if (sz < 0) {
    perror("Failed to move cursor in log file");
    return -3;
  }
  if (sz < (ssize_t)rotate_file_size_limit_) {
    // still have some space, just use this fd
    current_file_size_ = sz;
    log_file_fd_ = fd;
    return 0;
  }
  close(fd);

  // rotate files
  std::string base_name(rotate_file_base_file_);
  for (uint32_t dst = rotate_file_number_; dst > 0; dst--) {
    uint32_t src = dst - 1;
    std::string src_name = base_name;
    std::string dst_name = base_name + "." + std::to_string(dst);
    if (src > 0) {
      src_name += "." + std::to_string(src);
    }
    if (dst == rotate_file_number_) {
      remove(dst_name.c_str());
    } else {
      rename(src_name.c_str(), dst_name.c_str());
    }
  }

  // open new file
  fd = open(rotate_file_base_file_.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return -2;
  }
  current_file_size_ = 0;
  log_file_fd_ = fd;
  return 0;
}

}  // namespace innolog
}  // namespace innovusion
