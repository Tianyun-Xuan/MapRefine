/*** 
 * @Copyright [2021] <Innovusion Inc.>
 * @LastEditTime: 2021-12-14 11:28:26
 * @LastEditors: Tianyun Xuan
 * @FilePath: /apollo/modules/omnisense/innolog/innolog_base.cpp
 */
#include "innolog_base.h"
#include <sys/uio.h>
#include <string.h>
#include <string>

namespace innovusion {
namespace innolog {

/*** 
 * @description: Export from buffer
 * @param {int} fd file descriptor
 * @param {char} *h1 Time_header ex:2021-10-28 18:07:30.428
 * @param {char} *h2 Level+Module+File+Line ex: [FATAL] [VLOG] 6560 logger/vlog_test.cpp:27
 * @param {char} *body Additional input Info
 * @return {*}
 */
int innolog_write(int fd, const char *h1,
                        const char *h2, const char *body) {
  char space = ' ';
  char ret = '\n';
  struct iovec ios[6];  // pointer + size
  ios[0].iov_base = const_cast<char *>(h1);
  ios[0].iov_len = strlen(h1);
  ios[1].iov_base = &space;
  ios[1].iov_len = sizeof(space);
  ios[2].iov_base = const_cast<char *>(h2);
  ios[2].iov_len = strlen(h2);
  ios[3].iov_base = &space;
  ios[3].iov_len = sizeof(space);
  ios[4].iov_base = const_cast<char *>(body);
  ios[4].iov_len = strlen(body);
  ios[5].iov_base = &ret;
  ios[5].iov_len = sizeof(ret);
  int written = writev(fd, ios, 6);  // total number of bytes
  if (written < 0) {
    perror("writev");
  }
  return written;
}
}  // namespace innolog
}  // namespace innovusion

