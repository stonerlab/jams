//
// Created by Joe Barker on 2017/05/25.
//

#include <iostream>
#include <sstream>
#include <string>


#include "jams/core/utils.h"

std::string word_wrap(const char *text, size_t line_length = 72) {
// https://www.rosettacode.org/wiki/Word_wrap#C.2B.2B
  std::istringstream words(text);
  std::ostringstream wrapped;
  std::string word;

  if (words >> word) {
    wrapped << word;
    size_t space_left = line_length - word.length();
    while (words >> word) {
      if (space_left < word.length() + 1) {
        wrapped << '\n' << word;
        space_left = line_length - word.length();
      } else {
        wrapped << ' ' << word;
        space_left -= word.length() + 1;
      }
    }
  }
  return wrapped.str();
}
