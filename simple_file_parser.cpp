// ================================================================
// $Id: simple_file_parser.cc,v 1.2 2010/08/02 16:59:30 jdl Exp jdl $
//
// Copyright (C) 2010 Joe Linoff
//
// This source code is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This code is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see http://www.gnu.org/licenses/.
//
// ----------------------------------------------------------------
//
// Simple file parser that carries over some of the semantics
// from the LALR(n) parser. Namely it supports C++ style
// single line comments (//), multi-line C style comments
// and it ignores blank lines.
//
// It was developed to allow use to use a single paradigm
// for parsing the sites, tnames and nodes files.
// ================================================================
#include <iostream>
#include "simple_file_parser.h"
using namespace std;

#define MYDEBUG(msg) cout << "DEBUG:" << __FILE__ << ":" << __LINE__ << ": " << msg << endl

/* //Parameter not used!
simple_file_parser::simple_file_parser(const simple_file_parser& obj)
{
}

simple_file_parser& simple_file_parser::operator=(const simple_file_parser& obj)
{
  return *this;
}
*/

simple_file_parser::simple_file_parser()
  : m_ifs(0),
    m_lineno(0),
    m_bufsz(65536),
    m_in_comment(false)
{
  m_buf = new char[m_bufsz];
  size_t sz = sizeof(m_single_char_tokens)/sizeof(bool);
  for(size_t i=0;i<sz;++i)
    m_single_char_tokens[i] = false;
}

simple_file_parser::~simple_file_parser()
{
  close();
  delete [] m_buf;
  m_bufsz = 0;
}

bool simple_file_parser::file_exists(const char* fn)
{
  ifstream ifs(fn);
  if (!ifs || ifs.bad())
    return false;
  return true;
}

bool simple_file_parser::file_exists(const string& fn) {
  return file_exists(fn.c_str());
}

bool simple_file_parser::open(const string& fn)
{
  if (!file_exists(fn))
    return false;
  close();
  m_ifs = new ifstream(fn.c_str());
  if (!*m_ifs || m_ifs->bad()) {
    close();
    return false;
  }
  m_fn = fn;
  return true;
}

void simple_file_parser::close()
{
  if (m_ifs) {
    m_lineno = 0;
    delete m_ifs;
    m_ifs = 0;
    m_tokens.clear();
    m_fn = "";
  }
}

bool simple_file_parser::eof() const
{
  return m_ifs ? m_ifs->eof() : true;
}

void simple_file_parser::rewind()
{
  if (m_ifs) {
    m_ifs->seekg(0);
    m_lineno = 0;
    m_tokens.clear();
  }
}

void simple_file_parser::set_single_char_tokens(const char* tokens)
{
  size_t sz = sizeof(m_single_char_tokens)/sizeof(bool);
  for(size_t i=0;i<sz;++i)
    m_single_char_tokens[i] = false;
  if (tokens) {
    for(const char* p = tokens;*p;++p) {
      size_t i = size_t(*p);
      m_single_char_tokens[i] = true;
    }
  }
}

const char* simple_file_parser::get_single_char_tokens() const
{
  const size_t cache_size = sizeof(m_single_char_tokens)/sizeof(bool);
  static char cache[cache_size];
  char* p = cache;
  for(size_t i=0;i<cache_size;++i) {
    if (m_single_char_tokens[i]) {
      char ch = char(i);
      *p++ = ch;
    }
  }
  *p = 0;
  return cache;
}

char* simple_file_parser::skip_to_end_of_comment(char* p)
{
  for(;*p;++p) {
    if (*p == '*' && p[1] == '/') {
      p++;
      p++; // the first char after the '*/'
      m_in_comment=false;
      break;
    }
  }
  return p;
}

char* simple_file_parser::skip_comments(char* p)
{
  if (!*p) {
    return p;
  }
  
  if (m_in_comment) {
    p = skip_to_end_of_comment(p);
    if (m_in_comment) {
      // A multi-line comment that did not end on this line.
      return p;
    }
  }
  
  if (*p != '/') {
    // This can't possibly be a comment.
    return p;
  }
  
  if (p[1] == '/') {
    // C++ style -- everything to the EOL can be ignored.
    for(;*p;++p);
  }
  else if (p[1] == '*') {
    // C style -- /* .. */ everything until the next EOL can be
    //            ignored.
    p++;
    p++;
    m_in_comment=true;
    p = skip_to_end_of_comment(p);
  }
  return p;
}

char* simple_file_parser::skip_whitespace(char* p)
{
  if (!*p)
    return p;
  for(;*p!=0 && (*p<=' ' || *p>=127);++p);
  p = skip_comments(p);
  if (*p && (*p<=' ' || *p>=127)) {
    p = skip_whitespace(p); // handles nested comments
  }
  return p;
}

char* simple_file_parser::skip_to_whitespace(char* p)
{
  for(;*p!=0 && *p>' ' && *p<127;++p);
  return p;
}

char* simple_file_parser::is_continuation_line(char* line) const
{
  char* p = line;
  for(;*p;++p); // skip to the EOL
  if ((p-line)<2) {
    return 0; // 0 or 1 characters, cannot be a continuation line
  }
  if (p[-2] <= ' ' && p[-1] == '\\') {
    return p;
  }
  return 0; // not a continuation line
}

void simple_file_parser::add_token(char* token)
{
  //MYDEBUG("add_token \""<<token<<"\"");
  char* cur = token;
  char* beg = token;
  for(;*cur;++cur) {
    size_t i = size_t(*cur);
    if (m_single_char_tokens[i]) {
      char single_char_token[2] = {*cur,0};
      single_char_token[0] = *cur;
      *cur = 0;
      //MYDEBUG("found single character token '" << single_char_token << "' in \"" << token << "\"");
      //MYDEBUG("preceding token \"" << beg << "\"");
      if (*beg) // handle the case of a single character token
        m_tokens.push_back(beg);
      m_tokens.push_back(single_char_token);
      beg = cur+1;
      *cur = single_char_token[0];
    }
  }
  if (!*cur && *beg) {
    //MYDEBUG("adding token \""<<beg<<"\"");
    m_tokens.push_back(beg);
  }
}

size_t simple_file_parser::get_next_line()
{
  // Parsing happens here.
  m_buf[0] = 0;
  m_tokens.clear();
  char *p = m_buf;
  while (!m_ifs->eof() && m_ifs->getline(m_buf,m_bufsz)) {
    m_lineno++;

    // Check to see whether there is a backslash with preceding
    // whitespace at the end of the line, if so it is a continuation
    // line so data from the next line must be concatenated before
    // processing the tokens. All of the data is stored in m_buf.
    p = m_buf;
    char* eol = is_continuation_line(p);
    size_t sz = (size_t)m_ifs->gcount();
    while (eol && !m_ifs->eof()) {
      // Note that the starting point and the sz need to be adjusted
      // because we ignore the actual backslash character.
      --sz;
      p = eol-1; // ignore the backslash character
      if (sz>=m_bufsz) {
        break;
      }
      m_ifs->getline(p,m_bufsz-sz);
      m_lineno++;
      sz += (size_t)m_ifs->gcount();
      eol = is_continuation_line(p);
    }
    if (sz>=m_bufsz) {
      // We have a serious problem here because the buffer is not
      // large enough to contain the tokens for this record. There
      // are two choices, throw an exception or generate a fatal
      // error. For now I will generate a fatal error because these
      // should be an incredibly unlikely event.
      cerr << endl
           << "ERROR:" << __FILE__ << ":" << __LINE__ << ": "
           << "Internal buffer overflow, maximum of characters per line is "
           << (m_bufsz-1) << ", found " << sz
           << " at line " << m_lineno << " in " << get_file_name()
           << endl;
      exit(1);
    }

    // Process the line.
    // This loop skips w/s only lines.
    p = skip_whitespace(m_buf);
    if (*p) {
      while (*p) {
        char* end = p;
        if (*end=='"') {
          // Special handling for double quoted entities.
          // Nested quote characters are not allowed.
          // Use the single quote if you want to nest double quotes.
          // Legal examples:
          //    1 "A"
          //    2 "A B"
          //    3 "A B's"
          //    4 "A // embedded comment is not a comment"
          //    5 "A /* xx */ C"
          //
          // Illegal examples:
          //    1 "A \"B\""
          p++; // skip the initial "
          for(end++;*end!=0 && *end!='"' ;++end);
        }
        else if (false && *p=='\'') {
          // Special handling for single quoted entities.
          // Nested quote characters are not allowed.
          // Use the single quote if you want to nest double quotes.
          // Legal examples:
          //    1 'A'
          //    2 'A B'
          //    3 'A "B"'
          //    4 'A // embedded comment is not a comment'
          //    5 'A /* xx */ C'
          //
          // Illegal examples:
          //    1 'A \'B\''
          p++; // skip the initial '
          for(end++;*end!=0 && *end!='\'' ;++end);
        }
        else {
          // Any characters but w/s and comments.
          end = skip_to_whitespace(p);
        }
        char ec = *end;
        *end = 0;
        add_token(p);
        if (ec) {
          p = skip_whitespace(end+1);
        }
        else {
          *p = 0;
        }
      }
      break;
    }
  }
  return get_num_tokens();
}

bool simple_file_parser::get_next_line(size_t num_tokens,
                                       const string& tok1,
                                       const string& tok2,
                                       const string& tok3,
                                       const string& tok4,
                                       const string& tok5,
                                       const string& tok6,
                                       const string& tok7,
                                       const string& tok8)
{
  size_t n = get_next_line();
  if (num_tokens>0 && num_tokens!=n)
    return false;
  if (num_tokens>=1 && tok1!="" && tok1!=m_tokens[0])
    return false;
  if (num_tokens>=2 && tok2!="" && tok2!=m_tokens[1])
    return false;
  if (num_tokens>=3 && tok3!="" && tok3!=m_tokens[2])
    return false;
  if (num_tokens>=4 && tok4!="" && tok4!=m_tokens[3])
    return false;
  if (num_tokens>=5 && tok5!="" && tok5!=m_tokens[4])
    return false;
  if (num_tokens>=6 && tok6!="" && tok6!=m_tokens[5])
    return false;
  if (num_tokens>=7 && tok7!="" && tok7!=m_tokens[6])
    return false;
  if (num_tokens>=8 && tok8!="" && tok8!=m_tokens[7])
    return false;
  return true;
}

bool simple_file_parser::is_token_uint(size_t i) const
{
  const char* p = m_tokens[i].c_str();
  for(;*p;p++) {
    if (*p<'0' || *p>'9')
      return false;
  }

  return true;
}

bool simple_file_parser::is_token_int(size_t i) const
{
  const char* p = m_tokens[i].c_str();

  // leading '-' is okay
  if (*p!='-' && (*p<'0' || *p>'9'))
    return false;

  for(++p;*p;p++) {
    if (*p<'0' || *p>'9')
      return false;
  }

  return true;
}
