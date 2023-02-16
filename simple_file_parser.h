// ================================================================
// $Id: simple_file_parser.h,v 1.3 2010/08/02 16:59:30 jdl Exp jdl $
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
#ifndef include_simple_file_parser_h
#define include_simple_file_parser_h
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>

/**
 * A very simple line oriented parser that tokenizes the data
 * on each line.
 *
 * It is convenient for handling very simple data file formats.
 *
 * It ignores comments and blank lines automatically.
 *
 * It supports C++-style single line commands as well as C-style
 * multi-line comments. Nested C-style comments are NOT supported.
 * Each line must be terminated by a semicolon. The semicolon at
 * the end of the line is treated as a token.
 *
 * Note that the semantics of the tokens is determmined by the 
 * user.
 *
 * Here is a an example of an input file:
 * <pre>
 *   // Sample file
 *   token1 token2 token3;
 *   token4 token5 ;
 *
 *   "this is one token" 'this is "another"'
 *
 *   // multi-line example
 *   tokena tokenb \
 *      tokenc \
 *      tokend
 * </pre>
 *
 * Note that double quotes cannot be embedded in a doubly quoted string and
 * single quotes cannot be embedded in a singly quoted string.
 *
 * Here is an example of how to use it in your code:
 * \code
 * #include "simple_file_parser.h"
 * #include <iostream>
 * #include <iomanip>
 * #include <map>
 * #include <string>
 * using namespace std;
 *
 * #define MYERR cerr << endl << "ERROR:" << __FILE__ << ":" << __LINE__ << ": "
 *
 * // Parse my data file with this format
 * // [var] = [integer value] ;
 * void parse(const string& file_name,map<string,unsigned>& vars)
 * {
 *    simple_file_parser sfp;
 *    if (!sfp.open(file_name)) {
 *       MYERR << "cannot read file: '" << file_name << "'" << endl;
 *       exit(1);
 *    }
 *
 *    // Enforce the syntax.
 *    //  [var] = [uint val] ;
 *    //  ^     ^ ^          ^
 *    //  |     | |          +--- end of record
 *    //  |     | +-------------- value
 *    //  |     +---------------- equals sign
 *    //  +---------------------- var name
 *    while (!sfp.eof()) {
 *       if (!sfp.get_next_line(4,"","=","",";")) {
 *          MYERR << "syntax error -- expected 4 tokens"
 *                << " at line " << sfp.get_line_number()
 *                << " in " << sfp.get_file_name()
 *                << endl;
 *       }
 *       if (!sfp.is_token_uint(2)) {
 *          MYERR << "syntax error -- expected an unsigned integer"
 *                << " but found '" << sfp.get_token(2) << "'"
 *                << " at line " << sfp.get_line_number()
 *                << " in " << sfp.get_file_name()
 *                << endl;
 *       }
 *       string var = sfp.get_token(0);
 *       unsigned int val = sfp.get_token_uint(2);
 *       vars[var] = val; // store it in the collection
 *    }
 * }
 * \endcode
 * Here is what a general parsing (tokenizing) routine would look like:
 * \code
 * #include "simple_file_parser.h"
 * #include <iostream>
 * #include <iomanip>
 * using namespace std;
 *
 * #define MYERR cerr << endl << "ERROR:" << __FILE__ << ":" << __LINE__ << ": "
 *
 * void parse(const string& file_name)
 * {
 *    simple_file_parser sfp;
 *    if (!sfp.open(file_name)) {
 *       MYERR << "cannot read file: '" << file_name << "'" << endl;
 *       exit(1);
 *    }
 *
 *    // Read each line. Comments and blank lines are ignored.
 *    while (sfp.get_next_line()) {
 *       // Print out the tokens for each line.
 *       cout << "Line " << setw(4) << right << sfp.get_line_number() << left << " ";
 *       for(size_t i=0;i<sfp.get_num_tokens();i++) {
 *          if (i) { // print a blank prefix
 *             cout << "          ";
 *          }
 *          cout << "token[" 
 *               << setw(2) << setfill('0') << right << i
 *               << left << setfill(' ') << "] = " 
 *               << "'" << sfp.get_token(i) << "'"
 *               << endl;
 *       }
 *    }
 * }
 * \endcode
 */
class simple_file_parser
{
public:
  /**
   * Container for the tokens.
   */
  typedef std::vector<std::string> tokens_t;
private:
  /**
   * Private copy constructor.
   * Not copyable.
   */
  simple_file_parser(const simple_file_parser& obj);
   
  /**
   * Private assignment operator.
   * Not assignable.
   */
  simple_file_parser& operator=(const simple_file_parser& obj);
public:
  /**
   * Default constructor.
   */
  simple_file_parser();

  /**
   * Destructor.
   */
  ~simple_file_parser();
public:
  /**
   * Does the file exist?
   * @param fn The file name to check (const char*).
   * @returns True if it does or false otherwise.
   */
  static bool file_exists(const char* fn);

  /**
   * Does the file exist (const string parameter)?
   * @param fn The file name to check (const string&).
   * @returns True if it does or false otherwise.
   */
  static bool file_exists(const std::string& fn);
public:
  /**
   * Begin the parsing operation by opening the file and
   * setting up the internal state.
   * @param fn The file to parse.
   * @returns true if the file was opened and is ready to parse or
   *               false otherwise.
   */
  bool open(const std::string& fn);

  /**
   * End the parsing operation.
   * This method can be called multiple times.
   */
  void close();

  /**
   * Get the line number.
   * @return the current line number.
   */
  inline unsigned get_line_number() const {return m_lineno;}

  /**
   * Get the file name.
   * @return the file name.
   */
  inline const std::string& get_file_name() const {return m_fn;}

  /**
   * Are we at the end of the file?
   * @returns True if we are the eof() or false otherwise.
   */
  bool eof() const;

  /**
   * Rewind to the beginning of the file.
   */
  void rewind();
public:
  /**
   * Get tokens from the next line.
   * Multi-line records are coalsced into a single record and comments are stripped out.
   * Blank lines are ignored.
   * @returns The number of tokens.
   */
  size_t get_next_line();

  /**
   * Get tokens from next line with expected values.
   * Multi-line records are coalsced into a single record and comments are stripped out.
   * Blank lines are ignored.
   * If the expected values are not found, false is returned.
   * If a token is "", it is not tested.
   *
   * This function is extremely useful for looking for specific keywords
   * in the data file.
   *
   * The maximum of eight tokens was completely arbitrary.
   * @param num_tokens  The expected number of tokens.
   * @param tok1        Expected value of the first token.
   *                    It is blank if there is no expected value.
   * @param tok2        Expected value of the second token.
   *                    It is blank if there is no expected value.
   * @param tok3        Expected value of the third token.
   *                    It is blank if there is no expected value.
   * @param tok4        Expected value of the fourth token.
   *                    It is blank if there is no expected value.
   * @param tok5        Expected value of the fifth token.
   *                    It is blank if there is no expected value.
   * @param tok6        Expected value of the sixth token.
   *                    It is blank if there is no expected value.
   * @param tok7        Expected value of the seventh token.
   *                    It is blank if there is no expected value.
   * @param tok8        Expected value of the eighth token.
   *                    It is blank if there is no expected value.
   * @returns True if all of the conditions passed.
   */
  bool get_next_line(size_t num_tokens,
                     const std::string& tok1 = "",
                     const std::string& tok2 = "",
                     const std::string& tok3 = "",
                     const std::string& tok4 = "",
                     const std::string& tok5 = "",
                     const std::string& tok6 = "",
                     const std::string& tok7 = "",
                     const std::string& tok8 = "");
   
  /**
   * Get the number of tokens just parsed.
   * @returns The number of parsed tokens.
   */
  size_t get_num_tokens() const {return m_tokens.size();}

  /**
   * Get the i-th token.
   * @param i The token index in the range [0..(N-1)] where 
   *          N == get_num_tokens(). If i is out of range,
   *          an exception is thrown from the underlying
   *          std::vector.
   * @returns The i-th token.
   */
  const std::string& get_token(size_t i) const {return m_tokens[i];}

  /**
   * Get the i-th token as an unsigned integer.
   * @param i The token index in the range [0..(N-1)] where 
   *          N == get_num_tokens(). If i is out of range,
   *          an exception is thrown from the underlying
   *          std::vector.
   * @returns The i-th token as an unsigned integer.
   */
  unsigned get_token_uint(size_t i) const {return atoi(m_tokens[i].c_str());}

  /**
   * Is the i-th token an unsigned integer?
   * @param i The token index in the range [0..(N-1)] where 
   *          N == get_num_tokens(). If i is out of range,
   *          an exception is thrown from the underlying
   *          std::vector.
   * @returns true if it is or false otherwise.
   */
  bool is_token_uint(size_t i) const;

  /**
   * Get the i-th token as a signed integer.
   * @param i The token index in the range [0..(N-1)] where 
   *          N == get_num_tokens(). If i is out of range,
   *          an exception is thrown from the underlying
   *          std::vector.
   * @returns The i-th token as a signed integer.
   */
  unsigned get_token_int(size_t i) const {return atoi(m_tokens[i].c_str());}

  /**
   * Is the i-th token an integer?
   * @param i The token index in the range [0..(N-1)] where 
   *          N == get_num_tokens(). If i is out of range,
   *          an exception is thrown from the underlying
   *          std::vector.
   * @returns true if it is or false otherwise.
   */
  bool is_token_int(size_t i) const;
public:
  /**
   * Define the single character tokens.
   * This is used to define single character tokens.
   * Here is an example usage:
   * \code
   *    // Make ; and . separate tokens.
   *    sfp.set_single_char_tokens(";.");
   * \endcode
   * @param tokens A string the contains the list of single character tokens.
   */
  void set_single_char_tokens(const char* tokens);

  /**
   * Get the single character tokens.
   * @returns the single character tokens in a string. If there are no
   * tokens, an empty string it returned.
   */
  const char* get_single_char_tokens() const;
private:
  /**
   * Skip to the end of a C-style comment.
   * @param p Pointer to the character just past the comment.
   * @returns the updated pointer.
   */
  char* skip_to_end_of_comment(char* p);

  /**
   * Remove comments from the string.
   * @param p Pointer to the token.
   * @returns the updated pointer.
   */
  char* skip_comments(char* p);

  /**
   * Skip white space.
   * @param p Pointer to the string.
   * @returns The pointer to the next token.
   */
  char* skip_whitespace(char* p);

  /**
   * Skip to white space.
   * @param p Pointer to the string.
   * @returns The pointer to the next white space.
   */
  char* skip_to_whitespace(char* p);

  /**
   * Does this line continue?
   * A continuation line has a backslash preceded by whitespace as the
   * last character.
   * @returns The end of the line if it is a continuation line or NULL
   *          if it is not.
   */
  char* is_continuation_line(char* line) const;

  /**
   * Add a token.
   * This is where the processing takes place to determine whether
   * there are embedded single character tokens.
   * @param token The token to add. It may contain embedded single
   *              character tokens.
   */
  void add_token(char* token);
private:
  std::string    m_fn;
  std::ifstream* m_ifs;
  tokens_t       m_tokens;
  unsigned       m_lineno;
  char*          m_buf;
  unsigned       m_bufsz;
  unsigned       m_v;
  bool           m_single_char_tokens[256];
  bool           m_in_comment;
};

#endif
