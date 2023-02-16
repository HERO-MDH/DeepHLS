/*
Copyright 2022 Mohammad Riazati

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef UTILS_H_
#define UTILS_H_
#include <string>
#include <vector>

using std::string;
using std::vector;

#define ERRORLOG ErrorLog(__FILE__, __LINE__)
#define ERRORLOGT(a) ErrorLog(__FILE__, __LINE__, a)
#define INFOLOG(a) InfoLog(__FILE__, __LINE__, a)
#define ASSERT(a) {if (!(a)) ERRORLOG;}
#define ASSERTA ERRORLOG
#define ASSERTT(a, t) {if (!(a)) ERRORLOGT(t);}

int ErrorLog(string File, int Line, string text = "", string log_type = "ERROR");
int InfoLog(string File, int Line, string text = "");
	
bool IsSkipChar(const char * SkipCharList, char pChar);
bool IsNumber(const std::string& s);
string StringSubstituteAll(string source, string find, string replace);
void StringToFile(string pStr, string pFilePath);
int OccurancesInString(const string &pString, const char& pChar);

bool FileExists(const string& pFileName);
bool DirectoryExists(const string& pFileName);
string CurrentDirectory();
string GetWorkspaceDir();
string FindFileFullPath(string pFileName);
vector<string> SplitString(string pString, string pDelimeters);
string GetCurrentDateTime();

#endif //UTILS_H_