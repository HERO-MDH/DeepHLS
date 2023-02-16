/*
Copyright 2022 Mohammad Riazati

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include "utils.h"
#include <iostream>
#include <fstream>
#include <filesystem> //since c++ 17 //Project Properties - C / C++ - Language - C++ Language Standard - ISO C++17 Standard(/ std:c++17)
#include <algorithm>
#ifndef __linux__
#include <Windows.h> //To add colors to display messages
#endif

//using namespace std;

using std::cout;
using std::endl;
using std::ofstream;

int ErrorLog(string File, int Line, string text, string log_type) {
	ofstream f_ErrorLog;
	extern string log_location;

	string error_text = GetCurrentDateTime() + " ";
	error_text += log_type + ": ";
	error_text += File + "(" + std::to_string(Line) + ")";
	error_text += ": " + text;

	if (log_type == "ERROR") {
    #ifndef __linux__
		HANDLE m_hConsole;
		WORD m_currentConsoleAttr = 0;
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		m_hConsole=GetStdHandle(STD_OUTPUT_HANDLE);
		if(GetConsoleScreenBufferInfo(m_hConsole, &csbi)) m_currentConsoleAttr = csbi.wAttributes;

		SetConsoleTextAttribute (m_hConsole, FOREGROUND_RED);
		cout << error_text << endl;
		SetConsoleTextAttribute (m_hConsole, m_currentConsoleAttr);
    #else
		cout << error_text << endl;
    #endif
	}
	else cout << error_text << endl;

	f_ErrorLog.open(log_location, std::ios::app);
	f_ErrorLog << error_text << endl;
	f_ErrorLog.close();

	return 0;
}

int InfoLog(string File, int Line, string text) {
	return ErrorLog(File, Line, text, "INFO");
}

bool IsSkipChar(const char * SkipCharList,char pChar) {
	for(const char* p = SkipCharList; *p; ++p) 
	{
		if (*p == pChar) return true;
	}
	return false;
}

bool IsNumber(const std::string& s) {
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

string StringSubstituteAll(string source, string find, string replace) {
	size_t index = 0;
	string result = source;
	while (true) 
	{
		/* Locate the substring to replace. */
		index = result.find(find, index); //second parameter is offset
		if (index == string::npos) break;

		/* Make the replacement. */
		result.replace(index, find.length(), replace);

		/* Advance index forward so the next iteration doesn't pick it up as well. */
		index += replace.length();
	}		
	return result;
}

bool FileExists(const string& pFileName) {
	return std::filesystem::exists(pFileName);
}

bool DirectoryExists(const string& pFileName) {
	return FileExists(pFileName);
}

string CurrentDirectory() {
	extern bool running_in_vs_environment;
  string directory_delimiter;
  #ifdef __linux__
  directory_delimiter = "/";
  #else
  directory_delimiter = "\\";
  #endif
	string result = std::filesystem::current_path().string() + directory_delimiter;
	if (running_in_vs_environment) {
		extern string workspace_dir;
		result = workspace_dir;
	}
  else {
    extern string workspace_dir;
    workspace_dir = result + "WorkspaceDir" + directory_delimiter;
    if (!std::filesystem::exists(workspace_dir)) {
      std::filesystem::create_directories(workspace_dir);
    }

    extern string output_dir;
    output_dir = workspace_dir;
    extern string input_file_name;
    input_file_name = workspace_dir + "output_arch.py";
  }

	return result;
}

string GetWorkspaceDir() {
  extern string workspace_dir;
	static string workspace_dir_st;
	if (workspace_dir_st == "") workspace_dir_st = workspace_dir;
  return workspace_dir_st;
}

//if the file address is not absolute, finds the file from current directory or the workspace directory
string FindFileFullPath(string pFileName) {
	extern string workspace_dir;
	string current_directory = CurrentDirectory();
  #ifndef __linux__
  string directory_delimiter = "\\";
  #else
  string directory_delimiter = "/";
  #endif
	if (pFileName.find(directory_delimiter) != string::npos) return pFileName;
	else if (FileExists(current_directory + pFileName)) return current_directory + pFileName;
	else return workspace_dir + pFileName;
}

void StringToFile(string pStr, string pFilePath){
	string save_address;
	save_address = pFilePath;

	ofstream f_stream;
	f_stream.open(save_address, std::ios::out);
	f_stream << pStr;
	f_stream.close();
}

vector<string> SplitString(string pString, string pDelimeters) {
	vector<string> result;

	size_t last_loc = 0;
	size_t loc = pString.find_first_of(pDelimeters);
	while (loc != string::npos)	{
		result.push_back(pString.substr(last_loc, loc - last_loc));
		last_loc = loc + 1;
		loc = pString.find_first_of(pDelimeters, last_loc);
	}
	result.push_back(pString.substr(last_loc));

	return result;
}

int OccurancesInString(const string& pString, const char& pChar) {
	size_t loc = 0;
	int result = 0;

	while ((loc = pString.find(pChar, loc)) != string::npos) {
		++result;
		++loc;
	}

	return result;
}

string GetCurrentDateTime() {
	//https://www.cplusplus.com/reference/ctime/strftime/

  #ifndef __linux__
	time_t curr_time;
	char result[100];
  struct tm buf;

	time(&curr_time);
	localtime_s(&buf, &curr_time);
	
	strftime(result, sizeof result, "%Y-%m-%d %H:%M:%S", &buf);

	return result;

  #else

  time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];

  time (&rawtime);
  timeinfo = localtime (&rawtime);

	strftime(buffer, sizeof buffer, "%Y-%m-%d %H:%M:%S", timeinfo);
  
  return buffer;

  #endif
}