/*
Copyright 2022 Mohammad Riazati

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

/*
Tab size for this code to be more readable: 2

//Visual Studio: Setting C++ 17 as the language. (needed for <filesystem>).
//	Project - Properties (Alt + F7) - C/C++ - Language - C++ Language Standard - ISO C++17 Standard(/ std:c++17)

//exe file name: DeepHLS.exe
//Visual Studio: Project - Properties (Alt + F7) - Debugging - Command line arguments

Example runs:
deephls -options-json-file vgg.json
deephls -options-json-file lenet.json
deephls -keras-source keras-text -keras-source-text "model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))"
deephls -keras-source-text "model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))"

deephls
		* keras source: keras-source.py in the current directory
		* other settings as default

deephls -add-main-function -store-analysis-data
		* store-analysis-data will be effective only if add-main-function is active

deephls -loop-orders oz-oy-ox-iz-kx-ky#*#*#*#*#*#*#*#*#*#*#*#*#*#*
deephls -loop-orders *#*#*#*#*#*#oz-oy-iz-ox-kx-ky#*#*#*#*#*#*#*#* -single-layer 7
*/

//VS_ON_VM: Windows VM, VSC_ON_VM: Linux VM
#define VS_ON_VM
//#define VSC_ON_VM


#include <iostream>
#include <fstream>
#ifndef __linux__
#include <conio.h>
#endif
#include <iomanip>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "simple_file_parser.h"
#include "utils.h"
#include "enums.h"
#include "json.hpp"

#if defined(VS_ON_VM)
string workspace_dir = "C:\\DeepHLSWorkspace\\";
string output_dir = workspace_dir; //"C:\\Projects\\Temp\\Cfiles\\"; //Later overwritten in ReadOptions
string input_file_name = workspace_dir + "output_arch.py"; //"C:\\Projects\\Temp\\KerasInputs\\output_arch.py";
string log_location;
/* set PATH=%PATH%;C:\Projects\DeepHLS\Debug\ */

#elif defined(VSC_ON_VM)
string workspace_dir;
string output_dir;
string input_file_name; // = "...\\Temp\\KerasInputs\\output_arch.py"; //The result of KerasPreprocessings (Python) must be copied here //From: ...\Python\KerasPreprocessings\KerasPreprocessings\outputs
string log_location;
#else
string output_dir = "...\\Temp\\Cfiles\\";
string input_file_name = "...\\Temp\\KerasInputs\\output_arch.py"; //The result of KerasPreprocessings (Python) must be copied here //From: ...\Python\KerasPreprocessings\KerasPreprocessings\outputs
#endif // VS_ON_VM

using namespace std;
using json = nlohmann::json;

//#pragma warning(disable : 26812) //Warning C26812 : Prefer 'enum class' over 'enum' (Enum.3)

void ReadOptions(int argc = 0, char* argv[] = {});
void SetArbitraryParameters();
void CheckAndCorrectLoopOrders();
void DumpLayers(bool ExportFile = false);
bool ParseKerasFile(string SourceFile);
void CompleteLayersInfo();
void GenerateCFiles();
void GenerateHFileDataTypes();
void GenerateHFileParamList();
void GenerateDeepClConfigH();

string Tabs(int count);
void AddToCFile_Text(ofstream &f_stream, string text, int indent = 0);
void AddToCFile_EmptyLine(ofstream &f_stream, int count = 1);

void AddToCFile_Conv2dLayer(ofstream &f_stream, int layer_number);
void AddToCFile_Pooling2dLayer(ofstream &f_stream, int layer_number);
void AddToCFile_FlattenLayer(ofstream &f_stream, int layer_number);
void AddToCFile_DenseLayer(ofstream &f_stream, int layer_number);
void AddToCFile_ActivationFunction(ofstream& f_stream, ActivationFunctions function);
void AddToCFile_QMinMax(ofstream& f_stream);

void AddToCFile_MainAndPredict(ofstream &f_stream);

string LayerDataLocation(int pLayerNumber);
void SetNetworkGuess();

class Layer {
public:
	LayerTypes layer_type;
	PoolingTypes pooling_type = NO_POOLING;
	PaddingTypes padding_type = NO_PADDING;
	ActivationFunctions activation_function = NO_ACTIVATION;
	int filters_count = 0;
	int kernel_size_rows = 0;
	int kernel_size_cols = 0;
	int input_size_x = 0, input_size_y = 0, input_size_z = 0;
	int output_size_x = 0, output_size_y = 0, output_size_z = 0;
	int node_count = 0;
	int stride_size = 0;
};

vector<Layer*> Layers;
map<string, string> map_options;
string network_name, network_guess;
vector<string> loop_orders;
bool biases_enabled;
bool store_alanysis_data;
bool add_main_function;
bool data_type_mode_floating_point, data_type_mode_fixed_point_single, data_type_mode_fixed_point_multi;
string data_type_mode_detail;
bool numbered_loop_labels;
bool running_in_vs_environment;
int single_layer; //1 based, 0: disabled
bool dump_layers;
bool fault_simulation;
bool approximate_multipliers;
string approximate_multipliers_configuration;
string approximate_multipliers_type;
bool create_deepcl_config_h;

int main(int argc, char* argv[]) {
	ReadOptions(argc, argv);

	if (!ParseKerasFile(input_file_name)) {
		ERRORLOGT("ParseKerasFile failed. Exiting the program.");
		exit(-1);
	}
	SetNetworkGuess();
	SetArbitraryParameters();
	CheckAndCorrectLoopOrders();

	DumpLayers();
	CompleteLayersInfo();
	if (dump_layers) cout << endl << endl << "After CompleteLayersInfo: " << endl;
	DumpLayers(true);

	if (!biases_enabled) cout << endl << endl << endl << "****** ATTENTION: Network biases are disabled. ******* " << endl;

	GenerateHFileDataTypes();
	GenerateHFileParamList();
	GenerateCFiles();
	if (create_deepcl_config_h) GenerateDeepClConfigH();

	if (running_in_vs_environment) {
    #ifndef __linux__
		_getch();
    #endif
	}
}

void DumpLayers(bool ExportFile) {
	if (!dump_layers) return;

	ofstream fstr;
	string f_location = output_dir + "DumpLayers.txt";
	char d = '\t';

	cout << "type" << d << d << "padding" << d << "filters" << d << "k(rows)" << d << "k(cols)" << d << "activ" << d << "nodes" << d << "stride" << d << "in_x" << d << "in_y" << d << "in_z" << d << "out_x" << d << "out_y" << d << "out_z" << endl;
	if(ExportFile) {
		fstr.open(f_location, (ios::out));
		fstr << "type" << d << d << "padding" << d << "filters" << d << "k(rows)" << d << "k(cols)" << d << "activ" << d << "nodes" << d << "stride" << d << "in_x" << d << "in_y" << d << "in_z" << d << "out_x" << d << "out_y" << d << "out_z" << endl;
		fstr << endl;
	}

	Layer* current_layer;
	for(size_t i = 0; i < Layers.size(); i++) {
		current_layer = Layers[i];

		if (dump_layers) {
			string layer_type = LayerTypesToString(current_layer->layer_type);
			if (current_layer->layer_type == POOLING2D) {layer_type += "(" + PoolingTypesToString(current_layer->pooling_type) + ")";}
			cout << layer_type;  cout << d; {if (layer_type.size() < 8) cout << d;}
			if (current_layer->padding_type != NO_PADDING) {cout << PaddingTypesToString(current_layer->padding_type);} cout << d;
			if (current_layer->filters_count != 0) {cout << current_layer->filters_count;} cout << d;
			if (current_layer->kernel_size_rows != 0 && current_layer->kernel_size_cols != 0) {cout << current_layer->kernel_size_rows << d << current_layer->kernel_size_cols;} cout << d;
			if (current_layer->activation_function != NO_ACTIVATION) {cout << ActivationFunctionsToString(current_layer->activation_function);} cout << d;
			if (current_layer->node_count != 0) {cout << current_layer->node_count;} cout << d;
			if (current_layer->stride_size != 0) {cout << current_layer->stride_size;} cout << d;
			if (current_layer->input_size_x != 0 && current_layer->input_size_y != 0 && current_layer->input_size_z != 0) {cout << current_layer->input_size_x << d << current_layer->input_size_y << d << current_layer->input_size_z;} cout << d;
			if (current_layer->output_size_x != 0 && current_layer->output_size_y != 0 && current_layer->output_size_z != 0) {cout << current_layer->output_size_x << d << current_layer->output_size_y << d << current_layer->output_size_z;} cout << d;
			cout << endl;
		}

		if (ExportFile)	{
			fstr << LayerTypesToString(current_layer->layer_type); if (current_layer->layer_type == POOLING2D) {fstr << "(" << PoolingTypesToString(current_layer->pooling_type) << ")";} fstr << d;
			if (current_layer->padding_type != NO_PADDING) {fstr << PaddingTypesToString(current_layer->padding_type);} fstr << d;
			if (current_layer->filters_count != 0) {fstr << current_layer->filters_count;} fstr << d;
			if (current_layer->kernel_size_rows != 0 && current_layer->kernel_size_cols != 0) {fstr << current_layer->kernel_size_rows << d << current_layer->kernel_size_cols;} fstr << d;
			if (current_layer->activation_function != NO_ACTIVATION) {fstr << ActivationFunctionsToString(current_layer->activation_function);} fstr << d;
			if (current_layer->node_count != 0) {fstr << current_layer->node_count;} fstr << d;
			if (current_layer->stride_size != 0) {fstr << current_layer->stride_size;} fstr << d;
			if (current_layer->input_size_x != 0 && current_layer->input_size_y != 0 && current_layer->input_size_z != 0) {fstr << current_layer->input_size_x << d << current_layer->input_size_y << d << current_layer->input_size_z;} fstr << d;
			if (current_layer->output_size_x != 0 && current_layer->output_size_y != 0 && current_layer->output_size_z != 0) {fstr << current_layer->output_size_x << d << current_layer->output_size_y << d << current_layer->output_size_z;} fstr << d;
			fstr << endl;
		}
	}

	if(ExportFile) {
		fstr.close();
	}
}

bool ParseKerasFile(string SourceFile) {
	simple_file_parser sfp;
  if (!sfp.open(SourceFile)) {
    ERRORLOGT("Opening source file failed: " + SourceFile);
		return false;
	}
	const char *Delimiters = "'();+,.=#";

  sfp.set_single_char_tokens(Delimiters);

	string current_token;
	int current_valuable_token_index;
	size_t current_token_index;

	Layer *current_layer = NULL;
	int current_num_tokens;
	//int current_line_number;
  while (sfp.get_next_line()) {
		current_num_tokens = sfp.get_num_tokens();
		//current_line_number = sfp.get_line_number();
		if (current_num_tokens == 0) {
			ERRORLOG; //The parser is supposed to eliminate empty lines
			continue;
		}
		
		current_valuable_token_index = 0;
		current_token_index = 0;
    while(current_token_index < sfp.get_num_tokens()) {
			current_token = sfp.get_token(current_token_index);
			current_token_index++;

			if (current_token == "#") break;
			if (current_token.length() == 1 && IsSkipChar(Delimiters, current_token[0])) continue;
			if (current_valuable_token_index == 0 && current_token != "model") break; //only model.add lines are supported
			if (current_valuable_token_index == 1 && current_token != "add") break; //only model.add lines are supported

			if (current_token == "model") {
				ASSERT(current_valuable_token_index == 0);
				current_valuable_token_index++;
			}
			else if (current_token == "add") {
				ASSERT(current_valuable_token_index == 1);
				current_valuable_token_index++;
				current_layer = new Layer;
				Layers.push_back(current_layer);
			}
			else if (current_token == "layers") {
				ASSERT(current_valuable_token_index == 2);
				//current_valuable_token_index++;
			}
			else if (current_token == "Conv2D") {
				ASSERT(current_valuable_token_index == 2);
				current_valuable_token_index++;
				current_layer->layer_type = CONV2D;
			}
			else if (current_token == "MaxPool2D") {
				ASSERT(current_valuable_token_index == 2);
				current_valuable_token_index++;
				current_layer->layer_type = POOLING2D;
				current_layer->pooling_type = MAX_POOLING;
			}
			else if (current_token == "AveragePooling2D") {
				ASSERT(current_valuable_token_index == 2);
				current_valuable_token_index++;
				current_layer->layer_type = POOLING2D;
				current_layer->pooling_type = AVERAGE_POOLING;
			}
			else if (current_token == "Flatten") {
				ASSERT(current_valuable_token_index == 2);
				current_valuable_token_index++;
				current_layer->layer_type = FLATTEN;
			}
			else if (current_token == "Dense") {
				ASSERT(current_valuable_token_index == 2);
				current_valuable_token_index++;
				current_layer->layer_type = DENSE;
			}
			else if (current_token == "filters") {
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == CONV2D);

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->filters_count = atoi(current_token.c_str());
			}
			else if (current_token == "kernel_size") {
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == CONV2D);

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //"("
				current_token_index++;

				current_token = sfp.get_token(current_token_index);
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->kernel_size_rows = atoi(current_token.c_str());

				current_token = sfp.get_token(current_token_index); //","
				current_token_index++;

				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->kernel_size_cols = atoi(current_token.c_str());
			}
			else if (current_token == "activation") {
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == CONV2D || current_layer->layer_type == DENSE);

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //"'"
				current_token_index++;
				current_token = sfp.get_token(current_token_index);
				current_token_index++;
				current_valuable_token_index++;
				if (current_token == "relu") current_layer->activation_function = RELU;
				else if (current_token == "softmax") current_layer->activation_function = SOFTMAX;
				else if (current_token == "linear") current_layer->activation_function = LINEAR;
				else ERRORLOG;
			}
			else if (current_token == "units") {
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == DENSE);

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->node_count = atoi(current_token.c_str());
			}
			else if (current_token == "strides") {
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == POOLING2D || current_layer->layer_type == CONV2D);

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				if(IsNumber(current_token)) { //strides = 2
					current_layer->stride_size = atoi(current_token.c_str());
					ASSERT(current_layer->kernel_size_rows == 0);
					ASSERT(current_layer->kernel_size_cols == 0);
				}
				else { //strides = (2, 2)
					ASSERT(current_token == "(");
					current_token = sfp.get_token(current_token_index);
					current_token_index++;
					ASSERT(IsNumber(current_token));
					current_layer->stride_size = atoi(current_token.c_str());
					current_token = sfp.get_token(current_token_index); //","
					current_token_index++;
					current_token = sfp.get_token(current_token_index);
					current_token_index++;
					ASSERT(IsNumber(current_token));
					ASSERT(current_layer->stride_size == atoi(current_token.c_str()));
				}
			}
			else if (current_token == "padding") {
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == CONV2D);

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //"'"
				current_token_index++;
				current_token = sfp.get_token(current_token_index);
				current_token_index++;
				current_valuable_token_index++;
				if (current_token == "same") current_layer->padding_type = PADDING_SAME;
				else if (current_token == "valid") current_layer->padding_type = PADDING_VALID;
				else ERRORLOG;
			}
			else if (current_token == "input_shape") {
				current_valuable_token_index++;

				current_token = sfp.get_token(current_token_index); //"="
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //"("
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->input_size_x = atoi(current_token.c_str());

				current_token = sfp.get_token(current_token_index); //","
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->input_size_y= atoi(current_token.c_str());

				current_token = sfp.get_token(current_token_index); //","
				current_token_index++;
				current_token = sfp.get_token(current_token_index); //The value for the filter
				current_token_index++;
				current_valuable_token_index++;
				ASSERT(IsNumber(current_token));
				current_layer->input_size_z= atoi(current_token.c_str());
			}
			else if (IsNumber(current_token)) { 
				//to support model.add(Dense(120, activation='relu')), which is equivalent to model.add(layers.Dense(units=120, activation='relu'))
				current_valuable_token_index++;
				ASSERT(current_layer->layer_type == DENSE);
				ASSERT(current_layer->node_count == 0);
				current_layer->node_count = atoi(current_token.c_str());
			}
    }
  }

	return true;
}

void CompleteLayersInfo() {
	for(size_t i = 0; i < Layers.size(); i++)
	{
		if (Layers[i]->stride_size == 0) Layers[i]->stride_size = 1;
		if (Layers[i]->layer_type == CONV2D && Layers[i]->padding_type == NO_PADDING) Layers[i]->padding_type = PADDING_VALID;
		if (Layers[i]->kernel_size_rows == 0 && Layers[i]->kernel_size_cols == 0 && Layers[i]->stride_size != 0) Layers[i]->kernel_size_rows = Layers[i]->kernel_size_cols = Layers[i]->stride_size;

		if (Layers[i]->input_size_x == 0) {
			ASSERT(i > 0);
			ASSERT(Layers[i]->input_size_y == 0);
			ASSERT(Layers[i]->input_size_z == 0);
			
			Layers[i]->input_size_x = Layers[i-1]->output_size_x;
			Layers[i]->input_size_y = Layers[i-1]->output_size_y;
			Layers[i]->input_size_z = Layers[i-1]->output_size_z;
		}

		if (Layers[i]->filters_count == 0 && Layers[i]->layer_type == POOLING2D) {
			Layers[i]->filters_count = Layers[i-1]->output_size_z;
		}

		if (Layers[i]->layer_type == FLATTEN) {
			ASSERT(Layers[i+1]->layer_type == DENSE);

			ASSERT(Layers[i]->node_count == 0);

			if (i == 0) { 
				//The first layer is a FLATTEN layer (The case for an MLP where the real first layer is a multidimentional DENSE)
				//In this case the input size should be specified in the flatten layer
				ASSERT(Layers[i]->input_size_x != 0);
				ASSERT(Layers[i]->input_size_y != 0);
				ASSERT(Layers[i]->input_size_z != 0);
				
				Layers[i]->node_count = Layers[i]->input_size_x * Layers[i]->input_size_y * Layers[i]->input_size_z;;
			}
			else {
				ASSERT(Layers[i-1]->output_size_x != 0 && Layers[i-1]->output_size_y != 0 && Layers[i-1]->output_size_z != 0);
				Layers[i]->input_size_x = Layers[i-1]->output_size_x;
				Layers[i]->input_size_y = Layers[i-1]->output_size_y;
				Layers[i]->input_size_z = Layers[i-1]->output_size_z;

				Layers[i]->node_count = Layers[i-1]->output_size_x * Layers[i-1]->output_size_y * Layers[i-1]->output_size_z;;
			}
		}

		if (Layers[i]->node_count != 0) {
			ASSERT(Layers[i]->layer_type == DENSE || Layers[i]->layer_type == FLATTEN);
			Layers[i]->output_size_z = 1;
			Layers[i]->output_size_x = Layers[i]->node_count;
			Layers[i]->output_size_y = 1;
		}
		else if (Layers[i]->kernel_size_rows != 0) {
			ASSERT(Layers[i]->kernel_size_rows == Layers[i]->kernel_size_cols); //Currently, just square kernel is supported

			ASSERT(Layers[i]->filters_count != 0);
			Layers[i]->output_size_z = Layers[i]->filters_count;

			int LastFilteredBoxLen, StrideCount;
			if (Layers[i]->padding_type == PADDING_VALID || Layers[i]->layer_type == POOLING2D) {
				LastFilteredBoxLen = Layers[i]->input_size_x - Layers[i]->kernel_size_rows; 

				//ASSERT(LastFilteredBoxLen % Layers[i]->stride_size == 0); //otherwise floor(LastFilteredBoxLen / Layers[i]->stride_size) + 1;
				StrideCount = (LastFilteredBoxLen / Layers[i]->stride_size) + 1; //NOTE: Integer division to implemet the floor
				Layers[i]->output_size_x = StrideCount;
			
				LastFilteredBoxLen = Layers[i]->input_size_y - Layers[i]->kernel_size_cols; 
				StrideCount = (LastFilteredBoxLen / Layers[i]->stride_size) + 1;
				Layers[i]->output_size_y = StrideCount;
			}
			else {
				ASSERT(Layers[i]->padding_type == PADDING_SAME && Layers[i]->layer_type == CONV2D);
				ASSERT(Layers[i]->kernel_size_rows % 2 == 1);
				ASSERT(Layers[i]->kernel_size_cols % 2 == 1);

				int padding_size_x, padding_size_y;
				Layer* layer = Layers[i];

				padding_size_x = (layer->kernel_size_rows - 1) / 2;
				padding_size_y = (layer->kernel_size_cols - 1) / 2;

				if (Layers[i]->stride_size == 1)
				{
					Layers[i]->output_size_x = Layers[i]->input_size_x;
					Layers[i]->output_size_y = Layers[i]->input_size_y;
				}
				else
				{
					// Division returns int. So, even if stride is unaligned, it still works.
					Layers[i]->output_size_x = ((Layers[i]->input_size_x + 2 * padding_size_x - Layers[i]->kernel_size_rows) / Layers[i]->stride_size) + 1;
					Layers[i]->output_size_y = ((Layers[i]->input_size_y + 2 * padding_size_y - Layers[i]->kernel_size_cols) / Layers[i]->stride_size) + 1;
				}
			}
		}
		else ERRORLOG;
	}
}

void GenerateCFiles() {
	ofstream f_stream;
	string f_location = output_dir + "main.cpp";
	f_stream.open(f_location, (ios::out));
	int temp_int;
	string temp_string;
	int current_indent = 0;

	int LayersCount = Layers.size();
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");

	if (add_main_function) {
		AddToCFile_Text(f_stream, "//#define _HLS_RUN");
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN");
		//AddToCFile_Text(f_stream, "#include \"conio.h\"");
		AddToCFile_Text(f_stream, "#include \"stdlib.h\"");
		AddToCFile_Text(f_stream, "#include \"stdio.h\"");
		AddToCFile_Text(f_stream, "#include \"time.h\"");
		AddToCFile_Text(f_stream, "#include <typeinfo>");
		AddToCFile_Text(f_stream, "#include <thread>");
		AddToCFile_Text(f_stream, "#endif //_HLS_RUN");
		AddToCFile_EmptyLine(f_stream);
	}
	else {
		AddToCFile_Text(f_stream, "#define _HLS_RUN");
		AddToCFile_EmptyLine(f_stream);
	}


	AddToCFile_Text(f_stream, "#include \"param-list.h\"");
	if (add_main_function) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN");
		AddToCFile_Text(f_stream, "#include \"fixed-point-analysis.h\"");
		if (fault_simulation) AddToCFile_Text(f_stream, "#include \"fault_simulation.h\"");
		if (data_type_mode_fixed_point_single && data_type_mode_detail == "eight-bit-int") AddToCFile_Text(f_stream, "#include \"quantization.h\"");
		if (store_alanysis_data) {
			AddToCFile_Text(f_stream, "int RunCounter = 0;");
			AddToCFile_Text(f_stream, "#define STORE_DATA(p1, p2, p3, p4, p5, p6) {StoreData(p1, p2, p3, RunCounter < 1, p4, p5, p6);}");
			AddToCFile_Text(f_stream, "//#define STORE_DATA(p1, p2, p3, p4, p5, p6) {StoreData(p1, p2, p3, false, p4, p5, p6);}");
			AddToCFile_Text(f_stream, "//#define STORE_DATA(p1, p2, p3, p4, p5, p6)");
		}
		AddToCFile_Text(f_stream, "extern map<string, string> arguments;");
		AddToCFile_Text(f_stream, "#else");
		if (store_alanysis_data) AddToCFile_Text(f_stream, "#define STORE_DATA(p1, p2, p3, p4, p5, p6)");
		if (data_type_mode_fixed_point_single && data_type_mode_detail == "eight-bit-int") AddToCFile_Text(f_stream, "void quantize_layer_output(void *layer_data_base, void *layer_data_quantized, int size, int config = 0) {};");
		if (fault_simulation) AddToCFile_Text(f_stream, "void fault_injection(void *activation, int current_layer, int faulty_layer, int faulty_fmap, int faulty_bit) {};");
		AddToCFile_Text(f_stream, "#endif");
		AddToCFile_EmptyLine(f_stream);

		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN");
		AddToCFile_Text(f_stream, "#include \"datainterface.h\"");
		AddToCFile_Text(f_stream, "#include \"paraminterface.h\"");
		AddToCFile_Text(f_stream, "#ifndef __linux__");
		AddToCFile_Text(f_stream, "#pragma warning(disable: 4102) //To suppress unused label warnings");
		AddToCFile_Text(f_stream, "#endif //__linux__");
		AddToCFile_Text(f_stream, "#endif //_HLS_RUN");
	}
	AddToCFile_EmptyLine(f_stream, 2);

	vector<ActivationFunctions> list_of_funtions;
	for(size_t i = 0; i < Layers.size(); i++) {
		list_of_funtions.push_back(Layers[i]->activation_function);
	}
	sort(list_of_funtions.begin(), list_of_funtions.end());
	list_of_funtions.erase(unique(list_of_funtions.begin(), list_of_funtions.end() ), list_of_funtions.end() );

	for(size_t i = 0; i < list_of_funtions.size(); i++)	{
		AddToCFile_ActivationFunction(f_stream, list_of_funtions[i]);
	}
	AddToCFile_EmptyLine(f_stream);

	if (quantized) { AddToCFile_QMinMax(f_stream); AddToCFile_EmptyLine(f_stream);}

	if (approximate_multipliers) {
		AddToCFile_Text(f_stream, "#include \"multipliers.h\"");
		AddToCFile_EmptyLine(f_stream);

		AddToCFile_Text(f_stream, "//#define MULTIPLIER_BASE(a, b) (a*b)");
		AddToCFile_Text(f_stream, "#define MULTIPLIER_BASE mul_base");
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "#define MULTIPLIER_EXACT mul8s_1KV6");
		AddToCFile_EmptyLine(f_stream);

		AddToCFile_Text(f_stream, "//#define MULTIPLIER_NAME MULTIPLIER_EXACT");
		temp_string = ""; if (approximate_multipliers_type != "mul8s_1KVP") temp_string = "//"; AddToCFile_Text(f_stream, temp_string + "#define MULTIPLIER_NAME mul8s_1KVP");
		temp_string = ""; if (approximate_multipliers_type != "mul8s_1KV9") temp_string = "//"; AddToCFile_Text(f_stream, temp_string + "#define MULTIPLIER_NAME mul8s_1KV9");
		temp_string = ""; if (approximate_multipliers_type != "mul8s_1KV8") temp_string = "//"; AddToCFile_Text(f_stream, temp_string + "#define MULTIPLIER_NAME mul8s_1KV8");
		AddToCFile_EmptyLine(f_stream);

		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN");
		AddToCFile_Text(f_stream, R"(#define MUL_LAYER(a, b, layer_id) (mul_general<int16_t>(a, b, layer_id, arguments["mul-name"], arguments["mul-layers-config"])))");

		for(size_t i = 0; i < Layers.size(); i++) {
			if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) AddToCFile_Text(f_stream, "#define MUL_LAYER_" + to_string(i + 1) + "(a, b) MUL_LAYER(a, b, " + to_string(i + 1) + ")");
		}

		AddToCFile_Text(f_stream, "#else //_HLS_RUN");

		for(size_t i = 0; i < Layers.size(); i++) {
			if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) {
				temp_string = "";  if (approximate_multipliers_configuration[i] == '0') temp_string = "//";
				AddToCFile_Text(f_stream, temp_string + "#define MUL_LAYER_" + to_string(i + 1) + " MULTIPLIER_NAME");
			}
		}
		AddToCFile_EmptyLine(f_stream);

		for(size_t i = 0; i < Layers.size(); i++) {
			if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) {
				temp_string = "";  if (approximate_multipliers_configuration[i] == '1') temp_string = "//";
				AddToCFile_Text(f_stream, temp_string + "#define MUL_LAYER_" + to_string(i + 1) + " MULTIPLIER_EXACT");
			}
		}

		AddToCFile_Text(f_stream, "#endif //_HLS_RUN");
		AddToCFile_EmptyLine(f_stream);
	}

	int temp_input_size_x, temp_input_size_y, temp_input_size_z;
	if (single_layer == 0 || single_layer == 1 ||
			(single_layer >= 2 
						&& (Layers[single_layer]->layer_type == CONV2D || Layers[single_layer]->layer_type == POOLING2D || Layers[single_layer]->layer_type == FLATTEN)
			)
		 ){
		temp_int = 0;
		if (single_layer >= 2) temp_int = single_layer - 1;
		temp_input_size_x = Layers[temp_int]->input_size_x;
		temp_input_size_y = Layers[temp_int]->input_size_y;
		temp_input_size_z = Layers[temp_int]->input_size_z;
		ASSERT(temp_input_size_x > 0 && temp_input_size_y > 0 && temp_input_size_z > 0);

		AddToCFile_Text(f_stream, "typedef DataType_input InputType[" + to_string(temp_input_size_x) + "]"
																														+ "[" + to_string(temp_input_size_y) + "]"
																														+ "[" + to_string(temp_input_size_z) + "];");
	}
	else { //Single_layer >= 2 and DENSE
			temp_int = single_layer - 1;
			temp_input_size_x = Layers[temp_int]->input_size_x;
			ASSERT(temp_input_size_x > 0);

			AddToCFile_Text(f_stream, "typedef DataType_input InputType[" + to_string(temp_input_size_x) + "];");
	}

	int temp_output_size_x, temp_output_size_y, temp_output_size_z;
	if (single_layer == 0 || single_layer == LayersCount ||
			(single_layer >= 2 
						&& (Layers[single_layer]->layer_type == DENSE || Layers[single_layer]->layer_type == FLATTEN)
			)
		 ){
		temp_int = LayersCount - 1;
		if (single_layer >= 2) temp_int = single_layer - 1;
		temp_output_size_x = Layers[temp_int]->output_size_x;
		ASSERT(temp_output_size_x > 0);
		temp_string = "typedef DataType_output@ OutputType[" + to_string(temp_output_size_x) + "];";
		temp_string = StringSubstituteAll(temp_string, "@", quantized?"_short":"");
		AddToCFile_Text(f_stream, temp_string);
	}
	else { //Single_layer >= 2 and CONV2D or POOLING2D
			temp_int = single_layer - 1;
			temp_output_size_x = Layers[temp_int]->output_size_x;
			temp_output_size_y = Layers[temp_int]->output_size_y;
			temp_output_size_z = Layers[temp_int]->output_size_z;
			ASSERT(temp_output_size_x > 0 && temp_output_size_y > 0 && temp_output_size_z > 0);

			AddToCFile_Text(f_stream, "typedef DataType_output OutputType[" + to_string(temp_output_size_x) + "]"
																																+ "[" + to_string(temp_output_size_y) + "]"
																																+ "[" + to_string(temp_output_size_z) + "];");
	}

	AddToCFile_EmptyLine(f_stream);

  int layers_size = Layers.size();
	
	AddToCFile_Text(f_stream, "void forward(InputType inputs, OutputType &outputs,");
	if (fault_simulation) AddToCFile_Text(f_stream, "int faulty_layer, int faulty_fmap, int faulty_bit,", 3);

	if (quantized) {
		temp_string = "";
		bool conv_or_dense ;
		int conv_or_dense_count = 0;
		for(int i = 0; i < layers_size; i++) {
			conv_or_dense = Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE;
			if (conv_or_dense) ++conv_or_dense_count;
		}
		temp_string += "DataType_IZP input_zero_points[" + to_string(conv_or_dense_count) + "], ";
		temp_string += "DataType_OZP output_zero_points[" + to_string(conv_or_dense_count) + "], ";
		temp_string += "DataType_ISF input_scale_factors[" + to_string(conv_or_dense_count) + "], ";
		temp_string += "DataType_OSF output_scale_factors[" + to_string(conv_or_dense_count) + "], ";
		
		AddToCFile_Text(f_stream, temp_string, 3);

		temp_string = "";
		for(int i = 0; i < layers_size; i++) {
			bool conv = Layers[i]->layer_type == CONV2D;
			bool dense = Layers[i]->layer_type == DENSE;
			conv_or_dense = conv || dense;
			if (!conv_or_dense) continue;
			string temp_string2 = "";

			temp_string2 += "DataType_WSF weight_scales_#[@], ";
			temp_string2 = StringSubstituteAll(temp_string2, "#", to_string(i + 1));
			temp_string2 = StringSubstituteAll(temp_string2, "@", to_string(conv?Layers[i]->output_size_z:Layers[i]->output_size_x));

			temp_string += temp_string2;			
		}

		AddToCFile_Text(f_stream, temp_string, 3);
	}

	string indent = Tabs(3);
	temp_int = 0;
	for(int i = 0; i < layers_size; i++) {
		if(Layers[i]->layer_type == CONV2D) {
			if (single_layer && i != single_layer -1) f_stream  << " /*";
			if (temp_int > 0) f_stream  << ", ";
			if (temp_int > 0 && temp_int % 3 == 0) f_stream  << endl;
			if (temp_int % 3 == 0) f_stream << indent;
			f_stream  << "DataType_weights weights_" << i + 1 
								<< "[" + to_string(Layers[i]->kernel_size_rows) + "]"
								<< "[" + to_string(Layers[i]->kernel_size_cols) + "]"
								<< "[" + to_string(Layers[i]->input_size_z) + "]"
								<< "[" + to_string(Layers[i]->output_size_z) + "]"
				;
			if (biases_enabled) f_stream  << ", DataType_biases biases_" << i + 1 << "[" + to_string(Layers[i]->output_size_z) + "]";
			if (single_layer && i != single_layer -1) f_stream  << "*/ ";
			temp_int++;
		}
		else if (Layers[i]->layer_type == DENSE) {
			if (single_layer && i != single_layer -1) f_stream  << " /*";
			if (temp_int > 0) f_stream  << ", ";
			if (temp_int > 0 && temp_int % 3== 0) f_stream  << endl;
			if (temp_int % 3== 0) f_stream << indent;

			f_stream  << "DataType_weights weights_" << i + 1 
								<< "[" + to_string(Layers[i]->input_size_x) + "]"
								<< "[" + to_string(Layers[i]->output_size_x) + "]";
			if (biases_enabled) f_stream  << ", DataType_biases biases_" << i + 1 << "[" + to_string(Layers[i]->output_size_x) + "]";
			if (single_layer && i != single_layer -1) f_stream  << "*/ ";
			temp_int++;
		}
	}

	temp_int = 0;
	for(size_t i = 0; i < Layers.size(); i++)	{
		if (i == Layers.size() - 1) continue; //output

		int layer_number = i;
		Layer* layer = Layers[i];
		string layer_datatype_suffix = "Layer" + to_string(layer_number + 1);

		if (layer->layer_type == CONV2D || layer->layer_type == POOLING2D)
			temp_string = ", DataType_" + layer_datatype_suffix + " l#"
										+ "[" + to_string(layer->output_size_x) + "]"
										+ "[" + to_string(layer->output_size_y) + "]"
										+ "[" + to_string(layer->output_size_z) + "]"
			;
		else //FLATTEN, DENSE
			temp_string = ", DataType_" + layer_datatype_suffix + " l#[" + to_string(layer->output_size_x) + "]";

		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		if (LayerDataLocation(i + 1) == "local") {
			//either comment or totally remove
			temp_string = " /*" + temp_string + "*/";
			//temp_string = "";
		}

		if (temp_int == 0) f_stream << endl;
		if (temp_int > 0 && temp_int % 6 == 0) f_stream << endl;
		if (temp_int % 6 == 0) f_stream << indent;
		f_stream << temp_string;

		temp_int++;
	}

	AddToCFile_Text(f_stream, ")");
	AddToCFile_Text(f_stream, "{", current_indent++);

	if (store_alanysis_data) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", current_indent);
		AddToCFile_Text(f_stream, "for (int input_x = 0; input_x < " + to_string(Layers[0]->input_size_x) + "; input_x++)", current_indent++);
		AddToCFile_Text(f_stream, "for (int input_y = 0; input_y < " + to_string(Layers[0]->input_size_y) + "; input_y++)", current_indent++);
		AddToCFile_Text(f_stream, "for (int input_z = 0; input_z < " + to_string(Layers[0]->input_size_z) + "; input_z++)", current_indent++);
		AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(1) + ", \"inputs\", " + "(float)inputs[input_x][input_y][input_z]" + ", input_x, input_y, input_z);", current_indent);
		-- -- --current_indent;
		AddToCFile_Text(f_stream, "#endif", current_indent);
		AddToCFile_EmptyLine(f_stream, 2);
	}

	for(int i = 0; i < layers_size; i++)	{
		if (single_layer && single_layer - 1 != i) continue;
		switch(Layers[i]->layer_type)
		{
			case CONV2D:
				AddToCFile_Conv2dLayer(f_stream, i);
				break;
			case POOLING2D:
				AddToCFile_Pooling2dLayer(f_stream, i);
				break;
			case FLATTEN:
				AddToCFile_FlattenLayer(f_stream, i);
				break;
			case DENSE:
				AddToCFile_DenseLayer(f_stream, i);
				break;
			default:
				ASSERTA;
				break;
		}

		if (false && quantized /* old quantization method*/) {
			AddToCFile_EmptyLine(f_stream, 1);
			string output_size = to_string(Layers[i]->output_size_x);
			if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == POOLING2D) output_size += "*" + to_string(Layers[i]->output_size_y) + "*" + to_string(Layers[i]->output_size_z);

			string output_name = i == layers_size - 1 ? "outputs" : ("l" + to_string(i+1));
			string config_param = i == layers_size - 1 ? ", 1" : "";

			AddToCFile_Text(f_stream, "quantize_layer_output(&" + output_name + "_base, &" + output_name + ", " + output_size + config_param + ");", current_indent);
			if (i < layers_size - 1) AddToCFile_EmptyLine(f_stream, 1);
		}

		if (i < layers_size - 1) AddToCFile_EmptyLine(f_stream, 1);
		if (fault_simulation && i < layers_size - 1) {
			AddToCFile_Text(f_stream, "fault_injection(&l" + to_string(i+1) + ", " + to_string(i+1) + ", faulty_layer, faulty_fmap, faulty_bit);", current_indent);
			if (i < layers_size - 1) AddToCFile_EmptyLine(f_stream, 1);
		}
		else
			if (i < layers_size - 1) AddToCFile_EmptyLine(f_stream, 2);
	}

	AddToCFile_Text(f_stream, "}");
	AddToCFile_EmptyLine(f_stream, 3);

	if (add_main_function) AddToCFile_MainAndPredict(f_stream);

	f_stream.close();
}

void AddToCFile_Text(ofstream &f_stream, string text, int indent) {
	f_stream << Tabs(indent) << text << endl;
}

void AddToCFile_EmptyLine(ofstream &f_stream, int count) {
	for(int i = 0; i < count; i++) {
		AddToCFile_Text(f_stream, "");
	}
}

void AddToCFile_ActivationFunction(ofstream &f_stream, ActivationFunctions function) {
	switch(function) {
		case RELU:
			AddToCFile_Text(f_stream, "DataType_relu relu(DataType_relu x)");
			AddToCFile_Text(f_stream, "{");
			//AddToCFile_Text(f_stream, "return x > ZeroB ? x : ZeroB;", 1);
			AddToCFile_Text(f_stream, "return x > 0 ? x : (DataType_relu)0;", 1);
			AddToCFile_Text(f_stream, "}");
			break;
		case LINEAR:
			AddToCFile_Text(f_stream, "DataType_relu linear(DataType_relu x)");
			AddToCFile_Text(f_stream, "{");
			//AddToCFile_Text(f_stream, "return x > ZeroB ? x : ZeroB;", 1);
			AddToCFile_Text(f_stream, "return x;", 1);
			AddToCFile_Text(f_stream, "}");
			break;
		case NO_ACTIVATION:
			break;
		case SOFTMAX:
			break;
		default:
			ASSERTA;
			break;
	}
}

void AddToCFile_QMinMax(ofstream& f_stream) {
	AddToCFile_Text(f_stream, "#define Q_MAX(x, y) (x>y?x:y)");
	AddToCFile_Text(f_stream, "#define Q_MIN(x, y) (x>y?y:x)");
	AddToCFile_Text(f_stream, "#define Q_MIN_MAX(x) ( Q_MIN(Q_MAX(x, -128), 127) )");
}

void AddToCFile_Conv2dLayer(ofstream &f_stream, int layer_number) {
	int current_indent = 1;
	string temp_string, temp_string2;
	int temp_int;

	Layer* layer = Layers[layer_number];
  int layers_size = Layers.size();
	string layer_loop_order = loop_orders[layer_number];
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");
	int q_index = -1;
	for (int i = 0; i <= layer_number; ++i) 
		if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) q_index++;

	if (quantized) ASSERT(!single_layer); //Not tested yet

	AddToCFile_Text(f_stream, "//Layer " + to_string(layer_number+1) + ": Conv2D(Padding: " + PaddingTypesToString(layer->padding_type) + ", Stride: " + to_string(layer->stride_size) + ")" , current_indent);
	AddToCFile_Text(f_stream, "//Input: X:" + to_string(layer->input_size_x) + ", Y: " + to_string(layer->input_size_y) + ", Z: " + to_string(layer->input_size_z), current_indent);
	AddToCFile_Text(f_stream, "//Output: X:" + to_string(layer->output_size_x) + ", Y: " + to_string(layer->output_size_y) + ", Z: " + to_string(layer->output_size_z), current_indent);

	string layer_datatype_suffix = "Layer" + to_string(layer_number + 1);
	if (layer_number == layers_size - 1) layer_datatype_suffix = "output";

	if (!single_layer) {
		temp_string = "";
		if (LayerDataLocation(layer_number + 1) != "local") temp_string = "//"; else temp_string = "";
		string static_text = ""; //add_main_function?"STATIC ":"";
		temp_string += static_text + "DataType_" + layer_datatype_suffix + "@ l#[" + to_string(layer->output_size_x) + "]"
																										 										+ "[" + to_string(layer->output_size_y) + "]"
																																				+ "[" + to_string(layer->output_size_z) + "];";
		if (quantized) {
			temp_string2 = temp_string;
			temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1) + "_base");
			temp_string = StringSubstituteAll(temp_string, "@", "");
			AddToCFile_Text(f_stream, temp_string, current_indent);

			temp_string = temp_string2;
			temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
			temp_string = StringSubstituteAll(temp_string, "@", "_short");
			AddToCFile_Text(f_stream, temp_string, current_indent);
		}
		else {
			temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
			temp_string = StringSubstituteAll(temp_string, "@", "");
			AddToCFile_Text(f_stream, temp_string, current_indent);
		}
	}

	//Note copied from DSE: //Note: Labels in the code MUST not contain "_", as for1_1 is cosidered for1 ...

	string input_name = (layer_number == 0 || single_layer) ? "inputs" : ("l" + to_string(layer_number-1+1));
	string weights_tensor_name = "weights_" +  to_string(layer_number + 1);
	string biases_tensor_name = "biases_" +  to_string(layer_number + 1);
	string output_name = (layer_number == layers_size - 1 || single_layer) ? "outputs" : ("l" + to_string(layer_number+1));
	
	string output_name_full = output_name + "@[output_x][output_y][output_z]";
	string output_name_full_q = output_name + "[output_x][output_y][output_z]";

	if (quantized) {
		output_name_full = StringSubstituteAll(output_name_full, "@", "_base");
	}
	else {
		output_name_full = StringSubstituteAll(output_name_full, "@", "");
	}

	string base_for_label = "for" + to_string(layer_number+1);
	if (layer_number+1 >= 10) base_for_label += "t";

	string label_oz, label_oy, label_ox, label_iz, label_kx, label_ky;
	if (numbered_loop_labels) {label_oz = "";		label_oy = "1";		label_ox = "2";		label_iz = "3";		label_kx = "4";		label_ky = "5";}
	else											{label_oz = "Oz"; label_oy = "Oy";	label_ox = "Ox";	label_iz = "Iz";	label_kx = "Kx";	label_ky = "Ky";}

	string oz_for = base_for_label + label_oz + ": for (int output_z = 0; output_z < " + to_string(layer->output_size_z) + "; output_z++)";
	string oy_for = base_for_label + label_oy + ": for (int output_y = 0; output_y < " + to_string(layer->output_size_y) + "; output_y++)";
	string ox_for = base_for_label + label_ox + ": for (int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++)";

	string iz_for = base_for_label + label_iz + ": for (int input_z = 0; input_z < " + to_string(layer->input_size_z) + "; input_z++)";
	string kx_for = base_for_label + label_kx + ": for (int kernel_x = 0; kernel_x < " + to_string(layer->kernel_size_rows) + "; kernel_x++)";
	string ky_for = base_for_label + label_ky + ": for (int kernel_y = 0; kernel_y < " + to_string(layer->kernel_size_cols) + "; kernel_y++)";

	string right_side = "", right_side_q = "";
	string input_name_full, weights_tensor_name_full;
	string row_index, col_index, row_index_def, col_index_def, cond; //Used for valid same

	if(layer->padding_type == PADDING_VALID) {
		string stride_text = "";
		if(layer->stride_size != 1) {
			stride_text = " * " + to_string(layer->stride_size);
		}

		input_name_full = input_name + "[output_x# + kernel_x][output_y# + kernel_y][input_z]";
		input_name_full = StringSubstituteAll(input_name_full, "#", stride_text);

		weights_tensor_name_full = weights_tensor_name + "[kernel_x][kernel_y][input_z][output_z]";
		if (!approximate_multipliers)
			right_side = input_name_full + " * " + weights_tensor_name_full;
		else
			right_side = "MUL_LAYER_" + to_string(layer_number + 1) + "(" + input_name_full + ", " + weights_tensor_name_full + ")";
	}
	else {
		ASSERT(layer->padding_type == PADDING_SAME);

		int padding_size_x, padding_size_y;
		
		ASSERT((layer->kernel_size_rows - 1) % 2 == 0);
		ASSERT((layer->kernel_size_cols - 1) % 2 == 0);

		padding_size_x = (layer->kernel_size_rows - 1) / 2;
		padding_size_y = (layer->kernel_size_cols - 1) / 2;

		string stride_text = "";
		if (layer->stride_size != 1) {
			 if ((layer->input_size_x + 2 * padding_size_x - layer->kernel_size_rows) % layer->stride_size != 0)	cout << endl << "****** ATTENTION: unaligned stride X. ******* " << endl;
			 if ((layer->input_size_y + 2 * padding_size_y - layer->kernel_size_cols) % layer->stride_size != 0)	cout << endl << "****** ATTENTION: unaligned stride Y. ******* " << endl;

			 stride_text = " * " + to_string(layer->stride_size);
		}

		row_index = "output_x# + kernel_x - " + to_string((layer->kernel_size_rows - 1) / 2);
		col_index = "output_y# + kernel_y - " + to_string((layer->kernel_size_cols - 1) / 2);

		row_index = StringSubstituteAll(row_index, "#", stride_text);
		col_index = StringSubstituteAll(col_index, "#", stride_text);

		row_index_def = "int row_index = " + row_index + ";";
		col_index_def = "int col_index = " + col_index + ";";

		weights_tensor_name_full = weights_tensor_name + "[kernel_x][kernel_y][input_z][output_z]";

		input_name_full = input_name + "[row_index][col_index][input_z]";
		cond = "if (row_index >= 0 && row_index < " + to_string(layer->input_size_x) + " && col_index >= 0 && col_index < " + to_string(layer->input_size_y) + ")";

		if (!approximate_multipliers)
			right_side = input_name_full + " * " + weights_tensor_name_full;
		else
			right_side = "MUL_LAYER_" + to_string(layer_number + 1) + "(" + input_name_full + ", " + weights_tensor_name_full + ")";

		right_side_q = "input_zero_points[" + to_string(q_index) + "]" + " * " + weights_tensor_name_full;;
	}

	string loop1, loop2, loop3, loop4, loop5, loop6;

	temp_int = ((string)"oz").size();
	ASSERT(layer_loop_order.substr(0, temp_int) == "oz"); //Currently, only loop orders starting with oz are supported.
	loop1 = oz_for;
	
	loop5 = kx_for;
	loop6 = ky_for;

	string temp_element_definition, zero_definition;
	string temp_element_initializer, temp_element_assignment_to_output;
	string mac_operation, mac_operation_q;

	string temp_element_assignment_to_output_quantized;

	int output_format = 0;

	//if (layer->padding_type == PADDING_SAME) zero_definition = "const DataType_Zero" + to_string(layer_number + 1) + " Zero = 0;";
	string temp_element_name = "temp_element" + to_string(layer_number + 1);

	temp_int = ((string)"oy-ox-iz").size();
	string layer_loop_order_2_to_4 = layer_loop_order.substr(((string)"oz-").size(), temp_int);
	if (layer_loop_order_2_to_4				== "oy-ox-iz") {
		loop2 = oy_for; 
		loop3 = ox_for; 
		loop4 = iz_for; 
		temp_element_definition = "DataType_temp_element" + to_string(layer_number + 1) + " " + temp_element_name + ";";
		temp_element_initializer = temp_element_name + " = " +  (biases_enabled?(biases_tensor_name + "[output_z]"):"0") + ";";
		temp_element_assignment_to_output = output_name_full + " = " + ActivationFunctionsToString(layer->activation_function) + "(" + temp_element_name + ");";
		if (quantized) temp_element_assignment_to_output_quantized = output_name_full_q + " = DataType_Layer" + to_string(layer_number+1) + "_short(Q_MIN_MAX(" + output_name_full + "*input_scale_factors[" + to_string(q_index) + "]*weight_scales_"+to_string(layer_number+1)+"[output_z]/output_scale_factors["+to_string(q_index)+"] + output_zero_points["+to_string(q_index)+"]));";
		mac_operation = temp_element_name + " += " + right_side + ";";
		mac_operation_q = temp_element_name + " += " + right_side_q + ";";
		output_format = 1;
	}
	else if (layer_loop_order_2_to_4	== "ox-oy-iz") {
		loop2 = ox_for; 
		loop3 = oy_for; 
		loop4 = iz_for; 
		temp_element_definition = "DataType_temp_element" + to_string(layer_number + 1) + " " + temp_element_name + ";";
		temp_element_initializer = temp_element_name + " = " +  (biases_enabled?(biases_tensor_name + "[output_z]"):"0") + ";";
		temp_element_assignment_to_output = output_name_full + " = " + ActivationFunctionsToString(layer->activation_function) + "(" + temp_element_name + ");";
		if (quantized) temp_element_assignment_to_output_quantized = output_name_full_q + " = DataType_Layer" + to_string(layer_number+1) + "_short(Q_MIN_MAX(" + output_name_full + "*input_scale_factors[" + to_string(q_index) + "]*weight_scales_"+to_string(layer_number+1)+"[output_z]/output_scale_factors["+to_string(q_index)+"] + output_zero_points["+to_string(q_index)+"]));";
		mac_operation = temp_element_name + " += " + right_side + ";";
		output_format = 1;
	}
	else if (layer_loop_order_2_to_4	== "ox-iz-oy" || layer_loop_order_2_to_4	== "oy-iz-ox") { //oz-ox-iz-oy, oz-oy-iz-ox
		if (layer_loop_order_2_to_4.substr(0, 2) == "ox") {
			loop2 = ox_for; 
			loop3 = iz_for;
			loop4 = oy_for;
			temp_element_definition = "DataType_temp_element" + to_string(layer_number + 1) + " " + temp_element_name + "[" + to_string(layer->output_size_y) + "];";
			temp_element_initializer = base_for_label + "OyI: for(int output_y = 0; output_y < " + to_string(layer->output_size_y) + "; output_y++) " + temp_element_name + "[output_y] = " +  (biases_enabled?(biases_tensor_name + "[output_z]"):"0") + ";";
			temp_element_assignment_to_output = base_for_label + "OyO: for(int output_y = 0; output_y < " + to_string(layer->output_size_y) + "; output_y++) " + output_name_full + " = " + ActivationFunctionsToString(layer->activation_function) + "(" + temp_element_name + "[output_y]);";
			mac_operation = temp_element_name + "[output_y] += " + right_side + ";";
			output_format = 2;
		}
		else { //oy
			loop2 = oy_for; 
			loop3 = iz_for;
			loop4 = ox_for;
			temp_element_definition = "DataType_temp_element" + to_string(layer_number + 1) + " " + temp_element_name + "[" + to_string(layer->output_size_x) + "];";
			temp_element_initializer = base_for_label + "OxI: for(int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++) " + temp_element_name + "[output_x] = " +  (biases_enabled?(biases_tensor_name + "[output_z]"):"0") + ";";
			temp_element_assignment_to_output = base_for_label + "OxO: for(int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++) " + output_name_full + " = " + ActivationFunctionsToString(layer->activation_function) + "(" + temp_element_name + "[output_x]);";
			mac_operation = temp_element_name + "[output_x] += " + right_side + ";";
			output_format = 2;
		}
	}
	else if (layer_loop_order_2_to_4	== "iz-ox-oy" || layer_loop_order_2_to_4	== "iz-oy-ox") { //oz-iz-ox-oy, oz-iz-oy-ox
		loop2 = iz_for;
		temp_element_definition = "DataType_temp_element" + to_string(layer_number + 1) + " " + temp_element_name + "[" + to_string(layer->output_size_x) + "][" + to_string(layer->output_size_y) + "];";
		temp_element_initializer = base_for_label + "OxI: for(int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++)\n\t" + base_for_label + "OyI: for(int output_y = 0; output_y < " + to_string(layer->output_size_y) + "; output_y++)\n\t\t" + temp_element_name + "[output_x][output_y] = " +  (biases_enabled?(biases_tensor_name + "[output_z]"):"0") + ";";
		temp_element_assignment_to_output = base_for_label + "OxO: for(int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++)\n\t" + base_for_label + "OyO: for(int output_y = 0; output_y < " + to_string(layer->output_size_y) + "; output_y++)\n\t\t" + output_name_full + " = " + ActivationFunctionsToString(layer->activation_function) + "(" + temp_element_name + "[output_x][output_y]);";
		mac_operation = temp_element_name + "[output_x][output_y] += " + right_side + ";";
		output_format = 3;

		if (layer_loop_order_2_to_4.substr(3, 2) == "ox") {
			loop3 = ox_for; 
			loop4 = oy_for;
		}
		else { //oy
			loop3 = oy_for; 
			loop4 = ox_for;
		}
	}
	else
		ASSERTA;

	if (quantized) ASSERT(output_format == 1); //only tested in this format

	if (output_format == 1) {
		AddToCFile_Text(f_stream, loop1, current_indent++);
		AddToCFile_Text(f_stream, loop2, current_indent++);
		AddToCFile_Text(f_stream, loop3, current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		if (biases_enabled && store_alanysis_data)
			AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"biases\", (float)" + biases_tensor_name + "[output_z], -1, -1, output_z);", current_indent); 
		AddToCFile_Text(f_stream, temp_element_definition, current_indent);
		if (zero_definition != "") AddToCFile_Text(f_stream, zero_definition, current_indent);
		AddToCFile_Text(f_stream, temp_element_initializer, current_indent);
		AddToCFile_Text(f_stream, loop4, current_indent++);
		AddToCFile_Text(f_stream, loop5, current_indent++);
		AddToCFile_Text(f_stream, loop6, current_indent);
		if (store_alanysis_data && layer->padding_type != PADDING_SAME) {
			AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", current_indent);
			AddToCFile_Text(f_stream, "{", current_indent);
			AddToCFile_Text(f_stream, "#endif", current_indent);
			current_indent++;
		}
		
		if (layer->padding_type == PADDING_SAME)	{
			AddToCFile_Text(f_stream, "{", current_indent++);
			AddToCFile_Text(f_stream, row_index_def, current_indent);
			AddToCFile_Text(f_stream, col_index_def, current_indent);
		}
		else {
			if (!store_alanysis_data) current_indent++;
		}

		if (layer->padding_type == PADDING_SAME) { 
			AddToCFile_Text(f_stream, cond, current_indent++);
		}
		mac_operation = StringSubstituteAll(mac_operation, "\n", "\n" + Tabs(current_indent));
		AddToCFile_Text(f_stream, mac_operation, current_indent);
		if (layer->padding_type == PADDING_SAME) { 
			current_indent--;
			AddToCFile_Text(f_stream, "else", current_indent++);
			AddToCFile_Text(f_stream, mac_operation_q, current_indent);
		}
		
		if (store_alanysis_data) {
			if(layer->padding_type == PADDING_SAME) current_indent--;
			AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", current_indent);
			AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"weights\", (float)" + weights_tensor_name_full + ", -1, -1, -1);", current_indent);
			AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"temp_element\", (float)" + temp_element_name + ", -1, -1, -1);", current_indent);
			if (layer->padding_type != PADDING_SAME) AddToCFile_Text(f_stream, "}", --current_indent); //"{" will be added anyways if padding same
			if(layer->padding_type != PADDING_SAME) current_indent++;
			AddToCFile_Text(f_stream, "#endif", current_indent);
		}

		if (layer->padding_type == PADDING_SAME && store_alanysis_data) current_indent++;
		if (layer->padding_type == PADDING_SAME) AddToCFile_Text(f_stream, "}", -- -- current_indent);
		else current_indent--;

		-- -- current_indent;
		AddToCFile_Text(f_stream, temp_element_assignment_to_output, current_indent --);
		if (quantized) {current_indent++; AddToCFile_Text(f_stream, temp_element_assignment_to_output_quantized, current_indent --);}
		if (store_alanysis_data && layer->padding_type == PADDING_SAME) current_indent++;
		if (store_alanysis_data && layer->padding_type != PADDING_SAME) current_indent++;
		if (store_alanysis_data && quantized) AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"LayerOutputBase\", (float)" + output_name_full + ", output_x, output_y, output_z);", current_indent);
		if (store_alanysis_data) AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"LayerOutput\", (float)" + (quantized?output_name_full_q:output_name_full) + ", output_x, output_y, output_z);", current_indent);
		if (store_alanysis_data && layer->padding_type == PADDING_SAME) current_indent--;
		if (store_alanysis_data && layer->padding_type != PADDING_SAME) current_indent--;
		AddToCFile_Text(f_stream, "}", current_indent --);
	}
	else if (output_format == 2) {
		ASSERT(!store_alanysis_data); //Not supported in this format

		AddToCFile_Text(f_stream, loop1, current_indent++);
		AddToCFile_Text(f_stream, loop2, current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		AddToCFile_Text(f_stream, temp_element_definition, current_indent);
		AddToCFile_Text(f_stream, zero_definition, current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, temp_element_initializer, current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, loop3, current_indent++);
		AddToCFile_Text(f_stream, loop4, current_indent++);
		AddToCFile_Text(f_stream, loop5, current_indent++);
		AddToCFile_Text(f_stream, loop6, current_indent);

		if (layer->padding_type == PADDING_SAME)
		{
			AddToCFile_Text(f_stream, "{", current_indent++);
			AddToCFile_Text(f_stream, row_index_def, current_indent);
			AddToCFile_Text(f_stream, col_index_def, current_indent);
			AddToCFile_Text(f_stream, cond, current_indent++);
		}
		else current_indent++;

		mac_operation = StringSubstituteAll(mac_operation, "\n", "\n" + Tabs(current_indent));
		AddToCFile_Text(f_stream, mac_operation, current_indent);

		if (layer->padding_type == PADDING_SAME) AddToCFile_Text(f_stream, "}", -- current_indent);
		else current_indent--;

		-- -- -- current_indent;

		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, temp_element_assignment_to_output, current_indent --);
		AddToCFile_Text(f_stream, "}", current_indent --);
	}
	else if (output_format == 3) {
		ASSERT(!store_alanysis_data); //Not supported in this format
		AddToCFile_Text(f_stream, loop1, current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		AddToCFile_Text(f_stream, temp_element_definition, current_indent);
		AddToCFile_Text(f_stream, zero_definition, current_indent);
		AddToCFile_EmptyLine(f_stream);
		temp_element_initializer = StringSubstituteAll(temp_element_initializer, "\n", "\n" + Tabs(current_indent));
		AddToCFile_Text(f_stream, temp_element_initializer, current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, loop2, current_indent++);
		AddToCFile_Text(f_stream, loop3, current_indent++);
		AddToCFile_Text(f_stream, loop4, current_indent++);
		AddToCFile_Text(f_stream, loop5, current_indent++);
		AddToCFile_Text(f_stream, loop6, current_indent);

		if (layer->padding_type == PADDING_SAME)
		{
			AddToCFile_Text(f_stream, "{", current_indent++);
			AddToCFile_Text(f_stream, row_index_def, current_indent);
			AddToCFile_Text(f_stream, col_index_def, current_indent);
			AddToCFile_Text(f_stream, cond, current_indent++);
		}
		else current_indent++;

		mac_operation = StringSubstituteAll(mac_operation, "\n", "\n" + Tabs(current_indent));
		AddToCFile_Text(f_stream, mac_operation, current_indent);

		if (layer->padding_type == PADDING_SAME) AddToCFile_Text(f_stream, "}", -- -- current_indent);
		else current_indent--;

		-- -- -- -- current_indent;

		AddToCFile_EmptyLine(f_stream);
		temp_element_assignment_to_output = StringSubstituteAll(temp_element_assignment_to_output, "\n", "\n" + Tabs(current_indent));
		AddToCFile_Text(f_stream, temp_element_assignment_to_output, current_indent --);
		AddToCFile_Text(f_stream, "}", current_indent --);
	}
}

void AddToCFile_Pooling2dLayer(ofstream &f_stream, int layer_number) {
	int current_indent = 1;
	string temp_string, temp_string2;

	Layer* layer = Layers[layer_number];
  int layers_size = Layers.size();
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");
	bool last_layer = layer_number == layers_size - 1;

	ASSERT(layer->pooling_type == MAX_POOLING);


	AddToCFile_Text(f_stream, "//Layer " + to_string(layer_number+1) + ": " +  PoolingTypesToString(layer->pooling_type) + " Pooling", current_indent);
	AddToCFile_Text(f_stream, "//Input: X:" + to_string(layer->input_size_x) + ", Y: " + to_string(layer->input_size_y) + ", Z: " + to_string(layer->input_size_z), current_indent);
	AddToCFile_Text(f_stream, "//Output: X:" + to_string(layer->output_size_x) + ", Y: " + to_string(layer->output_size_y) + ", Z: " + to_string(layer->output_size_z), current_indent);

	string layer_datatype_suffix = "Layer" + to_string(layer_number + 1);
	if (quantized && !last_layer) layer_datatype_suffix += "_short";
	if (layer_number == layers_size - 1) layer_datatype_suffix = "output";

	temp_string = "";
	if (LayerDataLocation(layer_number + 1) != "local") temp_string = "//"; else temp_string = "";
	string static_text = ""; //add_main_function?"STATIC ":"";
	temp_string += static_text + "DataType_" + layer_datatype_suffix + "@ l#[" + to_string(layer->output_size_x) + "]"
																										 									+ "[" + to_string(layer->output_size_y) + "]"
																																			+ "[" + to_string(layer->output_size_z) + "];";
	if (data_type_mode_fixed_point_single && data_type_mode_detail == "eight-bit-int") {
		temp_string2 = temp_string;
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1) + "_base");
		temp_string = StringSubstituteAll(temp_string, "@", "");
		AddToCFile_Text(f_stream, temp_string, current_indent);

		temp_string = temp_string2;
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		temp_string = StringSubstituteAll(temp_string, "@", "_short");
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}
	else {
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		temp_string = StringSubstituteAll(temp_string, "@", "");
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}

	ASSERT(layer->input_size_z == layer->output_size_z);

	//Note copied from DSE: //Note: Labels in the code MUST not contain "_", as for1_1 is cosidered for1 ...

	string base_for_label = "for" + to_string(layer_number+1);
	if (layer_number+1 >= 10) base_for_label += "t";
	AddToCFile_Text(f_stream, base_for_label + ": for (int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++)", current_indent++);
	AddToCFile_Text(f_stream, base_for_label + "1: for (int output_y = 0; output_y < " + to_string(layer->output_size_y) + "; output_y++)", current_indent++);
	AddToCFile_Text(f_stream, base_for_label + "2: for (int output_z = 0; output_z < " + to_string(layer->input_size_z) + "; output_z++)", current_indent); //I put the output_z after output_x, output_y because z is usually small enough to enable us to use: (#pragma HLS array_partition variable=layer_output complete dim=3)
																																																																												 //Note that for a pooling layer input_z is equal to output_z
	AddToCFile_Text(f_stream, "{", current_indent++);
	
	string input_name = layer_number == 0 ? "inputs" : ("l" + to_string(layer_number-1+1));

	AddToCFile_Text(f_stream, "DataType_" + layer_datatype_suffix + " current_cell, max_value;", current_indent);
	AddToCFile_Text(f_stream, "");
	if (Layers[layer_number - 1]->activation_function == RELU) {//Min value will be zero
		AddToCFile_Text(f_stream, "max_value = 0;", current_indent);
	}
	else if (Layers[layer_number - 1]->activation_function == LINEAR && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t")) {
		AddToCFile_Text(f_stream, "max_value = -128;", current_indent);
	}
	else
	{
		ASSERTA;
		AddToCFile_Text(f_stream, "max_value = " + input_name + "[output_x * 2 + 0][output_y * 2 + 0][output_z];", current_indent);
	}


	ASSERT(layer->stride_size == layer->kernel_size_rows && layer->stride_size == layer->kernel_size_cols);

	AddToCFile_Text(f_stream, base_for_label + "3: for (int kernel_x = 0; kernel_x < " + to_string(layer->kernel_size_rows) + "; kernel_x++)", current_indent++);
	AddToCFile_Text(f_stream, base_for_label + "4: for (int kernel_y = 0; kernel_y < " + to_string(layer->kernel_size_cols) + "; kernel_y++)", current_indent);
	AddToCFile_Text(f_stream, "{", current_indent++);


	AddToCFile_Text(f_stream, "current_cell = " + input_name + "[output_x * " + to_string(layer->stride_size) + " + kernel_x][output_y * " + to_string(layer->stride_size) + " + kernel_y][output_z];", current_indent);
	AddToCFile_Text(f_stream, "if (current_cell >	max_value) max_value = current_cell;", current_indent);
	AddToCFile_Text(f_stream, "}", --current_indent);
	--current_indent;
	string output_name_full = "l" + to_string(layer_number+1) + "@[output_x][output_y][output_z]";
	if (data_type_mode_fixed_point_single && data_type_mode_detail == "eight-bit-int") {
		output_name_full = StringSubstituteAll(output_name_full, "@", "_base");
	}
	else {
		output_name_full = StringSubstituteAll(output_name_full, "@", "");
	}

	AddToCFile_Text(f_stream, output_name_full + " = max_value;", current_indent);
	if (store_alanysis_data) AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"LayerOutput\", (float)" + "max_value" + ", output_x, output_y, output_z);", current_indent); 

	AddToCFile_Text(f_stream, "}", --current_indent);
}

void AddToCFile_FlattenLayer(ofstream &f_stream, int layer_number) {
	int current_indent = 1;
	string temp_string, temp_string2;

	Layer* layer = Layers[layer_number];
  int layers_size = Layers.size();
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");
	bool last_layer = layer_number == layers_size - 1;

	AddToCFile_Text(f_stream, "//Layer " + to_string(layer_number+1) + ": Flatten", current_indent);
	AddToCFile_Text(f_stream, "//Input: X:" + to_string(layer->input_size_x) + ", Y: " + to_string(layer->input_size_y) + ", Z: " + to_string(layer->input_size_z), current_indent);
	AddToCFile_Text(f_stream, "//Output: X:" + to_string(layer->output_size_x) + ", Y: " + to_string(layer->output_size_y) + ", Z: " + to_string(layer->output_size_z), current_indent);

	string layer_datatype_suffix = "Layer" + to_string(layer_number + 1);
	if (quantized && !last_layer) layer_datatype_suffix += "_short";
	if (layer_number == layers_size - 1) layer_datatype_suffix = "output";

	temp_string = "";
	if (LayerDataLocation(layer_number + 1) != "local") temp_string = "//"; else temp_string = "";
	string static_text = ""; //add_main_function?"STATIC ":"";

	temp_string += static_text + "DataType_" + layer_datatype_suffix + "@ l#[" + to_string(layer->output_size_x) + "];";
	if (data_type_mode_fixed_point_single && data_type_mode_detail == "eight-bit-int") {
		temp_string2 = temp_string;
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1) + "_base");
		temp_string = StringSubstituteAll(temp_string, "@", "");
		AddToCFile_Text(f_stream, temp_string, current_indent);

		temp_string = temp_string2;
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		temp_string = StringSubstituteAll(temp_string, "@", "_short");
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}
	else {
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		temp_string = StringSubstituteAll(temp_string, "@", "");
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}

	ASSERT(layer->output_size_y == 1 && layer->output_size_z == 1);

	//Note copied from DSE: //Note: Labels in the code MUST not contain "_", as for1_1 is cosidered for1 ...

	string input_name = layer_number == 0 ? "inputs" : ("l" + to_string(layer_number-1+1) + "");
	string base_for_label = "for" + to_string(layer_number+1);
	if (layer_number+1 >= 10) base_for_label += "t";

	string label_ix, label_iy, label_iz;
	if (numbered_loop_labels) {label_ix = "";	label_iy = "1";	label_iz = "2";}
	else											{label_ix = "Ix";	label_iy = "Iy";	label_iz = "Iz";}

	AddToCFile_Text(f_stream, base_for_label + label_ix + ": for (int input_x = 0; input_x < " + to_string(layer->input_size_x) + "; input_x++)", current_indent++);
	AddToCFile_Text(f_stream, base_for_label + label_iy + ": for (int input_y = 0; input_y < " + to_string(layer->input_size_y) + "; input_y++)", current_indent++);
	AddToCFile_Text(f_stream, base_for_label + label_iz + ": for (int input_z = 0; input_z < " + to_string(layer->input_size_z) + "; input_z++)", current_indent);
	if (store_alanysis_data) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", current_indent);
		AddToCFile_Text(f_stream, "{", current_indent);
		AddToCFile_Text(f_stream, "#endif", current_indent);
	}
	current_indent++;

	string flatten_formula = "input_x * " + to_string(layer->input_size_y) + " * " + to_string(layer->input_size_z) + " + input_y * " + to_string(layer->input_size_z) + " + input_z";
	string output_name_full = "l" + to_string(layer_number + 1)
		+ "@[" + flatten_formula + "]";
	if (data_type_mode_fixed_point_single && data_type_mode_detail == "eight-bit-int") {
		output_name_full = StringSubstituteAll(output_name_full, "@", "_base");
	}
	else {
		output_name_full = StringSubstituteAll(output_name_full, "@", "");
	}

	AddToCFile_Text(f_stream, output_name_full + " = " + input_name + "[input_x][input_y][input_z];", current_indent);
	
	if (store_alanysis_data) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", --current_indent);
		AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"LayerOutput\", (float)" + output_name_full + ", " + flatten_formula + ", -1, -1);", ++current_indent);
		AddToCFile_Text(f_stream, "}", --current_indent);
		AddToCFile_Text(f_stream, "#endif", current_indent++);
	}
}

void AddToCFile_DenseLayer(ofstream &f_stream, int layer_number) {
	int current_indent = 1;
	string temp_string, temp_string2;

	Layer* layer = Layers[layer_number];
  int layers_size = Layers.size();
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");
	bool last_layer = layer_number == layers_size - 1;

	int q_index = -1;
	for (int i = 0; i <= layer_number; ++i) 
		if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) q_index++;

	if (quantized) ASSERT(!single_layer); //Not tested yet

	if (layer_number == 0) {
		ASSERT(layer->input_size_y == 0 && layer->input_size_z == 0); //For MLPs with no conv layer
	}
	else 
		ASSERT(Layers[layer_number - 1]->layer_type == FLATTEN || Layers[layer_number - 1]->layer_type == DENSE);

	AddToCFile_Text(f_stream, "//Layer " + to_string(layer_number+1) + ": Dense(Fully connected)", current_indent);
	AddToCFile_Text(f_stream, "//Input: X:" + to_string(layer->input_size_x) + ", Y: " + to_string(layer->input_size_y) + ", Z: " + to_string(layer->input_size_z), current_indent);
	AddToCFile_Text(f_stream, "//Output: X:" + to_string(layer->output_size_x) + ", Y: " + to_string(layer->output_size_y) + ", Z: " + to_string(layer->output_size_z), current_indent);

	if(!last_layer)	{
		string layer_datatype_suffix = "Layer" + to_string(layer_number + 1);
		if (LayerDataLocation(layer_number + 1) != "local") temp_string = "//"; else temp_string = "";
		string static_text = ""; //add_main_function?"STATIC ":"";
		
		temp_string += static_text + "DataType_" + layer_datatype_suffix + "@ l#[" + to_string(layer->output_size_x) + "];";
		if (quantized) {
			temp_string2 = temp_string;
			temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1) + "_base");
			temp_string = StringSubstituteAll(temp_string, "@", "");
			AddToCFile_Text(f_stream, temp_string, current_indent);

			temp_string = temp_string2;
			temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
			temp_string = StringSubstituteAll(temp_string, "@", "_short");
			AddToCFile_Text(f_stream, temp_string, current_indent);
		}
		else {
			temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
			temp_string = StringSubstituteAll(temp_string, "@", "");
			AddToCFile_Text(f_stream, temp_string, current_indent);
		}
	}
	else {
		if (quantized) {
			temp_string = "DataType_output outputs_base[" + to_string(layer->output_size_x) + "];";
			AddToCFile_Text(f_stream, temp_string, current_indent);
		}
	}

	ASSERT(layer->output_size_y == 1 && layer->output_size_z == 1);

	//Note copied from DSE: //Note: Labels in the code MUST not contain "_", as for1_1 is cosidered for1 ...

	string input_name = layer_number == 0 ? "inputs" : ("l" + to_string(layer_number-1+1));
	string weights_tensor_name = "weights_" +  to_string(layer_number + 1);
	string biases_tensor_name = "biases_" +  to_string(layer_number + 1);
	string base_for_label = "for" + to_string(layer_number+1);
	string activation_function = ActivationFunctionsToString(layer->activation_function);

	if (layer_number+1 >= 10) base_for_label += "t";

	string label_ox, label_ix;
	if (numbered_loop_labels) {label_ox = "";		label_ix = "1";}
	else											{label_ox = "Ox";		label_ix = "Ix";}


	string temp_element_name = "temp_element" + to_string(layer_number + 1);

	AddToCFile_Text(f_stream, base_for_label + label_ox + ": for (int output_x = 0; output_x < " + to_string(layer->output_size_x) + "; output_x++)", current_indent);
	AddToCFile_Text(f_stream, "{", current_indent++);
	AddToCFile_Text(f_stream, "DataType_temp_element" + to_string(layer_number + 1) + " " + temp_element_name + ";", current_indent); 
	if (biases_enabled) 
		AddToCFile_Text(f_stream, temp_element_name + " = " + biases_tensor_name + "[output_x];", current_indent);
	else
		AddToCFile_Text(f_stream, temp_element_name + " = 0;", current_indent);

	if (biases_enabled && store_alanysis_data) 
		AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"biases\", (float)" + biases_tensor_name + "[output_x], output_x, -1, -1);", current_indent); 

	AddToCFile_Text(f_stream, base_for_label + label_ix + ": for (int input_x = 0; input_x < " + to_string(layer->input_size_x) + "; input_x++)", current_indent);
	if (store_alanysis_data) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", current_indent);
		AddToCFile_Text(f_stream, "{", current_indent);
		AddToCFile_Text(f_stream, "#endif", current_indent);
	}
	current_indent++;
	
	string weights_tensor_name_full = weights_tensor_name + "[input_x][output_x]";


	string right_side = "";
	string input_name_full = input_name + "[input_x]";
	if (!approximate_multipliers)
		right_side = input_name_full + " * " + weights_tensor_name_full;
	else
		right_side = "MUL_LAYER_" + to_string(layer_number + 1) + "(" + input_name_full + ", " + weights_tensor_name_full + ")";

	AddToCFile_Text(f_stream, temp_element_name + " += " + right_side + ";", current_indent);
	
	--current_indent;
	if (store_alanysis_data) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN", current_indent);
		AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"weights\", (float)" + weights_tensor_name_full + ", -1, -1, -1);", ++current_indent);
		AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"temp_element\", (float)" + temp_element_name + ", -1, -1, -1);", current_indent);
		AddToCFile_Text(f_stream, "}", --current_indent);
		AddToCFile_Text(f_stream, "#endif", current_indent);
	}

	if(activation_function == "softmax")
	{
		ASSERT(layer_number == layers_size - 1);
		activation_function = "relu"; //changed from "" to "relu" on 2021-06-29, check 
	}

	string output_name = last_layer ? "outputs@" : ("l" + to_string(layer_number+1) + "@");
	string output_name_q = last_layer ? "outputs" : ("l" + to_string(layer_number+1) + "");
	string output_name_full = output_name + "[output_x]";
	string output_name_full_q = output_name_q + "[output_x]";

	if (quantized) {
		output_name_full = StringSubstituteAll(output_name_full, "@", "_base");
	}
	else {
		output_name_full = StringSubstituteAll(output_name_full, "@", "");
	}

	string temp_element_assignment_to_output;
	string temp_element_assignment_to_output_quantized;

	temp_element_assignment_to_output = output_name_full + " = " + activation_function + "(" + temp_element_name + ");";
	AddToCFile_Text(f_stream, temp_element_assignment_to_output, current_indent);
	if (quantized) {
		temp_element_assignment_to_output_quantized = output_name_full_q;
		temp_element_assignment_to_output_quantized += " = " + (last_layer?"DataType_output":"DataType_Layer" + to_string(layer_number+1)) + "_short(Q_MIN_MAX(" + output_name_full + "*input_scale_factors[" + to_string(q_index) + "]";
		temp_element_assignment_to_output_quantized += 	"*weight_scales_"+to_string(layer_number+1)+"[output_x]/output_scale_factors["+to_string(q_index)+"] + output_zero_points["+to_string(q_index)+"]));";
		AddToCFile_Text(f_stream, temp_element_assignment_to_output_quantized, current_indent);
	}

	if (store_alanysis_data && quantized) AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"LayerOutputBase\", (float)" + output_name_full + ", output_x, -1, -1);", current_indent); 
	if (store_alanysis_data) AddToCFile_Text(f_stream, (string)"STORE_DATA(" + to_string(layer_number + 1) + ", \"LayerOutput\", (float)" + (quantized?output_name_full_q:output_name_full) + ", output_x, -1, -1);", current_indent); 
	AddToCFile_Text(f_stream, "}", -- current_indent);
}

string Tabs(int count) {
	string result;
	for(int i = 0; i < count; i++) {
		result += "\t";
	}
	return result;
}

void AddToCFile_MainAndPredict(ofstream &f_stream) {
	string temp_string;

	AddToCFile_Text(f_stream, "#ifndef _HLS_RUN");
	AddToCFile_EmptyLine(f_stream);

  int layers_size = Layers.size();
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");

	string q_factors_string;
	if (quantized) {
		temp_string = "";
		bool conv_or_dense ;
		int conv_or_dense_count = 0;
		for(int i = 0; i < layers_size; i++) {
			conv_or_dense = Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE;
			if (conv_or_dense) ++conv_or_dense_count;
		}
		temp_string += "input_zero_points_src, ";
		temp_string += "output_zero_points_src, ";
		temp_string += "input_scale_factors_src, ";
		temp_string += "output_scale_factors_src, ";
		temp_string += "\n" + Tabs(3);
		
		for(int i = 0; i < layers_size; i++) {
			bool conv = Layers[i]->layer_type == CONV2D;
			bool dense = Layers[i]->layer_type == DENSE;
			conv_or_dense = conv || dense;
			if (!conv_or_dense) continue;
			string temp_string2 = "";

			temp_string2 += "weight_scales_#_src, ";
			temp_string2 = StringSubstituteAll(temp_string2, "#", to_string(i + 1));

			temp_string += temp_string2;
		}
		q_factors_string = temp_string;
	}

	AddToCFile_Text(f_stream, "void Predict(InputType input, int *p" + string(fault_simulation?", int faulty_layer, int faulty_fmap, int faulty_bit":"") + ")"	);
	AddToCFile_Text(f_stream, "{");

	int current_indent = 1;

	for(int i = 0; i < layers_size - 1; i++)	{
		string temp_string;
		int layer_number = i;
		Layer* layer = Layers[i];
		if (layer_number == layers_size - 1) continue; //output

		string layer_datatype_suffix = "Layer" + to_string(layer_number + 1);

		if (LayerDataLocation(layer_number + 1) == "local") temp_string = "//"; else temp_string = "";
		if (layer->layer_type == CONV2D || layer->layer_type == POOLING2D)
			temp_string += /* STATIC */ "DataType_" + layer_datatype_suffix + " l#_src[" + to_string(layer->output_size_x) + "]"
																																			 + "[" + to_string(layer->output_size_y) + "]"
																																			 + "[" + to_string(layer->output_size_z) + "];";
		else //FLATTEN, DENSE
			temp_string += /* STATIC */ "DataType_" + layer_datatype_suffix + " l#_src[" + to_string(layer->output_size_x) + "];";

		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}
	AddToCFile_EmptyLine(f_stream);

	AddToCFile_Text(f_stream, "\tOutputType outputs = {0};");
	AddToCFile_Text(f_stream, "\tforward(input, outputs,");
	
	if (fault_simulation) AddToCFile_Text(f_stream, "faulty_layer, faulty_fmap, faulty_bit,", 3);
	if (quantized) AddToCFile_Text(f_stream, q_factors_string, 3);

	string indent = Tabs(3);
	int temp_int = 0;
	for(int i = 0; i < layers_size; i++)	{
		if(Layers[i]->layer_type == CONV2D)	{
			if (temp_int > 0) f_stream  << ", ";
			if (temp_int > 0 && temp_int % 3 == 0) f_stream  << endl;
			if (temp_int % 3 == 0) f_stream << indent;

			f_stream  << "weights_" << i + 1 << "_src";
			if (biases_enabled) f_stream  << ", biases_" << i + 1 << "_src";
			temp_int++;
		}
		else if (Layers[i]->layer_type == DENSE) {
			if (temp_int > 0) f_stream  << ", ";
			if (temp_int > 0 && temp_int % 3 == 0) f_stream  << endl;
			if (temp_int % 3 == 0) f_stream << indent;
			
			f_stream  << "weights_" << i + 1 << "_src";
			if (biases_enabled) f_stream  << ", biases_" << i + 1 << "_src";
			temp_int++;
		}
	}

	temp_int = 0;
	for(int i = 0; i < layers_size; i++)	{
		if (i == layers_size - 1) continue; //output
		if (LayerDataLocation(i + 1) == "local") continue;

		if (temp_int == 0) f_stream  << endl << "\t\t\t";
		//if (temp_int > 0 && temp_int % 6 == 0) f_stream  << endl;
		//if (temp_int % 6 == 0) f_stream << indent;

		int layer_number = i;
		//Layer* layer = Layers[i];
		temp_string = ", l#_src";
		temp_string = StringSubstituteAll(temp_string, "#", to_string(layer_number+1));
		f_stream << temp_string;
		temp_int++;
	}

	AddToCFile_Text(f_stream, ");");

	AddToCFile_Text(f_stream, "");
	AddToCFile_Text(f_stream, "\t// output decoding");
	AddToCFile_Text(f_stream, "\tint result = 0;");
	AddToCFile_Text(f_stream, "\tDataType_relu maxvalue = outputs[0];");
	AddToCFile_Text(f_stream, "\tfor (int i = 1; i < " + to_string(Layers[Layers.size() - 1]->output_size_x) + "; i++)");
	AddToCFile_Text(f_stream, "\t{");
	AddToCFile_Text(f_stream, "\t\tif (outputs[i] > maxvalue)");
	AddToCFile_Text(f_stream, "\t\t{");
	AddToCFile_Text(f_stream, "\t\t\tmaxvalue = outputs[i];");
	AddToCFile_Text(f_stream, "\t\t\tresult = i;");
	AddToCFile_Text(f_stream, "\t\t}");
	AddToCFile_Text(f_stream, "\t}");
	AddToCFile_Text(f_stream, "");
	AddToCFile_Text(f_stream, "\t*p = result;");
	AddToCFile_Text(f_stream, "}");
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "int main(int argc, char* argv[])");
	AddToCFile_Text(f_stream, "{");

	current_indent = 1;
		
	AddToCFile_Text(f_stream, "double exec_time_sum_wall_clock = 0;", current_indent);
	AddToCFile_Text(f_stream, "Timer t1_wall_clock(Timer::CHR);", current_indent);
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "ReadOptions(argc, argv);", current_indent);
	AddToCFile_Text(f_stream, R"(//arguments["mul-name"] = "mul8s_1L12";)", current_indent);
	AddToCFile_Text(f_stream, R"(//arguments["mul-layers-config"] = "10100111";)", current_indent);
	AddToCFile_Text(f_stream, R"(//arguments["mul-layers-config"] = "00000000";)", current_indent);
	AddToCFile_Text(f_stream, R"(//arguments["fault-simulation"] = ""; //ACTIVE)", current_indent);
	AddToCFile_Text(f_stream, R"(//arguments["threads"] = "8";)", current_indent);
	AddToCFile_EmptyLine(f_stream);

	AddToCFile_Text(f_stream, "string filename = GetLogFileName();", current_indent);
	AddToCFile_Text(f_stream, "string filename_elements_s = GetLogAllElementsFileName();", current_indent);
	AddToCFile_EmptyLine(f_stream);

	AddToCFile_Text(f_stream, R"(AddToLog(filename, "DataType_relu: %s", TypeName(DataType_relu));)", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "DataType_input: %s", TypeName(DataType_input));)", current_indent);
	if (biases_enabled) AddToCFile_Text(f_stream, R"(AddToLog(filename, "DataType_biases: %s", TypeName(DataType_biases));)", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "DataType_weights: %s", TypeName(DataType_weights));)", current_indent);
	
	for(int i = 0; i < layers_size; i++)	{
		string temp_string = R"(AddToLog(filename, "DataType_Layer#: %s", TypeName(DataType_Layer#));)";
		string temp_string2 = to_string(i + 1);
		if (i == layers_size - 1) temp_string2 = "_output";
		temp_string = StringSubstituteAll(temp_string, "#", temp_string2);
		temp_string = StringSubstituteAll(temp_string, "_Layer_output", "_output");
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}
	AddToCFile_EmptyLine(f_stream);

	current_indent = 1;
	if (approximate_multipliers) {
		AddToCFile_Text(f_stream, R"(if (arguments["mul-name"] != "") {)", current_indent++);
		AddToCFile_Text(f_stream, R"(AddToLog(filename, "\nMultiplier Type: " + arguments["mul-name"]);)", current_indent);
		AddToCFile_Text(f_stream, R"(AddToLog(filename, "Multiplier Layer Config: " + arguments["mul-layers-config"] + "\n");)", current_indent);
		AddToCFile_Text(f_stream, R"(})", --current_indent);
		AddToCFile_EmptyLine(f_stream);
	}

	string initialize_param_argument;
	if (quantized) {
		for(int i = 0; i < layers_size; i++)	{
			initialize_param_argument += LayerTypesToString(Layers[i]->layer_type)[0];
		}
		initialize_param_argument = "\"" + initialize_param_argument + "\"";
	}

	AddToCFile_Text(f_stream, "\tInitializeParam(" + initialize_param_argument + ");");
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Parameters loaded (%s) ", GetCurrentTimeAsString());)", current_indent);
	AddToCFile_EmptyLine(f_stream);

	string number_of_inputs;
	number_of_inputs = "10000";
	if(network_guess == "vgg") number_of_inputs = "100";

	AddToCFile_Text(f_stream, "int image_count = GetImageCount(" + number_of_inputs + "); //Number of test images in data.h that is going to be processed", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Number of test images: %s ", to_string(image_count));)", current_indent);
	AddToCFile_Text(f_stream, "InitilizeData(image_count, " + to_string(Layers[0]->input_size_x) + ", " + to_string(Layers[0]->input_size_y) + ", " + to_string(Layers[0]->input_size_z) + ");" + " //Change first parameter based on the number of test images exist in the data.h", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Data loaded (%s) ", GetCurrentTimeAsString());)", current_indent);
	AddToCFile_EmptyLine(f_stream);

	current_indent = 1;
	AddToCFile_Text(f_stream, "int thread_count = GetThreadCount();", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Number of threads: " + to_string(thread_count));)", current_indent);

	AddToCFile_Text(f_stream, "\tvector<thread> threads;");
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "\tInputType* temp_data = new InputType[thread_count];");
	AddToCFile_Text(f_stream, "\tint *p = new int[thread_count];");
	AddToCFile_Text(f_stream, "\tint truePredict = 0;");
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "double exec_time_sum = 0;", current_indent);
	AddToCFile_Text(f_stream, "Timer t1(Timer::CHR);", current_indent);
	AddToCFile_EmptyLine(f_stream);
	if (fault_simulation){
		AddToCFile_Text(f_stream, R"(if (arguments["fault-simulation"] != "ACTIVE") {)", current_indent);
		AddToCFile_EmptyLine(f_stream);
	}
	
	AddToCFile_Text(f_stream, "int i = 0;", current_indent);
	AddToCFile_Text(f_stream, "int run_range = image_count;", current_indent);
	AddToCFile_Text(f_stream, "while (i < run_range)", current_indent);
	AddToCFile_Text(f_stream, "{", current_indent++);
	AddToCFile_Text(f_stream, "for (int t = 0; t < thread_count && i + t < run_range; t++)", current_indent++);
	AddToCFile_Text(f_stream, "for(int j = 0; j < " + to_string(Layers[0]->input_size_x) + "; j++)", current_indent++);
	AddToCFile_Text(f_stream, "for(int k = 0; k < " + to_string(Layers[0]->input_size_y) + "; k++)" + string((Layers[0]->input_size_z != 1)?" {":""), current_indent++);
	if(Layers[0]->input_size_z == 1) {
		AddToCFile_Text(f_stream, "temp_data[t][j][k][0] = testdata[i+t][j][k][0];", current_indent);
	}
	else {
		AddToCFile_Text(f_stream, "temp_data[t][j][k][0] = testdata[i+t][j][k][0];", current_indent);
		AddToCFile_Text(f_stream, "temp_data[t][j][k][1] = testdata[i+t][j][k][1];", current_indent);
		AddToCFile_Text(f_stream, "temp_data[t][j][k][2] = testdata[i+t][j][k][2];", current_indent);
	}
	--current_indent;
	if(Layers[0]->input_size_z != 1) AddToCFile_Text(f_stream, "}", current_indent);

	AddToCFile_EmptyLine(f_stream);

	-- --current_indent;

	AddToCFile_Text(f_stream, "t1.Start();", current_indent);
	AddToCFile_EmptyLine(f_stream);

	AddToCFile_Text(f_stream, "if (thread_count == 1)", current_indent);
	AddToCFile_Text(f_stream, "{", current_indent++);
	AddToCFile_Text(f_stream, "Predict(temp_data[0], p+0" + string(fault_simulation?", 0, 0, 0":"") + ");", current_indent);
	AddToCFile_Text(f_stream, "}", --current_indent);
	AddToCFile_Text(f_stream, "else", current_indent);
	AddToCFile_Text(f_stream, "{", current_indent++);
	AddToCFile_Text(f_stream, "for (int t = 0; t < thread_count && i + t < run_range; t++) threads.push_back(thread(Predict, temp_data[t], p+t" + string(fault_simulation?", 0, 0, 0":"") + "));", current_indent);
	AddToCFile_Text(f_stream, "for (auto &th : threads) th.join();", current_indent);
	AddToCFile_Text(f_stream, "threads.clear();", current_indent);
	AddToCFile_Text(f_stream, "}", --current_indent);
	AddToCFile_EmptyLine(f_stream);

	AddToCFile_Text(f_stream, "exec_time_sum += t1.Stop(Timer::S);", current_indent);
	AddToCFile_EmptyLine(f_stream);

	if(store_alanysis_data) AddToCFile_Text(f_stream, "if (0 == RunCounter++) ExportData(filename_elements_s);", current_indent);
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "for (int t = 0; t < thread_count && i + t < run_range; t++) if (testlabels[i + t] == p[t]) truePredict++;", current_indent);
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "i = min(i + thread_count, run_range);", current_indent);
	AddToCFile_Text(f_stream, "PrintProgress(i, truePredict, run_range, thread_count, filename);", current_indent);

	current_indent--;
	AddToCFile_Text(f_stream, "}", current_indent);
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Accuracy: " + to_string(truePredict) + " / " + to_string(run_range) + "(" + FloatToString(truePredict * 1.0 / (run_range) * 100) + "%) ");)", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Average execution time: %s seconds\n", to_string(exec_time_sum / run_range));)", current_indent);
	AddToCFile_EmptyLine(f_stream);

	AddToCFile_Text(f_stream, "ExportData(filename, true);", current_indent);

	if (fault_simulation){
		AddToCFile_Text(f_stream, "} else { //fault_simulation", current_indent);

		string layer_dimentions = "{";
		for(int i = 0; i < layers_size - 1; i++)	{ //No fault injection for the last layer
			if (i > 0) layer_dimentions += ", ";

			if(Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == POOLING2D)	{
				layer_dimentions += to_string(Layers[i]->output_size_x) + "*" + to_string(Layers[i]->output_size_y) + "*" + to_string(Layers[i]->output_size_z);
			}
			else{
				layer_dimentions += to_string(Layers[i]->output_size_x);
			}
		}
		layer_dimentions += "}";

		AddToCFile_Text(f_stream, "vector<vector<int>> fault_list = CreateFaultInjectionList(" + to_string(Layers.size() - 1) + ", " + layer_dimentions + ", sizeof(temp_data[0][0][0][0])*8);", current_indent);
		AddToCFile_Text(f_stream, "int fault_count = fault_list.size();", current_indent);
		AddToCFile_Text(f_stream, R"(AddToLog(filename, "Fault count: %s\n", to_string(fault_count));)", current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "int i = 0;", current_indent);
		AddToCFile_Text(f_stream, "int run_range = image_count;", current_indent);
		AddToCFile_Text(f_stream, "while (i < run_range)", current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		AddToCFile_Text(f_stream, "for(int j = 0; j < " + to_string(Layers[0]->input_size_x) + "; j++)", current_indent++);
		AddToCFile_Text(f_stream, "for(int k = 0; k < " + to_string(Layers[0]->input_size_y) + "; k++)" + string((Layers[0]->input_size_z != 1)?" {":""), current_indent++);
		if(Layers[0]->input_size_z == 1) {
			AddToCFile_Text(f_stream, "temp_data[0][j][k][0] = testdata[i+0][j][k][0];", current_indent);
		}
		else {
			AddToCFile_Text(f_stream, "temp_data[0][j][k][0] = testdata[i+0][j][k][0];", current_indent);
			AddToCFile_Text(f_stream, "temp_data[0][j][k][1] = testdata[i+0][j][k][1];", current_indent);
			AddToCFile_Text(f_stream, "temp_data[0][j][k][2] = testdata[i+0][j][k][2];", current_indent);
		}
		--current_indent;
		if(Layers[0]->input_size_z != 1) AddToCFile_Text(f_stream, "}", current_indent);

		AddToCFile_EmptyLine(f_stream);

		--current_indent;

		AddToCFile_Text(f_stream, "int f = 0;", current_indent);
		AddToCFile_Text(f_stream, "while (f < fault_count)", current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		AddToCFile_Text(f_stream, "t1.Start();", current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "if (thread_count == 1)", current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		AddToCFile_Text(f_stream, "Predict(temp_data[0], p+0, fault_list[f+0][0], fault_list[f+0][1], fault_list[f+0][2]);", current_indent);
		AddToCFile_Text(f_stream, "}", --current_indent);
		AddToCFile_Text(f_stream, "else", current_indent);
		AddToCFile_Text(f_stream, "{", current_indent++);
		AddToCFile_Text(f_stream, "for (int t = 0; t < thread_count && f + t < fault_count; t++) threads.push_back(thread(Predict, temp_data[0], p+t, fault_list[f+t][0], fault_list[f+t][1], fault_list[f+t][2]));", current_indent);
		AddToCFile_Text(f_stream, "for (auto &th : threads) th.join();", current_indent);
		AddToCFile_Text(f_stream, "for (int t = 0; t < thread_count && f + t < fault_count; t++) if (testlabels[i] == p[t]) truePredict++;", current_indent);
		AddToCFile_Text(f_stream, "threads.clear();", current_indent);
		AddToCFile_Text(f_stream, "}", --current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "exec_time_sum += t1.Stop(Timer::S);", current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "f = min(f + thread_count, fault_count);", current_indent);
		AddToCFile_Text(f_stream, "}", --current_indent);

		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "++i;", current_indent);
		AddToCFile_Text(f_stream, "PrintProgress(i * fault_count, truePredict, run_range * fault_count, 1, filename);", current_indent);

		AddToCFile_Text(f_stream, "}", --current_indent);
		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, R"(AddToLog(filename, "Accuracy: " + to_string(truePredict) + " / " + to_string(run_range * fault_count) + "(" + FloatToString(truePredict * 1.0 / (run_range * fault_count) * 100) + "%) ");)", current_indent);
		AddToCFile_Text(f_stream, R"(AddToLog(filename, "Average execution time: %s seconds\n", to_string(exec_time_sum / (run_range * fault_count)));)", current_indent);
		AddToCFile_Text(f_stream, "} //fault_simulation", current_indent);
	}

	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "exec_time_sum_wall_clock = t1_wall_clock.Stop(Timer::S);", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "Total execution time (wall clock): %s\n", FormatTime((int)exec_time_sum_wall_clock));)", current_indent);
	AddToCFile_Text(f_stream, R"(AddToLog(filename, "All Done!");)", current_indent);
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "#ifdef _DEBUG", current_indent);
	AddToCFile_Text(f_stream, "char temp_char = _getche();", current_indent);
	AddToCFile_Text(f_stream, "#endif", current_indent);
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "return 0;", current_indent);
	AddToCFile_Text(f_stream, "}", --current_indent);
	AddToCFile_Text(f_stream, "#endif //_HLS_RUN", current_indent);
}

void GenerateHFileDataTypes() {
	bool data_type_mode_all = data_type_mode_floating_point && data_type_mode_fixed_point_single && data_type_mode_fixed_point_multi;

	cout << "*********************************************************" << endl;
	if (data_type_mode_all)
		cout << "Remember to set DataType. Default: float" << endl;
	cout << "Output location: " << output_dir << endl;
	cout << "*********************************************************" << endl;

	ofstream f_stream;
	string f_location = output_dir + "data-types.h";
	f_stream.open(f_location, ios::out);
	string temp_string;

	bool quantized = data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t";

	//AddToCFile_Text(f_stream, "#pragma once");
	AddToCFile_Text(f_stream, "#ifndef _DATA_TYPES_H");
	AddToCFile_Text(f_stream, "#define _DATA_TYPES_H");


	AddToCFile_EmptyLine(f_stream);

	if (data_type_mode_all) {
		AddToCFile_Text(f_stream, "#define FLOAT_DATATYPE");

		temp_string = "//#define FIXEDPOINT_DATATYPE_SINGLE"; 
		AddToCFile_Text(f_stream, temp_string);

		temp_string = "//#define FIXEDPOINT_DATATYPE_MULTI"; 
		AddToCFile_Text(f_stream, temp_string);
		AddToCFile_EmptyLine(f_stream);
	}

	if (add_main_function && (data_type_mode_fixed_point_single || data_type_mode_fixed_point_multi) && data_type_mode_detail != "eight-bit-int" && data_type_mode_detail != "default_int8_t") {
		AddToCFile_Text(f_stream, "#ifdef _MSC_VER");
		AddToCFile_Text(f_stream, "#include <algorithm>");
		AddToCFile_Text(f_stream, "#endif");
		AddToCFile_EmptyLine(f_stream);
	}

	if (data_type_mode_all) AddToCFile_EmptyLine(f_stream);

	if (data_type_mode_floating_point) {
		if (data_type_mode_all) AddToCFile_Text(f_stream, "#ifdef FLOAT_DATATYPE");
		//AddToCFile_Text(f_stream, "typedef float DataType;");
		//AddToCFile_Text(f_stream, "typedef float DataType_Zero;");
		AddToCFile_Text(f_stream, "typedef float DataType_relu;");
		AddToCFile_Text(f_stream, "");
		AddToCFile_Text(f_stream, "typedef float DataType_input; ");
		/*if (biases_enabled)*/ AddToCFile_Text(f_stream, "typedef float DataType_biases;");
		AddToCFile_Text(f_stream, "typedef float DataType_weights; ");

		for (size_t i = 0; i < Layers.size(); i++) {
			if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) {
				AddToCFile_Text(f_stream, "");
				AddToCFile_Text(f_stream, "typedef float DataType_temp_element" + to_string(i + 1) + ";");
				//if (Layers[i]->layer_type == CONV2D && Layers[i]->padding_type == PADDING_SAME) AddToCFile_Text(f_stream, "typedef float DataType_Zero" + to_string(i + 1) + ";");
			}
			AddToCFile_Text(f_stream, "typedef float DataType_" + ((i == Layers.size() - 1) ? ((string)"output") : ("Layer" + to_string(i + 1))) + ";");
		}
		if (data_type_mode_all) AddToCFile_Text(f_stream, "#endif");
		AddToCFile_EmptyLine(f_stream);
	}

	//if data-type-mode-detail exists, minor differences are applied here
	//relevant "data-type-mode-detail": "eight-bit-int"

	if (data_type_mode_fixed_point_single) {
		if (data_type_mode_all) AddToCFile_Text(f_stream, "#ifdef FIXEDPOINT_DATATYPE_SINGLE");
		if (!quantized) {
			if (add_main_function) {
				AddToCFile_Text(f_stream, "#ifndef __linux__");
				AddToCFile_Text(f_stream, "#pragma warning(push, 0)");
				AddToCFile_Text(f_stream, "#endif");
			}
			AddToCFile_Text(f_stream, "#include <ap_fixed.h>");
			if (add_main_function) {
				AddToCFile_Text(f_stream, "#ifndef __linux__");
				AddToCFile_Text(f_stream, "#pragma warning(pop)");
				AddToCFile_Text(f_stream, "#endif");
			}
			AddToCFile_EmptyLine(f_stream);
		}

		if (!quantized) {
			if (network_guess == "lenet") {
				AddToCFile_Text(f_stream, "#define FX_SIZE_W 		16");
				AddToCFile_Text(f_stream, "#define FX_SIZE_I 		7");
			}
			else if (network_guess == "vgg") {
				AddToCFile_Text(f_stream, "#define FX_SIZE_W 		26");
				AddToCFile_Text(f_stream, "#define FX_SIZE_I 		16");
			}
			else if (network_guess == "vgg-scalehls")	{
				AddToCFile_Text(f_stream, "#define FX_SIZE_W 		8");
				AddToCFile_Text(f_stream, "#define FX_SIZE_I 		8");
			}
			else {
				AddToCFile_Text(f_stream, "#define FX_SIZE_W 		16");
				AddToCFile_Text(f_stream, "#define FX_SIZE_I 		16");
			}
		}
		else {
				AddToCFile_Text(f_stream, "#include <stdint.h>");
		}

		AddToCFile_EmptyLine(f_stream);
		if (!quantized)	{
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W,FX_SIZE_I> DataType; //Total: FX_SIZE_W bits, decimal: FX_SIZE_I bits");
			//AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W*2,FX_SIZE_I*2> DataType_Zero; //Total: FX_SIZE_W bits, decimal: FX_SIZE_I bits");
		}
		else if (data_type_mode_detail == "default_int8_t") {
			AddToCFile_Text(f_stream, "typedef int8_t DataType_short;");
			AddToCFile_Text(f_stream, "typedef int64_t DataType;");
			//AddToCFile_Text(f_stream, "typedef int8_t DataType_Zero;");
		}
		else {
			AddToCFile_Text(f_stream, "typedef int8_t DataType_short;");
			if (network_guess == "alexnet") AddToCFile_Text(f_stream, "typedef int32_t DataType;"); //unknown error with int64_t, almost the same accuracy
			else AddToCFile_Text(f_stream, "typedef int64_t DataType;");
			
			//AddToCFile_Text(f_stream, "typedef int64_t DataType_Zero;");
		}

		AddToCFile_EmptyLine(f_stream);
		AddToCFile_Text(f_stream, "typedef DataType DataType_relu;");
		AddToCFile_EmptyLine(f_stream);
		if (data_type_mode_detail != "eight-bit-int" && data_type_mode_detail != "default_int8_t")	{
			AddToCFile_Text(f_stream, "typedef DataType DataType_input;");
			AddToCFile_Text(f_stream, "typedef DataType DataType_weights;");
		/* if (biases_enabled) */ AddToCFile_Text(f_stream, "typedef DataType DataType_biases;");
		}
		else {
			AddToCFile_Text(f_stream, "typedef DataType_short DataType_input;");
			AddToCFile_Text(f_stream, "typedef DataType_short DataType_weights;");
		/* if (biases_enabled) */ AddToCFile_Text(f_stream, "typedef DataType DataType_biases;");
		}

		for (size_t i = 0; i < Layers.size(); i++) {
			bool conv_or_dense = Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE;
			bool last_layer = i == Layers.size() - 1;
			string layer_output_name = (last_layer ? ((string)"output") : ("Layer" + to_string(i + 1)));

			AddToCFile_EmptyLine(f_stream);

			if (conv_or_dense) {
				AddToCFile_Text(f_stream, "typedef DataType DataType_temp_element" + to_string(i + 1) + ";");
				//if (Layers[i]->layer_type == CONV2D && Layers[i]->padding_type == PADDING_SAME) AddToCFile_Text(f_stream, "typedef DataType_Zero DataType_Zero" + to_string(i + 1) + ";");
			}

			if (conv_or_dense) {
				AddToCFile_Text(f_stream, "typedef DataType DataType_" + layer_output_name + ";");
				if (quantized)
					AddToCFile_Text(f_stream, "typedef DataType_short DataType_" + layer_output_name + "_short;");
			}
			else {
				if (quantized) {
					AddToCFile_Text(f_stream, "typedef DataType_short DataType_" + layer_output_name + "_short;");
					AddToCFile_Text(f_stream, "typedef DataType_short DataType_" + layer_output_name + ";"); //Just used in TypeName print
				}
				else {
					AddToCFile_Text(f_stream, "typedef DataType DataType_" + layer_output_name + ";");
				}
			}
		}

		if (quantized) {
			AddToCFile_EmptyLine(f_stream);
			AddToCFile_Text(f_stream, "#define QUANTIZATION_FACTORS"); //Used in the simulator to disable the part that reads the factors
			AddToCFile_Text(f_stream, "typedef DataType_short DataType_IZP;");
			AddToCFile_Text(f_stream, "typedef DataType_short DataType_OZP;");
			AddToCFile_Text(f_stream, "typedef float DataType_ISF;");
			AddToCFile_Text(f_stream, "typedef float DataType_OSF;");
			AddToCFile_Text(f_stream, "typedef float DataType_WSF;");
		}

		if (data_type_mode_all) AddToCFile_Text(f_stream, "#endif");
		AddToCFile_EmptyLine(f_stream);
	}

	if (data_type_mode_fixed_point_multi) {
		if (data_type_mode_all) AddToCFile_Text(f_stream, "#ifdef FIXEDPOINT_DATATYPE_MULTI");
		if (add_main_function) {
			AddToCFile_Text(f_stream, "#ifndef __linux__");
			AddToCFile_Text(f_stream, "#pragma warning(push, 0)");
			AddToCFile_Text(f_stream, "#endif");
		}
		AddToCFile_Text(f_stream, "#include <ap_fixed.h>");
		if (add_main_function) {
			AddToCFile_Text(f_stream, "#ifndef __linux__");
			AddToCFile_Text(f_stream, "#pragma warning(pop)");
			AddToCFile_Text(f_stream, "#endif");
		}
		AddToCFile_EmptyLine(f_stream);
		if (network_guess == "lenet") {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		16");
		}
		else if (network_guess == "vgg") {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		26");
		}
		else if (network_guess == "vgg-scalehls") {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		8");
		}
		else {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		16");
		}

		AddToCFile_EmptyLine(f_stream, 2);

		AddToCFile_Text(f_stream, "//Relu: Max temp_element W + 1, Max temp_element I + 1");

		if (network_guess == "lenet") {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		16");
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W+?, ?> DataType_relu; //Total: FX_SIZE_W bits, decimal: FX_SIZE_I bits");
		}
		else if (network_guess == "vgg") {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		26");
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W+2+1, 16+1> DataType_relu; //Total: FX_SIZE_W bits, decimal: FX_SIZE_I bits");
		}
		else if (network_guess == "vgg-scalehls") {
			AddToCFile_Text(f_stream, "#define FX_SIZE_W 		8");
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, 8> DataType_relu; //Total: FX_SIZE_W bits, decimal: FX_SIZE_I bits");
		}
		else {
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W+?, FX_SIZE_I> DataType_relu; //Total: FX_SIZE_W bits, decimal: FX_SIZE_I bits");
		}
		AddToCFile_EmptyLine(f_stream);

		int datatype_weights_size = 0;
		int datatype_previous_layer_size = 0;

		if (network_guess == "lenet") {
			datatype_weights_size = 2;
			datatype_previous_layer_size = 1; //same as input
			AddToCFile_Text(f_stream, "typedef ap_ufixed<FX_SIZE_W, 1> DataType_input;"); //Note the u: lenet(unsigned), vgg(signed)
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W_TEMP, " + to_string(datatype_weights_size) + "> DataType_weights;");
			/*if (biases_enabled)*/ AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W_TEMP, 2> DataType_biases;");
		}
		else if (network_guess == "vgg") {
			datatype_weights_size = 2;
			datatype_previous_layer_size = 9; //same as input
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, 9> DataType_input;");
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, " + to_string(datatype_weights_size) + "> DataType_weights;");
			/*if (biases_enabled)*/AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, 5> DataType_biases;");
		}
		else if (network_guess == "vgg-scalehls") {
			datatype_weights_size = 8;
			datatype_previous_layer_size = 8; //same as input
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, 8> DataType_input;");
			AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, " + to_string(datatype_weights_size) + "> DataType_weights;");
			/*if (biases_enabled)*/ AddToCFile_Text(f_stream, "typedef ap_fixed<FX_SIZE_W, 5> DataType_biases;");
		}

		int integer_part_size = 0;
		int integer_part_size_temp_element = 0;

		for (size_t i = 0; i < Layers.size(); i++) {
			if (network_guess == "lenet") {
				switch (i + 1) {
				case 1: integer_part_size = 4; break;
				case 3: integer_part_size = 5; break;
				case 6: integer_part_size = 5; break;
				case 7: integer_part_size = 5; break;
				case 8: integer_part_size = 6; break;
				default: break; //Keep previous size when tha layer is flatten or pooling
				}
				integer_part_size_temp_element = integer_part_size + 1;
			}
			else if (network_guess == "vgg") {
				switch (i + 1) {
				case 1: integer_part_size = 11; break;
				case 2: integer_part_size = 13; break;

				case 4: integer_part_size = 13; break;
				case 5: integer_part_size = 14; break;

				case 7: integer_part_size = 15; break;
				case 8: integer_part_size = 14; break;
				case 9: integer_part_size = 15; break;

				case 11: integer_part_size = 14; break;
				case 12: integer_part_size = 13; break;
				case 13: integer_part_size = 13; break;

				case 15: integer_part_size = 12; break;
				case 16: integer_part_size = 11; break;
				case 17: integer_part_size = 10; break;


				case 20: integer_part_size = 7; break;
				case 21: integer_part_size = 5; break;
				case 22: integer_part_size = 5; break;

				default: break; //Keep previous size when the layer is flatten or pooling
				}
				integer_part_size_temp_element = integer_part_size + 1;

				switch (i + 1) {
				case 4:
				case 20:
				case 21:
					integer_part_size_temp_element++; break;

				default: break; //Keep previous size when the layer is flatten or pooling
				}
			}
			else if (network_guess == "vgg-scalehls") {
				integer_part_size = 8;
				integer_part_size_temp_element = integer_part_size + 1;
			}

			string data_type_temp_element = "ap_fixed<FX_SIZE_W+" + to_string(integer_part_size_temp_element - integer_part_size) + ", " + to_string(integer_part_size) + "+" + to_string(integer_part_size_temp_element - integer_part_size) + ">"; //used for temp_element
			string data_type_ufixed = "ap_ufixed<FX_SIZE_W, " + to_string(integer_part_size) + ">"; //used for Layer output

			string data_type_fixed_zero = "ap_fixed<FX_SIZE_W*2, " + to_string(datatype_previous_layer_size) + "+" + to_string(datatype_weights_size) + ">"; //used for temp_element

			if (Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE) {
				AddToCFile_Text(f_stream, "");
				AddToCFile_Text(f_stream, "typedef " + data_type_temp_element + " DataType_temp_element" + to_string(i + 1) + ";");
				//if (Layers[i]->layer_type == CONV2D && Layers[i]->padding_type == PADDING_SAME)  AddToCFile_Text(f_stream, "typedef " + data_type_fixed_zero + " DataType_Zero" + to_string(i + 1) + ";");
			}
			AddToCFile_Text(f_stream, "typedef " + data_type_ufixed + " DataType_" + ((i == Layers.size() - 1) ? ((string)"output") : ("Layer" + to_string(i + 1))) + ";");
			datatype_previous_layer_size = integer_part_size;
		}
		if (data_type_mode_all) AddToCFile_Text(f_stream, "#endif");
		AddToCFile_EmptyLine(f_stream, 2);
	}

	AddToCFile_Text(f_stream, "#endif //_DATA_TYPES_H");

	f_stream.close();
}

void GenerateHFileParamList() {
	ofstream f_stream;
	string f_location = output_dir + "param-list.h";
	f_stream.open(f_location, ios::out);
	string temp_string;

  int layers_size = Layers.size();
	bool quantized = data_type_mode_fixed_point_single && (data_type_mode_detail == "eight-bit-int" || data_type_mode_detail == "default_int8_t");

	//AddToCFile_Text(f_stream, "#pragma once");
	AddToCFile_Text(f_stream, "#ifndef _PARAM_LIST_H");
	AddToCFile_Text(f_stream, "#define _PARAM_LIST_H");
	
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "#include \"data-types.h\"");
	AddToCFile_EmptyLine(f_stream);

	if (add_main_function) {
		AddToCFile_Text(f_stream, "#ifndef _HLS_RUN");
		if (quantized) {
			bool conv_or_dense ;
			int conv_or_dense_count = 0;
			for(int i = 0; i < layers_size; i++) {
				conv_or_dense = Layers[i]->layer_type == CONV2D || Layers[i]->layer_type == DENSE;
				if (conv_or_dense) ++conv_or_dense_count;
			}
			AddToCFile_Text(f_stream, "DataType_IZP input_zero_points_src[" + to_string(conv_or_dense_count) + "];");
			AddToCFile_Text(f_stream, "DataType_OZP output_zero_points_src[" + to_string(conv_or_dense_count) + "];");
			AddToCFile_Text(f_stream, "DataType_ISF input_scale_factors_src[" + to_string(conv_or_dense_count) + "];");
			AddToCFile_Text(f_stream, "DataType_OSF output_scale_factors_src[" + to_string(conv_or_dense_count) + "];");
			AddToCFile_EmptyLine(f_stream);
		}

		for(int i = 0; i < layers_size; i++)	{
			f_stream  << "//" << LayerTypesToString(Layers[i]->layer_type) << endl;
			if(Layers[i]->layer_type == CONV2D)	{
				f_stream  << "DataType_weights weights_" << i + 1 << "_src" << "[" + to_string(Layers[i]->kernel_size_rows) + "]"
																																		<< "[" + to_string(Layers[i]->kernel_size_cols) + "]"
																																		<< "[" + to_string(Layers[i]->input_size_z) + "]"
																																		<< "[" + to_string(Layers[i]->output_size_z) + "]"
																													<< ";" << endl;
				/*if (biases_enabled)*/ f_stream  << "DataType_biases biases_" << i + 1 << "_src" << "[" + to_string(Layers[i]->output_size_z) + "];" << endl;

				if (quantized) {
					temp_string = "DataType_WSF weight_scales_#_src[@];";
					temp_string = StringSubstituteAll(temp_string, "#", to_string(i + 1));
					temp_string = StringSubstituteAll(temp_string, "@", to_string(Layers[i]->output_size_z));
					AddToCFile_Text(f_stream, temp_string);
				}

				AddToCFile_EmptyLine(f_stream);
			}
			else if (Layers[i]->layer_type == DENSE) {
				f_stream  << "DataType_weights weights_" << i + 1 << "_src" << "[" + to_string(Layers[i]->input_size_x) + "]"
																																		<< "[" + to_string(Layers[i]->output_size_x) + "]"
																													<< ";" << endl;
				/*if (biases_enabled)*/ f_stream  << "DataType_biases biases_" << i + 1 << "_src" << "[" + to_string(Layers[i]->output_size_x) + "];" << endl;

				if (quantized) {
					temp_string = "DataType_WSF weight_scales_#_src[@];";
					temp_string = StringSubstituteAll(temp_string, "#", to_string(i + 1));
					temp_string = StringSubstituteAll(temp_string, "@", to_string(Layers[i]->output_size_x));
					AddToCFile_Text(f_stream, temp_string);
				}

				AddToCFile_EmptyLine(f_stream);
			}
			else {
				f_stream  << "void *weights_" << i + 1 << "_src" << ";" << endl;
				/*if (biases_enabled)*/ f_stream  << "void *biases_" << i + 1 << "_src" << ";" << endl;
				if(quantized) f_stream  << "void *weight_scales_" << i + 1 << "_src" << ";" << endl;
				f_stream  << endl;
			}
		}

		f_stream  << endl << endl << "/////////////////////////////" << endl;
		for(size_t i = 0; i < 50 - Layers.size(); i++) {
			f_stream  << "void *weights_" << i + 1 + Layers.size() << "_src" << ";" << endl;
			/*if (biases_enabled)*/ f_stream  << "void *biases_" << i + 1 + Layers.size() << "_src" << ";" << endl;
			if(quantized) f_stream  << "void *weight_scales_" << i + 1 + Layers.size() << "_src" << ";" << endl;
			f_stream  << endl;
		}
		AddToCFile_Text(f_stream, "#endif //_HLS_RUN");
	}

	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, "#endif //_PARAM_LIST_H");
	
	f_stream.close();
}

string LayerDataLocation(int pLayerNumber) {
  ASSERT(pLayerNumber >= 0);

	if (map_options.count("layer-data-location")) return map_options["layer-data-location"];

	if (network_guess == "lenet") return "local";
	if (network_guess == "vgg") {
		return "port";
		
		//if(pLayerNumber < 18) return "port";
		//else return "local";
	}

	return "local";
}

/*
"design-source": "keras-file" or "keras-text"
"layer-config": [{
		"layer-number": 1,
		"type": "Conv2D",
		"filters": 64,
		"kernel_size": "(3, 3)",
		"padding": "same",
		"activation": "relu",
		"input_shape": "(32, 32, 3)"

"keras-source-file": "C:\\Projects\\KerasToCWorkspace\\output_arch.py",

In Json file
"keras-source-text": [
	"model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))",
	"model.add(Flatten())",
	, ..., ...
],
In command line
	"model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))#model.add(Flatten())",
*/

void ReadOptions(int argc, char* argv[]) {
	running_in_vs_environment = false;
  string temp_string;
  #ifndef __linux__
	temp_string = std::filesystem::current_path().string() + "\\";
	if (FileExists(temp_string + "DeepHLS.sln")) //As an indicator file to detect if program is running in the Visual Studio GUI
		running_in_vs_environment = true;

	if (running_in_vs_environment) {
		//Check if VS_ON_VM is defined
		INFOLOG("Running in VS environment");
	}
  #endif

	INFOLOG("Current directory: " + CurrentDirectory());
  extern string workspace_dir;
	INFOLOG("Workspace directory: " + workspace_dir);
  extern string output_dir;
	INFOLOG("Output directory: " + output_dir);

	map<string, string> command_line_arguments;
	string parameter, value;
	int i = 1; //the first argument is the executable file name
	while (i < argc) {
		parameter = argv[i];
		++i;
		parameter = parameter.substr(1);
		if (i == argc || argv[i][0] == '-') {
			command_line_arguments.insert(make_pair(parameter, "ACTIVE"));
			INFOLOG("Parameter: " + parameter + ", Value: " + "ACTIVE");
			continue;
		}

		value = argv[i];
		++i;
		INFOLOG("Parameter: " + parameter + ", Value: " + value);
		command_line_arguments.insert(make_pair(parameter, StringSubstituteAll(value, "#", "\n")));
	}

	map_options.insert(command_line_arguments.begin(), command_line_arguments.end());

	if (map_options.count("options-json-file")) {
		map_options["options-json-file"] = FindFileFullPath(map_options["options-json-file"]);
		if (!FileExists(map_options["options-json-file"])) {
			ERRORLOGT("options-json-file does not exist: " + map_options["options-json-file"]);
			exit(0);
		}
		else {
			string temp_string = map_options["options-json-file"];
			INFOLOG("Options json file: " + temp_string);
			ifstream json_file_stream(map_options["options-json-file"]);
			json json_object;
			try
			{
				json_file_stream >> json_object;
			}
			catch (const std::exception& e)
			{
				ERRORLOGT(string("Reading options-json-file failed.") + e.what());
			}

			//Note: json iterators are not in the order of the json file
			for (auto json_iterator = json_object.begin(); json_iterator != json_object.end(); ++json_iterator)	{
				string json_iterator_key = json_iterator.key();

				if (	 json_iterator_key == "keras-source-file" 
						|| json_iterator_key == "design-source" 
						|| json_iterator_key == "output-directory"
						|| json_iterator_key == "network-name" 
						|| json_iterator_key == "layer-data-location" //local, ports
						|| json_iterator_key == "data-type-mode" //floating-point, fixed-point-single, fixed-point-multi, all-modes (default)
						|| json_iterator_key == "data-type-mode-detail" //eight-bit-int, ...
						|| json_iterator_key == "loop-hierarchy-labels" //numbers, names
						|| json_iterator_key == "single-layer" //"1", "7", or 1, 7
						|| json_iterator_key == "approximate-multipliers-configuration" //e.g., "10100111"
						|| json_iterator_key == "approximate-multipliers-type" //e.g., "base", "mul8s_1L12"
						|| json_iterator_key == "create-deepcl-config-h" //e.g., "base", "mul8s_1L12"
				
						//If present in the JSON file, the value should have been set as "ACTIVE"
						|| json_iterator_key == "store-analysis-data" 
						|| json_iterator_key == "disable-biases" //The network will just have weights, not biases.
						|| json_iterator_key == "add-main-function" //Add main() and predict() functions
						|| json_iterator_key == "dump-layers" //on screen and file
						|| json_iterator_key == "fault-simulation"
						|| json_iterator_key == "approximate-multipliers"
					) {
					ASSERT(json_iterator.value().is_string());

					if (map_options.count(json_iterator_key) != 0)
						INFOLOG(json_iterator_key + " in options-json-file ignored.");
					else {
						map_options[json_iterator_key] = json_iterator.value().get<string>();
						INFOLOG("JSON entry: " + json_iterator_key + ", Value: " + map_options[json_iterator_key]);
					}
				}
				else if (json_iterator_key == "keras-source-text" || json_iterator_key == "loop-orders") {
					if (map_options.count(json_iterator_key) != 0) {
						INFOLOG(json_iterator_key + " in options-json-file ignored.");
					}
					else {
						string json_iterator_value = "";
						if (json_iterator.value().is_string()) {
							//Single layer
							json_iterator_value = json_iterator.value().get<string>();
						}
						else {
							ASSERT(json_iterator.value().is_array());
							for (auto& it : json_iterator.value().items())
							{
								ASSERT(it.value().is_string());
								//cout << it.value().get<string>() << endl;
								if (!json_iterator_value.empty()) json_iterator_value += "\n";
								json_iterator_value += it.value().get<string>();
							}
						}
						map_options[json_iterator_key] = json_iterator_value;
						INFOLOG("JSON entry: " + json_iterator_key + ", Value: " + (json_iterator.value().is_string()?"":"\n") + map_options[json_iterator_key]);
					}

					//Save keras source to temp: NO, later, after that all processing is done, if necessary
				}
				else if (json_iterator.key() == "layer-config") {
					ERRORLOGT("layer-config parameter not supported yet. It can be implemented using keras-source-text");
				}
			}
		}
	}

	if (!map_options.count("design-source")) {
		if (!map_options.count("keras-source-text")) map_options["design-source"] = "keras-file";
		else {
			map_options["design-source"] = "keras-text";
		}
	}

	if (map_options.count("design-source") && map_options["design-source"] == "keras-source-text") {
		if (!map_options.count("keras-source-text")) {
			ERRORLOGT("design-source is keras-text, but no keras-source-text is specified");
		}
		else {
			string temp_file_save_location = GetWorkspaceDir() + "keras-source.py";
			if (FileExists(temp_file_save_location)) INFOLOG(temp_file_save_location + " already exists. It will be overwritten.");
			INFOLOG("design-source is keras-text. Temporary source file created: " + temp_file_save_location);
			StringToFile(map_options["keras-source-text"], temp_file_save_location);
		}
	}

	ASSERT(map_options.count("design-source"));
	
	if (map_options["design-source"] == "keras-source-text") {
		input_file_name = GetWorkspaceDir() + "keras-source.py";
	}
	else if (map_options["design-source"] == "keras-file" && map_options.count("keras-source-file")) {
		input_file_name = map_options["keras-source-file"];
	}
	else if (map_options["design-source"] == "keras-file" && !map_options.count("keras-source-file")) {
		bool default_input_file_exists = FileExists(input_file_name);
		string current_directory_input_file = CurrentDirectory() + "keras-source.py";
		bool current_directory_input_file_exists = FileExists(current_directory_input_file);

		if (!default_input_file_exists && !current_directory_input_file_exists)	{
			ERRORLOGT("No input file set.");
			ERRORLOGT("default input file: " + input_file_name);
			ERRORLOGT("current directory input file: " + current_directory_input_file);
		}
		else {
			if (default_input_file_exists) {
				//backward compatibility
				input_file_name = input_file_name;
			}
			else if (FileExists(CurrentDirectory() + "keras-source.py")) {
				input_file_name = CurrentDirectory() + "keras-source.py";
			}
			else {
				ASSERTA;
			}
		}
	}

	INFOLOG("keras-source-file: " + input_file_name);

	if (!map_options.count("output-directory")) map_options["output-directory"] = GetWorkspaceDir();
	temp_string = map_options["output-directory"];

  #ifndef __linux__
  string directory_delimiter = "\\";
  #else
  string directory_delimiter = "/";
  #endif
  
	if (temp_string.substr(temp_string.size() - 1, 1) != directory_delimiter) temp_string += directory_delimiter;
	map_options["output-directory"] = temp_string;
	INFOLOG("Output directory: " + map_options["output-directory"]);
	output_dir = map_options["output-directory"];

	if (!map_options.count("log-location")) map_options["log-location"] = GetWorkspaceDir() + "DeepHLS.log";
	log_location = map_options["log-location"];
	INFOLOG("Log location: " + map_options["log-location"]);

	network_name = "";
	if (map_options.count("network-name")) {
		network_name = map_options["network-name"];
		INFOLOG("Network name: " + network_name);
	}

	if (map_options.count("disable-biases")) biases_enabled = false; else biases_enabled = true;
	if (map_options.count("store-analysis-data")) store_alanysis_data = true; else store_alanysis_data = false;
	if (map_options.count("add-main-function")) add_main_function = true; else add_main_function = false;
	if (map_options.count("dump-layers")) dump_layers = true; else dump_layers = false;
	if (map_options.count("fault-simulation")) fault_simulation = true; else fault_simulation = false;

	if (map_options.count("approximate-multipliers")) approximate_multipliers = true; else approximate_multipliers = false;
	if (map_options.count("approximate-multipliers-configuration")) {
		approximate_multipliers_configuration = map_options["approximate-multipliers-configuration"];
	}
	else {
		approximate_multipliers_configuration = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
	}

	if (map_options.count("approximate-multipliers-type")) {
		approximate_multipliers_type = map_options["approximate-multipliers-type"]; 
	}
	else {
		approximate_multipliers_type = "base";
	}

	if (store_alanysis_data && !add_main_function) {
		ERRORLOGT("store-analysis-data will be ignored since add-main-function is not active");
		store_alanysis_data = false;
	}

	if (fault_simulation && !add_main_function) {
		ERRORLOGT("fault_simulation will be ignored since add-main-function is not active");
		fault_simulation = false;
	}

	if (store_alanysis_data && map_options.count("loop-orders")) {
		ERRORLOGT("store-analysis-data will be ignored since it is not supported when loop-orders are selected");
		store_alanysis_data = false;
	}

	data_type_mode_floating_point = data_type_mode_fixed_point_single = data_type_mode_fixed_point_multi = false;
	if (!map_options.count("data-type-mode")) map_options["data-type-mode"] = "all-modes"; 
	if (map_options["data-type-mode"] == "all-modes") {
		data_type_mode_floating_point = data_type_mode_fixed_point_single = data_type_mode_fixed_point_multi = true;
	}
	else if (map_options["data-type-mode"] == "floating-point") data_type_mode_floating_point = true;
	else if (map_options["data-type-mode"] == "fixed-point-single") data_type_mode_fixed_point_single = true;
	else if (map_options["data-type-mode"] == "fixed-point-multi") data_type_mode_fixed_point_multi = true;

	if (map_options.count("data-type-mode-detail"))
		data_type_mode_detail = map_options["data-type-mode-detail"];

	if (map_options.count("loop-orders")) {
		loop_orders = SplitString(map_options["loop-orders"], "\n");
	}

	if (map_options.count("loop-hierarchy-labels") && map_options["loop-hierarchy-labels"] == "numbers") numbered_loop_labels = true; else numbered_loop_labels = false;
	if (map_options.count("single-layer")) {
		if (add_main_function) {
			ERRORLOGT("add-main-function will be ignored because single-layer is active.");
			add_main_function = false;
		}
		single_layer = stoi(map_options["single-layer"]);
	}
	else single_layer = 0;

	if (map_options.count("create-deepcl-config-h")) create_deepcl_config_h = true; else create_deepcl_config_h = false;

	return;
}

void SetNetworkGuess() {
	if (network_name != "") network_guess = network_name;
	else if (Layers.size() < 10) network_guess = "lenet";
	else if (Layers[0]->input_size_x == 32 && Layers.size() > 14) network_guess = "vgg-scalehls";
	else if (Layers[0]->input_size_x == 32 && Layers.size() == 13) network_guess = "alexnet";
	else if (Layers.size() > 14) network_guess = "vgg";
}

void SetArbitraryParameters() {
	if (network_name == "vgg-scalehls" || network_guess == "vgg-scalehls") {
		biases_enabled = false;
		data_type_mode_fixed_point_single = true;
		data_type_mode_floating_point = false;
		data_type_mode_fixed_point_multi = false;
		dump_layers = false;
	}

	//INFOLOG("Arbitrary parameter is set for ... ");
	//store_alanysis_data = false;
	//add_main_function = false;
}

void CheckAndCorrectLoopOrders() {
	if (Layers.size() != loop_orders.size()) {
		if (loop_orders.size() != 0) {
			ERRORLOGT("loop-orders argument is not set correctly. Will be ignored. Number of layers: " + to_string(Layers.size()) + ", number of loop-orders: " + to_string(loop_orders.size()));
			loop_orders.clear();
		}

		for (auto layer : Layers)	{
			if (layer->layer_type == CONV2D) loop_orders.push_back("oz-oy-ox-iz-kx-ky");
			else if (layer->layer_type == POOLING2D) loop_orders.push_back("oz-oy-ox-kx-ky");
			else if (layer->layer_type == FLATTEN) loop_orders.push_back("oz-oy-ox");
			else if (layer->layer_type == DENSE) loop_orders.push_back("ox-ix");
			else ASSERTA;
		}
	}
	else {
		int i = 0;
		for (auto layer : Layers)	{
			if (layer->layer_type == CONV2D) {
				string default_loop_order = "oz-oy-ox-iz-kx-ky";
				if (loop_orders[i] == "default" || loop_orders[i] == "*") loop_orders[i] = default_loop_order;
				else if (loop_orders[i].substr(0, 2) != "oz") {ERRORLOGT("Incorrect loop-order argument will be ignored. Only oz as top loop is supported"); loop_orders[i] = default_loop_order;}
				else if (loop_orders[i].substr(0, 2) != "oz") {ERRORLOGT("Incorrect loop-order argument will be ignored. Only oz as top loop is supported"); loop_orders[i] = default_loop_order;}
				else if (loop_orders[i].size() != default_loop_order.size()) {ERRORLOGT("Incorrect loop-order argument will be ignored."); loop_orders[i] = default_loop_order;}
				else if (loop_orders[i].substr(loop_orders[i].size() - 5) != "kx-ky") {ERRORLOGT("Incorrect loop-order argument will be ignored."); loop_orders[i] = default_loop_order;}
				else if (OccurancesInString(loop_orders[i], '-') != OccurancesInString(default_loop_order, '-')) {ERRORLOGT("Incorrect loop-order argument will be ignored."); loop_orders[i] = default_loop_order;}
			}
			else if (layer->layer_type == POOLING2D) {
				if (loop_orders[i] == "default" || loop_orders[i] == "*") loop_orders[i] = "oz-oy-ox-kx-ky";
				ASSERTT(OccurancesInString(loop_orders[i], '-') == OccurancesInString("oz-oy-ox-kx-ky", '-'), "Incorrect loop-order argument.");
			}
			else if (layer->layer_type == FLATTEN) {
				if (loop_orders[i] == "default" || loop_orders[i] == "*") loop_orders[i] = "oz-oy-ox";
				ASSERTT(OccurancesInString(loop_orders[i], '-') == OccurancesInString("oz-oy-ox", '-'), "Incorrect loop-order argument.");
			}
			else if (layer->layer_type == DENSE) {
				if (loop_orders[i] == "default" || loop_orders[i] == "*") loop_orders[i] = "ox-ix";
				ASSERTT(OccurancesInString(loop_orders[i], '-') == OccurancesInString("ox-ix", '-'), "Incorrect loop-order argument.");
			}
			else ASSERTA;
			
			i++;
		}
	}

	return;
}

void GenerateDeepClConfigH() {
	ofstream f_stream;
	string f_location = output_dir + "layer_config.h";
	f_stream.open(f_location, ios::out);
	string temp_string;
	int current_indent = 0;

	ASSERT(network_guess == "vgg");
	int memrd_src[] = {0, 1, 4, 1, 4, 1, 0, 4, 0, 1, 4, 1, 0, 4, 2, 3};
	int memwr_dst[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2};
	string layer_names[] = {"conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3", "fc6", "fc7", "fc8"};
	AddToCFile_Text(f_stream, R"(#include "../device/hw_param.h")");
	AddToCFile_EmptyLine(f_stream);
	AddToCFile_Text(f_stream, R"(#define NUM_CONFIG_ITEM  25)");
	AddToCFile_EmptyLine(f_stream);

  int layers_size = Layers.size();
	
	temp_string = "unsigned layer_config[][NUM_CONFIG_ITEM] = {{" + ((Layers[0]->layer_type ==  CONV2D) ? string("0") : string("1")) + ",";
	AddToCFile_Text(f_stream, temp_string);

	current_indent = 7;
	int current_conv_or_dense_layer_index = -1;
	Layer *current_layer;

	for (int i = 0; i < layers_size; i++)	{
		current_layer = Layers[i];
		if (current_layer->layer_type == POOLING2D || current_layer->layer_type == FLATTEN) continue;
		current_conv_or_dense_layer_index++;
		bool is_last_layer = (i == layers_size - 1);

		if (i != 0) AddToCFile_Text(f_stream, "{" + ((current_layer->layer_type ==  CONV2D) ? string("0") : string("1")) + ",", current_indent);

		int weights_sizes[4] = {0, 0, 0, 0};
		int input_sizes[3] = {current_layer->input_size_x, current_layer->input_size_y, current_layer->input_size_z};
		int bias_size = 0;
		Layer *previous_layer = (i > 0?Layers[i-1]:nullptr);
		if(current_layer->layer_type == CONV2D) {			
			weights_sizes[0] =  current_layer->kernel_size_rows;
			weights_sizes[1] =  current_layer->kernel_size_cols;
			weights_sizes[2] =  current_layer->input_size_z;
			weights_sizes[3] =  current_layer->output_size_z;
			bias_size = current_layer->output_size_z;
		}
		else if (current_layer->layer_type == DENSE) {
			if (previous_layer && previous_layer->layer_type == FLATTEN) {
				input_sizes[0] = previous_layer->input_size_x;
				input_sizes[1] = previous_layer->input_size_y;
				input_sizes[2] = previous_layer->input_size_z;

				weights_sizes[0] =  previous_layer->input_size_x;
				weights_sizes[1] =  previous_layer->input_size_y;
				weights_sizes[2] =  previous_layer->input_size_z;
				weights_sizes[3] =  current_layer->output_size_x;
				bias_size = current_layer->output_size_x;
			}
			else {
				input_sizes[0] = current_layer->input_size_z;
				input_sizes[1] = current_layer->input_size_y;
				input_sizes[2] = current_layer->input_size_x;

				weights_sizes[0] =  current_layer->input_size_z;
				weights_sizes[1] =  current_layer->input_size_y;
				weights_sizes[2] =  current_layer->input_size_x;
				weights_sizes[3] =  current_layer->output_size_x;
				bias_size = current_layer->output_size_x;
			}
		}

		temp_string = "";
		temp_string += to_string(input_sizes[0]) + ", ";
		temp_string += to_string(input_sizes[1]) + ", ";
		temp_string += to_string(input_sizes[2]) + ", ";

		temp_string += to_string(weights_sizes[0]) + ", ";
		temp_string += to_string(weights_sizes[1]) + ", ";
		temp_string += to_string(weights_sizes[2]) + ", ";
		temp_string += to_string(weights_sizes[3]) + ", ";
		temp_string += to_string(bias_size) + ",";
		AddToCFile_Text(f_stream, temp_string, current_indent);

		AddToCFile_Text(f_stream, to_string(memrd_src[current_conv_or_dense_layer_index]) + ",", current_indent);
		
		temp_string = "";
		int output_sizes[3] = {current_layer->output_size_x, current_layer->output_size_y, current_layer->output_size_z};
		if (current_layer->layer_type == DENSE) {
			output_sizes[0] = current_layer->output_size_z;
			output_sizes[2] = current_layer->output_size_x;
		}

		temp_string += to_string(output_sizes[0]) + ", ";
		temp_string += to_string(output_sizes[1]) + ", ";
		temp_string += to_string(output_sizes[2]) + ", ";
		temp_string += to_string(current_layer->stride_size) + ", ";
		temp_string += to_string(current_layer->padding_type == PADDING_SAME ? 1 : 0) + ", ";
		temp_string += to_string(0) + ", "; //conv_split
		temp_string += to_string(is_last_layer? 0 : 1) + ","; //conv_relu
		AddToCFile_Text(f_stream, temp_string, current_indent);

		temp_string = "";
		bool next_layer_is_pooling = (!is_last_layer && Layers[i+1]->layer_type == POOLING2D);

		Layer *next_layer;
		Layer *temp_layer;
		if (next_layer_is_pooling) {
			next_layer = Layers[i+1];
			temp_layer = next_layer;
		} 
		else {
			next_layer = nullptr;
			temp_layer = current_layer;
		};
		int pool_sizes[3] = {temp_layer->output_size_x, temp_layer->output_size_y, temp_layer->output_size_z};
		if (current_layer->layer_type == DENSE) {
			pool_sizes[0] = current_layer->output_size_z;
			pool_sizes[2] = current_layer->output_size_x;
		}

		temp_string += to_string(next_layer_is_pooling? 1 : 0) + ", ";
		temp_string += to_string(pool_sizes[0]) + ", ";
		temp_string += to_string(pool_sizes[1]) + ", ";
		temp_string += to_string(pool_sizes[2]) + ", ";
		temp_string += to_string(next_layer_is_pooling? next_layer->kernel_size_cols : 0) + ", ";
		temp_string += to_string(next_layer_is_pooling? next_layer->stride_size : 0) + ",";
		AddToCFile_Text(f_stream, temp_string, current_indent);

		AddToCFile_Text(f_stream, "0,", current_indent); //lrn_on

		temp_string = "";
		temp_string += to_string(memwr_dst[current_conv_or_dense_layer_index]);
		temp_string += string("}") + (is_last_layer?"":",") + "//Layer-" + to_string(current_conv_or_dense_layer_index + 1) + " (" + layer_names[current_conv_or_dense_layer_index] + ")";
		AddToCFile_Text(f_stream, temp_string, current_indent);
	}

	AddToCFile_Text(f_stream, "};", current_indent);
	AddToCFile_EmptyLine(f_stream);

	current_indent = 0;
	current_conv_or_dense_layer_index = -1;
	temp_string = "char precision_config[][3] ={";
	for (int i = 0; i < (int)Layers.size(); i++)	{
		Layer *current_layer = Layers[i];
		if (current_layer->layer_type == POOLING2D || current_layer->layer_type == FLATTEN) continue;
		current_conv_or_dense_layer_index++;

		temp_string += "{(char)8, (char) 0, (char)0},//Layer-" + to_string(current_conv_or_dense_layer_index + 1);
		AddToCFile_Text(f_stream, temp_string, current_indent);

		current_indent = 7;
		temp_string = "";
	}
	
	AddToCFile_Text(f_stream, "};", current_indent);
	AddToCFile_EmptyLine(f_stream);

	current_indent = 0;

	current_layer = Layers[0];
	temp_string = "unsigned input_config[4] = {" + to_string(current_layer->input_size_x) + ", " + to_string(current_layer->input_size_y) + ", " + to_string(current_layer->input_size_z) + ", 1};";
	AddToCFile_Text(f_stream, temp_string, current_indent);
	AddToCFile_EmptyLine(f_stream);

	current_conv_or_dense_layer_index = -1;
	for (int i = 0; i < (int)Layers.size(); i++)	{
		Layer *current_layer = Layers[i];
		if (current_layer->layer_type == POOLING2D || current_layer->layer_type == FLATTEN) continue;
		current_conv_or_dense_layer_index++;
	}

	current_layer = Layers[Layers.size() - 1];
	temp_string = "unsigned output_config[3] = {1, 1, 1024};//Layer-" + to_string(current_conv_or_dense_layer_index + 1);
	AddToCFile_Text(f_stream, temp_string, current_indent);

	f_stream.close();
}