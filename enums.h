/*
Copyright 2022 Mohammad Riazati

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef ENUMS_H_
#define ENUMS_H_

#include <string>
using std::string;

enum LayerTypes { CONV2D, POOLING2D, FLATTEN, DENSE };
enum PoolingTypes { NO_POOLING, AVERAGE_POOLING, MAX_POOLING };
enum ActivationFunctions { NO_ACTIVATION, RELU, SOFTMAX, LINEAR };
enum PaddingTypes { NO_PADDING, PADDING_VALID, PADDING_SAME };

string LayerTypesToString(LayerTypes param);
string PoolingTypesToString(PoolingTypes param);
string PaddingTypesToString(PaddingTypes param);
string ActivationFunctionsToString(ActivationFunctions param);


#endif //ENUMS_H_