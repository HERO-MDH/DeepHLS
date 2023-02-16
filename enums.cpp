/*
Copyright 2022 Mohammad Riazati

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include "enums.h"
#include "utils.h"

string LayerTypesToString(LayerTypes param) {
	switch (param)
	{
	case CONV2D:
		return "Conv2D";
		break;
	case POOLING2D:
		return "Pooling2D";
		break;
	case FLATTEN:
		return "Flatten";
		break;
	case DENSE:
		return "Dense";
		break;
	default:
		ERRORLOG;
		return "";
		break;
	}
}

string PoolingTypesToString(PoolingTypes param) {
	switch (param)
	{
	case AVERAGE_POOLING:
		return "AVG";
		break;
	case MAX_POOLING:
		return "MAX";
		break;
	default:
		ERRORLOG;
		return "";
		break;
	}
}

string PaddingTypesToString(PaddingTypes param) {
	switch (param)
	{
	case PADDING_SAME:
		return "same";
		break;
	case PADDING_VALID:
		return "valid";
		break;
	default:
		ERRORLOG;
		return "";
		break;
	}
}

string ActivationFunctionsToString(ActivationFunctions param) {
	switch (param)
	{
	case RELU:
		return "relu";
		break;
	case LINEAR:
		return "linear";
		break;
	case SOFTMAX:
		return "softmax";
		break;
	default:
		ERRORLOG;
		return "";
		break;
	}
}
