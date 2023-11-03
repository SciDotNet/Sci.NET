//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_API_H
#define SCI_NET_NATIVE_API_H

#include <cstdint>
#include "api_status_code.h"
#include "cfloat"
#include "cinttypes"

#define SDN_DLL_EXPORT_API extern "C" apiStatusCode_t __declspec(dllexport) __cdecl


#endif //SCI_NET_NATIVE_API_H
