//
// Created by reece on 17/10/2022.
//

#ifndef SCI_NET_COMMON_API_EXPORT
#define SCI_NET_COMMON_API_EXPORT

#include "SciDotNet/sdnApiStatusCode.h"

#define SDN_DLL_EXPORT_API extern "C" apiStatusCode_t __declspec(dllexport) __cdecl
#define SDN_DLL_EXPORT_CLASS extern "C" class __declspec(dllexport)
#define SDN_DLL_EXPORT_STRUCT extern "C" struct __declspec(dllexport)

#endif //SCI_NET_COMMON_API_EXPORT
