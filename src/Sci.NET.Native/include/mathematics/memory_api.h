//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_MEMORY_API_H
#define SCI_NET_NATIVE_MEMORY_API_H

#include "api.h"

SDN_DLL_EXPORT_API allocate_memory(void **ptr, size_t size);

SDN_DLL_EXPORT_API free_memory(void *ptr);

SDN_DLL_EXPORT_API copy_memory_to_device(void *dst, void *src, size_t size);

SDN_DLL_EXPORT_API copy_memory_to_host(void *dst, void *src, size_t size);

SDN_DLL_EXPORT_API copy_memory_device_to_device(void *dst, void *src, size_t size);

#endif //SCI_NET_NATIVE_MEMORY_API_H
