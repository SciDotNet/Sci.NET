//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_API_STATUS_CODE_H
#define SCI_NET_NATIVE_API_STATUS_CODE_H

enum sdnApiStatusCode {
    sdnStatusUnknown = 0,
    sdnSuccess = 1,
    sdnNotInitialized = 2,
    sdnInvalidValue = 3,
    sdnInternalError = 4,
};

using apiStatusCode_t = sdnApiStatusCode;

#endif //SCI_NET_NATIVE_API_STATUS_CODE_H
