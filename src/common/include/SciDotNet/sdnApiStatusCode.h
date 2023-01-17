//
// Created by reece on 17/10/2022.
//

#ifndef SCI_NET_COMMON_API_STATUS
#define SCI_NET_COMMON_API_STATUS

enum sdnApiStatusCode {
    sdnStatusUnknown = 0,
    sdnSuccess = 1,
    sdnNotInitialized = 2,
    sdnInvalidValue = 3,
    sdnInternalError = 4,
};

using apiStatusCode_t = sdnApiStatusCode;

#endif //SCI_NET_COMMON_API_STATUS
