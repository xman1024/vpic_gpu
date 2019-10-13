#ifndef __DEBUG__H__
#define __DEBUG__H__
#include <iostream>

#define DEBUG(x) \
    std::cerr << __FUNCTION__ << ":" << __LINE__ << " " << #x << " = " << x << std::endl;

#endif  // __DEBUG__H__