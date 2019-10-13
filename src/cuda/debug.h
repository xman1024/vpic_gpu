#ifndef __DEBUG__H__
#define __DEBUG__H__
#include <iostream>

#define DEBUG(x) std::cerr << #x << " = " << x << std::endl;

#endif // __DEBUG__H__