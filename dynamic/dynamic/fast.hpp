#ifndef FAST_H
#define FAST_H
#include<vector>
using std::vector;
typedef unsigned char fast_byte;
struct fast_xy
{
  short x, y;
  fast_xy(short x_, short y_) : x(x_), y(y_) {}
};
void fast_corner_detect_10(const fast_byte* img, int imgWidth, int imgHeight, int widthStep, short barrier, vector<fast_xy>& corners); 

#endif
