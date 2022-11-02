// narrowing.cpp

#include <iostream>

#include "vector.h"

int main(){

  //char c1(999);     
  //char c2= 999;
  //std::cout << "c1: " << c1 << std::endl;
  //std::cout << "c2: " << c2 << std::endl;
  
  //int i1(3.14); 
  //int i2= 3.14;
  //std::cout << "i1: " << i1 << std::endl;
  //std::cout << "i2: " << i2 << std::endl;

  Vector<double> v1(2);
  std::cout << "1" << std::endl;
  Vector<double> v2 = v1;
  std::cout << "2" << std::endl;
  Vector<double> v3 = v2;
  std::cout << "3" << std::endl;
  v3 = v1;
  std::cout << "4" << std::endl;


}
