/****************************************************************************
  FileName     [ myMinHeap.h ]
  PackageName  [ util ]
  Synopsis     [ Define MinHeap ADT ]
  Author       [ Chung-Yang (Ric) Huang ]
  Copyright    [ Copyleft(c) 2014-present LaDs(III), GIEE, NTU, Taiwan ]
****************************************************************************/

#ifndef MY_MIN_HEAP_H
#define MY_MIN_HEAP_H

#include <algorithm>
#include <vector>

template <class Data>
class MinHeap
{
public:
   MinHeap(size_t s = 0) { if (s != 0) _data.reserve(s); }
   ~MinHeap() {}

   void clear() { _data.clear(); }

   // For the following member functions,
   // We don't respond for the case vector "_data" is empty!
   const Data& operator [] (size_t i) const { return _data[i]; }   
   Data& operator [] (size_t i) { return _data[i]; }

   size_t size() const { return _data.size(); }

   // TODO
   const Data& min() const { return _data.front(); }

   void insert(const Data& d) { 
    int MyIndex = _data.size();
    if(_data.empty()){ _data.push_back(d); }
    else{ 
      _data.push_back(d); 
      while( _data[MyIndex]<_data[(MyIndex-1)/2]){
        if(MyIndex==0)return;
        swap( _data[MyIndex], _data[(MyIndex-1)/2] );
        MyIndex = (MyIndex-1)/2;
      }
      return;
    }
   }
   void delMin() { delData(0); }

   void delData(size_t i) {

    int LastIndex = _data.size()-1, temp;
    swap( _data[LastIndex], _data[i] );
    _data.pop_back();
    LastIndex = i;

    //look up
      while( _data[LastIndex]<_data[(LastIndex-1)/2] && LastIndex!=0){
        if(LastIndex==0)break;
        swap( _data[LastIndex], _data[(LastIndex-1)/2] );
        LastIndex = (LastIndex-1)/2;
      }

   //look down
    if(LastIndex*2+1 > _data.size()-1) return;
    while( _data[LastIndex*2+1]< _data[LastIndex]  || ( LastIndex*2+2 <= _data.size()-1 && _data[LastIndex*2+2]< _data[LastIndex] ) ){
      if(LastIndex*2+2 <= _data.size()-1 ){
        if(_data[LastIndex*2+1] < _data[LastIndex*2+2]) {temp = LastIndex*2+1;}
        else if (_data[LastIndex*2+2] < _data[LastIndex*2+1] ) {temp = LastIndex*2+2;}
        else temp = LastIndex*2+1; 
      }
      else{ temp = LastIndex*2+1; }
     
      swap( _data[LastIndex], _data[temp] );
      LastIndex = temp;
      if(LastIndex*2+1 > _data.size()-1)break;
    }

     


    }
  
private:
   // DO NOT add or change data members
   vector<Data>   _data;
};

#endif // MY_MIN_HEAP_H
