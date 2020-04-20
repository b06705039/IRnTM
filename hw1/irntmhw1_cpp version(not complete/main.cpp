#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "porter2_stemmer.h"
using namespace std;

bool ifAlphabet (char ch)
{
  if (( ch >= 65 && ch <= 90 )||( ch >=97 && ch <= 122))
  	return true;
  return false;
}
int main(int argc, char *argv[]){
    //read file
   ifstream file;
   file.open(argv[1]);

   vector<string> token;
   string word;

  //read
    char current;
    while(file.peek()!=EOF){
        file.get(current);
        current = tolower(current);
        //only token with the space
        if( !ifAlphabet(current) ){
        	token.push_back(word);
        	word.clear();
        }
        else
        	word.push_back(current);
    }
    if( !word.empty() ){
      //if (s)
        	token.push_back(word);
        	word.clear();
    }
fstream output;
output.open("result.txt", ios::out);

//test
	for ( int i = 0; i < token.size(); i++){
   		  Porter2Stemmer::trim(token[i]);
        Porter2Stemmer::stem(token[i]);
        output << token[i] << " ";
  }
output.close();

}


